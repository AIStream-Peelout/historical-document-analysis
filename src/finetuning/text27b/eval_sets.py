# File name: eval_sets.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Pre-registered eval sets of the text27b runs (``eval_v1.json``; plan §5, §9.1 E7, §9.5).

Each set records what is evaluated (pgpids / row ids), the sha256 of every source file it was drawn
from, the prompt and scorer versions, and its counts. Sets whose inputs do not exist yet are
written with ``status: pending`` and the list of missing inputs; re-running the CLI after the
phase-2 build fills them in (they are computed from the family files, never stored by hand).

Sets
    * **E1 probe-clean** -- the 10-18 probe sets (S1–S5, C1, C2) minus ``exclusions_v1``; stock,
      Opus, Gemini and Sonnet outputs are on file (re-scored by :mod:`scorer` under v2).
    * **E2 test stratified** -- edition rows of a seeded document sample per (family, language)
      cell of ``test_indist`` (sizes of plan §5), topped up from ``heldout_vision`` when a cell is
      short (each item keeps its split); AR→EN documents carry their ``edition_rescript`` twin and
      catalogue documents their ``describe_pgp`` rows (correlated, B5). A ``new_books`` block lists
      every ``test_indist`` translate document of the L5 file (own split, decision D14).
    * **E2-read** -- every ``read`` row of the vision-benchmark hold-outs (``heldout_vision``, plus
      ``heldout`` rows carrying ``held_out_vision_benchmark`` -- the PGP-131 documents; plan B10).
    * **E3 dev-monitor** -- all dev translate rows (PGP + L5) + 40 catalogue + 20 relations, exposed
      reads removed; materialised as ``dev/dev_monitor.jsonl`` (the harness ``DEV_FILE``).
    * **E4 PGP-131** -- the 131 benchmark ids with their pgpids and keys from ``holdouts.json``.
    * **E5 excluded treatments** -- book treatment spans with a ≥300-character human analysis
      target whose document is held out / test / dev (dev = monitor only), with the non-train
      ``analyse_document`` rows (L2 file, L5 analysis rows) of the same documents attached.
    * **E6 RAG** -- test documents with leak-safe discussion pages (``rag_test_docs.json``).
    * **E7 noise bands** -- vision-benchmark hold-out documents (as E2-read), best unexposed read per
      document with coverage 0.6–1.5 and CER ≤ 0.9, binned by CER (0–.15 / .15–.3 / .3–.45 / .45–.6 / .6–.9).
    * ``thresholds`` -- the §5 / §9 bars (structured, plus the plan text verbatim when the local
      plan file is found) and the v2 re-based stock bars from ``scores_v2/summary_v2.json``.

Rows that ``excluded_rows.json`` removes with a ``split_conflict:<split>:…`` reason (their document is
held out by another family file) take that split here (:func:`split_overrides`). ``analyse_document``
rows come from ``analyse_document_v2.jsonl`` when it exists (decision D11).

The manifest names no book titles, page keys, shelfmarks, hosts or absolute paths: files are
given relative to the repository, treatments and new-book documents by a hash id.

Hub folder (decision D13; built locally, never pushed): ``hub/genizah_text27b/`` with
``<arm>/train.jsonl`` (the materialised epoch in file order without ``meta.target_raw``: same text
fingerprint and row-id order as the mixture, its own sha256), ``<arm>/mixture_manifest.json`` (the arm's
mixture-manifest section + a ``hub`` block with the copy's sha256, the harness text fingerprint and the
row-id digest), ``dev/dev_monitor.jsonl`` (also without ``meta.target_raw``), ``README.md`` (no holdings,
no quoted rows) and ``SHA256SUMS``. The pre-strip target can name documents the strips removed from the
trained text, so it stays with the build.

Public API: :func:`build_manifest`, :func:`write_manifest`, the set builders :func:`build_e1` …
:func:`build_e7` and :func:`build_e2_new_books`, :func:`thresholds`, loaders :func:`load_family_refs`,
:func:`load_new_book_refs`, :func:`load_refs_file`, :func:`load_v01_refs`, :func:`split_overrides`,
:func:`load_exposure`, :func:`doc_reads`; dev monitor :func:`dev_monitor_rows`,
:func:`dev_monitor_problems`, :func:`write_dev_monitor`, :func:`materialise_dev_monitor`; Hub
:func:`build_hub`, :func:`hub_arm`, :func:`hub_row`, :func:`write_hub_rows`, :func:`readme_text`,
:func:`write_sha256sums`, :func:`hub_problems`, :func:`stream_fingerprint`, :func:`stream_identity`,
:func:`row_ids_digest`; helpers :class:`RowRef`, :func:`file_sig`, :func:`seeded_rank`,
:func:`base_lang`, :func:`band_of`, :func:`is_vision_holdout`, :func:`strongest_split`, :func:`doc_hash`,
:func:`train_prompt_version`.

CLI::

    .venv/bin/python -m src.finetuning.text27b.eval_sets build            # → out_dir/eval_v1.json + dev/dev_monitor.jsonl
    .venv/bin/python -m src.finetuning.text27b.eval_sets build --dry-run-v01  # E2/E3/E7 from v0_1 splits
    .venv/bin/python -m src.finetuning.text27b.eval_sets hub              # → out_dir/hub/genizah_text27b/
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import scorer as _scorer
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)

EVAL_VERSION = "eval_v1"
EVAL_FILE = "eval_v1.json"
SEED = 3407
#: PGP families an eval set can draw from (literature families have their own probes).
PGP_EVAL_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "describe_pgp",
                                      "extract_relations", "analyse_document")
#: E2 document targets per (family, language cell) (plan §5); ``ALL`` = every language.
E2_TARGETS: Dict[Tuple[str, str], int] = {
    ("translate_en", "JA"): 110, ("translate_he", "JA"): 150, ("translate_en", "AR"): 79, ("translate_en", "HE"): 36,
    ("catalogue_json", "JA"): 300, ("catalogue_json", "HE"): 80, ("catalogue_json", "AR"): 60,
    ("extract_relations", "ALL"): 150}
#: Splits E2 may top up from when ``test_indist`` is short.
E2_TOP_UP: Tuple[str, ...] = ("heldout_vision",)
#: E3 sample sizes besides the dev translate rows.
E3_SAMPLES: Dict[str, int] = {"catalogue_json": 40, "extract_relations": 20}
E3_TRANSLATE: Tuple[str, ...] = ("translate_en", "translate_he")
#: Conditions whose input is a real machine read (exposure applies).
REAL_READ_CONDITIONS: Tuple[str, ...] = ("read", "read_hard", "read_partial")
E7_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "extract_relations")
E7_BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9))
E7_COVERAGE: Tuple[float, float] = (0.6, 1.5)
#: Plan §9.1 E7 document counts per band (v0_1 estimate), for comparison only.
E7_PLAN_COUNTS: Dict[str, List[int]] = {"translate_en": [12, 20, 24, 20, 22], "translate_he": [4, 13, 32, 27, 18],
                                        "catalogue_json": [52, 121, 130, 73, 66],
                                        "extract_relations": [39, 91, 105, 62, 49]}
E5_MIN_CHARS = 300
#: Splits whose vision-flagged documents form the clean read pool of E2-read and E7 (plan B10 / §9.1:
#: "test ∩ held-out-vision"). The hold-out rebuild put the vision-flagged PGP-131 documents in ``heldout``;
#: ``heldout_vision`` rows count without the flag, ``heldout`` rows only with it. Items keep their split.
VISION_SPLITS: Tuple[str, ...] = ("heldout_vision", "heldout")
#: The widened L2 file that replaces the v0_1-carried ``analyse_document.jsonl`` when present (decision D11).
ANALYSE_V2 = "analyse_document_v2"
#: The L5 file (rows of several families: translate_en / translate_he / analyse_document).
BOOK_PAIRS = "book_pairs"
#: Families of the new-book (L5) rows that E2 / E3 use.
NEW_BOOK_TRANSLATE: Tuple[str, ...] = ("translate_en", "translate_he")
#: Prefix of the ``excluded_rows.json`` reason of a row whose document another family file holds out.
SPLIT_CONFLICT = "split_conflict"
#: Strength of a split for a document's status (the strongest over its rows wins).
SPLIT_STRENGTH: Dict[str, int] = {"train": 0, "probe": 1, "dev": 2, "test_indist": 3, "heldout_vision": 4, "heldout": 5}
#: Longest rendered row the harness accepts (derive_t27 G5 applies it to the dev monitor too).
MAX_SEQ = 6144
#: The materialised E3 rows, under ``out_dir`` and in the Hub folder (derive_t27 ``DEV_FILE``).
DEV_MONITOR_FILE = "dev/dev_monitor.jsonl"


# ===================================================================================== helpers
@dataclass(frozen=True)
class RowRef:
    """The meta fields of one family row that the set builders need.

    ``doc`` is the document identity (the pgpid, else the row's ``doc_id`` / ``doc_key``); ``origin`` the
    family file the row came from; ``split_file`` the file's split when ``split`` was overridden by a
    ``split_conflict`` exclusion (None otherwise).
    """

    row_id: str
    family: str
    pgpid: str
    split: str
    condition: str
    lang_class: str
    read_model: Optional[str] = None
    input_cer: Optional[float] = None
    read_coverage: Optional[float] = None
    reader_exposed: Optional[bool] = None
    held_out_vision: bool = False
    side: Optional[str] = None
    doc: str = ""
    tokens: Optional[int] = None
    origin: str = ""
    split_file: Optional[str] = None


def sha256_file(path: Path) -> str:
    """sha256 of a file.

    :param path: File.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sig(path: Path, repo: Path) -> Dict[str, Any]:
    """Repository-relative path, existence, size and sha256 of an input file.

    :param path: File.
    :param repo: Repository root.
    :returns: ``{"path", "exists", "bytes", "sha256"}`` (only ``path`` / ``exists`` when missing).
    """
    path = Path(path)
    rel = _scorer.repo_relative(path, repo)
    if not path.exists():
        return {"path": rel, "exists": False}
    return {"path": rel, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def seeded_rank(seed: int, *parts: Any) -> str:
    """Deterministic sort key (sha1 of the seed and parts) for seeded, addition-stable sampling.

    :param seed: Seed.
    :param parts: Key parts (family, pgpid ...).
    :returns: Hex digest.
    """
    return hashlib.sha1("|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")).hexdigest()


def base_lang(lang_class: Optional[str]) -> str:
    """Language cell of a lang_class (``JA?`` → ``JA``; None → ``UNKNOWN``).

    :param lang_class: PGP language class.
    :returns: The cell label.
    """
    return (lang_class or "UNKNOWN").rstrip("?") or "UNKNOWN"


def band_of(cer: Optional[float], bands: Sequence[Tuple[float, float]] = E7_BANDS) -> Optional[str]:
    """E7 band label of a CER (intervals ``(lo, hi]``; a CER of 0 falls in the first band).

    :param cer: CER.
    :param bands: Band edges.
    :returns: ``"lo-hi"`` or None outside every band.
    """
    if cer is None:
        return None
    for lo, hi in bands:
        if lo < cer <= hi or (lo == 0.0 and cer == 0.0):
            return f"{lo:g}-{hi:g}"
    return None


def train_prompt_version() -> str:
    """Version tag of the training-format prompts (hash of the system prompt, instructions and notes).

    :returns: ``t27-prompt-<sha12>``.
    """
    blob = json.dumps({"system": _rows.SYSTEM, "instruction": _rows.INSTRUCTION, "cond_note": _rows.COND_NOTE},
                      ensure_ascii=False, sort_keys=True)
    return "t27-prompt-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _opt_float(v: Any) -> Optional[float]:
    """Float or None.

    :param v: Value.
    :returns: The float, or None for None / non-numeric values.
    """
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def ref_from_row(row: Mapping[str, Any], family: Optional[str] = None, origin: str = "",
                 split_override: Optional[str] = None) -> RowRef:
    """:class:`RowRef` of a family row (v0_2 schema; v0_1 rows work with ``family`` given).

    :param row: Row with ``meta``.
    :param family: Family override (v0_1 rows carry only ``task``).
    :param origin: Name of the family file the row came from.
    :param split_override: Split to use instead of the row's own (a ``split_conflict`` exclusion), or None.
    :returns: The reference.
    """
    m = row.get("meta") or {}
    fam = family or m.get("family") or m.get("task")
    pid = str(m.get("pgpid") or "")
    rid = m.get("row_id") or _rows.row_id(fam, pid, m.get("condition"), m.get("read_model"), m.get("side"))
    exposed = m.get("reader_exposed")
    split = str(m.get("split"))
    tokens = m.get("tokens")
    return RowRef(row_id=str(rid), family=str(fam), pgpid=pid,
                  split=split_override if split_override and split_override != split else split,
                  condition=str(m.get("condition")), lang_class=str(m.get("lang_class") or "UNKNOWN"),
                  read_model=m.get("read_model"), input_cer=_opt_float(m.get("input_cer")),
                  read_coverage=_opt_float(m.get("read_coverage")),
                  reader_exposed=None if exposed is None else bool(exposed),
                  held_out_vision=bool(m.get("held_out_vision_benchmark")), side=m.get("side"),
                  doc=pid or str(m.get("doc_id") or m.get("doc_key") or ""),
                  tokens=int(tokens) if isinstance(tokens, (int, float)) else None, origin=origin,
                  split_file=split if split_override and split_override != split else None)


def doc_hash(doc: str) -> str:
    """Opaque id of a document identity that may name a book (no holdings in the manifest).

    :param doc: Document identity (pgpid or ``book:<source>:<doc>`` style).
    :returns: The pgpid unchanged when ``doc`` is all digits, else ``d`` + 15 hex of its sha1.
    """
    return doc if doc.isdigit() else "d" + hashlib.sha1(doc.encode("utf-8")).hexdigest()[:15]


def split_overrides(excluded_rows: Optional[Path]) -> Dict[str, str]:
    """Rows whose document another family file holds out, with the split they take in the evals.

    The contamination module excludes such TRAIN rows from the mixtures with the reason
    ``split_conflict:<split>:<identity>``; the eval sets treat the row as that split.

    :param excluded_rows: ``excluded_rows.json`` (``{row_id: reason or [reasons]}``; None / missing = none).
    :returns: ``{row_id: split}``.
    """
    if not excluded_rows or not Path(excluded_rows).exists():
        return {}
    out: Dict[str, str] = {}
    for rid, reasons in _read_json(excluded_rows).items():
        for reason in [reasons] if isinstance(reasons, str) else list(reasons or []):
            parts = str(reason).split(":")
            if parts[0] == SPLIT_CONFLICT and len(parts) > 1 and parts[1] in SPLIT_STRENGTH:
                if rid not in out or SPLIT_STRENGTH[parts[1]] > SPLIT_STRENGTH[out[rid]]:
                    out[rid] = parts[1]
    return out


def family_file(families_dir: Path, family: str) -> Path:
    """File a family's rows are read from (``analyse_document`` → the L2 file when present, decision D11).

    :param families_dir: ``out_dir/families``.
    :param family: Family.
    :returns: The path (it may not exist).
    """
    v2 = Path(families_dir) / f"{ANALYSE_V2}.jsonl"
    if family == "analyse_document" and v2.exists():
        return v2
    return Path(families_dir) / f"{family}.jsonl"


def load_refs_file(path: Path, overrides: Optional[Mapping[str, str]] = None,
                   family: Optional[str] = None) -> Dict[str, List[RowRef]]:
    """Row references of one family file, grouped by ``meta.family`` (streaming; texts are not kept).

    :param path: JSONL file.
    :param overrides: :func:`split_overrides` output.
    :param family: Family override for every row (None = each row's own).
    :returns: ``{family: [RowRef]}``.
    """
    out: Dict[str, List[RowRef]] = defaultdict(list)
    origin = Path(path).stem
    for r in _rows.read_jsonl(path):
        rid = str((r.get("meta") or {}).get("row_id") or "")
        ref = ref_from_row(r, family, origin, (overrides or {}).get(rid))
        out[ref.family].append(ref)
    return dict(out)


def load_family_refs(families_dir: Path, families: Sequence[str] = PGP_EVAL_FAMILIES,
                     overrides: Optional[Mapping[str, str]] = None) -> Dict[str, List[RowRef]]:
    """Row references of the family files that exist (``analyse_document`` from the L2 file when present).

    :param families_dir: ``out_dir/families``.
    :param families: Families to read.
    :param overrides: :func:`split_overrides` output.
    :returns: ``{family: [RowRef]}`` for the files found.
    """
    out: Dict[str, List[RowRef]] = {}
    for fam in families:
        f = family_file(families_dir, fam)
        if f.exists():
            out[fam] = load_refs_file(f, overrides, fam).get(fam, [])
    return out


def load_new_book_refs(families_dir: Path, overrides: Optional[Mapping[str, str]] = None) -> Dict[str, List[RowRef]]:
    """Row references of the L5 file (``book_pairs.jsonl``), grouped by family; empty when it is absent.

    :param families_dir: ``out_dir/families``.
    :param overrides: :func:`split_overrides` output.
    :returns: ``{family: [RowRef]}``.
    """
    f = Path(families_dir) / f"{BOOK_PAIRS}.jsonl"
    return load_refs_file(f, overrides) if f.exists() else {}


def strongest_split(refs: Iterable[RowRef]) -> Dict[str, str]:
    """Strongest split per pgpid over rows (``heldout`` > ``heldout_vision`` > ``test_indist`` > ``dev`` > ``train``).

    :param refs: Row references.
    :returns: ``{pgpid: split}``.
    """
    out: Dict[str, str] = {}
    for r in refs:
        if r.pgpid and (r.pgpid not in out or SPLIT_STRENGTH.get(r.split, 0) > SPLIT_STRENGTH.get(out[r.pgpid], 0)):
            out[r.pgpid] = r.split
    return out


def is_vision_holdout(ref: RowRef) -> bool:
    """Whether a row belongs to the clean read pool (``heldout_vision``, or ``heldout`` with the vision flag).

    :param ref: Row reference.
    :returns: True for rows E2-read / E7 may use.
    """
    return ref.split == "heldout_vision" or (ref.split in VISION_SPLITS and ref.held_out_vision)


def v01_split(split: str, held_out_vision: bool) -> str:
    """v0_2-style split of a v0_1 row for the dry run (test → heldout_vision / test_indist).

    :param split: v0_1 split.
    :param held_out_vision: The row's ``held_out_vision_benchmark`` flag.
    :returns: The mapped split.
    """
    if split != "test":
        return split
    return "heldout_vision" if held_out_vision else "test_indist"


def load_v01_refs(v01_dir: Path, families: Sequence[str] = PGP_EVAL_FAMILIES) -> Dict[str, List[RowRef]]:
    """Row references of the v0_1 build mapped to v0_2 splits (dry run only; no H rebuild, no exposure).

    :param v01_dir: Directory of ``train/dev/test.jsonl``.
    :param families: Families to keep.
    :returns: ``{family: [RowRef]}``.
    """
    out: Dict[str, List[RowRef]] = defaultdict(list)
    for split in ("dev", "test"):
        f = Path(v01_dir) / f"{split}.jsonl"
        if not f.exists():
            continue
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            if m.get("task") not in families:
                continue
            ref = ref_from_row(r, m["task"])
            out[ref.family].append(RowRef(**{**asdict(ref), "split": v01_split(split, ref.held_out_vision)}))
    return dict(out)


def _sigs(paths_: Iterable[Path], repo: Path) -> List[Dict[str, Any]]:
    """File signatures of several inputs.

    :param paths_: Files.
    :param repo: Repository root.
    :returns: :func:`file_sig` records.
    """
    return [file_sig(p, repo) for p in paths_]


def _pending(definition: str, requires: Sequence[str]) -> Dict[str, Any]:
    """A set whose inputs are missing.

    :param definition: Set definition.
    :param requires: Missing inputs (repository-relative).
    :returns: ``{"status": "pending", "definition", "requires"}``.
    """
    return {"status": "pending", "definition": definition, "requires": list(requires)}


# ========================================================================================== E1
E1_DEF = ("10-18 probe sets S1 JA→EN, S2 JA→EN from reads, S3 JA→HE, S4 AR→EN (+ translit / translit_plain twins), "
          "S5 HE→EN, C1 catalogue, C2 catalogue from reads, minus the (set, pgpid) rows of exclusions_v1. Stored "
          "outputs: stock 27B, Opus (S1/S2/C1/C2), Gemini, Sonnet, 35B-A3B; fine-tunes are run with the training "
          "prompt and the stock model under both prompts.")


def build_e1(samples: Mapping[str, Any], exclusions: Mapping[Tuple[str, str], List[str]]) -> Dict[str, Any]:
    """E1 probe-clean: the probe sets minus the exclusions.

    :param samples: Parsed ``samples.json``.
    :param exclusions: :func:`scorer.load_exclusions` output.
    :returns: The set dict (``items`` per probe set: conditions, kept pgpids, excluded pgpids with reasons).
    """
    items, counts = {}, {}
    for name, spec in samples["sets"].items():
        pids = [str(e["pgpid"]) for e in spec["entries"]]
        kept = [p for p in pids if (name, p) not in exclusions]
        items[name] = {"task": spec.get("task"), "target": spec.get("target"),
                       "conditions": list(spec.get("conditions") or []), "pgpids": kept,
                       "excluded": {p: exclusions[(name, p)] for p in pids if (name, p) in exclusions}}
        counts[name] = {"kept": len(kept), "excluded": len(pids) - len(kept)}
    return {"status": "ready", "definition": E1_DEF, "counts": counts, "items": items}


# ========================================================================================== E2
E2_DEF = ("Edition rows of a seeded document sample per (family, language) cell of test_indist, sizes of plan §5; "
          "cells short of their size are topped up from heldout_vision (items keep their split, so test_indist-only "
          "results can be reported). AR→EN documents carry their edition_rescript twin rows; catalogue documents "
          "their describe_pgp edition rows (B5: correlated with the catalogue summary).")


def _docs_by_cell(refs: Sequence[RowRef], split: str, condition: str = "edition") -> Dict[str, Dict[str, List[RowRef]]]:
    """Rows of one split and condition grouped by language cell and document.

    :param refs: Rows of one family.
    :param split: Split.
    :param condition: Condition.
    :returns: ``{cell: {pgpid: [RowRef]}}``.
    """
    out: Dict[str, Dict[str, List[RowRef]]] = defaultdict(lambda: defaultdict(list))
    for r in refs:
        if r.split == split and r.condition == condition and r.pgpid:
            out[base_lang(r.lang_class)][r.pgpid].append(r)
    return out


def build_e2(refs: Mapping[str, Sequence[RowRef]], targets: Mapping[Tuple[str, str], int] = E2_TARGETS,
             seed: int = SEED, top_up: Sequence[str] = E2_TOP_UP) -> Dict[str, Any]:
    """E2 test stratified sample.

    :param refs: Family row references.
    :param targets: Document targets per (family, cell).
    :param seed: Sampling seed.
    :param top_up: Splits used to fill short cells, in order.
    :returns: The set dict (``cells`` with counts, ``items`` per cell).
    """
    cells, items = {}, {}
    for (fam, cell), target in targets.items():
        fam_refs = refs.get(fam) or []
        chosen: List[Tuple[str, List[RowRef]]] = []
        available: Dict[str, int] = {}
        for split in ("test_indist",) + tuple(top_up):
            by_cell = _docs_by_cell(fam_refs, split)
            docs = {}
            for c, d in by_cell.items():
                if cell == "ALL" or c == cell:
                    docs.update(d)
            available[split] = len(docs)
            taken = {p for p, _ in chosen}
            for pid in sorted(docs, key=lambda p: seeded_rank(seed, fam, p)):
                if len(chosen) >= target:
                    break
                if pid not in taken:
                    chosen.append((pid, docs[pid]))
        twins: Dict[str, List[str]] = {}
        pids = {p for p, _ in chosen}
        if fam == "translate_en" and cell == "AR":
            for r in fam_refs:
                if r.condition == "edition_rescript" and r.pgpid in pids:
                    twins.setdefault(r.pgpid, []).append(r.row_id)
        if fam == "catalogue_json":
            for r in refs.get("describe_pgp") or []:
                if r.condition == "edition" and r.pgpid in pids:
                    twins.setdefault(r.pgpid, []).append(r.row_id)
        key = f"{fam}:{cell}"
        by_split = Counter(rs[0].split for _, rs in chosen)
        cells[key] = {"target": target, "docs": len(chosen), "rows": sum(len(rs) for _, rs in chosen),
                      "by_split": dict(by_split), "available": available,
                      "shortfall": max(0, target - len(chosen)), "twin_rows": sum(len(v) for v in twins.values())}
        items[key] = [{"pgpid": pid, "split": rs[0].split, "row_ids": [r.row_id for r in rs],
                       **({"twin_row_ids": twins[pid]} if pid in twins else {})} for pid, rs in chosen]
    return {"status": "ready", "definition": E2_DEF, "seed": seed, "cells": cells, "items": items}


E2NB_DEF = ("New-book documents (L5, decision D14: their own sha1 80/10/10 split): every test_indist translate "
            "document with all its rows (edition, edition_rescript). Rows whose document another family file holds "
            "out (split_conflict exclusions) take that split. Document ids are hashed (no holdings).")


def build_e2_new_books(new_book_refs: Mapping[str, Sequence[RowRef]],
                       families: Sequence[str] = NEW_BOOK_TRANSLATE,
                       splits: Sequence[str] = ("test_indist",)) -> Dict[str, Any]:
    """E2 new-book block: the L5 translate documents of the test split (all of them; no plan size).

    :param new_book_refs: :func:`load_new_book_refs` output.
    :param families: Translate families.
    :param splits: Splits to take.
    :returns: ``{"definition", "cells", "items"}``; ``heldout_available`` counts the H-matched L5 documents.
    """
    cells, items = {}, {}
    for fam in families:
        docs: Dict[str, List[RowRef]] = defaultdict(list)
        held: Set[str] = set()
        for r in new_book_refs.get(fam) or []:
            if r.split in splits and r.doc:
                docs[r.doc].append(r)
            elif r.split == "heldout" and r.doc:
                held.add(r.doc)
        key = f"{fam}:new_books"
        chosen = sorted(docs, key=lambda d: seeded_rank(SEED, fam, d))
        cells[key] = {"docs": len(chosen), "rows": sum(len(docs[d]) for d in chosen),
                      "by_split": dict(Counter(docs[d][0].split for d in chosen)),
                      "by_lang": dict(Counter(base_lang(docs[d][0].lang_class) for d in chosen)),
                      "split_overridden_docs": sum(1 for d in chosen if any(r.split_file for r in docs[d])),
                      "heldout_available": len(held)}
        items[key] = [{"doc": doc_hash(d), "split": docs[d][0].split, "lang": base_lang(docs[d][0].lang_class),
                       "row_ids": [r.row_id for r in docs[d]],
                       **({"split_file": docs[d][0].split_file} if any(r.split_file for r in docs[d]) else {})}
                      for d in chosen]
    return {"definition": E2NB_DEF, "cells": cells, "items": items}


E2R_DEF = ("Every `read` row of the vision-benchmark hold-outs (split heldout_vision, plus split heldout rows that "
           "carry held_out_vision_benchmark, i.e. the PGP-131 documents; plan B10 'test ∩ held-out-vision'): the "
           "readers never trained on these pages. Items keep their split; rows tagged reader_exposed=True are "
           "dropped and counted.")


def build_e2_read(refs: Mapping[str, Sequence[RowRef]]) -> Dict[str, Any]:
    """E2-read: read rows of the vision-benchmark hold-outs (:func:`is_vision_holdout`).

    :param refs: Family row references.
    :returns: The set dict (counts per family × language and per family × split, ``items`` per family).
    """
    items: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    by_split: Dict[str, Dict[str, int]] = {}
    dropped = Counter()
    for fam, rs in refs.items():
        sel = [r for r in rs if is_vision_holdout(r) and r.condition == "read"]
        dropped[fam] = sum(1 for r in sel if r.reader_exposed)
        sel = [r for r in sel if not r.reader_exposed]
        if not sel:
            continue
        items[fam] = [{"row_id": r.row_id, "pgpid": r.pgpid, "split": r.split, "read_model": r.read_model,
                       "input_cer": r.input_cer, "band": band_of(r.input_cer)} for r in sel]
        counts[fam] = dict(Counter(base_lang(r.lang_class) for r in sel))
        by_split[fam] = dict(Counter(r.split for r in sel))
    return {"status": "ready", "definition": E2R_DEF, "splits": list(VISION_SPLITS), "counts": counts,
            "by_split": by_split, "exposed_dropped": dict(dropped), "items": items}


# ========================================================================================== E3
E3_DEF = ("Dev monitor for checkpoint choice only: all dev translate_en / translate_he rows (PGP and the L5 new-book "
          "dev documents) + a seeded 40 catalogue_json and 20 extract_relations dev rows; real-read rows tagged "
          "reader_exposed=True removed; rows longer than the harness max_seq and rows whose split was overridden "
          "are skipped (counted). Materialised as dev/dev_monitor.jsonl (the harness DEV_FILE).")


def build_e3(refs: Mapping[str, Sequence[RowRef]], seed: int = SEED,
             samples: Mapping[str, int] = E3_SAMPLES,
             new_book_refs: Optional[Mapping[str, Sequence[RowRef]]] = None,
             max_tokens: Optional[int] = MAX_SEQ) -> Dict[str, Any]:
    """E3 dev-monitor.

    :param refs: Family row references.
    :param seed: Sampling seed.
    :param samples: Sample sizes of the non-translate families.
    :param new_book_refs: L5 row references (:func:`load_new_book_refs`); their dev translate rows join the
        translate lists.
    :param max_tokens: Longest rendered row (``meta.tokens``) kept; None = no limit.
    :returns: The set dict (``row_ids`` per family, counts per family and origin, exposure bookkeeping).
    """
    items: Dict[str, List[str]] = {}
    removed, unknown, too_long, overridden = Counter(), Counter(), Counter(), Counter()
    origins: Dict[str, Counter] = defaultdict(Counter)

    def usable(r: RowRef) -> bool:
        """Dev row that is not an exposed real read, fits the harness and keeps its own split.

        :param r: Row reference.
        :returns: True when usable.
        """
        if r.split != "dev":
            return False
        if r.split_file is not None:
            overridden[r.family] += 1
            return False
        if r.condition in REAL_READ_CONDITIONS:
            if r.reader_exposed:
                removed[r.family] += 1
                return False
            if r.reader_exposed is None:
                unknown[r.family] += 1
        if max_tokens is not None and r.tokens is not None and r.tokens > max_tokens:
            too_long[r.family] += 1
            return False
        return True

    for fam in E3_TRANSLATE:
        sel = [r for r in list(refs.get(fam) or []) + list((new_book_refs or {}).get(fam) or []) if usable(r)]
        items[fam] = [r.row_id for r in sel]
        origins[fam].update(r.origin or fam for r in sel)
    for fam, k in samples.items():
        pool = [r for r in refs.get(fam) or [] if usable(r)]
        sel = sorted(pool, key=lambda r: seeded_rank(seed, fam, r.row_id))[:k]
        items[fam] = [r.row_id for r in sel]
        origins[fam].update(r.origin or fam for r in sel)
    counts = {f: len(v) for f, v in items.items()}
    return {"status": "ready" if not sum(unknown.values()) else "provisional", "definition": E3_DEF, "seed": seed,
            "counts": counts, "rows": sum(counts.values()), "origins": {f: dict(c) for f, c in origins.items()},
            "max_tokens": max_tokens, "exposed_removed": dict(removed), "exposure_unknown_read_rows": dict(unknown),
            "over_max_tokens": dict(too_long), "split_overridden": dict(overridden), "row_ids": items}


# ========================================================================================== E4
E4_DEF = ("PGP-131 documents (benchmark canonical ids → pgpids via the pgpid map, the merged corpus and the "
          "shelfmark key; holdouts.json), evaluated as describe-from-reads against the PGP description.")


def _base_reason(reason: str) -> str:
    """Reason without its ``group:`` / ``key:`` prefixes.

    :param reason: Stored reason.
    :returns: Base reason.
    """
    return reason.split(":")[-1]


def build_e4(benchmark_ids: Sequence[str], holdouts: Mapping[str, Any], merged_map: Mapping[str, Any],
             described: Optional[Set[str]] = None) -> Dict[str, Any]:
    """E4 PGP-131 from a parsed ``holdouts.json``.

    :param benchmark_ids: The 131 canonical ids.
    :param holdouts: Parsed ``holdouts.json`` (``benchmark``, ``canonical``, ``meta``).
    :param merged_map: ``merged_pgpid_map.json`` (canonical id → ``{"pgpids": [...]}``).
    :param described: pgpids with a non-empty PGP description (None = unknown).
    :returns: The set dict (one item per canonical id, pgpid totals).
    """
    bench = {str(k): list(v) for k, v in (holdouts.get("benchmark") or {}).items()}
    pgp131 = {p for p, rs in bench.items() if any(_base_reason(r) == "pgp131" for r in rs)}
    unmapped = {r["canonical_id"]: r for r in (holdouts.get("meta") or {}).get("pgp131_unmapped_resolution") or []}
    items, without = [], []
    for cid in benchmark_ids:
        pids = {str(p) for p in (merged_map.get(cid) or {}).get("pgpids", [])}
        if cid in unmapped:
            pids |= {str(p) for p in unmapped[cid].get("pgpids") or []}
        pids = sorted(pids & set(bench) if bench else pids, key=lambda s: int(s) if s.isdigit() else 0)
        item = {"canonical_id": cid, "key": _sm.canonical_to_key(cid), "pgpids": pids,
                "via": "pgpid_map" if (merged_map.get(cid) or {}).get("pgpids") else "key_or_corpus"}
        if described is not None:
            item["described_pgpids"] = [p for p in pids if p in described]
        if not pids:
            without.append(cid)
        items.append(item)
    all_pids = sorted({p for it in items for p in it["pgpids"]} | pgp131, key=lambda s: int(s) if s.isdigit() else 0)
    counts = {"canonical_ids": len(benchmark_ids), "pgpids": len(all_pids), "canonical_without_pgpid": len(without),
              "via_key_or_corpus": sum(1 for it in items if it["via"] == "key_or_corpus" and it["pgpids"])}
    if described is not None:
        counts["pgpids_with_description"] = sum(1 for p in all_pids if p in described)
    return {"status": "ready", "definition": E4_DEF, "counts": counts, "pgpids": all_pids,
            "canonical_without_pgpid": without, "items": items}


# ========================================================================================== E5
E5_DEF = ("Book treatment spans (boundary-clean, PGP id, introduction + notes ≥ 300 characters) whose document is "
          "a hold-out (benchmark tier), test or dev document: never trained on; dev items are for monitoring only. "
          "Items carry a hash id; the span itself is looked up in document_treatments.jsonl (sha256 recorded).")


def treatment_id(rec: Mapping[str, Any]) -> str:
    """Stable hash id of a treatment span (no book names in the manifest).

    :param rec: ``document_treatments.jsonl`` record.
    :returns: 16-hex id.
    """
    raw = "|".join(str(rec.get(k, "")) for k in ("book", "part", "doc_label")) + "|" + ",".join(rec.get("pages") or [])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def treatment_pgpids(rec: Mapping[str, Any], merged_map: Mapping[str, Any]) -> List[str]:
    """pgpids of a treatment: its own plus those of its merged canonical ids.

    :param rec: Treatment record.
    :param merged_map: Canonical id → ``{"pgpids"}``.
    :returns: Sorted pgpids.
    """
    pids = {str(p) for p in rec.get("pgpids") or []}
    for cid in rec.get("merged_doc_ids") or []:
        pids |= {str(p) for p in (merged_map.get(cid) or {}).get("pgpids", [])}
    return sorted(pids, key=lambda s: int(s) if s.isdigit() else 0)


def treatment_status(pgpids: Sequence[str], holdouts: Mapping[str, Any],
                     split_of: Optional[Mapping[str, str]] = None) -> str:
    """Status of a treatment's documents: ``heldout``, ``test``, ``dev``, ``train`` or ``not_in_build``.

    :param pgpids: The treatment's pgpids.
    :param holdouts: Parsed ``holdouts.json`` (benchmark tier → heldout; eval tier test / dev reasons).
    :param split_of: v0_2 split per pgpid from the family files (overrides the eval tier when given).
    :returns: The status (the strongest over the pgpids).
    """
    bench = holdouts.get("benchmark") or {}
    evald = holdouts.get("eval_docs") or {}
    found: Set[str] = set()
    for p in pgpids:
        if p in bench:
            found.add("heldout")
            continue
        sp = (split_of or {}).get(p)
        if sp:
            found.add({"test_indist": "test", "heldout_vision": "heldout", "heldout": "heldout"}.get(sp, sp))
            continue
        bases = {_base_reason(r) for r in evald.get(p, [])}
        if "test" in bases:
            found.add("test")
        elif "dev" in bases:
            found.add("dev")
        elif split_of is None:
            found.add("train")
    for status in ("heldout", "test", "dev", "train"):
        if status in found:
            return status
    return "not_in_build"


def analysis_eval_rows(analysis_refs: Iterable[RowRef]) -> Dict[str, Dict[str, Any]]:
    """Non-train ``analyse_document`` rows (L2 / L5) grouped by document.

    :param analysis_refs: Row references of the analysis files.
    :returns: ``{doc: {"pgpid", "split" (strongest), "origin", "row_ids"}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in analysis_refs:
        if r.split == "train" or not r.doc:
            continue
        d = out.setdefault(r.doc, {"pgpid": r.pgpid, "split": r.split, "origin": r.origin, "row_ids": []})
        d["row_ids"].append(r.row_id)
        if SPLIT_STRENGTH.get(r.split, 0) > SPLIT_STRENGTH.get(d["split"], 0):
            d["split"] = r.split
    return out


def build_e5(treatments: Iterable[Mapping[str, Any]], merged_map: Mapping[str, Any], holdouts: Mapping[str, Any],
             split_of: Optional[Mapping[str, str]] = None, min_chars: int = E5_MIN_CHARS,
             analysis_refs: Optional[Iterable[RowRef]] = None) -> Dict[str, Any]:
    """E5 excluded treatments, with the ready-built analysis rows of the same documents when they exist.

    :param treatments: ``document_treatments.jsonl`` records.
    :param merged_map: ``merged_pgpid_map.json``.
    :param holdouts: Parsed ``holdouts.json``.
    :param split_of: v0_2 split per pgpid (optional).
    :param min_chars: Minimum introduction + notes characters of a target.
    :param analysis_refs: ``analyse_document`` row references (L2 file + L5 analysis rows); their non-train
        rows are attached to the items sharing a pgpid (``row_ids``) and listed per document in
        ``analysis_rows`` (``in_item`` False for documents no treatment item covers).
    :returns: The set dict (items with hash id, pgpids, status, chars, ``monitor_only``, optional ``row_ids``).
    """
    items, counts = [], Counter()
    for rec in treatments:
        pids = [str(p) for p in rec.get("pgpids") or []]
        if not pids or not rec.get("boundary_clean") or (rec.get("intro_chars", 0) + rec.get("notes_chars", 0)) < min_chars:
            continue
        allp = treatment_pgpids(rec, merged_map)
        status = treatment_status(allp, holdouts, split_of)
        counts[status] += 1
        if status in ("heldout", "test", "dev"):
            items.append({"id": treatment_id(rec), "pgpids": allp, "status": status, "monitor_only": status == "dev",
                          "target_chars": rec.get("intro_chars", 0) + rec.get("notes_chars", 0),
                          "has_pgp_digital_edition": bool(rec.get("has_pgp_digital_edition")),
                          "has_our_read": bool(rec.get("has_our_read"))})
    out: Dict[str, Any] = {"status": "ready", "definition": E5_DEF,
                           "counts": {"targets_by_status": dict(counts), "excluded": len(items)}, "items": items}
    if analysis_refs is not None:
        docs = analysis_eval_rows(analysis_refs)
        by_pid: Dict[str, List[str]] = defaultdict(list)
        for d in docs.values():
            if d["pgpid"]:
                by_pid[d["pgpid"]].extend(d["row_ids"])
        covered: Set[str] = set()
        for it in items:
            rids = sorted({rid for p in it["pgpids"] for rid in by_pid.get(p, [])})
            if rids:
                it["row_ids"] = rids
                covered.update(p for p in it["pgpids"] if p in by_pid)
        out["analysis_rows"] = [{"doc": doc_hash(doc), "pgpid": d["pgpid"] or None, "split": d["split"],
                                 "origin": d["origin"], "row_ids": d["row_ids"], "in_item": d["pgpid"] in covered}
                                for doc, d in sorted(docs.items(), key=lambda kv: (kv[1]["split"], doc_hash(kv[0])))]
        out["counts"].update({"items_with_rows": sum(1 for it in items if it.get("row_ids")),
                              "analysis_docs_by_split": dict(Counter(d["split"] for d in docs.values())),
                              "analysis_rows_by_split": dict(Counter(d["split"] for d in docs.values()
                                                                     for _ in d["row_ids"])),
                              "analysis_docs_outside_items": sum(1 for d in docs.values() if d["pgpid"] not in covered)})
    return out


# ========================================================================================== E6
E6_DEF = ("Test documents with discussion pages in the literature, per family the ones with at least one leak-safe "
          "page (< 5 % of the target's 8-grams); paired with / without the literature block (describe, catalogue, "
          "translate_en). Pages are listed in rag_test_docs.json (sha256 recorded).")


def build_e6(rag_test_docs: Mapping[str, Any]) -> Dict[str, Any]:
    """E6 RAG test documents from module E's ``rag_test_docs.json``.

    :param rag_test_docs: Parsed file (``docs`` with ``pgpid``, ``families``, ``leak_safe_pages``).
    :returns: The set dict (pgpid → leak-safe page count per family).
    """
    items, counts = [], Counter()
    for d in rag_test_docs.get("docs") or []:
        safe = {f: len(v) for f, v in (d.get("leak_safe_pages") or {}).items() if v}
        if not safe:
            continue
        for f in safe:
            counts[f] += 1
        items.append({"pgpid": str(d.get("pgpid")), "held_out_vision_benchmark": bool(d.get("held_out_vision_benchmark")),
                      "leak_safe_pages": safe})
    return {"status": "ready", "definition": E6_DEF, "leak_max": rag_test_docs.get("leak_max"),
            "counts": {"docs": len(items), "docs_per_family": dict(counts)}, "items": items}


# ========================================================================================== E7
E7_DEF = ("Noise bands: vision-benchmark hold-out documents (split heldout_vision, plus split heldout with "
          "held_out_vision_benchmark: plan §9.1 'test ∩ held-out-vision ∩ reader-unexposed') with an edition row in "
          "the family; per document the lowest-CER read that is not reader-exposed, non-empty, coverage 0.6–1.5 and "
          "CER ≤ 0.9 (CER / coverage recomputed with textnorm on clean_read(read) vs strip_latin_lines(edition)); "
          "Δ = score(edition) − score(read), same target, per band. Items without read_row_id (reads outside the "
          "builder's read gate) need their read input built at eval time.")


def load_exposure(path: Optional[Path]) -> Optional[Dict[Tuple[str, str], str]]:
    """Strongest exposure level per (pgpid, read_model) from ``exposure.jsonl``.

    :param path: ``exposure.jsonl`` (None / missing = unknown).
    :returns: ``{(pgpid, read_model): level}`` or None.
    """
    if not path or not Path(path).exists():
        return None
    order = {lvl: i for i, lvl in enumerate(reversed(_rows.EXPOSURE_LEVELS))}
    out: Dict[Tuple[str, str], str] = {}
    for rec in _rows.read_jsonl(path):
        k = (str(rec.get("pgpid")), str(rec.get("read_model")))
        lvl = rec.get("read_level") or rec.get("level") or "none"
        if k not in out or order.get(lvl, 0) > order.get(out[k], 0):
            out[k] = lvl
    return out


def doc_reads(pairs_path: Path, pgpids: Set[str],
              exposure: Optional[Mapping[Tuple[str, str], str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Reads of the given documents with recomputed CER / coverage and exposure.

    :param pairs_path: ``pgp_parallel`` JSONL.
    :param pgpids: Documents to read.
    :param exposure: :func:`load_exposure` output (None = unknown).
    :returns: ``{pgpid: [{"read_model", "cer", "coverage", "empty", "exposed"}]}``.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in _rows.read_jsonl(pairs_path):
        pid = str(row.get("pgpid"))
        if pid not in pgpids:
            continue
        ed, _ = tn.strip_latin_lines(row.get("edition_text") or "")
        reads = []
        for rd in row.get("reads") or []:
            txt, _ = tn.clean_read(rd.get("text") or "")
            empty = not tn.normalize_for_cer(txt) or not tn.normalize_for_cer(ed)
            lvl = None if exposure is None else exposure.get((pid, str(rd.get("model"))), "none")
            reads.append({"read_model": rd.get("model"), "empty": empty,
                          "cer": None if empty else round(tn.cer(txt, ed), 4),
                          "coverage": None if empty else round(tn.coverage(txt, ed), 4),
                          "exposed": None if lvl is None else lvl != "none"})
        out[pid] = reads
    return out


def best_read(reads: Sequence[Mapping[str, Any]], coverage: Tuple[float, float] = E7_COVERAGE,
              max_cer: float = E7_BANDS[-1][1]) -> Optional[Mapping[str, Any]]:
    """Lowest-CER eligible read (non-empty, not exposed, coverage in range, CER ≤ ``max_cer``).

    :param reads: :func:`doc_reads` entries of one document.
    :param coverage: Coverage interval (inclusive).
    :param max_cer: CER ceiling.
    :returns: The read or None.
    """
    ok = [r for r in reads if not r["empty"] and not r.get("exposed") and r["coverage"] is not None
          and coverage[0] <= r["coverage"] <= coverage[1] and r["cer"] <= max_cer]
    return min(ok, key=lambda r: (r["cer"], str(r["read_model"]))) if ok else None


def build_e7(refs: Mapping[str, Sequence[RowRef]], reads: Mapping[str, Sequence[Mapping[str, Any]]],
             exposure_known: bool, families: Sequence[str] = E7_FAMILIES) -> Dict[str, Any]:
    """E7 noise bands.

    :param refs: Family row references.
    :param reads: :func:`doc_reads` output for the heldout_vision documents.
    :param exposure_known: Whether exposure was available (else the set is provisional).
    :param families: Families to band.
    :returns: The set dict (per family: band counts vs the plan estimate, items).
    """
    labels = [band_of(hi) for _, hi in E7_BANDS]
    fams: Dict[str, Any] = {}
    for fam in families:
        eds: Dict[str, List[str]] = defaultdict(list)
        split_of: Dict[str, str] = {}
        read_rows: Dict[Tuple[str, Optional[str]], str] = {}
        for r in refs.get(fam) or []:
            if not is_vision_holdout(r):
                continue
            if r.condition == "edition":
                eds[r.pgpid].append(r.row_id)
                split_of[r.pgpid] = r.split
            elif r.condition == "read":
                read_rows[(r.pgpid, r.read_model)] = r.row_id
        items, bands = [], Counter()
        for pid in sorted(eds, key=lambda s: int(s) if s.isdigit() else 0):
            b = best_read(reads.get(pid) or [])
            if b is None:
                continue
            band = band_of(b["cer"])
            bands[band] += 1
            items.append({"pgpid": pid, "split": split_of[pid], "read_model": b["read_model"], "cer": b["cer"],
                          "coverage": b["coverage"], "band": band, "edition_row_ids": eds[pid],
                          "read_row_id": read_rows.get((pid, b["read_model"]))})
        fams[fam] = {"docs_with_edition": len(eds), "docs_banded": len(items),
                     "bands": {lab: bands.get(lab, 0) for lab in labels},
                     "plan_estimate": dict(zip(labels, E7_PLAN_COUNTS.get(fam, []))),
                     "by_split": dict(Counter(it["split"] for it in items)),
                     "items_without_read_row": sum(1 for it in items if it["read_row_id"] is None),
                     "items": items}
    return {"status": "ready" if exposure_known else "provisional", "definition": E7_DEF,
            "splits": list(VISION_SPLITS), "coverage": list(E7_COVERAGE), "bands": labels, "families": fams}


# ================================================================================== thresholds
#: The §5 / §9 bars, structured (amendable; the verbatim plan text is attached when found).
THRESHOLDS: Dict[str, Any] = {
    "t27a": {"S1_ja_en_min_gain_over_stock": 5.0, "S1_ja_en_v1_abs": 42.4, "S1_ja_en_target": 45.0,
             "S3_ja_he_min_gain_over_stock": 5.0, "S3_ja_he_v1_abs": 32.0, "S4_ar_en_max_loss_vs_stock": 1.0,
             "S5_he_en_max_loss_vs_stock": 2.0, "S4_translit_min_gain_over_stock": 3.0,
             "S2_read_min_gain_over_stock": 3.0, "degenerate_rate_max": 0.01},
    "t27b": {"S1_S3_max_loss_vs_t27a": 1.0, "catalogue_json_valid_min": 0.99, "type_acc_min": 0.92,
             "date_pm2_min": 0.35, "date_pm2_min_dated_docs": 100, "dated_without_reference_max": 0.15,
             "people_recall_min": 0.50, "people_grounded_min": 0.95, "in_neither_specifics_max": 0.25,
             "places_recall_min": 0.50, "relations_f1_min_gain_over_stock": 10.0,
             "cell_regression_max": 2.0, "cell_regression_requires_ci_excluding_0": True},
    "promotion": {"mlx_8bit_drift_max_chrfpp": 1.0, "reasoning_tokens_max": 0, "reasoning_check_calls": 20,
                  "median_latency_s_max": 60, "regurgitation_rate_max": 0.02, "regurgitation_span_chars": 200,
                  "regurgitation_train_docs": 100, "scrub_strings_max": 0, "blind_ab_preferred_min": 0.70,
                  "blind_ab_items": [20, 20], "l1_verbatim_continuation_rate_max": 0.02,
                  "l1_verbatim_continuation_chars": 200, "l1_page_openings": 100},
    "e6": {"distractor_import_max": 0.05, "quote_words_max": 30, "site_format_prompts": 20},
    "e7": {"success": "t27b hard-band delta smaller than stock's, paired CI excluding 0; edition and in-gate read "
                      "scores within 2"},
    "notes": "t27a bars are offsets from the stock 27B; under scorer v2 use the re-based values (rebased_v2). "
             "people_grounded_min was set on the 10-18 judge; the deterministic scorer share is a proxy.",
}
#: Plan paragraphs copied verbatim into the manifest: (name, start marker, end marker or None = paragraph end).
_VERBATIM: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("bars_s5", "Bars (amendable):", "Optional baseline arm"),
    ("e7_success_s9_1", "Success: t27b", None),
    ("promotion_s9_5", "Promotion adds:", None),
)


def extract_verbatim(plan_text: str) -> Dict[str, Optional[str]]:
    """Copy the threshold paragraphs of the plan verbatim (whitespace of wrapped lines collapsed).

    :param plan_text: The plan markdown.
    :returns: ``{name: text or None}``.
    """
    out: Dict[str, Optional[str]] = {}
    for name, start, end in _VERBATIM:
        i = plan_text.find(start)
        if i < 0:
            out[name] = None
            continue
        j = plan_text.find(end, i) if end else -1
        if j < 0:
            m = re.search(r"\n\s*\n|\n(?=#)|\n(?=- \*\*)", plan_text[i:])
            j = i + m.start() if m else len(plan_text)
        out[name] = re.sub(r"\s*\n\s*", " ", plan_text[i:j]).strip()
    return out


def thresholds(plan_path: Optional[Path] = None, scores_summary: Optional[Path] = None) -> Dict[str, Any]:
    """The thresholds block: structured bars, verbatim plan text, v2 re-based stock bars.

    :param plan_path: Local plan markdown (None / missing = no verbatim text).
    :param scores_summary: ``scores_v2/summary_v2.json`` (None / missing = no re-based bars).
    :returns: The block.
    """
    block = json.loads(json.dumps(THRESHOLDS))
    if plan_path and Path(plan_path).exists():
        block["verbatim"] = extract_verbatim(Path(plan_path).read_text(encoding="utf-8"))
        block["verbatim_source"] = Path(plan_path).name
    if scores_summary and Path(scores_summary).exists():
        s = json.loads(Path(scores_summary).read_text(encoding="utf-8"))
        block["rebased_v2"] = {"scorer_version": s.get("scorer_version"), "base_system": s.get("base_system"),
                               "bars": [{k: b[k] for k in ("bar", "sample_set", "condition", "stock_v2", "bar_v2", "n")}
                                        for b in s.get("bars_rebased", [])]}
    return block


def latest_plan(directory: Path) -> Optional[Path]:
    """Newest ``PLAN_*.md`` in a directory.

    :param directory: Directory to search.
    :returns: The file or None.
    """
    plans = sorted(Path(directory).glob("PLAN_*.md"))
    return plans[-1] if plans else None


# ==================================================================================== manifest
def _read_json(path: Path) -> Any:
    """Parse a JSON file.

    :param path: File.
    :returns: Parsed content.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_id_list(path: Path) -> List[str]:
    """Canonical ids of a benchmark file (a JSON list, or ``{"ids": [...]}``).

    :param path: JSON file.
    :returns: Ids as strings.
    :raises ValueError: When the file holds neither shape.
    """
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("ids")
    if not isinstance(data, list):
        raise ValueError(f"{Path(path).name}: expected a list of ids or {{'ids': [...]}}")
    return [str(x) for x in data]


def described_pgpids(pairs_path: Path, pgpids: Set[str]) -> Set[str]:
    """pgpids among ``pgpids`` whose pairs row has a non-empty PGP description.

    :param pairs_path: ``pgp_parallel`` JSONL.
    :param pgpids: Candidates.
    :returns: The described ones.
    """
    return {str(r["pgpid"]) for r in _rows.read_jsonl(pairs_path)
            if str(r.get("pgpid")) in pgpids and (r.get("description") or "").strip()}


def build_manifest(p: Optional[_paths.Paths] = None, families_dir: Optional[Path] = None, dry_run_v01: bool = False,
                   seed: int = SEED, holdouts_path: Optional[Path] = None, plan_path: Optional[Path] = None,
                   e2_top_up: Sequence[str] = E2_TOP_UP) -> Dict[str, Any]:
    """Assemble ``eval_v1.json`` from whatever inputs exist (missing ones leave their sets pending).

    :param p: Build paths (default :func:`paths.load_paths`).
    :param families_dir: Family files (default ``out_dir/families``).
    :param dry_run_v01: Draw E2 / E2-read / E3 / E7 from the v0_1 build (splits mapped, no exposure tags).
    :param seed: Sampling seed.
    :param holdouts_path: ``holdouts.json`` (default ``out_dir/holdouts.json``).
    :param plan_path: Plan markdown for the verbatim thresholds (default: newest ``PLAN_*.md`` next to ``out_dir``).
    :param e2_top_up: Splits E2 tops up from (empty = test_indist only).
    :returns: The manifest.
    """
    p = p or _paths.load_paths()
    repo = p.repo
    families_dir = Path(families_dir or p.out("families"))
    holdouts_path = Path(holdouts_path or p.out("holdouts.json"))
    plan_path = plan_path or latest_plan(p.out_dir.parent)
    scores_summary = p.out("scores_v2", "summary_v2.json")
    rel = functools.partial(_scorer.repo_relative, repo=repo)
    sets: Dict[str, Any] = {}
    sources: Dict[str, Any] = {}

    # E1
    if p.samples.exists():
        excl = _scorer.load_exclusions(p.exclusions_v1 if p.exclusions_v1.exists() else None)
        sets["E1_probe_clean"] = build_e1(_read_json(p.samples), excl)
        sets["E1_probe_clean"]["source_files"] = _sigs([p.samples, p.exclusions_v1], repo)
        sets["E1_probe_clean"]["prompt_versions"] = [_scorer.PROBE_PROMPT_VERSION, train_prompt_version()]
        if scores_summary.exists():
            sets["E1_probe_clean"]["baseline_scores"] = file_sig(scores_summary, repo)
    else:
        sets["E1_probe_clean"] = _pending(E1_DEF, [rel(p.samples)])

    # family-based sets
    new_book: Dict[str, List[RowRef]] = {}
    if dry_run_v01:
        refs = load_v01_refs(p.text_ft_v0_1)
        fam_sources = _sigs([p.text_ft_v0_1 / "dev.jsonl", p.text_ft_v0_1 / "test.jsonl"], repo)
    else:
        excluded_rows = p.out("excluded_rows.json")
        overrides = split_overrides(excluded_rows)
        refs = load_family_refs(families_dir, overrides=overrides)
        new_book = load_new_book_refs(families_dir, overrides)
        extra = ([families_dir / f"{BOOK_PAIRS}.jsonl"] if new_book else []) + ([excluded_rows] if overrides else [])
        fam_sources = _sigs([family_file(families_dir, f) for f in refs] + extra, repo)
        sources["split_overrides"] = {"rows": len(overrides), "by_split": dict(Counter(overrides.values()))}
    exposure_path = p.out("exposure.jsonl")
    need = {"E2_test_stratified": ("translate_en", "translate_he", "catalogue_json", "extract_relations"),
            "E2_read": ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations"),
            "E3_dev_monitor": ("translate_en", "translate_he", "catalogue_json", "extract_relations"),
            "E7_noise_bands": E7_FAMILIES}
    defs = {"E2_test_stratified": E2_DEF, "E2_read": E2R_DEF, "E3_dev_monitor": E3_DEF, "E7_noise_bands": E7_DEF}
    for name, fams in need.items():
        missing = [rel(families_dir / f"{f}.jsonl") for f in fams if f not in refs]
        if missing:
            sets[name] = _pending(defs[name], missing)
            continue
        if name == "E2_test_stratified":
            sets[name] = build_e2(refs, seed=seed, top_up=e2_top_up)
            if new_book:
                sets[name]["new_books"] = build_e2_new_books(new_book)
        elif name == "E2_read":
            sets[name] = build_e2_read(refs)
        elif name == "E3_dev_monitor":
            sets[name] = build_e3(refs, seed=seed, new_book_refs=new_book)
        else:
            exposure = None if dry_run_v01 else load_exposure(exposure_path)
            pool = {r.pgpid for f in E7_FAMILIES for r in refs.get(f) or [] if is_vision_holdout(r)}
            reads = doc_reads(p.pgp_parallel, pool, exposure)
            sets[name] = build_e7(refs, reads, exposure is not None)
            sets[name]["source_files_extra"] = _sigs([p.pgp_parallel] + ([exposure_path] if exposure else []), repo)
        sets[name]["source_files"] = fam_sources
        sets[name]["prompt_version"] = train_prompt_version()
        if dry_run_v01:
            sets[name]["status"] = "dry_run_v0_1"

    # E4 / E5 (holdouts)
    if holdouts_path.exists():
        h = _read_json(holdouts_path)
        bench_ids = load_id_list(p.benchmark_ids) if p.benchmark_ids.exists() else []
        merged_map = _read_json(p.merged_pgpid_map) if p.merged_pgpid_map.exists() else {}
        e4 = build_e4(bench_ids, h, merged_map)
        e4 = build_e4(bench_ids, h, merged_map, described_pgpids(p.pgp_parallel, set(e4["pgpids"])))
        e4["source_files"] = _sigs([holdouts_path, p.benchmark_ids, p.merged_pgpid_map, p.pgp_parallel], repo)
        e4["prompt_version"] = train_prompt_version()
        sets["E4_pgp131"] = e4
        if p.document_treatments.exists():
            all_refs = [r for rs in list(refs.values()) + list(new_book.values()) for r in rs]
            split_of = None if dry_run_v01 or not refs else strongest_split(all_refs)
            analysis = None if dry_run_v01 else (list(refs.get("analyse_document") or [])
                                                 + list(new_book.get("analyse_document") or []))
            e5 = build_e5(_rows.read_jsonl(p.document_treatments), merged_map, h, split_of,
                          analysis_refs=analysis)
            e5["source_files"] = _sigs([p.document_treatments, holdouts_path, p.merged_pgpid_map], repo)
            if analysis is not None:
                e5["source_files_rows"] = _sigs([family_file(families_dir, "analyse_document")]
                                                + ([families_dir / f"{BOOK_PAIRS}.jsonl"] if new_book else []), repo)
            sets["E5_excluded_treatments"] = e5
        else:
            sets["E5_excluded_treatments"] = _pending(E5_DEF, [rel(p.document_treatments)])
    else:
        sets["E4_pgp131"] = _pending(E4_DEF, [rel(holdouts_path)])
        sets["E5_excluded_treatments"] = _pending(E5_DEF, [rel(holdouts_path)])

    # E6
    rag = p.out("rag_test_docs.json")
    if rag.exists():
        sets["E6_rag"] = build_e6(_read_json(rag))
        sets["E6_rag"]["source_files"] = _sigs([rag], repo)
        sets["E6_rag"]["prompt_version"] = train_prompt_version()
    else:
        sets["E6_rag"] = _pending(E6_DEF, [rel(rag)])

    sources["plan"] = Path(plan_path).name if plan_path else None
    return {"name": EVAL_VERSION, "build": _rows.BUILD,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scorer_version": _scorer.SCORER_VERSION, "train_prompt_version": train_prompt_version(),
            "probe_prompt_version": _scorer.PROBE_PROMPT_VERSION, "seed": seed, "dry_run_v01": dry_run_v01,
            "status": {k: v["status"] for k, v in sets.items()}, "sets": sets,
            "thresholds": thresholds(plan_path, scores_summary), "sources": sources}


def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write the manifest atomically.

    :param manifest: Output of :func:`build_manifest`.
    :param path: Destination.
    :returns: The path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# ================================================================================= dev monitor
def select_rows(files: Sequence[Path], row_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Full rows for the given ids, in the order of ``row_ids`` (first occurrence over the files).

    :param files: JSONL files to search (missing files are skipped).
    :param row_ids: Wanted ``meta.row_id`` values.
    :returns: The rows.
    :raises ValueError: When an id is not found.
    """
    wanted, found = set(row_ids), {}
    for f in files:
        if not Path(f).exists():
            continue
        for r in _rows.read_jsonl(f):
            rid = (r.get("meta") or {}).get("row_id")
            if rid in wanted and rid not in found:
                found[rid] = r
    missing = [rid for rid in row_ids if rid not in found]
    if missing:
        raise ValueError(f"{len(missing)} row ids not found in {[Path(f).name for f in files]}, e.g. {missing[:3]}")
    return [found[rid] for rid in row_ids]


def dev_monitor_rows(e3: Mapping[str, Any], families_dir: Path) -> List[Dict[str, Any]]:
    """The E3 rows, translate families first, each family in E3 order.

    :param e3: The built E3 set (``row_ids`` per family).
    :param families_dir: ``out_dir/families`` (the family files and the L5 file).
    :returns: Full rows.
    """
    fams = [f for f in E3_TRANSLATE if f in e3["row_ids"]] + [f for f in e3["row_ids"] if f not in E3_TRANSLATE]
    files = [family_file(families_dir, f) for f in fams] + [Path(families_dir) / f"{BOOK_PAIRS}.jsonl"]
    return select_rows(files, [rid for f in fams for rid in e3["row_ids"][f]])


def dev_monitor_problems(rows: Sequence[Mapping[str, Any]], max_tokens: Optional[int] = MAX_SEQ) -> List[str]:
    """Problems of dev-monitor rows against the harness contract (derive_t27 G2 / G5) and decision D13.

    :param rows: Dev-monitor rows.
    :param max_tokens: Longest rendered row (``meta.tokens``); None = unchecked.
    :returns: One line per problem (empty = clean): split dev, build, family, condition, roles, row id present and
        unique, no vision-benchmark flag, a ``reader_exposed`` tag on every row (a bool on real reads), no exposed
        read, known token length within ``max_tokens``, NFC message contents, no project-name string anywhere in the row.
    """
    problems: List[str] = []
    seen: Set[str] = set()
    for i, r in enumerate(rows):
        m = r.get("meta") or {}
        rid = m.get("row_id")
        tag = f"row {i} ({rid})"
        roles = [x.get("role") for x in r.get("messages") or []]
        if roles != ["system", "user", "assistant"]:
            problems.append(f"{tag}: roles {roles}")
        if m.get("split") != "dev":
            problems.append(f"{tag}: split {m.get('split')!r}")
        if m.get("build") != _rows.BUILD:
            problems.append(f"{tag}: build {m.get('build')!r}")
        if m.get("family") not in _rows.FAMILIES:
            problems.append(f"{tag}: family {m.get('family')!r}")
        if m.get("condition") not in _rows.CONDITIONS:
            problems.append(f"{tag}: condition {m.get('condition')!r}")
        if not rid or rid in seen:
            problems.append(f"{tag}: missing or duplicate row_id")
        seen.add(rid)
        if m.get("held_out_vision_benchmark") is True:
            problems.append(f"{tag}: vision-benchmark hold-out")
        if "reader_exposed" not in m:
            problems.append(f"{tag}: no reader_exposed tag")
        elif m.get("condition") in REAL_READ_CONDITIONS and not isinstance(m.get("reader_exposed"), bool):
            problems.append(f"{tag}: read row with reader_exposed={m.get('reader_exposed')!r}")
        if m.get("reader_exposed") is True:
            problems.append(f"{tag}: reader-exposed read")
        if max_tokens is not None and not (isinstance(m.get("tokens"), int) and m["tokens"] <= max_tokens):
            problems.append(f"{tag}: tokens {m.get('tokens')!r} (max {max_tokens})")
        bad_nfc = _rows.non_nfc_roles(dict(r))
        if bad_nfc:
            problems.append(f"{tag}: message content not NFC ({', '.join(bad_nfc)})")
        if tn.has_source_names(json.dumps(r, ensure_ascii=False)):
            problems.append(f"{tag}: project-name string in the row")
    return problems


def write_dev_monitor(rows: Sequence[Dict[str, Any]], path: Path, repo: Path) -> Dict[str, Any]:
    """Write the dev monitor (rows unchanged) and describe it.

    :param rows: Rows from :func:`dev_monitor_rows` (checked with :func:`dev_monitor_problems`).
    :param path: Destination (``out_dir/dev/dev_monitor.jsonl``).
    :param repo: Repository root (for the relative path).
    :returns: :func:`file_sig` plus ``rows``, ``by_family`` and ``by_condition``.
    """
    _rows.write_jsonl(path, rows)
    return {**file_sig(path, repo), "rows": len(rows),
            "by_family": dict(Counter(r["meta"]["family"] for r in rows)),
            "by_condition": dict(Counter(r["meta"]["condition"] for r in rows))}


# ================================================================================== Hub folder
HUB_NAME = "genizah_text27b"
HUB_ARMS: Tuple[str, ...] = ("t27a", "t27b")
SHA256_FILE = "SHA256SUMS"
#: Message roles of the harness text fingerprint (derive_t27.text_fingerprint == rows.fingerprint on these).
FINGERPRINT_KEYS: Tuple[str, ...] = ("system", "user", "assistant")
#: Files of the Hub folder the harness reads (derive_t27 data cell: TRAIN_FILE, MANIFEST_FILE, DEV_FILE, README).
HUB_TRAIN = "{arm}/train.jsonl"
HUB_MANIFEST = "{arm}/mixture_manifest.json"
#: ``meta`` fields the Hub copies drop: the pre-strip target can still name documents the strips removed.
HUB_DROP_META: Tuple[str, ...] = ("target_raw",)


def stream_fingerprint(path: Path, keys: Sequence[str] = FINGERPRINT_KEYS) -> Tuple[str, int]:
    """Text fingerprint of a JSONL file in file order, and its row count (streaming).

    :param path: JSONL file.
    :param keys: Fingerprint columns.
    :returns: ``(sha256 hex, rows)``.
    """
    n = [0]

    def counted() -> Iterable[Dict[str, Any]]:
        """Rows of the file, counted.

        :returns: Row iterator.
        """
        for r in _rows.read_jsonl(path):
            n[0] += 1
            yield r

    return _rows.fingerprint(counted(), keys), n[0]


def row_ids_digest(row_ids: Iterable[Any]) -> str:
    """Order-sensitive sha256 over row ids (one id per line).

    :param row_ids: ``meta.row_id`` values in file order.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    for rid in row_ids:
        h.update(str(rid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def stream_identity(path: Path, keys: Sequence[str] = FINGERPRINT_KEYS,
                    drop: Sequence[str] = HUB_DROP_META) -> Dict[str, Any]:
    """Identity of a JSONL file in one streaming pass: text fingerprint, rows, ordered row ids, dropped-field rows.

    :param path: JSONL file.
    :param keys: Fingerprint columns.
    :param drop: ``meta`` fields counted in ``rows_with_dropped``.
    :returns: ``{"text_fingerprint_sha256", "rows", "row_ids", "row_ids_sha256", "rows_with_dropped"}``.
    """
    ids: List[Any] = []
    dropped = [0]

    def tracked() -> Iterable[Dict[str, Any]]:
        """Rows of the file, their ids and dropped fields recorded.

        :returns: Row iterator.
        """
        for r in _rows.read_jsonl(path):
            m = r.get("meta") or {}
            ids.append(m.get("row_id"))
            if any(k in m for k in drop):
                dropped[0] += 1
            yield r

    fp = _rows.fingerprint(tracked(), keys)
    return {"text_fingerprint_sha256": fp, "rows": len(ids), "row_ids": ids, "row_ids_sha256": row_ids_digest(ids),
            "rows_with_dropped": dropped[0]}


def hub_row(row: Mapping[str, Any], drop: Sequence[str] = HUB_DROP_META) -> Dict[str, Any]:
    """A row as the Hub carries it: messages unchanged, ``meta`` without the ``drop`` fields (key order kept).

    :param row: Mixture or dev-monitor row.
    :param drop: ``meta`` fields to remove.
    :returns: A new row (the input is not modified).
    """
    out = dict(row)
    if isinstance(row.get("meta"), Mapping):
        out["meta"] = {k: v for k, v in row["meta"].items() if k not in drop}
    return out


def write_hub_rows(src: Path, dst: Path, drop: Sequence[str] = HUB_DROP_META) -> int:
    """Stream a JSONL file into its Hub copy (atomic), each row through :func:`hub_row`.

    :param src: Source JSONL (a materialised epoch or the dev monitor).
    :param dst: Destination (parents are created).
    :param drop: ``meta`` fields to remove.
    :returns: Rows written.
    """
    return _rows.write_jsonl(dst, (hub_row(r, drop) for r in _rows.read_jsonl(src)))


def _write_json(obj: Any, path: Path) -> None:
    """Write JSON atomically (indent 1, UTF-8).

    :param obj: Value.
    :param path: Destination.
    :returns: None
    """
    write_manifest(obj, path)


def hub_arm(arm: str, section: Mapping[str, Any], mixtures_dir: Path, hub_dir: Path) -> Dict[str, Any]:
    """Write one arm's materialised epoch into the Hub folder without ``meta.target_raw`` and write its manifest.

    The source epoch must be the one the mixture manifest describes (its sha256 equals ``output_sha256`` and its
    text fingerprint, the harness G2 digest, equals ``fingerprint``). The Hub copy is identified with the mixture
    by the same text fingerprint and the same ordered row-id list, not by bytes: it drops :data:`HUB_DROP_META`.
    Its own sha256 goes into the ``hub`` block (and ``SHA256SUMS`` via :func:`build_hub`).

    :param arm: ``t27a`` or ``t27b``.
    :param section: The arm's section of ``mixtures/mixture_manifest.json``.
    :param mixtures_dir: ``out_dir/mixtures``.
    :param hub_dir: The Hub folder.
    :returns: Summary (rows, sha256, fingerprint, row-id digest, tokens, steps, components, noise-slice share).
    :raises ValueError: When the epoch file does not match its manifest or the copy does not match the epoch.
    """
    src = Path(mixtures_dir) / Path(str(section["output"])).name
    sha = sha256_file(src)
    if section.get("output_sha256") and sha != section["output_sha256"]:
        raise ValueError(f"{arm}: {src.name} sha256 {sha[:12]}… != mixture manifest {str(section['output_sha256'])[:12]}…")
    src_id = stream_identity(src)
    fp = src_id["text_fingerprint_sha256"]
    if section.get("fingerprint") and fp != section["fingerprint"]:
        raise ValueError(f"{arm}: {src.name} fingerprint {fp[:12]}… != mixture manifest {str(section['fingerprint'])[:12]}…")
    train_rel, man_rel = HUB_TRAIN.format(arm=arm), HUB_MANIFEST.format(arm=arm)
    dst = Path(hub_dir) / train_rel
    write_hub_rows(src, dst)
    got = stream_identity(dst)
    if got["text_fingerprint_sha256"] != fp or got["row_ids"] != src_id["row_ids"] or got["rows_with_dropped"]:
        raise ValueError(f"{arm}: Hub copy differs from {src.name} (fingerprint, row-id order or "
                         f"{'/'.join(HUB_DROP_META)} left on {got['rows_with_dropped']} rows)")
    hub = {"train_file": train_rel, "rows": got["rows"], "sha256": sha256_file(dst), "text_fingerprint_sha256": fp,
           "fingerprint_keys": list(FINGERPRINT_KEYS), "row_ids_sha256": got["row_ids_sha256"],
           "dropped_meta": list(HUB_DROP_META), "source_output": str(section["output"]), "source_sha256": sha}
    _write_json({**dict(section), "hub": hub}, Path(hub_dir) / man_rel)
    return {"arm": arm, **hub, "manifest_file": man_rel, "totals": section.get("totals"),
            "projected_steps": section.get("projected_steps"), "components": section.get("components"),
            "noise_slice": section.get("noise_slice"), "counts_by_family": _family_rows(section)}


def _family_rows(section: Mapping[str, Any]) -> Dict[str, int]:
    """Rows per family of a mixture section (its ``counts.rows`` keys are ``family|condition|lang|split``).

    :param section: Mixture manifest section.
    :returns: ``{family: rows}``.
    """
    rows = (section.get("counts") or {}).get("rows") or {}
    out: Counter = Counter()
    for key, n in rows.items():
        if isinstance(n, int):
            out[str(key).split("|")[0]] += n
    return dict(out)


def write_sha256sums(folder: Path, name: str = SHA256_FILE) -> Dict[str, str]:
    """``shasum -a 256 -c``-compatible checksum file over every other file of a folder.

    :param folder: Folder.
    :param name: Checksum file name (excluded from itself).
    :returns: ``{relative path: sha256}``.
    """
    folder = Path(folder)
    sums = {p.relative_to(folder).as_posix(): sha256_file(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != name and not p.name.startswith(".")}
    text = "".join(f"{h}  {rel}\n" for rel, h in sorted(sums.items()))
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=str(folder))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, folder / name)
    return sums


def _fmt_m(tokens: Optional[float]) -> str:
    """Tokens in millions.

    :param tokens: Token count.
    :returns: ``"x.xx M"`` or ``"–"``.
    """
    return "–" if tokens is None else f"{tokens / 1e6:.2f} M"


def readme_text(arms: Sequence[Mapping[str, Any]], dev: Mapping[str, Any], e3: Mapping[str, Any],
                eval_info: Mapping[str, Any]) -> str:
    """Dataset card of the Hub folder: layout, arms, schema, pinning, provenance. Names no holdings, hosts,
    index names or machine paths, and quotes no rows.

    :param arms: :func:`hub_arm` summaries.
    :param dev: The dev-monitor description (:func:`write_dev_monitor`).
    :param e3: The E3 set of ``eval_v1.json``.
    :param eval_info: ``{"name", "built_at", "scorer_version", "train_prompt_version"}`` of the eval manifest.
    :returns: Markdown.
    """
    configs = "".join(f"- config_name: {a['arm']}\n  data_files:\n  - split: train\n    path: {a['train_file']}\n"
                      for a in arms)
    configs += f"- config_name: dev_monitor\n  data_files:\n  - split: validation\n    path: {DEV_MONITOR_FILE}\n"
    lines = ["---", "license: other", "pretty_name: genizah_text27b", "configs:", configs.rstrip("\n"), "---", "",
             f"# {HUB_NAME} — text fine-tune mixtures (build {_rows.BUILD})", "",
             "**Private.** Chat rows for the text-only Genizah fine-tunes (arms `t27a`, `t27b`). Keep this dataset "
             "private: targets include in-copyright scholarly translations and literature excerpts used for internal "
             "training only. This card quotes no rows; source lists, hold-out lists and exclusion manifests stay with "
             "the build and are not part of this dataset.", "",
             "## Layout", "", "| file | rows | sha256 (first 16) |", "|---|---:|---|"]
    for a in arms:
        lines.append(f"| `{a['train_file']}` | {a['rows']:,} | `{a['sha256'][:16]}` |")
        lines.append(f"| `{a['manifest_file']}` | – | mixture manifest of `{a['arm']}` |")
    lines += [f"| `{DEV_MONITOR_FILE}` | {dev['rows']:,} | `{dev['sha256'][:16]}` |",
              f"| `{SHA256_FILE}` | – | sha256 of every file above and of this card |", "",
              "Each `train.jsonl` is ONE materialised, seeded epoch (`text27b.mixture`, seed 3407): train it once, in "
              "file order (map-style, no shuffling, no streaming). The copies here drop `meta.target_raw` (the "
              "build's pre-strip target, kept for its audit); messages, every other field, the text fingerprint and "
              "the row-id order equal the build's mixture. The harness (`derive_t27`) downloads exactly "
              "`<arm>/train.jsonl`, `<arm>/mixture_manifest.json`, `dev/dev_monitor.jsonl` and this card.", "",
              "## Arms", "", "| arm | rows | tokens | unpacked steps (16 rows) | packed steps (16.6 k tok) | noise-slice share |",
              "|---|---:|---:|---:|---:|---:|"]
    for a in arms:
        tot, st, ns = a.get("totals") or {}, a.get("projected_steps") or {}, a.get("noise_slice") or {}
        lines.append(f"| `{a['arm']}` | {a['rows']:,} | {_fmt_m(tot.get('tokens'))} | {st.get('unpacked_steps', '–')} | "
                     f"{st.get('packed_steps', '–')} | {ns.get('share', 0) or 0:.3f} |")
    for a in arms:
        lines += ["", f"`{a['arm']}` components (rows / tokens / share of tokens):", ""]
        lines += [f"- `{c}`: {v.get('rows', 0):,} / {_fmt_m(v.get('tokens'))} / {v.get('share', 0):.3f}"
                  for c, v in sorted((a.get("components") or {}).items())]
    lines += ["", "`t27a` = the translate rows (every condition, ×2) plus the new-book translate pairs (×2): the "
              "translation pilot and the no-slice, no-literature baseline. `t27b` = the joint mixture: translate ×2, "
              "catalogue ∪ describe one row per document, relations 40 %, analysis ×2, literature-conditioned rows, "
              "domain reading, with the noise slice swapped in for noised copies.", "",
              "## Row schema", "",
              "`{\"messages\": [system, user, assistant], \"meta\": {...}}`. Render with the model's chat template and "
              "`enable_thinking=False` (one empty think block, no reasoning-effort sentence); loss on the assistant "
              "turn only. Key `meta` fields: `family`, `condition` (edition, noised_edition, read, read_hard, "
              "read_partial, edition_rescript, literature), `lang_class`, `split` (`train` in the arm files, `dev` in "
              "the monitor), `row_id`, `tokens` (rendered length, ≤ 6,144), `reader_exposed` / `exposure_level` on "
              "machine-read inputs, `build` = `v0_2`, and the mixture fields `mixture`, `mix_component`, `mix_rule`, "
              "`mix_copy`, `mix_repeats`.", "",
              "Targets are human-written text (editions' translations, catalogue records, scholars' commentary, "
              "literature pages). Machine reads (our page readers' transcriptions) appear only as inputs.", "",
              "## Dev monitor (E3)", "",
              f"{E3_DEF}", "",
              f"Rows per family: {json.dumps(dev.get('by_family', {}), sort_keys=True)}; by condition: "
              f"{json.dumps(dev.get('by_condition', {}), sort_keys=True)}. Reader-exposed reads removed before "
              f"sampling: {json.dumps(e3.get('exposed_removed', {}), sort_keys=True)}; every row carries a "
              "`reader_exposed` tag and none is `true`. No dev document is a train document of either arm. From the "
              f"pre-registered eval manifest `{eval_info.get('name')}` (built {eval_info.get('built_at')}, scorer "
              f"`{eval_info.get('scorer_version')}`, prompts `{eval_info.get('train_prompt_version')}`).", "",
              "## Pinning", "",
              "The harness refuses to train until the arm's dataset revision (40-hex commit of this dataset) and "
              "the text fingerprint of `<arm>/train.jsonl` (sha256 over the system / user / assistant texts in file "
              "order; `derive_t27.text_fingerprint`) are pinned. After pushing, regenerate the notebooks with "
              "`derive_t27.py --<arm>-revision <commit> --<arm>-mixture <local copy of <arm>/train.jsonl>`.", "",
              "| arm | text fingerprint (sha256) |", "|---|---|"]
    lines += [f"| `{a['arm']}` | `{a['text_fingerprint_sha256']}` |" for a in arms]
    lines += ["", "## Integrity", "", f"`shasum -a 256 -c {SHA256_FILE}` in this folder.", ""]
    return "\n".join(lines)


def hub_problems(hub_dir: Path, arms: Sequence[str] = HUB_ARMS, max_tokens: Optional[int] = MAX_SEQ) -> List[str]:
    """Re-check a Hub folder from its files: checksums, each train file against its ``hub`` block (sha256, text
    fingerprint, rows, row-id digest), train rows (split, build, vision flag, tokens, NFC message contents, no
    project-name string anywhere, no :data:`HUB_DROP_META` field), dev rows (:func:`dev_monitor_problems`, no
    :data:`HUB_DROP_META` field), train/dev disjointness.

    :param hub_dir: The Hub folder.
    :param arms: Arms expected.
    :param max_tokens: Longest rendered row (``meta.tokens``).
    :returns: One line per problem (empty = clean).
    """
    hub_dir = Path(hub_dir)
    problems: List[str] = []
    sums = hub_dir / SHA256_FILE
    if not sums.exists():
        return [f"{SHA256_FILE} missing"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1)
        f = hub_dir / rel
        if not f.exists() or sha256_file(f) != h:
            problems.append(f"checksum mismatch: {rel}")
    dev = list(_rows.read_jsonl(hub_dir / DEV_MONITOR_FILE)) if (hub_dir / DEV_MONITOR_FILE).exists() else []
    if not dev:
        problems.append(f"{DEV_MONITOR_FILE} missing or empty")
    problems += [f"dev: {p}" for p in dev_monitor_problems(dev, max_tokens)]
    dev_dropped = sum(1 for r in dev if any(k in (r.get("meta") or {}) for k in HUB_DROP_META))
    if dev_dropped:
        problems.append(f"dev: {dev_dropped} rows carry {'/'.join(HUB_DROP_META)}")
    dev_docs = {str(r["meta"]["pgpid"]) for r in dev if r["meta"].get("pgpid") not in (None, "")}
    dev_ids = {r["meta"].get("row_id") for r in dev}
    for arm in arms:
        f, man = hub_dir / HUB_TRAIN.format(arm=arm), hub_dir / HUB_MANIFEST.format(arm=arm)
        if not f.exists() or not man.exists():
            problems.append(f"{arm}: train file or manifest missing")
            continue
        info = _read_json(man).get("hub") or {}
        got = stream_identity(f)
        if (got["text_fingerprint_sha256"] != info.get("text_fingerprint_sha256") or got["rows"] != info.get("rows")
                or got["row_ids_sha256"] != info.get("row_ids_sha256")):
            problems.append(f"{arm}: fingerprint / rows / row-id order differ from its manifest")
        if sha256_file(f) != info.get("sha256"):
            problems.append(f"{arm}: sha256 differs from its manifest")
        bad, docs, ids = Counter(), set(), set()
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            if m.get("split") != "train":
                bad["split"] += 1
            if m.get("build") != _rows.BUILD:
                bad["build"] += 1
            if m.get("held_out_vision_benchmark") is True:
                bad["vision_flag"] += 1
            if max_tokens is not None and not (isinstance(m.get("tokens"), int) and m["tokens"] <= max_tokens):
                bad["tokens"] += 1
            if _rows.non_nfc_roles(r):
                bad["non_nfc"] += 1
            if tn.has_source_names(json.dumps(r, ensure_ascii=False)):
                bad["source_names"] += 1
            if any(k in m for k in HUB_DROP_META):
                bad["/".join(HUB_DROP_META)] += 1
            if m.get("pgpid") not in (None, ""):
                docs.add(str(m["pgpid"]))
            ids.add(m.get("row_id"))
        problems += [f"{arm}: {n_bad} rows with bad {k}" for k, n_bad in bad.items()]
        if docs & dev_docs:
            problems.append(f"{arm}: {len(docs & dev_docs)} documents in train and the dev monitor")
        if ids & dev_ids:
            problems.append(f"{arm}: {len(ids & dev_ids)} row ids in train and the dev monitor")
    return problems


def build_hub(p: _paths.Paths, eval_manifest: Mapping[str, Any], out: Optional[Path] = None,
              arms: Sequence[str] = HUB_ARMS, dev_monitor: Optional[Path] = None,
              mixtures_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build the Hub-ready dataset folder (decision D13; nothing is pushed).

    Layout: ``<arm>/train.jsonl`` (the materialised epoch without :data:`HUB_DROP_META`, :func:`hub_arm`),
    ``<arm>/mixture_manifest.json`` (the arm's mixture-manifest section + a ``hub`` block with rows, the copy's
    sha256, the harness text fingerprint and the row-id digest), ``dev/dev_monitor.jsonl`` (the E3 rows, also
    without :data:`HUB_DROP_META`; same fingerprint and row-id order as the dev monitor), ``README.md`` (no
    holdings) and ``SHA256SUMS``.

    :param p: Build paths.
    :param eval_manifest: Parsed ``eval_v1.json`` (its E3 set and versions go into the card).
    :param out: Hub folder (default ``out_dir/hub/genizah_text27b``).
    :param arms: Arms to include.
    :param dev_monitor: Dev-monitor file (default ``out_dir/dev/dev_monitor.jsonl``).
    :param mixtures_dir: Mixture directory (default ``out_dir/mixtures``).
    :returns: Summary (arms, dev, checksums, problems from :func:`hub_problems`).
    :raises ValueError: When the dev monitor fails :func:`dev_monitor_problems`, an epoch file does not match
        its manifest, or a Hub copy does not match its source.
    """
    out = Path(out or p.out("hub", HUB_NAME))
    mixtures_dir = Path(mixtures_dir or p.out("mixtures"))
    dev_monitor = Path(dev_monitor or p.out(DEV_MONITOR_FILE))
    dev_rows = list(_rows.read_jsonl(dev_monitor))
    bad = dev_monitor_problems(dev_rows)
    if bad:
        raise ValueError(f"dev monitor: {len(bad)} problems, e.g. {bad[:3]}")
    mix = _read_json(mixtures_dir / "mixture_manifest.json")
    summaries = [hub_arm(a, mix[a], mixtures_dir, out) for a in arms]
    write_hub_rows(dev_monitor, out / DEV_MONITOR_FILE)
    src_dev, got_dev = stream_identity(dev_monitor), stream_identity(out / DEV_MONITOR_FILE)
    if (got_dev["text_fingerprint_sha256"] != src_dev["text_fingerprint_sha256"]
            or got_dev["row_ids"] != src_dev["row_ids"] or got_dev["rows_with_dropped"]):
        raise ValueError("dev monitor: Hub copy differs from its source (fingerprint, row-id order or dropped fields)")
    dev = {**file_sig(out / DEV_MONITOR_FILE, out), "rows": len(dev_rows),
           "text_fingerprint_sha256": got_dev["text_fingerprint_sha256"], "row_ids_sha256": got_dev["row_ids_sha256"],
           "by_family": dict(Counter(r["meta"]["family"] for r in dev_rows)),
           "by_condition": dict(Counter(r["meta"]["condition"] for r in dev_rows))}
    e3 = (eval_manifest.get("sets") or {}).get("E3_dev_monitor") or {}
    info = {k: eval_manifest.get(k) for k in ("name", "built_at", "scorer_version", "train_prompt_version")}
    (out / "README.md").write_text(readme_text(summaries, dev, e3, info), encoding="utf-8")
    sums = write_sha256sums(out)
    return {"hub": _scorer.repo_relative(out, p.repo), "arms": summaries, "dev": dev, "sha256": sums,
            "problems": hub_problems(out, arms)}


def summary_lines(manifest: Mapping[str, Any]) -> List[str]:
    """Human-readable counts of a manifest (for the CLI log).

    :param manifest: The manifest.
    :returns: Lines.
    """
    out = []
    for name, s in manifest["sets"].items():
        if s["status"] == "pending":
            out.append(f"{name}: pending (needs {', '.join(s['requires'])})")
        elif name == "E1_probe_clean":
            out.append(f"{name}: " + ", ".join(f"{k} {v['kept']}" for k, v in s["counts"].items()))
        elif name == "E2_test_stratified":
            out.append(f"{name}: " + ", ".join(f"{k} {c['docs']}/{c['target']} ({c['by_split']})" for k, c in s["cells"].items()))
            if s.get("new_books"):
                out.append("  new books: " + ", ".join(f"{k} {c['docs']} docs / {c['rows']} rows ({c['by_split']}, "
                                                       f"overridden {c['split_overridden_docs']})"
                                                       for k, c in s["new_books"]["cells"].items()))
        elif name == "E2_read":
            out.append(f"{name}: " + ", ".join(f"{f} {sum(c.values())} {s.get('by_split', {}).get(f, {})}"
                                               for f, c in s["counts"].items())
                       + f"; exposed dropped {s['exposed_dropped']}")
        elif name == "E3_dev_monitor":
            out.append(f"{name}: {s['rows']} rows {s['counts']}; origins {s.get('origins')}; exposed removed "
                       f"{s['exposed_removed']}; exposure unknown {s['exposure_unknown_read_rows']}; over max tokens "
                       f"{s.get('over_max_tokens')}; split overridden {s.get('split_overridden')}")
            if s.get("materialised"):
                out.append(f"  materialised {s['materialised']['path']} ({s['materialised']['rows']} rows)")
        elif name == "E7_noise_bands":
            out.append(f"{name} [{s['status']}]: " + "; ".join(f"{f} {list(v['bands'].values())} (plan {list(v['plan_estimate'].values())})"
                                                              for f, v in s["families"].items()))
        else:
            out.append(f"{name}: {s.get('counts')}")
    return out


def materialise_dev_monitor(manifest: Dict[str, Any], p: _paths.Paths, families_dir: Optional[Path] = None,
                            path: Optional[Path] = None) -> Dict[str, Any]:
    """Write the E3 rows as the dev monitor and record the file in the manifest (``E3.materialised``).

    :param manifest: Output of :func:`build_manifest` (E3 built from the v0_2 family files).
    :param p: Build paths.
    :param families_dir: Family files (default ``out_dir/families``).
    :param path: Destination (default ``out_dir/dev/dev_monitor.jsonl``).
    :returns: The ``materialised`` record.
    :raises ValueError: When E3 is not built or its rows fail :func:`dev_monitor_problems`.
    """
    e3 = manifest["sets"]["E3_dev_monitor"]
    if e3["status"] not in ("ready", "provisional"):
        raise ValueError(f"E3 is {e3['status']}: no dev monitor")
    rows = dev_monitor_rows(e3, Path(families_dir or p.out("families")))
    bad = dev_monitor_problems(rows)
    if bad:
        raise ValueError(f"dev monitor: {len(bad)} problems, e.g. {bad[:3]}")
    e3["materialised"] = write_dev_monitor(rows, Path(path or p.out(DEV_MONITOR_FILE)), p.repo)
    return e3["materialised"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: ``build`` writes ``eval_v1.json`` (or the dry-run file) and ``dev/dev_monitor.jsonl`` under ``out_dir``;
    ``hub`` builds the Hub-ready folder ``out_dir/hub/genizah_text27b`` from the mixtures and the dev monitor.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: None
    """
    ap = argparse.ArgumentParser(description="text27b eval manifest (eval_v1.json), dev monitor and Hub folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--families", type=Path, default=None)
    b.add_argument("--holdouts", type=Path, default=None)
    b.add_argument("--plan", type=Path, default=None)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--dry-run-v01", action="store_true",
                   help="draw E2/E2-read/E3/E7 from the v0_1 build (writes eval_v1_dryrun_v01.json)")
    b.add_argument("--e2-no-top-up", action="store_true", help="E2 from test_indist only")
    b.add_argument("--no-dev-monitor", action="store_true", help="do not write dev/dev_monitor.jsonl")
    h = sub.add_parser("hub")
    h.add_argument("--eval", type=Path, default=None, help="eval manifest (default out_dir/eval_v1.json)")
    h.add_argument("--out", type=Path, default=None, help="Hub folder (default out_dir/hub/genizah_text27b)")
    h.add_argument("--arms", nargs="+", default=list(HUB_ARMS))
    h.add_argument("--mixtures", type=Path, default=None)
    h.add_argument("--dev", type=Path, default=None, help="dev monitor (default out_dir/dev/dev_monitor.jsonl)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = _paths.load_paths()
    if args.cmd == "hub":
        summary = build_hub(p, _read_json(args.eval or p.out(EVAL_FILE)), args.out, args.arms, args.dev, args.mixtures)
        for a in summary["arms"]:
            print(f"{a['arm']}: {a['rows']:,} rows, sha256 {a['sha256'][:16]}…, fingerprint {a['text_fingerprint_sha256']}")
        print(f"dev monitor: {summary['dev']['rows']} rows {summary['dev']['by_family']}")
        print(f"files: {len(summary['sha256'])} + {SHA256_FILE}; problems: {summary['problems'] or 'none'}")
        print(f"wrote {summary['hub']}")
        if summary["problems"]:
            raise SystemExit(1)
        return
    m = build_manifest(p, args.families, args.dry_run_v01, args.seed, args.holdouts, args.plan,
                       () if args.e2_no_top_up else E2_TOP_UP)
    if not args.dry_run_v01 and not args.no_dev_monitor:
        materialise_dev_monitor(m, p, args.families)
    out = args.out or p.out("eval_v1_dryrun_v01.json" if args.dry_run_v01 else EVAL_FILE)
    write_manifest(m, out)
    for line in summary_lines(m):
        print(line)
    print(f"wrote {_scorer.repo_relative(out, p.repo)}")


if __name__ == "__main__":
    main()
