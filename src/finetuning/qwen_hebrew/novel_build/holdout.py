"""Stage 5 (hold-out and split) of the novel-rows build.

Every side that passed the quality gates gets one :class:`~.schema.HoldoutResult`: held out (``split="excluded"``)
with every reason that applies, or ``train`` / ``val`` by a hash of its document group. Shelf-mark keys come from the
facade :mod:`src.datasets.shelfmarks` only (``canonical_id``, ``loose_key``, ``split``, ``resolve`` through
``build_legacy_store_rows.es_ids``).

Identity (per candidate; any hit holds out every side of its document group):

* ``benchmark:<registry>:<id>``: the registered benchmarks (``benchmark_registry.BENCHMARK_REGISTRIES``): their ids,
  their PGP ids (``<id>`` = ``pgpid:<n>``), their KTIV manuscripts (``sys:<n>``), the ES ids those resolve to through
  the merged-corpus index, and the facade loose keys of the ids and of the registered PGP documents' shelf marks;
* ``benchmark:pgp131:<id>``: the frozen 131-fragment benchmark (``decontam/benchmark_ids.json`` and its key inventory
  ``benchmark_keys.json``);
* ``benchmark:religious:<doc id>``: the religious page benchmark (its KTIV manuscripts and shelf marks);
* ``exclusion:derive_v25a_apd:<id>``: the APD exclusion lists given with ``--apd-exclusions`` (JSONL);
* ``exclusion:named:<name>``: :data:`NAMED_EXCLUSIONS`.

A candidate is matched by its ES ids (as recorded and as resolved), its PGP ids (as recorded and as the index links
them), its KTIV system number and the institution-qualified ``loose_key`` of every id and shelf-mark part.

Text (per side): ``text:<set>:<id>`` when the final target shares 25-letter Hebrew shingles
(``DecontamGate.content_contaminated``) or 20-letter rasm shingles with a ground-truth document of
``build_legacy_store_rows.GT_SETS`` (``load_gt_sets`` / ``build_gt_index`` / ``text_leakage``); ``<id>`` is the GT
document with the most shared shingles.

Text rule (``--text-rule``): ``strict`` is the rule above as it stands. ``boilerplate`` (the default) is the same rule
with one exception for formulae: a shared shingle held by at least ``--common-df`` (3) independent document groups of
the pool (the candidates' targets, counted once per ``document_group``) is boilerplate, an epistolary opening, a deed
clause or a basmala, and does not count towards the shared-shingle thresholds. A side still holds out, boilerplate or
not, when its longest shared run spans at least ``--long-run-letters`` (100) letters or at least
``--share-threshold`` (0.30) of its shingles are shared with one GT set. The text matches the exception discounted are
recorded in the result's ``flags`` (``boilerplate:<set>:<id>``); identity reasons are unchanged.

Split: ``val`` when ``int(sha1(document_group)[:8], 16) / 0xFFFFFFFF < val_share`` (default 5 %), else ``train``;
both sides of a sheet and every candidate of a group land together. ``rnl`` is copied from the candidate.

Usage::

    python -m src.finetuning.qwen_hebrew.novel_build.holdout --work-dir DIR [--val-share 0.05]
        [--apd-exclusions FILE ...] [--no-index]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.datasets import shelfmarks as shelf
from src.datasets.evaluations import benchmark_registry as br
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import _HEB, DecontamGate
from src.finetuning.qwen_hebrew import build_legacy_store_rows as lsr
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, HoldoutResult, QualityResult, read_jsonl,
                                                           write_jsonl)

VAL_SHARE = 0.05
TEXT_RULES = ("strict", "boilerplate")
COMMON_DF = 3                   # a shingle held by this many pool document groups, or more, is boilerplate
LONG_RUN_LETTERS = 100          # a shared run of this many letters holds out whatever its shingles' status
SHARE_THRESHOLD = 0.30          # this share of a side's shingles in one GT set holds out whatever their status
BOILERPLATE_FLAG = "boilerplate"
FROZEN_NAME = "pgp131"
RELIGIOUS_NAME = "religious"
APD_EXCLUSION = "derive_v25a_apd"
FROZEN_IDS = br.REGISTRY_DIR / "benchmark_ids.json"
FROZEN_KEYS = br.REGISTRY_DIR / "benchmark_keys.json"
RELIGIOUS = lsr.EVAL_DIR / "genizah_religious_v1" / "genizah_religious_v1.json"
#: optional default APD exclusion list(s), ``os.pathsep``-separated (the CLI flag adds to them)
APD_EXCLUSIONS_ENV = "NOVEL_BUILD_APD_EXCLUSIONS"
#: documents held out by name: a training row that copies a benchmark document verbatim, and that document
NAMED_EXCLUSIONS: Dict[str, Dict[str, List[str]]] = {}
#: The names themselves are a private list (benchmark membership): pass them with ``--named-exclusions`` or
#: the env var ``NOVEL_BUILD_NAMED_EXCLUSIONS``; each entry maps a name to ``{"pgpids": [...], "marks": [...]}``.
NAMED_EXCLUSIONS_ENV = "NOVEL_BUILD_NAMED_EXCLUSIONS"
#: fields of an APD exclusion record that hold one identifier, and the ones that hold lists
APD_ID_FIELDS = ("id", "doc_id", "doc", "stem", "canonical_id")
APD_LIST_FIELDS = ("ids",)
KTIV_PAGE_ID = re.compile(r"KTIV_(\d{12,})_FL\d+")
CANDIDATE_MISSING = "candidate_missing"


# ===================================================================================================================
# Identity
# ===================================================================================================================

def key_of(mark: str) -> str:
    """The facade loose key of a mark or an ES id (``""`` for a namespaced non-ES id such as ``apd:<doc>``).

    :param mark: Shelf mark or id.
    :type mark: str
    :return: ``shelfmarks.loose_key(mark)`` or ``""``.
    :rtype: str
    """
    mark = (mark or "").strip()
    if not mark or re.match(r"^[a-z_]+:", mark):
        return ""
    return shelf.loose_key(mark)


def mark_parts(mark: str) -> List[str]:
    """Single marks of a (possibly joined) shelf mark.

    :param mark: Raw shelf mark.
    :type mark: str
    :return: ``shelfmarks.split`` parts (empty for a blank mark).
    :rtype: List[str]
    """
    return shelf.split(mark or "")


@dataclass
class Identity:
    """Everything a candidate is known by.

    :ivar ids: ES ids, as recorded and as resolved.
    :ivar pgpids: PGP ids, as recorded and as the index links them.
    :ivar keys: Facade loose keys of the ids and the shelf-mark parts.
    :ivar sys_nums: KTIV system numbers.
    :ivar names: Other identifiers (row id and its tail, row stem, namespaced ids without the namespace, ``pgpid:<n>``).
    """
    ids: Set[str] = field(default_factory=set)
    pgpids: Set[str] = field(default_factory=set)
    keys: Set[str] = field(default_factory=set)
    sys_nums: Set[str] = field(default_factory=set)
    names: Set[str] = field(default_factory=set)


def candidate_identity(cand: Candidate, index: Optional[shelf.ShelfmarkIndex] = None) -> Identity:
    """A candidate's identity (``build_legacy_store_rows.identity_leakage``'s expansion, every id kept).

    :param cand: Candidate.
    :type cand: Candidate
    :param index: Merged-corpus index for ``shelfmarks.resolve``, or None (ids as recorded only).
    :type index: Optional[shelf.ShelfmarkIndex]
    :return: The identity.
    :rtype: Identity
    """
    ident = Identity()
    ident.ids = {c for c in cand.canonical_ids if c and not re.match(r"^[a-z_]+:", c)}
    for part in mark_parts(cand.shelfmark):
        cid = shelf.canonical_id(part)
        if cid:
            ident.ids.add(cid)
    ident.pgpids = {str(p) for p in cand.pgpids if str(p).strip()}
    if index is not None:
        for cid in list(ident.ids):
            ident.ids |= lsr.es_ids(cid, index)
        ident.ids |= lsr.es_ids(cand.shelfmark, index)
        for pgpid in list(ident.pgpids):
            ident.ids |= lsr.es_ids(f"pgpid:{pgpid}", index)
        for cid in ident.ids:
            ident.pgpids |= {str(p) for p in index.pgpids.get(cid) or ()}
    ident.keys = {key_of(x) for x in ident.ids | set(mark_parts(cand.shelfmark))} - {""}
    if cand.ktiv_id:
        ident.sys_nums.add(str(cand.ktiv_id))
    ident.names = {cand.row_id} | {f"pgpid:{p}" for p in ident.pgpids}
    if ":" in cand.row_id:
        ident.names.add(cand.row_id.split(":", 1)[1])
    stem = (cand.notes or {}).get("stem")
    if isinstance(stem, str) and stem:
        ident.names.add(stem)
    for cid in cand.canonical_ids:
        if cid:
            ident.names.add(cid)
            if ":" in cid:
                ident.names.add(cid.split(":", 1)[1])
    return ident


@dataclass
class BenchmarkSet:
    """One benchmark's documents, each identifier mapped to the benchmark id it stands for (the reason's ``<id>``).

    :ivar name: Registry name, ``pgp131`` or ``religious``.
    :ivar ids: ES id -> benchmark id.
    :ivar pgpids: PGP ids.
    :ivar sys_nums: KTIV system number -> benchmark id.
    :ivar keys: Loose key -> benchmark id.
    """
    name: str
    ids: Dict[str, str] = field(default_factory=dict)
    pgpids: Set[str] = field(default_factory=set)
    sys_nums: Dict[str, str] = field(default_factory=dict)
    keys: Dict[str, str] = field(default_factory=dict)

    def add_id(self, cid: str, label: str, index: Optional[shelf.ShelfmarkIndex]) -> None:
        """Add a benchmark id (an ES id or a KTIV page id), its loose key and what it resolves to.

        :param cid: The id.
        :type cid: str
        :param label: Benchmark id it stands for.
        :type label: str
        :param index: Merged-corpus index, or None.
        :type index: Optional[shelf.ShelfmarkIndex]
        """
        if not cid:
            return
        page = KTIV_PAGE_ID.fullmatch(cid)
        if page:
            self.sys_nums.setdefault(page.group(1), label)
        self.ids.setdefault(cid, label)
        self.add_mark(cid, label, index)

    def add_mark(self, mark: str, label: str, index: Optional[shelf.ShelfmarkIndex]) -> None:
        """Add the loose keys of a shelf mark (or id) and the ES ids it resolves to.

        :param mark: Shelf mark, possibly joined, or an id.
        :type mark: str
        :param label: Benchmark id it stands for.
        :type label: str
        :param index: Merged-corpus index, or None.
        :type index: Optional[shelf.ShelfmarkIndex]
        """
        for part in mark_parts(mark):
            key = key_of(part)
            if key:
                self.keys.setdefault(key, label)
        if index is not None:
            for resolved in sorted(lsr.es_ids(mark, index)):
                self.ids.setdefault(resolved, label)
                key = key_of(resolved)
                if key:
                    self.keys.setdefault(key, label)

    def reasons(self, ident: Identity) -> List[str]:
        """Reasons a candidate is one of this benchmark's documents.

        :param ident: The candidate's identity.
        :type ident: Identity
        :return: ``benchmark:<name>:<id>`` codes, without duplicates.
        :rtype: List[str]
        """
        hits = [self.ids[c] for c in sorted(ident.ids) if c in self.ids]
        hits += [f"pgpid:{p}" for p in sorted(ident.pgpids & self.pgpids)]
        hits += [self.sys_nums[s] for s in sorted(ident.sys_nums) if s in self.sys_nums]
        hits += [self.keys[k] for k in sorted(ident.keys) if k in self.keys]
        return list(dict.fromkeys(f"benchmark:{self.name}:{hit}" for hit in hits))


def registry_set(name: str, registry_dir: Path, pgp_marks: Mapping[str, str],
                 index: Optional[shelf.ShelfmarkIndex]) -> BenchmarkSet:
    """One registered benchmark (``benchmark_registry`` file), as ``build_legacy_store_rows.held_out_ids`` expands it.

    :param name: Registry name.
    :type name: str
    :param registry_dir: Directory of the registry files (the default resolves registries kept elsewhere).
    :type registry_dir: Path
    :param pgp_marks: ``{pgpid: shelf mark}`` of the pinned PGP export (``build_legacy_store_rows.pgp_shelf_marks``).
    :type pgp_marks: Mapping[str, str]
    :param index: Merged-corpus index, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :return: The set.
    :rtype: BenchmarkSet
    :raises FileNotFoundError: When the registry file is missing (a build must know what is held out).
    """
    ids, pgpids = br.registered_benchmark_documents([name], registry_dir)
    out = BenchmarkSet(name, pgpids=set(pgpids))
    for cid in sorted(ids):
        out.add_id(cid, cid, index)
    for pgpid in sorted(pgpids):
        label = f"pgpid:{pgpid}"
        if index is not None:
            for resolved in sorted(lsr.es_ids(label, index)):
                out.add_id(resolved, label, None)
        out.add_mark(pgp_marks.get(pgpid, ""), label, index)
    for sys_num in sorted(br.registered_benchmark_manuscripts([name], registry_dir)):
        out.sys_nums.setdefault(sys_num, f"sys:{sys_num}")
    return out


def frozen_set(ids_path: Path, keys_path: Optional[Path], index: Optional[shelf.ShelfmarkIndex]) -> BenchmarkSet:
    """The frozen 131-fragment benchmark: its ES ids and its key inventory.

    :param ids_path: ``benchmark_ids.json`` (a JSON list of ids).
    :type ids_path: Path
    :param keys_path: ``benchmark_keys.json`` (a JSON list of loose keys), or None.
    :type keys_path: Optional[Path]
    :param index: Merged-corpus index, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :return: The set (``pgp131``).
    :rtype: BenchmarkSet
    :raises FileNotFoundError: When the id list is missing.
    """
    out = BenchmarkSet(FROZEN_NAME)
    for cid in json.loads(ids_path.read_text(encoding="utf-8")):
        out.add_id(str(cid), str(cid), index)
    if keys_path is not None and keys_path.exists():
        for key in json.loads(keys_path.read_text(encoding="utf-8")):
            out.keys.setdefault(str(key), str(key))
    return out


def religious_set(path: Path) -> BenchmarkSet:
    """The religious page benchmark: its pages' KTIV manuscripts and shelf marks.

    :param path: The benchmark JSON (``docs`` with ``doc_id``, ``sys_num``, ``shelf_mark``).
    :type path: Path
    :return: The set (``religious``).
    :rtype: BenchmarkSet
    :raises FileNotFoundError: When the benchmark file is missing.
    """
    out = BenchmarkSet(RELIGIOUS_NAME)
    for doc in json.loads(path.read_text(encoding="utf-8"))["docs"]:
        label = str(doc["doc_id"])
        out.ids.setdefault(label, label)
        if doc.get("sys_num"):
            out.sys_nums.setdefault(str(doc["sys_num"]), label)
        for part in mark_parts(doc.get("shelf_mark") or ""):
            key = key_of(part)
            if key:
                out.keys.setdefault(key, label)
    return out


def load_apd_exclusions(files: Sequence[Path]) -> Dict[str, str]:
    """Identifiers of the APD exclusion lists.

    :param files: JSONL files; a line is an id string or a record whose :data:`APD_ID_FIELDS` hold one id each, whose
        :data:`APD_LIST_FIELDS` hold lists of ids and whose ``pgpids`` hold PGP ids.
    :type files: Sequence[Path]
    :return: ``{identifier: identifier}`` (``pgpid:<n>`` for PGP ids).
    :rtype: Dict[str, str]
    :raises FileNotFoundError: When a given file is missing.
    """
    out: Dict[str, str] = {}
    for path in files:
        for record in read_jsonl(path):
            if isinstance(record, str):
                out.setdefault(record, record)
                continue
            for name in APD_ID_FIELDS:
                value = record.get(name)
                if isinstance(value, str) and value:
                    out.setdefault(value, value)
            for name in APD_LIST_FIELDS:
                for value in record.get(name) or []:
                    out.setdefault(str(value), str(value))
            for pgpid in record.get("pgpids") or []:
                out.setdefault(f"pgpid:{pgpid}", f"pgpid:{pgpid}")
    return out


@dataclass
class NamedExclusion:
    """A document held out by name.

    :ivar name: Its name (the reason's ``<name>``).
    :ivar pgpids: PGP ids.
    :ivar ids: ES ids (given, and of the marks).
    :ivar keys: Loose keys of the marks and ids.
    """
    name: str
    pgpids: Set[str] = field(default_factory=set)
    ids: Set[str] = field(default_factory=set)
    keys: Set[str] = field(default_factory=set)

    def matches(self, ident: Identity) -> bool:
        """Whether a candidate is this document.

        :param ident: The candidate's identity.
        :type ident: Identity
        :return: True on a shared PGP id, ES id or loose key, or when the name is one of the candidate's names.
        :rtype: bool
        """
        return bool(ident.pgpids & self.pgpids or ident.ids & self.ids or ident.keys & self.keys
                    or self.name in ident.names)


def compile_named(named: Mapping[str, Mapping[str, Sequence[str]]]) -> List[NamedExclusion]:
    """Named exclusions with their ids and keys.

    :param named: ``{name: {"pgpids": [...], "marks": [...], "ids": [...]}}``.
    :type named: Mapping[str, Mapping[str, Sequence[str]]]
    :return: The exclusions.
    :rtype: List[NamedExclusion]
    """
    out = []
    for name, spec in named.items():
        entry = NamedExclusion(name, {str(p) for p in spec.get("pgpids") or []}, set(spec.get("ids") or []))
        for mark in list(spec.get("marks") or []) + list(spec.get("ids") or []):
            for part in mark_parts(mark):
                cid = shelf.canonical_id(part)
                if cid:
                    entry.ids.add(cid)
                key = key_of(part)
                if key:
                    entry.keys.add(key)
        out.append(entry)
    return out


# ===================================================================================================================
# Resources and decisions
# ===================================================================================================================

@dataclass(frozen=True)
class TextRule:
    """How shingles shared with a GT document hold a side out.

    :ivar name: ``strict`` (``build_legacy_store_rows.text_leakage`` as it stands) or ``boilerplate`` (the same
        thresholds over the shared shingles that are not boilerplate, plus the long-run and share guards).
    :ivar common_df: Pool document groups holding a shingle from which it is boilerplate (at least 2: the side's own
        group always holds its shingles).
    :ivar long_run_letters: Shared run length, in letters, that holds out whatever the shingles' status.
    :ivar share_threshold: Share of a side's shingles found in one GT set that holds out whatever their status.
    """

    name: str = "boilerplate"
    common_df: int = COMMON_DF
    long_run_letters: int = LONG_RUN_LETTERS
    share_threshold: float = SHARE_THRESHOLD

    def __post_init__(self) -> None:
        """Validate the parameters.

        :raises ValueError: On an unknown name, ``common_df`` below 2, or thresholds that cannot fire or always fire.
        """
        if self.name not in TEXT_RULES:
            raise ValueError(f"text rule {self.name!r}: expected one of {TEXT_RULES}")
        if self.common_df < 2:
            raise ValueError(f"common_df {self.common_df}: a shingle is always held by the side's own group")
        if self.long_run_letters < lsr.HEBREW_SHINGLE:
            raise ValueError(f"long_run_letters {self.long_run_letters}: shorter than one shingle")
        if not 0.0 < self.share_threshold <= 1.0:
            raise ValueError(f"share_threshold {self.share_threshold}: expected a share in (0, 1]")


@dataclass
class PoolFrequency:
    """Document frequency of the GT shingles over the pool: in how many independent document groups each occurs.

    Only the shingles of the GT index are counted (no other shingle can match a GT document), which keeps the
    counters small.

    :ivar hebrew: Hebrew 25-letter shingle -> number of document groups holding it.
    :ivar arabic: Rasm 20-letter shingle -> number of document groups holding it.
    :ivar groups: Document groups counted.
    """

    hebrew: Counter
    arabic: Counter
    groups: int


def pool_frequency(cands: Iterable[Candidate], gt: lsr.GtIndex) -> PoolFrequency:
    """Count, per GT shingle, the pool document groups whose targets hold it.

    Every candidate of a group contributes to one count (two editions of one fragment are one group), so a passage
    specific to one document that the pool holds twice through two sources stays rare.

    :param cands: Every candidate of the build (their ``targets``).
    :type cands: Iterable[Candidate]
    :param gt: GT index (its shingles are the ones counted).
    :type gt: lsr.GtIndex
    :return: The frequencies.
    :rtype: PoolFrequency
    """
    relevant_hebrew: Set[str] = set().union(*gt.hebrew.values()) if gt.hebrew else set()
    relevant_arabic: Set[str] = set().union(*(index.keys() for index in gt.arabic.values())) if gt.arabic else set()
    per_group_hebrew: Dict[str, Set[str]] = defaultdict(set)
    per_group_arabic: Dict[str, Set[str]] = defaultdict(set)
    groups: Set[str] = set()
    for cand in cands:
        groups.add(cand.document_group)
        for target in cand.targets:
            if relevant_hebrew:
                per_group_hebrew[cand.document_group] |= lsr.hebrew_shingles(target.text) & relevant_hebrew
            if relevant_arabic:
                per_group_arabic[cand.document_group] |= lsr.arabic_shingles(target.text) & relevant_arabic
    hebrew: Counter = Counter(s for shingles in per_group_hebrew.values() for s in shingles)
    arabic: Counter = Counter(s for shingles in per_group_arabic.values() for s in shingles)
    return PoolFrequency(hebrew, arabic, len(groups))


def _default_contaminated(text: str, shingles: set) -> bool:
    """The Hebrew content rule of ``DecontamGate.content_contaminated``: at least two shared 25-letter shingles.

    :param text: Target text.
    :type text: str
    :param shingles: GT shingles.
    :type shingles: set
    :return: True when contaminated.
    :rtype: bool
    """
    return lsr.hebrew_hits(text, shingles) >= 2


@dataclass
class HoldoutResources:
    """Everything the hold-out reads.

    :ivar benchmarks: Benchmark sets (registries, ``pgp131``, ``religious``).
    :ivar apd: APD exclusion identifiers.
    :ivar named: Named exclusions.
    :ivar gt_sets: GT texts per set (``build_legacy_store_rows.load_gt_sets``).
    :ivar gt_index: Their shingles (``build_legacy_store_rows.build_gt_index``).
    :ivar contaminated: The Hebrew content rule (``DecontamGate.content_contaminated``).
    :ivar index: Merged-corpus index, or None.
    :ivar shingle_cache: Hebrew shingles per GT document, filled on first use.
    :ivar text_rule: The text rule (:class:`TextRule`).
    :ivar pool: Pool document frequencies of the GT shingles (filled by :func:`decide_holdout` under the
        ``boilerplate`` rule; None under ``strict``).
    """
    benchmarks: List[BenchmarkSet]
    apd: Dict[str, str]
    named: List[NamedExclusion]
    gt_sets: Dict[str, List[Tuple[str, str]]]
    gt_index: lsr.GtIndex
    contaminated: Callable[[str, set], bool] = _default_contaminated
    index: Optional[shelf.ShelfmarkIndex] = None
    shingle_cache: Dict[Tuple[str, int], Set[str]] = field(default_factory=dict)
    text_rule: TextRule = TextRule()
    pool: Optional[PoolFrequency] = None


def make_resources(benchmarks: Sequence[BenchmarkSet], gt_sets: Mapping[str, Sequence[Tuple[str, str]]], *,
                   apd: Optional[Mapping[str, str]] = None,
                   named: Mapping[str, Mapping[str, Sequence[str]]] = NAMED_EXCLUSIONS,
                   contaminated: Callable[[str, set], bool] = _default_contaminated,
                   index: Optional[shelf.ShelfmarkIndex] = None, text_rule: TextRule = TextRule()) -> HoldoutResources:
    """Assemble the resources (builds the GT index).

    :param benchmarks: Benchmark sets.
    :type benchmarks: Sequence[BenchmarkSet]
    :param gt_sets: GT texts per set.
    :type gt_sets: Mapping[str, Sequence[Tuple[str, str]]]
    :param apd: APD exclusion identifiers.
    :type apd: Optional[Mapping[str, str]]
    :param named: Named exclusions.
    :type named: Mapping[str, Mapping[str, Sequence[str]]]
    :param contaminated: The Hebrew content rule.
    :type contaminated: Callable[[str, set], bool]
    :param index: Merged-corpus index, or None.
    :type index: Optional[shelf.ShelfmarkIndex]
    :param text_rule: The text rule.
    :type text_rule: TextRule
    :return: The resources.
    :rtype: HoldoutResources
    """
    sets = {name: list(items) for name, items in gt_sets.items()}
    return HoldoutResources(list(benchmarks), dict(apd or {}), compile_named(named), sets, lsr.build_gt_index(sets),
                            contaminated, index, text_rule=text_rule)


def identity_reasons(cand: Candidate, res: HoldoutResources) -> List[str]:
    """Identity reasons of one candidate.

    :param cand: Candidate.
    :type cand: Candidate
    :param res: Resources.
    :type res: HoldoutResources
    :return: ``benchmark:...`` and ``exclusion:...`` codes.
    :rtype: List[str]
    """
    ident = candidate_identity(cand, res.index)
    reasons = [r for bench in res.benchmarks for r in bench.reasons(ident)]
    reasons += [f"exclusion:{APD_EXCLUSION}:{res.apd[n]}" for n in sorted(ident.names | ident.ids) if n in res.apd]
    reasons += [f"exclusion:named:{entry.name}" for entry in res.named if entry.matches(ident)]
    return list(dict.fromkeys(reasons))


def _hebrew_doc(text: str, set_name: str, res: HoldoutResources) -> Optional[str]:
    """The GT document of a set sharing the most 25-letter Hebrew shingles with a text.

    :param text: Target text.
    :type text: str
    :param set_name: GT set.
    :type set_name: str
    :param res: Resources (shingles cached per document).
    :type res: HoldoutResources
    :return: Its label, or None when none shares a shingle.
    :rtype: Optional[str]
    """
    best, label = 0, None
    for i, (doc, gt) in enumerate(res.gt_sets.get(set_name, [])):
        shingles = res.shingle_cache.get((set_name, i))
        if shingles is None:
            shingles = res.shingle_cache[(set_name, i)] = lsr.hebrew_shingles(gt)
        hits = lsr.hebrew_hits(text, shingles)
        if hits > best:
            best, label = hits, doc
    return label


def text_reasons(text: str, res: HoldoutResources) -> List[str]:
    """Text reasons of one side: ``text:<set>:<GT document>`` for every GT set whose shingles it shares.

    :param text: Final target text.
    :type text: str
    :param res: Resources.
    :type res: HoldoutResources
    :return: Codes (``build_legacy_store_rows.text_leakage``'s decisions, with the GT document named).
    :rtype: List[str]
    """
    leak = lsr.text_leakage(text, res.gt_index, res.contaminated)
    reasons = [f"text:{name}:{_hebrew_doc(text, name, res) or f'hebrew_{lsr.HEBREW_SHINGLE}_letter_shingles'}"
               for name in leak["hebrew"]]
    reasons += [f"text:{name}:{info.get('doc') or f'rasm_{lsr.ARABIC_SHINGLE}_shingles'}"
                for name, info in leak["arabic"].items()]
    return list(dict.fromkeys(reasons))


def _hebrew_positions(text: str, n: int = lsr.HEBREW_SHINGLE) -> List[str]:
    """The Hebrew shingles of a text in position order (the gate's letter class, no folding).

    :param text: Text.
    :type text: str
    :param n: Shingle length.
    :type n: int
    :return: One shingle per position.
    :rtype: List[str]
    """
    letters = "".join(_HEB.findall(text or ""))
    return [letters[i:i + n] for i in range(len(letters) - n + 1)]


def _rasm_positions(text: str, n: int = lsr.ARABIC_SHINGLE) -> List[str]:
    """The rasm shingles of a text in position order (``build_legacy_store_rows.arabic_shingles``'s skeleton).

    :param text: Text.
    :type text: str
    :param n: Shingle length.
    :type n: int
    :return: One shingle per position.
    :rtype: List[str]
    """
    rasm = lsr.rasm_skeleton(text or "")
    return [rasm[i:i + n] for i in range(len(rasm) - n + 1)]


def longest_shared_run(positions: Sequence[str], shared: Callable[[str], bool], n: int) -> int:
    """Letters spanned by the longest run of consecutive positions whose shingle is shared.

    :param positions: Shingles in position order.
    :type positions: Sequence[str]
    :param shared: Whether a shingle is shared.
    :type shared: Callable[[str], bool]
    :param n: Shingle length.
    :type n: int
    :return: ``run + n - 1`` letters for a run of ``run`` positions; 0 when no position is shared.
    :rtype: int
    """
    best = run = 0
    for shingle in positions:
        run = run + 1 if shared(shingle) else 0
        best = max(best, run)
    return best + n - 1 if best else 0


def hebrew_match(text: str, set_name: str, res: HoldoutResources) -> Tuple[bool, bool, Optional[str]]:
    """One side against one GT set's Hebrew shingles: the strict decision, the rule's decision, the document.

    Strict: at least two shingle positions shared with the set (``DecontamGate.content_contaminated``'s threshold).
    Boilerplate: at least two shared positions whose shingle fewer than ``common_df`` pool groups hold, or a shared
    run of ``long_run_letters``, or ``share_threshold`` of the positions shared.

    :param text: Final target text.
    :type text: str
    :param set_name: GT set.
    :type set_name: str
    :param res: Resources (``pool`` filled under the boilerplate rule).
    :type res: HoldoutResources
    :return: ``(strict, held, doc)``: whether the strict rule holds the side out, whether the resources' rule does,
        and the GT document sharing the most shingles (None when none is shared).
    :rtype: Tuple[bool, bool, Optional[str]]
    """
    shingles = res.gt_index.hebrew.get(set_name) or set()
    positions = _hebrew_positions(text)
    hits = [shingle in shingles for shingle in positions]
    n_hits = sum(hits)
    strict = n_hits >= 2
    if not strict:
        return False, False, None
    doc = _hebrew_doc(text, set_name, res)
    rule = res.text_rule
    if rule.name == "strict":
        return True, True, doc
    pool = res.pool.hebrew if res.pool is not None else Counter()
    specific = sum(1 for shingle, hit in zip(positions, hits) if hit and pool[shingle] < rule.common_df)
    run = longest_shared_run(positions, shingles.__contains__, lsr.HEBREW_SHINGLE)
    share = n_hits / len(positions)
    held = specific >= 2 or run >= rule.long_run_letters or share >= rule.share_threshold
    return True, held, doc


def arabic_match(text: str, set_name: str, res: HoldoutResources) -> Tuple[bool, bool, Optional[str]]:
    """One side against one GT set's rasm shingles: the strict decision, the rule's decision, the document.

    Strict (``build_legacy_store_rows.text_leakage``): at least ``ARABIC_MIN_SHARED_ONE_DOC`` shingles shared with
    one GT document, or ``ARABIC_MIN_SHARE_UNION`` of the side's shingles shared with the set. Boilerplate: the
    one-document count over the shingles fewer than ``common_df`` pool groups hold, the share guard over every shared
    shingle (``share_threshold``), and the long-run guard (``long_run_letters``, rasm letters).

    :param text: Final target text.
    :type text: str
    :param set_name: GT set.
    :type set_name: str
    :param res: Resources (``pool`` filled under the boilerplate rule).
    :type res: HoldoutResources
    :return: ``(strict, held, doc)`` as :func:`hebrew_match`; the document is the one sharing the most counted
        shingles.
    :rtype: Tuple[bool, bool, Optional[str]]
    """
    index = res.gt_index.arabic.get(set_name) or {}
    positions = _rasm_positions(text)
    mine = set(positions)
    if not mine or not index:
        return False, False, None
    per_doc: Counter = Counter()
    union = 0
    for shingle in mine:
        holders = index.get(shingle)
        if holders:
            union += 1
            per_doc.update(holders)
    top = per_doc.most_common(1)
    max_one = top[0][1] if top else 0
    share = union / len(mine)
    strict = max_one >= lsr.ARABIC_MIN_SHARED_ONE_DOC or share >= lsr.ARABIC_MIN_SHARE_UNION
    doc = res.gt_index.docs[set_name][top[0][0]] if top else None
    if not strict:
        return False, False, doc
    rule = res.text_rule
    if rule.name == "strict":
        return True, True, doc
    pool = res.pool.arabic if res.pool is not None else Counter()
    per_doc_specific: Counter = Counter()
    for shingle in mine:
        holders = index.get(shingle)
        if holders and pool[shingle] < rule.common_df:
            per_doc_specific.update(holders)
    top_specific = per_doc_specific.most_common(1)
    max_specific = top_specific[0][1] if top_specific else 0
    run = longest_shared_run(positions, index.__contains__, lsr.ARABIC_SHINGLE)
    held = (max_specific >= lsr.ARABIC_MIN_SHARED_ONE_DOC or share >= rule.share_threshold
            or run >= rule.long_run_letters)
    if held and top_specific:
        doc = res.gt_index.docs[set_name][top_specific[0][0]]
    return True, held, doc


def text_decision(text: str, res: HoldoutResources) -> Tuple[List[str], List[str]]:
    """Text reasons and boilerplate flags of one side under the resources' text rule.

    Under ``strict`` this is :func:`text_reasons` with no flags. Under ``boilerplate`` every GT set is judged by
    :func:`hebrew_match` and :func:`arabic_match`.

    :param text: Final target text.
    :type text: str
    :param res: Resources.
    :type res: HoldoutResources
    :return: ``(reasons, flags)``: ``text:<set>:<doc>`` for every set the rule holds the side out on, and
        ``boilerplate:<set>:<doc>`` for every set the strict rule would have held it out on but the exception admits.
    :rtype: Tuple[List[str], List[str]]
    """
    if res.text_rule.name == "strict":
        return text_reasons(text, res), []
    reasons: List[str] = []
    flags: List[str] = []
    for name in res.gt_sets:
        hebrew = hebrew_match(text, name, res)
        arabic = arabic_match(text, name, res)
        for (strict, held, doc), fallback in ((hebrew, f"hebrew_{lsr.HEBREW_SHINGLE}_letter_shingles"),
                                              (arabic, f"rasm_{lsr.ARABIC_SHINGLE}_shingles")):
            if held:
                reasons.append(f"text:{name}:{doc or fallback}")
            elif strict:
                flags.append(f"{BOILERPLATE_FLAG}:{name}:{doc or fallback}")
    return list(dict.fromkeys(reasons)), list(dict.fromkeys(flags))


def split_of(group: str, val_share: float = VAL_SHARE) -> str:
    """Deterministic split of a document group.

    :param group: Document group.
    :type group: str
    :param val_share: Share of groups in validation.
    :type val_share: float
    :return: ``val`` when the first 8 hex digits of the group's SHA-1, over ``0xFFFFFFFF``, fall below ``val_share``;
        else ``train``.
    :rtype: str
    """
    return "val" if int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < val_share else "train"


def decide_holdout(cands: Mapping[str, Candidate], quality: Iterable[QualityResult], res: HoldoutResources,
                   val_share: float = VAL_SHARE) -> List[HoldoutResult]:
    """Hold-out and split of every quality-passed side.

    Identity reasons of every candidate of a document group hold out every side of the group (a benchmark fragment
    gives no row through another source or edition); text reasons hold out the side whose text matched. Under the
    ``boilerplate`` text rule the pool document frequencies are counted here from ``cands`` (once per resources).

    :param cands: Candidates by row id (every candidate of the build, so a group's other candidates count and the
        pool frequencies cover the whole pool).
    :type cands: Mapping[str, Candidate]
    :param quality: Quality results (only ``ok`` ones get a record).
    :type quality: Iterable[QualityResult]
    :param res: Resources.
    :type res: HoldoutResources
    :param val_share: Share of document groups in validation.
    :type val_share: float
    :return: One record per passed side, in input order.
    :rtype: List[HoldoutResult]
    """
    passed = [q for q in quality if q.ok]
    if res.text_rule.name == "boilerplate" and res.pool is None:
        res.pool = pool_frequency(cands.values(), res.gt_index)
    groups = {cands[q.row_id].document_group for q in passed if q.row_id in cands}
    by_group: Dict[str, List[str]] = defaultdict(list)
    for cand in cands.values():
        if cand.document_group in groups:
            by_group[cand.document_group].extend(identity_reasons(cand, res))
    out = []
    for q in passed:
        cand = cands.get(q.row_id)
        if cand is None:
            out.append(HoldoutResult(q.row_id, q.side_key, True, [CANDIDATE_MISSING], "excluded", False))
            continue
        text, flags = text_decision(q.text, res)
        reasons = list(dict.fromkeys(by_group[cand.document_group] + text))
        split = "excluded" if reasons else split_of(cand.document_group, val_share)
        out.append(HoldoutResult(q.row_id, q.side_key, bool(reasons), reasons, split, bool(cand.rnl), flags))
    return out


# ===================================================================================================================
# CLI
# ===================================================================================================================

def pgp_fragments(pgp_export: Path) -> List[Dict[str, str]]:
    """Rows of the PGP export's ``fragments.csv`` (PGP ids of multi-document fragments for the index).

    :param pgp_export: Export directory.
    :type pgp_export: Path
    :return: Rows (empty when the file is absent).
    :rtype: List[Dict[str, str]]
    """
    path = pgp_export / "fragments.csv"
    if not path.exists():
        return []
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_resources(*, apd_files: Sequence[Path] = (),
                   named: Mapping[str, Mapping[str, Sequence[str]]] = NAMED_EXCLUSIONS,
                   use_index: bool = True, merged: Path = shelf.MERGED_SHELFMARKS, pgp_export: Path = lsr.PGP_EXPORT,
                   index_cache: Optional[Path] = None, registry_dir: Path = br.REGISTRY_DIR,
                   registries: Sequence[str] = br.BENCHMARK_REGISTRIES,
                   text_rule: TextRule = TextRule()) -> HoldoutResources:
    """The real registries, benchmark lists, exclusion lists, GT texts and merged-corpus index (read only).

    :param apd_files: APD exclusion lists.
    :type apd_files: Sequence[Path]
    :param named: Named exclusions.
    :type named: Mapping[str, Mapping[str, Sequence[str]]]
    :param use_index: Build the merged-corpus index (False: ids as recorded and loose keys only).
    :type use_index: bool
    :param merged: Merged corpus JSONL.
    :type merged: Path
    :param pgp_export: Pinned PGP export (``documents.csv``, ``fragments.csv``).
    :type pgp_export: Path
    :param index_cache: Cache directory of the index, or None.
    :type index_cache: Optional[Path]
    :param registry_dir: Registry directory.
    :type registry_dir: Path
    :param registries: Registry names.
    :type registries: Sequence[str]
    :param text_rule: The text rule.
    :type text_rule: TextRule
    :return: The resources.
    :rtype: HoldoutResources
    """
    index = (shelf.ShelfmarkIndex(merged, fragments=pgp_fragments(pgp_export), cache_dir=index_cache)
             if use_index else None)
    documents = pgp_export / "documents.csv"
    marks = lsr.pgp_shelf_marks(documents) if documents.exists() else {}
    benchmarks = [registry_set(name, registry_dir, marks, index) for name in registries]
    benchmarks += [frozen_set(FROZEN_IDS, FROZEN_KEYS, index), religious_set(RELIGIOUS)]
    return make_resources(benchmarks, lsr.load_gt_sets(), apd=load_apd_exclusions(apd_files), named=named,
                          contaminated=DecontamGate().content_contaminated, index=index, text_rule=text_rule)


def summarize(results: Sequence[HoldoutResult]) -> str:
    """Split counts and the reason families.

    :param results: Results.
    :type results: Sequence[HoldoutResult]
    :return: Printable text.
    :rtype: str
    """
    splits = Counter(r.split for r in results)
    families = Counter(":".join(reason.split(":")[:2]) for r in results for reason in r.reasons)
    admitted = [r for r in results if r.flags and not r.held_out]
    flagged = Counter(":".join(flag.split(":")[:2]) for r in results for flag in r.flags)
    text = (f"{len(results):,} passed sides: " + ", ".join(f"{s} {splits[s]:,}" for s in ("train", "val", "excluded"))
            + f"; rnl {sum(r.rnl for r in results):,}\n  reasons: "
            + ", ".join(f"{k} {n:,}" for k, n in families.most_common()))
    if flagged:
        text += (f"\n  boilerplate exception: {len(admitted):,} side(s) admitted that the strict rule holds out; "
                 "matches discounted: " + ", ".join(f"{k} {n:,}" for k, n in flagged.most_common()))
    return text


def text_rule_path(holdout_file: Path) -> Path:
    """The sidecar recording the text rule next to a holdout stage file.

    :param holdout_file: The stage file.
    :type holdout_file: Path
    :return: ``<its directory>/<basename of paths.TEXT_RULE_FILE>``.
    :rtype: Path
    """
    return holdout_file.with_name(Path(paths.TEXT_RULE_FILE).name)


def text_rule_record(rule: TextRule, res: HoldoutResources, results: Sequence[HoldoutResult]) -> Dict[str, Any]:
    """What the sidecar records: the rule's parameters and the exception's effect.

    :param rule: The text rule.
    :type rule: TextRule
    :param res: Resources (``pool`` under the boilerplate rule).
    :type res: HoldoutResources
    :param results: The decisions.
    :type results: Sequence[HoldoutResult]
    :return: ``{name, common_df, long_run_letters, share_threshold, pool_groups, admitted, flagged_sides,
        matches_discounted: {set: n}}``.
    :rtype: Dict[str, Any]
    """
    flagged = [r for r in results if r.flags]
    matches = Counter(":".join(flag.split(":")[:2]) for r in results for flag in r.flags)
    return {"name": rule.name, "common_df": rule.common_df, "long_run_letters": rule.long_run_letters,
            "share_threshold": rule.share_threshold, "pool_groups": res.pool.groups if res.pool else None,
            "admitted": sum(1 for r in flagged if not r.held_out), "flagged_sides": len(flagged),
            "matches_discounted": dict(sorted(matches.items()))}


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Stage 5: hold-outs and the train / val split of the passed sides.")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="build work dir")
    parser.add_argument("--candidates", type=Path, default=None, help="candidates (default: the images stage file)")
    parser.add_argument("--quality", type=Path, default=None, help="quality results (default: the quality stage file)")
    parser.add_argument("--out", type=Path, default=None, help="output (default: the holdout stage file)")
    parser.add_argument("--val-share", type=float, default=VAL_SHARE, help="share of document groups in validation")
    parser.add_argument("--apd-exclusions", type=Path, action="append", default=[],
                        help=f"APD exclusion list (JSONL; repeatable; ${APD_EXCLUSIONS_ENV} adds defaults)")
    parser.add_argument("--named-exclusions", type=Path, default=None,
                        help="JSON {name: {pgpids, marks, ids}} replacing the built-in named exclusions")
    parser.add_argument("--no-index", action="store_true", help="do not build the merged-corpus index")
    parser.add_argument("--merged", type=Path, default=shelf.MERGED_SHELFMARKS, help="merged corpus JSONL")
    parser.add_argument("--pgp-export", type=Path, default=lsr.PGP_EXPORT, help="pinned PGP export directory")
    parser.add_argument("--index-cache", type=Path, default=None,
                        help="cache directory of the index (default: <work dir>/cache/shelfmark_index)")
    parser.add_argument("--sources", default=None, help="comma-separated sources (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="decide at most this many passed sides")
    parser.add_argument("--text-rule", choices=TEXT_RULES, default="boilerplate",
                        help="text rule: strict, or boilerplate (shared shingles held by --common-df pool document "
                             "groups do not count; the long-run and share guards still hold out)")
    parser.add_argument("--common-df", type=int, default=COMMON_DF,
                        help="pool document groups holding a shingle from which it is boilerplate")
    parser.add_argument("--long-run-letters", type=int, default=LONG_RUN_LETTERS,
                        help="a shared run of this many letters holds out whatever its shingles' status")
    parser.add_argument("--share-threshold", type=float, default=SHARE_THRESHOLD,
                        help="this share of a side's shingles in one GT set holds out whatever their status")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Hold out and split the passed sides of a work dir; write the holdout stage file.

    The stage is recomputed from its inputs on every run, so there is nothing to resume.

    :param argv: Command-line arguments (None: ``sys.argv``).
    :type argv: Optional[List[str]]
    """
    args = build_parser().parse_args(argv)
    cands = {c.row_id: c for c in read_jsonl(args.candidates or paths.stage_path(args.work_dir, "images"), Candidate)}
    sources = {s.strip() for s in args.sources.split(",") if s.strip()} if args.sources else None
    quality = [q for q in read_jsonl(args.quality or paths.stage_path(args.work_dir, "quality"), QualityResult)
               if q.ok and (sources is None or getattr(cands.get(q.row_id), "source", None) in sources)]
    if args.limit is not None:
        quality = quality[:args.limit]
    apd_files = [Path(p) for p in os.environ.get(APD_EXCLUSIONS_ENV, "").split(os.pathsep) if p] + args.apd_exclusions
    if not apd_files:
        print("no APD exclusion list given: no exclusion:derive_v25a_apd reasons")
    _named_path = args.named_exclusions or (Path(os.environ[NAMED_EXCLUSIONS_ENV]) if os.environ.get(NAMED_EXCLUSIONS_ENV) else None)
    named = (json.loads(_named_path.read_text(encoding="utf-8")) if _named_path is not None else NAMED_EXCLUSIONS)
    text_rule = TextRule(args.text_rule, args.common_df, args.long_run_letters, args.share_threshold)
    res = load_resources(apd_files=apd_files, named=named, use_index=not args.no_index, merged=args.merged,
                         pgp_export=args.pgp_export,
                         index_cache=args.index_cache or args.work_dir / "cache" / "shelfmark_index",
                         text_rule=text_rule)
    results = decide_holdout(cands, quality, res, args.val_share)
    print(f"text rule: {text_rule.name}" + (f" (common_df {text_rule.common_df} document groups, long run "
                                            f"{text_rule.long_run_letters} letters, share {text_rule.share_threshold}; "
                                            f"pool groups counted {res.pool.groups:,})"
                                            if res.pool is not None else ""))
    out = args.out or paths.stage_path(args.work_dir, "holdout")
    write_jsonl(out, results)
    sidecar = text_rule_path(out)
    sidecar.write_text(json.dumps(text_rule_record(text_rule, res, results), ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(summarize(results))
    print(f"wrote {out} and {sidecar}")


if __name__ == "__main__":
    main()
