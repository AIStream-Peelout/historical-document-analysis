# File name: holdouts.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Hold-out set H for the text27b build (plan §9.0 / §9.4.1, build brief rule 8).

A document is held out when its PGP id, a canonical id OR a normalised shelfmark key matches H.
The 2026-10-18 builder trusted the canonical-id → pgpid map alone, which silently dropped every
benchmark id the map does not carry (PGP-131's Bodleian leaves are spelled without ``Bodl`` there),
so those documents leaked into train. Here every benchmark id is resolved three ways -- the
pgpid map, the merged corpus, and the institution-free shelfmark key
(:func:`shelfmarks.canonical_to_key`) looked up in every shelfmark source -- and then closed over
shelfmark groups (pgpids sharing a fragment or a merged id; the v0 builder's union-find).

Two tiers:

* **benchmark** -- documents that must never be trained on (they become the ``heldout`` split):
  PGP-131 (``paths.benchmark_ids``), the registered benchmarks
  (:func:`benchmark_registry.registered_benchmark_documents`), v0_1's
  ``held_out_vision_benchmark`` documents, the zero-shot probe documents (``samples.json`` /
  ``exclusions_v1.json``).
* **eval_docs** -- v0_1 dev and test documents, the ``pgp_parallel`` documents whose split is ``dev`` /
  ``test`` but that have no v0_1 rows (the 10-22 fix pass, F1: literature naming such a document is
  kept out too), plus the extra benchmark files not in the registry (``paths.json`` entry
  ``extra_benchmarks``, default :data:`DEFAULT_EXTRA_BENCHMARKS`). These keep their split in the PGP
  families; the tier exists to keep literature text about them out of training (plan §9.4.2).

Reasons. Stored reasons are ``pgp131``, ``registry``, ``vision_flag``, ``probe``, ``dev``, ``test``,
``extra:<benchmark>``; a ``group:`` prefix marks a document added by the group closure, a ``key:``
prefix one resolved (or matched at query time) through a shelfmark key only.

Public API (other text27b modules import these names):

* :class:`Holdouts` -- ``Holdouts.load()``, ``Holdouts.build()``, ``.save()``,
  ``.is_benchmark(pgpid, shelfmark, canonical) -> reason | None``, ``.is_eval_doc(...)``,
  ``.is_held(...)``, ``.match_any(pgpids, canonicals, shelfmarks)``, ``.keys``, ``.eval_keys``,
  ``.pgpids``, ``.eval_pgpids``, ``.reasons_of(pgpid)``.
* :func:`build_holdouts` -- assemble H from the configured sources (the CLI's ``build``).
* Source readers: :func:`load_id_list`, :func:`walk_pgpids`, :func:`probe_pgpids`,
  :func:`load_extra_benchmark`, :func:`scan_v01`, :func:`scan_pairs`.
* Grouping and keys: :class:`UnionFind`, :func:`build_groups`, :class:`KeyIndex`,
  :func:`build_key_index`, :func:`cached_keys`.
* Reasons: :func:`best_reason`, :func:`base_reason`, :func:`derive`.

CLI::

    python -m src.finetuning.text27b.holdouts build [--out PATH] [--no-extra]
    python -m src.finetuning.text27b.holdouts show [--holdouts PATH]
    python -m src.finetuning.text27b.holdouts check [--pgpid ID] [--shelfmark SM] [--canonical CID]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from src.datasets.evaluations import benchmark_registry as _registry
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm

LOG = logging.getLogger(__name__)

#: File name of the hold-out manifest under ``paths.out_dir``.
HOLDOUTS_FILE = "holdouts.json"
#: Format version of ``holdouts.json``.
FORMAT_VERSION = 1
#: Benchmark reasons, most important first (used to pick one reason to report).
BENCHMARK_REASONS: Tuple[str, ...] = ("pgp131", "registry", "vision_flag", "probe")
#: Eval-doc reasons, most important first (``extra:<name>`` reasons rank between them).
EVAL_REASONS: Tuple[str, ...] = ("test", "dev")
#: Prefix of a reason inherited through the shelfmark-group closure.
GROUP_PREFIX = "group:"
#: Prefix of a reason established through a shelfmark key only.
KEY_PREFIX = "key:"
#: Prefix of the reasons of the extra benchmark files.
EXTRA_PREFIX = "extra:"
#: Repository-relative benchmark files that are not in ``BENCHMARK_REGISTRIES`` (plan §9.4.1);
#: ``paths.json`` may override them with an ``extra_benchmarks`` list.
DEFAULT_EXTRA_BENCHMARKS: Tuple[str, ...] = (
    "src/datasets/raw_data/cairo_genizah/evaluations/sealed_v1/sealed_v1.json",
    "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/genizah_religious_v1.json",
)
_V01_SPLITS: Tuple[str, ...] = ("train", "dev", "test")


# ------------------------------------------------------------------------------------ helpers
@functools.lru_cache(maxsize=262144)
def cached_keys(shelfmark: Optional[str]) -> Tuple[str, ...]:
    """Shelfmark keys of a (possibly joined) shelfmark, memoised (:func:`shelfmarks.keys`).

    :param shelfmark: Shelfmark string (None or '' gives no keys).
    :returns: Keys in order, without duplicates.
    """
    return tuple(_sm.keys(shelfmark)) if shelfmark else ()


@functools.lru_cache(maxsize=262144)
def _canonical_key(canonical_id: str) -> str:
    """Memoised :func:`shelfmarks.canonical_to_key`.

    :param canonical_id: Canonical id.
    :returns: Key or ''.
    """
    return _sm.canonical_to_key(canonical_id)


def base_reason(reason: str) -> str:
    """A reason without its ``group:`` / ``key:`` derivation prefixes.

    :param reason: Stored or derived reason.
    :returns: The base reason (``pgp131``, ``test``, ``extra:<name>`` ...).
    """
    while reason.startswith(GROUP_PREFIX) or reason.startswith(KEY_PREFIX):
        reason = reason[len(GROUP_PREFIX):] if reason.startswith(GROUP_PREFIX) else reason[len(KEY_PREFIX):]
    return reason


def derive(prefix: str, reason: str) -> str:
    """A derived reason: one prefix on the base reason (derivations never chain).

    :param prefix: :data:`GROUP_PREFIX` or :data:`KEY_PREFIX`.
    :param reason: Source reason (possibly derived itself).
    :returns: ``prefix + base_reason(reason)``.
    """
    return prefix + base_reason(reason)


def _rank(reason: str) -> Tuple[int, int, int]:
    """Sort key of a reason: base priority first, then direct < ``group:`` < ``key:``, then fewer prefixes.

    :param reason: A stored or derived reason.
    :returns: ``(base rank, derivation rank, number of prefixes)``.
    """
    deriv, depth, r = 0, 0, reason
    while r.startswith(GROUP_PREFIX) or r.startswith(KEY_PREFIX):
        if r.startswith(KEY_PREFIX):
            deriv, r = max(deriv, 2), r[len(KEY_PREFIX):]
        else:
            deriv, r = max(deriv, 1), r[len(GROUP_PREFIX):]
        depth += 1
    order = list(BENCHMARK_REASONS) + ["extra"] + list(EVAL_REASONS)
    head = "extra" if r.startswith(EXTRA_PREFIX) else r
    return (order.index(head) if head in order else len(order), deriv, depth)


def best_reason(reasons: Iterable[str]) -> Optional[str]:
    """Pick the most important reason (benchmark before eval, direct before group / key).

    :param reasons: Reasons.
    :returns: The best reason, or None for an empty input.
    """
    rs = sorted(set(reasons), key=lambda r: (_rank(r), r))
    return rs[0] if rs else None


def _add(mapping: Dict[str, Set[str]], item: str, reason: str) -> bool:
    """Add a reason to an item of a reason map.

    :param mapping: item → reasons.
    :param item: Key (pgpid, canonical id or shelfmark key).
    :param reason: Reason to add.
    :returns: True when the item was new to the map.
    """
    new = item not in mapping
    mapping.setdefault(item, set()).add(reason)
    return new


def _file_sig(path: Path, repo: Path, hash_limit: int = 20_000_000) -> Dict[str, Any]:
    """Describe an input file for the manifest (repository-relative path, size, sha256 when small).

    :param path: File path.
    :param repo: Repository root (paths are written relative to it when possible).
    :param hash_limit: Files up to this size get a sha256.
    :returns: ``{"path", "exists", "bytes", "sha256"?}``.
    """
    try:
        rel = str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        rel = path.name
    if not path.exists():
        return {"path": rel, "exists": False}
    out: Dict[str, Any] = {"path": rel, "exists": True}
    if path.is_file():
        out["bytes"] = path.stat().st_size
        if out["bytes"] <= hash_limit:
            out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------------------------ sources
def load_id_list(path: Union[str, Path]) -> List[str]:
    """Read a JSON list of canonical ids (``benchmark_ids.json``) or a registry-style ``{"ids": [...]}``.

    :param path: JSON file.
    :returns: Canonical ids as strings.
    :raises ValueError: When the file holds neither shape.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("ids"), list):
        return [str(x) for x in data["ids"]]
    raise ValueError(f"{path}: expected a list of ids or {{'ids': [...]}}")


def walk_pgpids(obj: Any) -> Set[str]:
    """Collect every ``pgpid`` / ``pgpids`` value in a JSON structure (recursive).

    :param obj: Parsed JSON.
    :returns: PGP ids as strings.
    """
    out: Set[str] = set()
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "pgpid" and isinstance(v, (str, int)) and str(v).strip():
                    out.add(str(v).strip())
                elif k == "pgpids" and isinstance(v, list):
                    out |= {str(x).strip() for x in v if isinstance(x, (str, int)) and str(x).strip()}
                else:
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return out


def probe_pgpids(samples: Optional[Path], exclusions: Optional[Path]) -> Set[str]:
    """PGP ids of the zero-shot probe: every ``pgpid`` under ``samples.json``'s ``sets`` and in ``exclusions_v1.json``.

    :param samples: ``samples.json`` (skipped when None or missing).
    :param exclusions: ``exclusions_v1.json`` (skipped when None or missing).
    :returns: PGP ids.
    """
    out: Set[str] = set()
    for path, sub in ((samples, "sets"), (exclusions, "entries")):
        if path is None or not Path(path).exists():
            LOG.warning("probe source missing: %s", path)
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        out |= walk_pgpids(data.get(sub, data) if isinstance(data, dict) else data)
    return out


@dataclass
class ExtraBenchmark:
    """An extra benchmark file (``{"benchmark", "ids", "pgpids"}`` or ``{"benchmark", "docs": [{"shelf_mark"}]}``)."""

    name: str
    ids: Set[str]
    pgpids: Set[str]
    shelfmarks: List[str]


def load_extra_benchmark(path: Union[str, Path]) -> ExtraBenchmark:
    """Read an extra benchmark file: canonical ids, pgpids and shelfmark strings, whatever it carries.

    :param path: JSON file.
    :returns: The parsed benchmark (name from ``benchmark``, else the file stem).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    name = str(data.get("benchmark") or Path(path).stem) if isinstance(data, dict) else Path(path).stem
    ids: Set[str] = set()
    sms: List[str] = []
    stack = [data]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("ids", "canonical_ids") and isinstance(v, list):
                    ids |= {str(x) for x in v if x}
                elif k in ("canonical_id", "merged_id") and isinstance(v, str) and v:
                    ids.add(v)
                elif k in ("shelf_mark", "shelfmark") and isinstance(v, str) and v.strip():
                    sms.append(v.strip())
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return ExtraBenchmark(name=name, ids=ids, pgpids=walk_pgpids(data), shelfmarks=sorted(set(sms)))


@dataclass
class V01Doc:
    """What v0_1 says about one pgpid: rows per split, vision flag, shelfmarks, group."""

    splits: Dict[str, int] = field(default_factory=dict)
    vision: bool = False
    shelfmarks: Set[str] = field(default_factory=set)
    group: Optional[str] = None
    tasks: Dict[str, int] = field(default_factory=dict)


def scan_v01(v01_dir: Union[str, Path], splits: Sequence[str] = _V01_SPLITS) -> Dict[str, V01Doc]:
    """Summarise the v0_1 row files per pgpid (``<split>.jsonl`` under ``v01_dir``; missing files skipped).

    :param v01_dir: Directory of the v0_1 build.
    :param splits: Split files to read.
    :returns: pgpid → :class:`V01Doc`.
    """
    out: Dict[str, V01Doc] = {}
    for sp in splits:
        f = Path(v01_dir) / f"{sp}.jsonl"
        if not f.exists():
            LOG.warning("v0_1 split file missing: %s", f)
            continue
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            pg = str(m.get("pgpid") or "")
            if not pg:
                continue
            d = out.setdefault(pg, V01Doc())
            split = str(m.get("split") or sp)
            d.splits[split] = d.splits.get(split, 0) + 1
            task = str(m.get("task") or "")
            d.tasks[task] = d.tasks.get(task, 0) + 1
            d.vision = d.vision or _truthy(m.get("held_out_vision_benchmark"))
            if m.get("shelfmark"):
                d.shelfmarks.add(str(m["shelfmark"]))
            if m.get("shelfmark_group"):
                d.group = str(m["shelfmark_group"])
    return out


def _truthy(value: Any) -> bool:
    """Interpret a JSON flag that may be a bool or a ``"True"`` / ``"False"`` string.

    :param value: Raw value.
    :returns: Its truth value.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


@dataclass
class PairDoc:
    """What ``pgp_parallel`` says about one pgpid."""

    shelfmark: str
    merged_ids: List[str]
    vision: bool
    split: Optional[str]


def scan_pairs(path: Optional[Union[str, Path]]) -> Dict[str, PairDoc]:
    """Read pgpid, shelfmark, merged ids, vision flag and split from ``pgp_parallel`` (missing file → empty).

    :param path: ``pgp_parallel_v1_1.jsonl``.
    :returns: pgpid → :class:`PairDoc`.
    """
    out: Dict[str, PairDoc] = {}
    if path is None or not Path(path).exists():
        LOG.warning("pgp_parallel missing: %s", path)
        return out
    for r in _rows.read_jsonl(path):
        pg = str(r.get("pgpid") or "")
        if not pg:
            continue
        mids = r.get("merged_doc_ids") or []
        if isinstance(mids, str):
            mids = [mids]
        out[pg] = PairDoc(shelfmark=str(r.get("shelfmark") or ""), merged_ids=[str(x) for x in mids],
                          vision=_truthy(r.get("held_out_vision_benchmark")), split=r.get("split"))
    return out


# ------------------------------------------------------------------------------------ groups
class UnionFind:
    """Union-find over numeric pgpid strings; a set's root is its smallest id (the v0 builder's rule)."""

    def __init__(self) -> None:
        """Create an empty structure."""
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        """Root of ``x`` (path halving; unseen ids become singletons).

        :param x: pgpid.
        :returns: Root pgpid.
        """
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Join the sets of ``a`` and ``b`` (the smaller numeric root wins).

        :param a: pgpid.
        :param b: pgpid.
        :returns: None.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=lambda s: (int(s) if s.isdigit() else sys.maxsize, s))
            self.parent[hi] = lo


def build_groups(edge_lists: Iterable[Iterable[str]], universe: Iterable[str] = ()) -> Dict[str, List[str]]:
    """Shelfmark groups: connected components of pgpids that co-occur in any list.

    :param edge_lists: Iterables of pgpids that belong together (a fragment's pgpids, a merged id's
        pgpids, the members of a v0_1 ``shelfmark_group``).
    :param universe: Extra pgpids that get a (possibly singleton) group.
    :returns: pgpid → sorted members of its group (only for ids seen).
    """
    uf = UnionFind()
    for x in universe:
        if x:
            uf.find(str(x))
    for ids in edge_lists:
        ids = [str(i) for i in ids if str(i).strip()]
        for i in ids:
            uf.find(i)
        for i in ids[1:]:
            uf.union(ids[0], i)
    comp: Dict[str, List[str]] = collections.defaultdict(list)
    for x in list(uf.parent):
        comp[uf.find(x)].append(x)
    out: Dict[str, List[str]] = {}
    for members in comp.values():
        ms = sorted(members, key=lambda s: (int(s) if s.isdigit() else sys.maxsize, s))
        for x in ms:
            out[x] = ms
    return out


# ------------------------------------------------------------------------------------ key index
@dataclass
class KeyIndex:
    """Shelfmark key → pgpids and → merged canonical ids, with the source of each pgpid link."""

    pgpids: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    cids: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    sources: Dict[Tuple[str, str], Set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    def add_shelfmark(self, shelfmark: str, pgpids: Iterable[str], source: str) -> None:
        """Link every key of a shelfmark to pgpids.

        :param shelfmark: Shelfmark string (joins are split).
        :param pgpids: PGP ids carried by the record.
        :param source: Name of the record's source (for the resolution report).
        :returns: None.
        """
        pgs = [str(p) for p in pgpids if str(p).strip()]
        for k in cached_keys(shelfmark):
            for p in pgs:
                self.pgpids[k].add(p)
                self.sources[(k, p)].add(source)

    def add_canonical(self, cid: str, pgpids: Iterable[str], source: str, display: str = "") -> None:
        """Link a merged canonical id (and its display string) to its key(s) and pgpids.

        :param cid: Canonical id.
        :param pgpids: PGP ids of the record.
        :param source: Source name.
        :param display: Display shelfmark of the record (keyed too).
        :returns: None.
        """
        ks = {k for k in (_canonical_key(cid),) if k} | set(cached_keys(display))
        pgs = [str(p) for p in pgpids if str(p).strip()]
        for k in ks:
            self.cids[k].add(cid)
            for p in pgs:
                self.pgpids[k].add(p)
                self.sources[(k, p)].add(source)

    def lookup(self, key: str) -> Set[str]:
        """PGP ids linked to a key.

        :param key: Shelfmark key.
        :returns: PGP ids (empty set when unknown).
        """
        return set(self.pgpids.get(key, ())) if key else set()


def build_key_index(corpus: Optional[_sm.Corpus], merged_map: Mapping[str, Mapping[str, Any]],
                    v01: Mapping[str, V01Doc], pairs: Mapping[str, PairDoc]) -> KeyIndex:
    """Index every shelfmark source by key: PGP documents and fragments, the merged corpus, the pgpid map, v0_1, pgp_parallel.

    :param corpus: Loaded :class:`shelfmarks.Corpus` (None skips its sources).
    :param merged_map: ``merged_pgpid_map.json`` (canonical id → ``{"pgpids": [...]}``).
    :param v01: :func:`scan_v01` output.
    :param pairs: :func:`scan_pairs` output.
    :returns: The index.
    """
    ki = KeyIndex()
    if corpus is not None:
        for pg, d in corpus.docs.items():
            ki.add_shelfmark(d.get("shelfmark") or "", [pg], "pgp_documents")
        for fr in corpus.fragments:
            ki.add_shelfmark(fr.get("shelfmark") or "", fr.get("pgpids") or [], "pgp_fragments")
        for cid, info in corpus.merged.items():
            ki.add_canonical(cid, info.get("pgpids") or [], "merged_corpus", info.get("disp") or "")
    for cid, info in merged_map.items():
        ki.add_canonical(cid, (info or {}).get("pgpids") or [], "pgpid_map")
    for pg, d in v01.items():
        for s in d.shelfmarks:
            ki.add_shelfmark(s, [pg], "v0_1")
    for pg, d in pairs.items():
        ki.add_shelfmark(d.shelfmark, [pg], "pgp_parallel")
    return ki


# ------------------------------------------------------------------------------------ the hold-out set
def _norm_pgpid(pgpid: Any) -> str:
    """Canonical string form of a pgpid (``None`` → '').

    :param pgpid: Raw value.
    :returns: Stripped string.
    """
    return str(pgpid).strip() if pgpid is not None else ""


@dataclass
class Holdouts:
    """The hold-out set H (two tiers) with reasons, keys and the build report.

    :ivar benchmark: pgpid → reasons (never in train; the ``heldout`` split).
    :ivar eval_docs: pgpid → reasons (v0_1 dev/test and extra benchmarks; literature exclusion).
    :ivar canonical: canonical id → reasons (benchmark tier).
    :ivar eval_canonical: canonical id → reasons (eval tier).
    :ivar key_map: shelfmark key → reasons (benchmark tier).
    :ivar eval_key_map: shelfmark key → reasons (eval tier).
    :ivar meta: Build report (counts, inputs, resolution of the PGP-131 ids).
    """

    benchmark: Dict[str, List[str]]
    eval_docs: Dict[str, List[str]]
    canonical: Dict[str, List[str]]
    eval_canonical: Dict[str, List[str]]
    key_map: Dict[str, List[str]]
    eval_key_map: Dict[str, List[str]]
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None, p: Optional[_paths.Paths] = None) -> "Holdouts":
        """Load ``holdouts.json``.

        :param path: File (default ``paths.out_dir / holdouts.json``).
        :param p: Build paths used for the default location.
        :returns: The hold-out set.
        :raises FileNotFoundError: When the file does not exist (run the ``build`` CLI first).
        """
        if path is None:
            path = (p or _paths.load_paths()).out(HOLDOUTS_FILE)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Holdouts":
        """Build from the parsed JSON of :meth:`to_dict`.

        :param data: Parsed ``holdouts.json``.
        :returns: The hold-out set.
        """
        def sec(name: str) -> Dict[str, List[str]]:
            """Read one reason map.

            :param name: Section name.
            :returns: item → reasons.
            """
            return {str(k): list(v) for k, v in (data.get(name) or {}).items()}

        return cls(benchmark=sec("benchmark"), eval_docs=sec("eval_docs"), canonical=sec("canonical"),
                   eval_canonical=sec("eval_canonical"), key_map=sec("keys"), eval_key_map=sec("eval_keys"),
                   meta=dict(data.get("meta") or {}))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form (``meta`` first, maps sorted).

        :returns: The dictionary written to ``holdouts.json``.
        """
        def srt(m: Mapping[str, Iterable[str]], numeric: bool = False) -> Dict[str, List[str]]:
            """Sort a reason map by item (numerically for pgpids).

            :param m: item → reasons.
            :param numeric: Sort pgpids numerically.
            :returns: Sorted copy with sorted reasons.
            """
            ks = sorted(m, key=(lambda s: (int(s) if s.isdigit() else sys.maxsize, s)) if numeric else None)
            return {k: sorted(set(m[k]), key=lambda r: (_rank(r), r)) for k in ks}

        return {"format_version": FORMAT_VERSION, "meta": self.meta,
                "benchmark": srt(self.benchmark, True), "eval_docs": srt(self.eval_docs, True),
                "canonical": srt(self.canonical), "eval_canonical": srt(self.eval_canonical),
                "keys": srt(self.key_map), "eval_keys": srt(self.eval_key_map)}

    def save(self, path: Optional[Union[str, Path]] = None, p: Optional[_paths.Paths] = None) -> Path:
        """Write ``holdouts.json`` atomically.

        :param path: Destination (default ``paths.out_dir / holdouts.json``).
        :param p: Build paths used for the default location.
        :returns: The written path.
        """
        dest = Path(path) if path is not None else (p or _paths.load_paths()).out(HOLDOUTS_FILE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, dest)
        return dest

    # -------------------------------------------------------------------------------- views
    @property
    def keys(self) -> Set[str]:
        """Shelfmark keys of the benchmark tier.

        :returns: Keys.
        """
        return set(self.key_map)

    @property
    def eval_keys(self) -> Set[str]:
        """Shelfmark keys of the eval tier.

        :returns: Keys.
        """
        return set(self.eval_key_map)

    @property
    def pgpids(self) -> Set[str]:
        """PGP ids of the benchmark tier.

        :returns: PGP ids.
        """
        return set(self.benchmark)

    @property
    def eval_pgpids(self) -> Set[str]:
        """PGP ids of the eval tier.

        :returns: PGP ids.
        """
        return set(self.eval_docs)

    def reasons_of(self, pgpid: Any) -> List[str]:
        """All stored reasons of a pgpid (both tiers).

        :param pgpid: PGP id.
        :returns: Reasons (empty when the document is not held).
        """
        p = _norm_pgpid(pgpid)
        return sorted(set(self.benchmark.get(p, [])) | set(self.eval_docs.get(p, [])), key=lambda r: (_rank(r), r))

    # -------------------------------------------------------------------------------- queries
    @staticmethod
    def _tier_reason(ids: Mapping[str, List[str]], cmap: Mapping[str, List[str]], kmap: Mapping[str, List[str]],
                     pgpid: Any, shelfmark: Optional[str], canonical: Optional[str]) -> Optional[str]:
        """Reason a document matches one tier, or None.

        :param ids: pgpid → reasons of the tier.
        :param cmap: canonical id → reasons of the tier.
        :param kmap: key → reasons of the tier.
        :param pgpid: PGP id (or None).
        :param shelfmark: Shelfmark string (or None; joins are split).
        :param canonical: Canonical id (or None).
        :returns: Best reason; key matches carry the ``key:`` prefix.
        """
        found: List[str] = []
        p = _norm_pgpid(pgpid)
        if p and p in ids:
            found += ids[p]
        if canonical:
            if canonical in cmap:
                found += cmap[canonical]
            k = _canonical_key(canonical)
            if k and k in kmap:
                found += [derive(KEY_PREFIX, r) for r in kmap[k]]
        if shelfmark:
            for k in cached_keys(shelfmark):
                if k in kmap:
                    found += [derive(KEY_PREFIX, r) for r in kmap[k]]
        return best_reason(found)

    def is_benchmark(self, pgpid: Any = None, shelfmark: Optional[str] = None,
                     canonical: Optional[str] = None) -> Optional[str]:
        """Whether a document is in the benchmark tier (never trained on).

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark (any spelling; every part of a join is checked).
        :param canonical: Canonical id (matched exactly and by key).
        :returns: The reason (``pgp131``, ``group:registry``, ``key:pgp131`` ...) or None.
        """
        return self._tier_reason(self.benchmark, self.canonical, self.key_map, pgpid, shelfmark, canonical)

    def is_eval_doc(self, pgpid: Any = None, shelfmark: Optional[str] = None,
                    canonical: Optional[str] = None) -> Optional[str]:
        """Whether a document is in the eval tier (v0_1 dev/test, extra benchmarks).

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Canonical id.
        :returns: The reason (``test``, ``dev``, ``extra:<name>`` with prefixes) or None.
        """
        return self._tier_reason(self.eval_docs, self.eval_canonical, self.eval_key_map, pgpid, shelfmark, canonical)

    def is_held(self, pgpid: Any = None, shelfmark: Optional[str] = None,
                canonical: Optional[str] = None) -> Optional[str]:
        """Benchmark tier first, then eval tier (the literature-exclusion test).

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Canonical id.
        :returns: The reason or None.
        """
        return self.is_benchmark(pgpid, shelfmark, canonical) or self.is_eval_doc(pgpid, shelfmark, canonical)

    def match_any(self, pgpids: Iterable[Any] = (), canonicals: Iterable[str] = (), shelfmarks: Iterable[str] = (),
                  include_eval: bool = True) -> Optional[str]:
        """Best reason over several identifiers of one mention (a literature page's resolved ids).

        :param pgpids: PGP ids.
        :param canonicals: Canonical ids.
        :param shelfmarks: Shelfmark strings.
        :param include_eval: Also test the eval tier.
        :returns: The best reason or None.
        """
        test = self.is_held if include_eval else self.is_benchmark
        found = [test(pgpid=p) for p in pgpids] + [test(canonical=c) for c in canonicals] + \
                [test(shelfmark=s) for s in shelfmarks]
        return best_reason(r for r in found if r)


# ------------------------------------------------------------------------------------ building
def _extra_benchmark_files(p: _paths.Paths) -> List[Path]:
    """Extra benchmark files from ``paths.json`` (``extra_benchmarks``) or :data:`DEFAULT_EXTRA_BENCHMARKS`.

    :param p: Build paths.
    :returns: Resolved paths (missing ones are kept; the builder logs and skips them).
    """
    raw = p.get("extra_benchmarks")
    if raw is None:
        raw = list(DEFAULT_EXTRA_BENCHMARKS)
    if isinstance(raw, (str, Path)):
        raw = [raw]
    out = []
    for x in raw:
        q = Path(os.path.expanduser(str(x)))
        out.append(q if q.is_absolute() else p.repo / q)
    return out


def _close_groups(tier: Dict[str, Set[str]], groups: Mapping[str, List[str]]) -> int:
    """Add every member of a held document's group to the tier (reason ``group:<reason>``).

    :param tier: pgpid → reasons (modified in place).
    :param groups: pgpid → group members.
    :returns: Number of pgpids added.
    """
    added = 0
    for pg, rs in list(tier.items()):
        bases = {base_reason(r) for r in rs}
        for q in groups.get(pg, ()):
            if q == pg:
                continue
            for b in bases:
                added += _add(tier, q, derive(GROUP_PREFIX, b))
    return added


def _resolve_ids(ids: Mapping[str, Set[str]], merged_map: Mapping[str, Mapping[str, Any]],
                 corpus: Optional[_sm.Corpus], ki: KeyIndex, tier: Dict[str, Set[str]],
                 keys: Dict[str, Set[str]]) -> Dict[str, Dict[str, Any]]:
    """Resolve canonical ids to pgpids (pgpid map, merged corpus, shelfmark key) and record their keys.

    :param ids: canonical id → reasons.
    :param merged_map: ``merged_pgpid_map.json``.
    :param corpus: Merged corpus (None skips it).
    :param ki: Key index.
    :param tier: pgpid → reasons (modified in place).
    :param keys: key → reasons (modified in place).
    :returns: canonical id → resolution record ``{"key", "map_pgpids", "corpus_pgpids", "key_pgpids", "key_cids"}``.
    """
    res: Dict[str, Dict[str, Any]] = {}
    for cid, rs in ids.items():
        k = _canonical_key(cid)
        map_pg = {str(x) for x in (merged_map.get(cid) or {}).get("pgpids", [])}
        corp_pg = {str(x) for x in ((corpus.merged.get(cid) or {}).get("pgpids", []) if corpus is not None else [])}
        key_pg = ki.lookup(k) - map_pg - corp_pg
        key_cids = sorted(ki.cids.get(k, ())) if k else []
        for r in rs:
            for pg in map_pg | corp_pg:
                _add(tier, pg, r)
            for pg in key_pg:
                _add(tier, pg, derive(KEY_PREFIX, r))
            if k:
                _add(keys, k, r)
            if corpus is not None:
                for c2 in key_cids:
                    for k2 in cached_keys((corpus.merged.get(c2) or {}).get("disp") or ""):
                        _add(keys, k2, r)
        res[cid] = {"key": k, "map_pgpids": sorted(map_pg), "corpus_pgpids": sorted(corp_pg),
                    "key_pgpids": sorted(key_pg), "key_cids": key_cids}
    return res


def _pgpid_keys(tier: Mapping[str, Set[str]], corpus: Optional[_sm.Corpus], v01: Mapping[str, V01Doc],
                pairs: Mapping[str, PairDoc], keys: Dict[str, Set[str]]) -> None:
    """Add the shelfmark keys of every held pgpid (PGP record, v0_1 meta, pgp_parallel) to a key map.

    :param tier: pgpid → reasons.
    :param corpus: Merged corpus / PGP export (None skips the PGP record).
    :param v01: :func:`scan_v01` output.
    :param pairs: :func:`scan_pairs` output.
    :param keys: key → reasons (modified in place; ``key:`` prefixes are not propagated).
    :returns: None.
    """
    for pg, rs in tier.items():
        sms: Set[str] = set()
        if corpus is not None and pg in corpus.docs:
            sms.add(corpus.docs[pg].get("shelfmark") or "")
        if pg in v01:
            sms |= v01[pg].shelfmarks
        if pg in pairs:
            sms.add(pairs[pg].shelfmark)
        base = {r[len(KEY_PREFIX):] if r.startswith(KEY_PREFIX) else r for r in rs}
        for s in sms:
            for k in cached_keys(s):
                for r in base:
                    _add(keys, k, r)


def build_holdouts(p: Optional[_paths.Paths] = None, corpus: Optional[_sm.Corpus] = None,
                   registry: Optional[Tuple[Set[str], Set[str]]] = None,
                   extra_files: Optional[Sequence[Path]] = None, use_corpus: bool = True) -> Holdouts:
    """Assemble H from the configured sources (PGP-131, registries, v0_1 vision flag, probe, v0_1 dev/test, the
    ``pgp_parallel`` dev/test documents without v0_1 rows, extras).

    :param p: Build paths (default :func:`paths.load_paths`).
    :param corpus: Pre-loaded corpus (default :func:`shelfmarks.load_corpus` when ``use_corpus``).
    :param registry: Pre-read ``(canonical ids, pgpids)`` of the registered benchmarks (default:
        :func:`benchmark_registry.registered_benchmark_documents`).
    :param extra_files: Extra benchmark files (default from ``paths.json`` / :data:`DEFAULT_EXTRA_BENCHMARKS`);
        ``[]`` disables them.
    :param use_corpus: Load the merged corpus / PGP export when ``corpus`` is None.
    :returns: The hold-out set with its build report in ``meta``.
    """
    p = p or _paths.load_paths()
    if corpus is None and use_corpus:
        corpus = _sm.load_corpus(p)
    with open(p.merged_pgpid_map, encoding="utf-8") as fh:
        merged_map: Dict[str, Dict[str, Any]] = json.load(fh)
    v01 = scan_v01(p.text_ft_v0_1)
    pairs = scan_pairs(p.pgp_parallel)
    ki = build_key_index(corpus, merged_map, v01, pairs)
    LOG.info("key index: %d keys", len(ki.pgpids))

    # ---- benchmark tier: canonical ids and direct pgpids
    b131 = load_id_list(p.benchmark_ids)
    reg_ids, reg_pg = registry if registry is not None else _registry.registered_benchmark_documents()
    cids: Dict[str, Set[str]] = {}
    for c in b131:
        _add(cids, c, "pgp131")
    for c in reg_ids:
        _add(cids, str(c), "registry")
    bench: Dict[str, Set[str]] = {}
    bkeys: Dict[str, Set[str]] = {}
    for pg in reg_pg:
        _add(bench, str(pg), "registry")
    vision = {pg for pg, d in v01.items() if d.vision} | {pg for pg, d in pairs.items() if d.vision}
    for pg in vision:
        _add(bench, pg, "vision_flag")
    probe = probe_pgpids(p.samples, p.exclusions_v1)
    for pg in probe:
        _add(bench, pg, "probe")
    resolution = _resolve_ids(cids, merged_map, corpus, ki, bench, bkeys)

    # ---- eval tier: v0_1 dev/test + extra benchmarks
    ev: Dict[str, Set[str]] = {}
    ecids: Dict[str, Set[str]] = {}
    ekeys: Dict[str, Set[str]] = {}
    for pg, d in v01.items():
        for sp in ("test", "dev"):
            if d.splits.get(sp):
                _add(ev, pg, sp)
    pairs_only = 0
    for pg, d in pairs.items():
        if d.split in ("test", "dev") and pg not in v01:
            pairs_only += _add(ev, pg, str(d.split))
    LOG.info("eval tier: %d pgp_parallel dev/test documents without v0_1 rows", pairs_only)
    extras_report: Dict[str, Dict[str, int]] = {}
    for f in (_extra_benchmark_files(p) if extra_files is None else list(extra_files)):
        if not Path(f).exists():
            LOG.warning("extra benchmark file missing, skipped: %s", Path(f).name)
            continue
        eb = load_extra_benchmark(f)
        reason = EXTRA_PREFIX + eb.name
        for c in eb.ids:
            _add(ecids, c, reason)
        n_before = len(ev)
        for pg in eb.pgpids:
            _add(ev, pg, reason)
        sm_pg = 0
        for s in eb.shelfmarks:
            for k in cached_keys(s):
                _add(ekeys, k, reason)
                for pg in ki.lookup(k):
                    sm_pg += _add(ev, pg, derive(KEY_PREFIX, reason))
        extras_report[eb.name] = {"ids": len(eb.ids), "pgpids": len(eb.pgpids), "shelfmarks": len(eb.shelfmarks),
                                  "pgpids_via_shelfmark_key": sm_pg, "new_eval_pgpids": len(ev) - n_before}
    eval_resolution = _resolve_ids(ecids, merged_map, corpus, ki, ev, ekeys)

    # ---- group closure (fragments, merged ids, v0_1 groups) and keys of every held pgpid
    edge_lists: List[List[str]] = []
    if corpus is not None:
        edge_lists += [list(fr.get("pgpids") or []) for fr in corpus.fragments]
        edge_lists += [list(info.get("pgpids") or []) for info in corpus.merged.values()]
    edge_lists += [list((info or {}).get("pgpids") or []) for info in merged_map.values()]
    by_group: Dict[str, List[str]] = collections.defaultdict(list)
    for pg, d in v01.items():
        if d.group:
            by_group[d.group].append(pg)
    edge_lists += list(by_group.values())
    groups = build_groups(edge_lists)
    direct_bench = set(bench)
    added_group = _close_groups(bench, groups)
    added_group_eval = _close_groups(ev, groups)
    _pgpid_keys(bench, corpus, v01, pairs, bkeys)
    _pgpid_keys(ev, corpus, v01, pairs, ekeys)

    h = Holdouts(benchmark={k: sorted(v) for k, v in bench.items()},
                 eval_docs={k: sorted(v) for k, v in ev.items()},
                 canonical={k: sorted(v) for k, v in cids.items()},
                 eval_canonical={k: sorted(v) for k, v in ecids.items()},
                 key_map={k: sorted(v) for k, v in bkeys.items()},
                 eval_key_map={k: sorted(v) for k, v in ekeys.items()})
    h.meta = _report(h, p, b131, resolution, eval_resolution, v01, pairs, ki, direct_bench, added_group,
                     added_group_eval, probe, vision, extras_report, reg_ids, reg_pg)
    h.meta.setdefault("counts", {})["eval_pgpids_pgp_parallel_split_only"] = pairs_only
    return h


def _v01_placement(pgpids: Iterable[str], v01: Mapping[str, V01Doc]) -> Dict[str, Dict[str, int]]:
    """Rows per v0_1 split for each pgpid.

    :param pgpids: PGP ids.
    :param v01: :func:`scan_v01` output.
    :returns: pgpid → {split: rows} (empty dict for pgpids absent from v0_1).
    """
    return {pg: dict(v01[pg].splits) if pg in v01 else {} for pg in pgpids}


def _report(h: Holdouts, p: _paths.Paths, b131: Sequence[str], resolution: Mapping[str, Dict[str, Any]],
            eval_resolution: Mapping[str, Dict[str, Any]], v01: Mapping[str, V01Doc], pairs: Mapping[str, PairDoc],
            ki: KeyIndex, direct_bench: Set[str], added_group: int, added_group_eval: int, probe: Set[str],
            vision: Set[str], extras: Mapping[str, Dict[str, int]], reg_ids: Set[str], reg_pg: Set[str]) -> Dict[str, Any]:
    """Build report stored in ``holdouts.json`` → ``meta``.

    :param h: The assembled hold-out set.
    :param p: Build paths.
    :param b131: PGP-131 canonical ids.
    :param resolution: Benchmark canonical-id resolution records.
    :param eval_resolution: Eval canonical-id resolution records.
    :param v01: v0_1 summary.
    :param pairs: pgp_parallel summary.
    :param ki: Key index.
    :param direct_bench: Benchmark pgpids before the group closure.
    :param added_group: pgpids the closure added to the benchmark tier.
    :param added_group_eval: pgpids the closure added to the eval tier.
    :param probe: Probe pgpids.
    :param vision: Vision-flag pgpids.
    :param extras: Per extra benchmark counts.
    :param reg_ids: Registry canonical ids.
    :param reg_pg: Registry pgpids.
    :returns: The report.
    """
    top = collections.Counter(best_reason(rs) for rs in h.benchmark.values())
    top_eval = collections.Counter(best_reason(rs) for rs in h.eval_docs.values())
    unmapped = [c for c in b131 if not resolution[c]["map_pgpids"]]
    pgp131_pg = {pg for pg, rs in h.benchmark.items() if any(base_reason(r) == "pgp131" for r in rs)}
    in_train = {pg for pg in h.benchmark if v01.get(pg) and v01[pg].splits.get("train")}
    key_only = {pg for pg, rs in h.benchmark.items() if all(r.startswith(KEY_PREFIX) for r in rs)}
    bodleian = []
    for c in unmapped:
        r = resolution[c]
        pgs = sorted(set(r["corpus_pgpids"]) | set(r["key_pgpids"]), key=lambda s: int(s) if s.isdigit() else 0)
        bodleian.append({"canonical_id": c, "key": r["key"], "merged_ids_with_key": r["key_cids"],
                         "pgpids": pgs, "via": {pg: sorted(ki.sources.get((r["key"], pg), ())) for pg in pgs},
                         "v0_1_rows": _v01_placement(pgs, v01),
                         "is_benchmark": {pg: h.is_benchmark(pgpid=pg) for pg in pgs}})
    v01_bench_rows = collections.Counter()
    for pg in h.benchmark:
        for sp, n in (v01[pg].splits.items() if pg in v01 else ()):
            v01_bench_rows[sp] += n
    no_pgpid_131 = [c for c in b131 if not (set(resolution[c]["map_pgpids"]) | set(resolution[c]["corpus_pgpids"])
                                            | set(resolution[c]["key_pgpids"]))]
    inputs = {name: _file_sig(Path(path), p.repo) for name, path in (
        ("benchmark_ids", p.benchmark_ids), ("merged_pgpid_map", p.merged_pgpid_map), ("samples", p.samples),
        ("exclusions_v1", p.exclusions_v1), ("pgp_parallel", p.pgp_parallel),
        ("v0_1_train", p.text_ft_v0_1 / "train.jsonl"), ("v0_1_dev", p.text_ft_v0_1 / "dev.jsonl"),
        ("v0_1_test", p.text_ft_v0_1 / "test.jsonl"), ("merged_shelfmarks", p.merged_shelfmarks))}
    return {
        "build": _rows.BUILD,
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "registries": list(_registry.BENCHMARK_REGISTRIES),
        "counts": {
            "benchmark_pgpids": len(h.benchmark),
            "benchmark_pgpids_by_reason": dict(sorted(top.items())),
            "benchmark_pgpids_direct": len(direct_bench),
            "benchmark_pgpids_added_by_group": added_group,
            "benchmark_pgpids_key_only": len(key_only),
            "benchmark_canonical_ids": len(h.canonical),
            "benchmark_keys": len(h.key_map),
            "pgp131_canonical_ids": len(b131),
            "pgp131_unmapped_in_pgpid_map": len(unmapped),
            "pgp131_without_any_pgpid": len(no_pgpid_131),
            "pgp131_pgpids": len(pgp131_pg),
            "registry_canonical_ids": len(reg_ids),
            "registry_pgpids": len(reg_pg),
            "vision_flag_pgpids": len(vision),
            "probe_pgpids": len(probe),
            "eval_pgpids": len(h.eval_docs),
            "eval_pgpids_by_reason": dict(sorted(top_eval.items())),
            "eval_pgpids_added_by_group": added_group_eval,
            "eval_canonical_ids": len(h.eval_canonical),
            "eval_keys": len(h.eval_key_map),
            "benchmark_pgpids_with_v0_1_train_rows": len(in_train),
            "v0_1_rows_of_benchmark_pgpids": dict(v01_bench_rows),
            "benchmark_and_eval_overlap": len(set(h.benchmark) & set(h.eval_docs)),
        },
        "extra_benchmarks": dict(extras),
        "pgp131_unmapped_resolution": bodleian,
        "pgp131_without_any_pgpid": no_pgpid_131,
        "benchmark_in_v0_1_train": {pg: {"reasons": h.benchmark[pg], "rows": v01[pg].splits}
                                    for pg in sorted(in_train, key=lambda s: int(s) if s.isdigit() else 0)},
        "eval_resolution_unresolved": sorted(c for c, r in eval_resolution.items()
                                             if not (r["map_pgpids"] or r["corpus_pgpids"] or r["key_pgpids"])),
    }


# ------------------------------------------------------------------------------------ CLI
def _print_summary(h: Holdouts, out: Any = None) -> None:
    """Print the counts and the resolution of the PGP-131 ids missing from the pgpid map.

    :param h: Hold-out set.
    :param out: Text stream (default ``sys.stdout`` at call time).
    :returns: None.
    """
    out = out if out is not None else sys.stdout
    c = h.meta.get("counts", {})
    print("hold-out set H", file=out)
    for k, v in c.items():
        print(f"  {k}: {v}", file=out)
    for name, v in (h.meta.get("extra_benchmarks") or {}).items():
        print(f"  extra benchmark {name}: {v}", file=out)
    res = h.meta.get("pgp131_unmapped_resolution") or []
    print(f"PGP-131 ids without a pgpid-map entry: {len(res)}", file=out)
    for r in res:
        rows = {pg: n for pg, n in r["v0_1_rows"].items()}
        print(f"  {r['canonical_id']} key={r['key']} pgpids={r['pgpids']} v0_1={rows} "
              f"held={r['is_benchmark']}", file=out)
    b = h.meta.get("benchmark_in_v0_1_train") or {}
    by = collections.Counter(best_reason(v["reasons"]) for v in b.values())
    print(f"benchmark pgpids with v0_1 TRAIN rows (move to heldout in v0_2): {len(b)} {dict(by)}", file=out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``build`` / ``show`` / ``check``.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit status.
    """
    ap = argparse.ArgumentParser(prog="python -m src.finetuning.text27b.holdouts", description=__doc__.split("\n")[0])
    ap.add_argument("--paths", default=None, help="paths.json (default $TEXT27B_PATHS or logs/text_finetune_27b/paths.json)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build holdouts.json")
    b.add_argument("--out", default=None, help="output file (default <out_dir>/holdouts.json)")
    b.add_argument("--no-extra", action="store_true", help="skip the extra benchmark files")
    s = sub.add_parser("show", help="print the counts of an existing holdouts.json")
    s.add_argument("--holdouts", default=None)
    c = sub.add_parser("check", help="test one document against H")
    c.add_argument("--holdouts", default=None)
    c.add_argument("--pgpid", default=None)
    c.add_argument("--shelfmark", default=None)
    c.add_argument("--canonical", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = _paths.load_paths(a.paths)
    if a.cmd == "build":
        h = build_holdouts(p, extra_files=[] if a.no_extra else None)
        h.meta["build_args"] = {"no_extra": bool(a.no_extra), "out": a.out}
        dest = h.save(a.out, p)
        _print_summary(h)
        print(f"wrote {dest}")
        return 0
    h = Holdouts.load(getattr(a, "holdouts", None), p)
    if a.cmd == "show":
        _print_summary(h)
        return 0
    print(json.dumps({"benchmark": h.is_benchmark(a.pgpid, a.shelfmark, a.canonical),
                      "eval_doc": h.is_eval_doc(a.pgpid, a.shelfmark, a.canonical)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
