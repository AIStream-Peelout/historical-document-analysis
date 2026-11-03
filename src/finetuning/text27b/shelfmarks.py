# File name: shelfmarks.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Shelfmark keys, corpus matching and the located-in-text rule (text27b).

Wraps the tracked normaliser (:class:`ShelfmarkNormalizer`, :func:`resolve_token`, :func:`combine`)
and ports the literature-inventory helpers of the 2026-10-18 probe (``analysis_lib``:
``prep_shelfmark``, ``match_shelfmark``, ``locate``, ``LEAKED_RAW``, the corpus loader).

Keys. :func:`key` maps any shelfmark spelling to an institution-free compact key; a canonical id
from the merge or a benchmark list maps to the same key through :func:`canonical_to_key`, e.g.
``key("Bodl. MS heb. d 99/7") == canonical_to_key("Oxford_Bodleian_MS_heb_d_99_7") ==
"mshebd99_7"`` although the merge spells such a leaf ``Oxford_Bodleian_Bodl_MS_heb_d_99_7``
(benchmark lists use the older form without ``Bodl``).
Construction: normaliser core → leading institution words dropped (every part of every
institution token, plus ``Bodl`` / ``Bod`` / ``ULC``) → lower-case, separators removed except
between two digit runs (``ENA NS 7.71`` → ``enans7_71`` ≠ ``ENA NS 77.1`` → ``enans77_1``). Keys
without a digit, or shorter than 4 characters (3 with a letter), are junk (``""``). Keys carry no institution, so two
libraries' identical cores share a key: a deliberate over-match for hold-out checks.

FROZEN 2026-11 (shelf-mark migration plan, PR-6). :func:`keys`, :func:`key`, :func:`core_key`, :func:`compact` and
:func:`canonical_to_key` are the key format of the t27b hold-outs / mixtures and stay byte-identical (snapshot test
``tests/test_text27b_shelfmarks_frozen.py``); :func:`core_of` and :func:`token_of` stay too, because the facade has
no byte-identical replacement (``core_id`` differs on 153 and ``split_id(canonical_id(...))[0]`` on 38 of the 309
probe inputs of that snapshot). New code must use ``src.datasets.shelfmarks``. :func:`prep_shelfmark` moved
verbatim into the facade and is re-imported here; :func:`match_shelfmark` is a call to the facade's ``resolve``
over :func:`corpus_index`.

Public API (other text27b modules import these names):

* :data:`ShelfmarkNormalizer`, :func:`resolve_token`, :func:`combine` -- the tracked normaliser,
  re-exported.
* :data:`INSTITUTION_WORDS` -- words dropped from the front of a core when keying.
* :data:`LEAKED_RAW`, :data:`LEAKED_CORES`, :data:`LEAKED_KEYS` -- shelfmarks the structuring
  prompt used as examples (they leak into ``shelf_marks_mentioned``; count only when located).
* :func:`prep_shelfmark`, :func:`core_of`, :func:`token_of`, :func:`compact`.
* :func:`key`, :func:`keys`, :func:`canonical_to_key`, :func:`core_key`.
* :class:`Corpus`, :func:`load_corpus`, :func:`match_shelfmark`, :func:`corpus_index`, :func:`pgp_flags`,
  :func:`compact_index`.
* :func:`tokens_with_pos`, :func:`strip_paren_tokens`, :func:`locate`, :func:`is_located`,
  :func:`located_keys`, :data:`STOP_TOKENS`.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import io
import logging
import os
import pickle
import re
import sys
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import orjson

with contextlib.redirect_stdout(io.StringIO()):
    from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as N
    from src.datasets.merging import institution_tokens as _it
    from src.datasets.merging.institution_tokens import combine, resolve_token

from src.datasets import shelfmarks as _shelf
from src.datasets.shelfmarks import prep_shelfmark  # noqa: F401  (moved to the facade in PR-6; re-exported)
from src.finetuning.text27b import paths as _paths

#: Re-exported for callers that need the full normaliser (``to_canonical_id``, ``parse_oxford`` ...).
ShelfmarkNormalizer = N
LOG = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

#: Shelfmarks the structuring prompt uses as examples (they leak into ``shelf_marks_mentioned``).
LEAKED_RAW: Tuple[str, ...] = ("T-S A43.1", "T-S NS 322.64", "T-S Ar.31.30", "ENA 2689.11", "ENA NS 12.5",
                               "MS Heb. d. 32", "AIU VI C 6", "Halper 331", "Rylands Genizah 2", "Ar. 400",
                               "A 589", "B 3239", "ENA2689")


# ------------------------------------------------------------------------------------ cleaning
def compact(core: str) -> str:
    """Compact a canonical core: lower-case, drop separators except between two digit runs.

    FROZEN 2026-11: key format of the t27b hold-outs/mixtures; new code must use src.datasets.shelfmarks.

    :param core: Canonical core (underscore-separated).
    :returns: Compact key (``T_S_NS_999_64`` → ``tsns999_64``).
    """
    c = core.lower()
    return re.sub(r"(?<=[a-z])_(?=[a-z0-9])|(?<=[0-9])_(?=[a-z])", "", c)


def core_of(sm: str) -> str:
    """Return the normaliser's canonical core for a shelfmark ('' for junk the normaliser rejects).

    FROZEN 2026-11 with the t27b keys (the facade's ``core_id`` is not byte-identical: it returns ``""`` for joins,
    garbage and NLI numbers and gives an ES id's core); new code must use ``src.datasets.shelfmarks.core_id``.

    :param sm: Shelfmark string.
    :returns: Canonical core.
    """
    try:
        return N.to_canonical_id(sm) or ""
    except Exception:  # the normaliser raises on some junk strings
        return ""


def token_of(sm: str) -> Optional[str]:
    """Resolve the institution token of a shelfmark.

    FROZEN 2026-11 with the t27b keys (no byte-identical facade function: the facade infers the token from
    series words as well); new code must use ``src.datasets.shelfmarks.canonical_id`` / ``split_id``.

    :param sm: Shelfmark string.
    :returns: Institution token (``Oxford_Bodleian`` ...) or None.
    """
    try:
        return resolve_token(sm)
    except Exception:  # unresolvable institution strings raise
        return None


LEAKED_CORES: Set[str] = {core_of(x) for x in LEAKED_RAW} - {""}

# ------------------------------------------------------------------------------------ keys
#: Words that are never part of a physical shelfmark core: every part of every institution
#: token (``Oxford``, ``Bodleian``, ``Cambridge``, ``CUL``, ``Mosseri``, ``JRL``, ``JTS`` ...)
#: plus the abbreviations the normaliser keeps in cores (``Bodl``).
INSTITUTION_WORDS: frozenset = frozenset(
    {p.lower() for tok, _ in list(_it._REGISTRY) + list(_it._FALLBACK_REGISTRY) for p in tok.split("_")}
    | {"bodl", "bod", "ulc"})


def core_key(core: str) -> str:
    """Key of a canonical core or canonical id (leading institution words dropped, compacted).

    FROZEN 2026-11: key format of the t27b hold-outs/mixtures; new code must use src.datasets.shelfmarks.

    A KTIV ``__sys<digits>`` suffix and a final side letter after a digit (``225.25r``) are
    dropped, as :func:`prep_shelfmark` drops ``r`` / ``v`` from shelfmark strings.

    :param core: Underscore-separated core or fully qualified canonical id.
    :returns: Compact key, or '' when the remainder has no digit or is shorter than 4 characters
        (3 when it mixes a letter and digits, e.g. ``l18`` for ``JRL L 18``).
    """
    core = re.sub(r"__sys\d+$", "", core.strip("_ "))
    parts = [p for p in re.split(r"_+", core) if p]
    while parts and parts[0].lower() in INSTITUTION_WORDS:
        parts.pop(0)
    k = re.sub(r"(?<=\d)[rv]$", "", compact("_".join(parts)))
    if not re.search(r"\d", k):
        return ""
    return k if len(k) >= 4 or (len(k) == 3 and re.search(r"[a-z]", k)) else ""


def key(shelfmark: Optional[str]) -> str:
    """Institution-free normalised key of one shelfmark (any source spelling).

    FROZEN 2026-11: key format of the t27b hold-outs/mixtures; new code must use src.datasets.shelfmarks.

    :param shelfmark: Shelfmark (``Bodl. MS heb. d 99/7``, ``T-S 99J1.2``, ``Oxford: MS heb. e.99/6`` ...).
        A joined shelfmark (``A + B``) gives the key of its first part; use :func:`keys` for all.
    :returns: The key, or '' for junk.
    """
    if not shelfmark:
        return ""
    ks = keys(shelfmark)
    return ks[0] if ks else ""


def keys(shelfmark: Optional[str]) -> List[str]:
    """Keys of every part of a (possibly joined) shelfmark, in order, without duplicates.

    FROZEN 2026-11: key format of the t27b hold-outs/mixtures; new code must use src.datasets.shelfmarks.

    :param shelfmark: Shelfmark string; parts joined by ``+`` (PGP joins) are keyed separately.
    :returns: Non-empty keys.
    """
    if not shelfmark:
        return []
    out: List[str] = []
    for part in re.split(r"\s\+\s|\s*\+\s*(?=[A-Z])", str(shelfmark)):
        k = core_key(core_of(prep_shelfmark(part)))
        if k and k not in out:
            out.append(k)
    return out


def canonical_to_key(canonical_id: Optional[str]) -> str:
    """Key of a merge / benchmark canonical id (``Oxford_Bodleian_MS_heb_d_99_7`` → ``mshebd99_7``).

    FROZEN 2026-11: key format of the t27b hold-outs/mixtures; new code must use src.datasets.shelfmarks.

    :param canonical_id: Fully qualified canonical id (institution token + core).
    :returns: The same key :func:`key` gives the shelfmark the id names, or '' for junk.
    """
    return core_key(canonical_id or "")


LEAKED_KEYS: Set[str] = {key(x) for x in LEAKED_RAW} - {""}


# ------------------------------------------------------------------------------------ corpus
@dataclass
class Corpus:
    """Merged-corpus and PGP lookups (port of ``analysis_lib.Corpus``)."""

    merged: Dict[str, Dict[str, Any]]
    core_idx: Dict[str, Set[str]]
    core_idx_lc: Dict[str, Set[str]]
    pgp_to_cids: Dict[str, Set[str]]
    frag_core_pgpids: Dict[str, Set[str]]
    docs: Dict[str, Dict[str, str]]
    fn_rel: Dict[str, Set[str]]
    fn_rows: List[Dict[str, Any]]
    read_ids: Set[str]
    read_ids_by_model: Dict[str, Set[str]]
    sources: Dict[str, Dict[str, str]]
    fragments: List[Dict[str, Any]]


def _load_merged(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Read the merged corpus once (slim fields).

    :param path: ``merged_shelfmarks.jsonl``.
    :returns: ``(merged, core_idx, core_idx_lc, pgp_to_cids)``.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    core_idx: Dict[str, Set[str]] = collections.defaultdict(set)
    core_lc: Dict[str, Set[str]] = collections.defaultdict(set)
    pgp_to: Dict[str, Set[str]] = collections.defaultdict(set)
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = orjson.loads(line)
            cid = rec.get("canonical_id")
            if not cid:
                continue
            pg = [str(x) for x in (rec.get("pgpids") or [])]
            merged[cid] = {"disp": rec.get("shelfmark_display") or "", "pgpids": pg,
                           "src": list(rec.get("sources_present") or []),
                           "desc": (rec.get("description") or "")[:300]}
            for p in pg:
                pgp_to[p].add(cid)
            c = core_of(N._strip_institution(rec.get("shelfmark_display") or ""))
            if c:
                core_idx[c].add(cid)
                core_lc[c.lower()].add(cid)
    return merged, dict(core_idx), dict(core_lc), dict(pgp_to)


def _read_doc_ids(path: Path) -> Set[str]:
    """Document ids of a reads JSONL (regex on the head of each line; no full parse).

    :param path: Reads file.
    :returns: ``doc_id`` values.
    """
    ids: Set[str] = set()
    pat = re.compile(rb'"doc_id":\s*"([^"]+)"')
    with open(path, "rb") as fh:
        for line in fh:
            m = pat.search(line[:400])
            if m:
                ids.add(m.group(1).decode("utf-8"))
    return ids


def _corpus_inputs(p: _paths.Paths, read_files: Optional[Sequence[Path]]) -> List[Path]:
    """Input files of the corpus (for the cache signature).

    :param p: Build paths.
    :param read_files: Reads files or None for the default set.
    :returns: Ordered input paths.
    """
    pgp = p.pgp_export
    reads = list(read_files) if read_files is not None else sorted(
        f for f in p.ai_reads.glob("*.jsonl") if not f.name.endswith(".failures.jsonl"))
    return [p.merged_shelfmarks, pgp / "documents.csv", pgp / "footnotes.csv", pgp / "fragments.csv",
            pgp / "sources.csv"] + reads


def load_corpus(p: Optional[_paths.Paths] = None, read_files: Optional[Sequence[Path]] = None,
                cache: bool = True) -> Corpus:
    """Load the corpus lookups (merged corpus, PGP export CSVs, reads' document ids).

    Reads ``paths.merged_shelfmarks`` and ``documents/footnotes/fragments/sources.csv`` of
    ``paths.pgp_export``. The result is pickled under ``out_dir/cache/`` and reused while every
    input file keeps its size and modification time and the facade's ``NORMALISER_VERSION`` is
    unchanged (shelf-mark migration plan, PR-6).

    :param p: Build paths (default :func:`paths.load_paths`).
    :param read_files: Reads JSONL files for ``read_ids_by_model`` (default: every ``*.jsonl`` in
        ``paths.ai_reads`` except ``*.failures.jsonl``), keyed by file name.
    :param cache: Use / write the pickle cache.
    :returns: The :class:`Corpus`.
    """
    p = p or _paths.load_paths()
    inputs = _corpus_inputs(p, read_files)
    sig = [("NORMALISER_VERSION", _shelf.NORMALISER_VERSION)] + [
        (str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in inputs]
    cp = p.out("cache", "corpus.pkl")
    if cache and cp.exists():
        with open(cp, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("sig") == sig:
            return blob["corpus"]
    merged, core_idx, core_lc, pgp_to = _load_merged(p.merged_shelfmarks)
    pgp = p.pgp_export
    docs: Dict[str, Dict[str, str]] = {}
    with open(pgp / "documents.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            docs[r["pgpid"]] = {k: r.get(k, "") for k in ("shelfmark", "type", "description", "languages_primary",
                                                          "doc_date_standard", "has_transcription", "has_translation",
                                                          "origin", "destination", "mentioned", "tags")}
    fn_rel: Dict[str, Set[str]] = collections.defaultdict(set)
    fn_rows: List[Dict[str, Any]] = []
    with open(pgp / "footnotes.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rels = {x.strip() for x in (r.get("doc_relation") or "").split(";") if x.strip()}
            fn_rel[r["document_id"]] |= rels
            fn_rows.append({"pgpid": r["document_id"], "slug": r.get("source_slug", ""), "location": r.get("location", ""),
                            "rels": sorted(rels),
                            "content": r.get("content", "") if rels & {"Digital Edition", "Digital Translation"} else ""})
    frags: List[Dict[str, Any]] = []
    frag_core: Dict[str, Set[str]] = collections.defaultdict(set)
    with open(pgp / "fragments.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            pg = {x.strip() for x in (r.get("pgpids") or "").split(";") if x.strip()}
            frags.append({"shelfmark": r["shelfmark"], "pgpids": sorted(pg), "historic": r.get("shelfmarks_historic", "")})
            c = core_of(N._strip_institution(r["shelfmark"]))
            if c:
                frag_core[c] |= pg
    sources: Dict[str, Dict[str, str]] = {}
    with open(pgp / "sources.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sources[r["slug"]] = {"title": r.get("title", ""), "year": r.get("year", ""), "authors": r.get("authors", "")}
    read_by = {f.name: _read_doc_ids(f) for f in inputs[5:]}
    corpus = Corpus(merged, core_idx, core_lc, pgp_to, dict(frag_core), docs, dict(fn_rel), fn_rows,
                    set().union(*read_by.values()) if read_by else set(), read_by, sources, frags)
    if cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".corpus.", suffix=".tmp", dir=str(cp.parent))
        with os.fdopen(fd, "wb") as fh:
            pickle.dump({"sig": sig, "corpus": corpus}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
    LOG.info("corpus: %d merged ids, %d PGP docs, %d fragments", len(merged), len(docs), len(frags))
    return corpus


_COMPACT_CACHE: Dict[int, Dict[str, Set[str]]] = {}


def compact_index(corpus: Corpus) -> Dict[str, Set[str]]:
    """Compact core → merged ids index (built once per corpus object).

    :param corpus: Loaded corpus.
    :returns: Index.
    """
    idx = _COMPACT_CACHE.get(id(corpus))
    if idx is None:
        idx = {}
        for c, ids in corpus.core_idx.items():
            idx.setdefault(compact(c), set()).update(ids)
        _COMPACT_CACHE[id(corpus)] = idx
    return idx


#: Single-entry cache of :func:`corpus_index`: ``key`` = ``(id(corpus), NORMALISER_VERSION, INDEX_FORMAT)``, a
#: WEAK reference ``ref`` to the corpus (so a reused ``id`` cannot alias another corpus, and the cache never keeps a
#: corpus alive) and its ``index``. The entry is dropped when its corpus is freed (weakref callback) or replaced by
#: another corpus' index, so the cache never outlives the caller's corpus.
_INDEX_CACHE: Dict[str, Any] = {}


def _drop_index(ref: "weakref.ReferenceType[Any]") -> None:
    """Weakref callback: empty :data:`_INDEX_CACHE` when the freed corpus is the cached one.

    :param ref: The dead reference.
    :type ref: weakref.ReferenceType
    :return: None.
    :rtype: None
    """
    if _INDEX_CACHE.get("ref") is ref:
        _INDEX_CACHE.clear()


def _build_corpus_index(corpus: Any) -> _shelf.ShelfmarkIndex:
    """Build the facade's ES index over a corpus' merged records (no cache).

    :param corpus: :class:`Corpus` (or a stand-in with ``merged`` / ``fragments``).
    :type corpus: Any
    :return: The index.
    :rtype: ShelfmarkIndex
    """
    records = [{"canonical_id": cid, "shelfmark_display": (rec or {}).get("disp") or "",
                "pgpids": list((rec or {}).get("pgpids") or [])}
               for cid, rec in (getattr(corpus, "merged", None) or {}).items()]
    return _shelf.ShelfmarkIndex(records, fragments=getattr(corpus, "fragments", None) or [])


def corpus_index(corpus: Any) -> _shelf.ShelfmarkIndex:
    """The facade's ES index over a corpus' merged records (built once per corpus object; the last one is cached).

    Records come from ``corpus.merged`` (canonical id, display form, PGP ids); ``corpus.fragments`` (PGP
    ``fragments.csv`` rows: shelf mark, PGP ids, historic marks) give the fragment marks of a ``pgpid:<n>`` that no
    merged record carries, and the historic spellings of a fragment's record. The cache holds only a weak
    reference to the corpus; a stand-in that cannot be weakly referenced (``types.SimpleNamespace``) is not
    cached (its index is rebuilt on every call).

    :param corpus: :class:`Corpus` (or a stand-in with ``merged`` / ``fragments``).
    :type corpus: Any
    :return: The index.
    :rtype: ShelfmarkIndex
    """
    key = (id(corpus), _shelf.NORMALISER_VERSION, _shelf.INDEX_FORMAT)
    ref = _INDEX_CACHE.get("ref")
    if _INDEX_CACHE.get("key") == key and ref is not None and ref() is corpus:
        return _INDEX_CACHE["index"]
    index = _build_corpus_index(corpus)
    try:
        ref = weakref.ref(corpus, _drop_index)
    except TypeError:  # not weakly referenceable: never cached, so never kept alive
        return index
    _INDEX_CACHE.clear()
    _INDEX_CACHE.update(key=key, ref=ref, index=index)
    return index


#: The facade's ``how`` (``how_detail``) -> the ``how`` name ``match_shelfmark`` emitted before PR-6 (``rag_l3``
#: counts ``match_<how>`` and writes ``match_how`` rows). ``parent`` has no pre-PR-6 counterpart and keeps its name.
_HOW_TO_LEGACY: Dict[str, str] = {"exact": "exact", "alias": "core", "loose": "core_compact", "core": "core",
                                  "parent": "parent", "pgpid": "exact", "unmatched": "unmatched"}


def _match_shelfmark_head(raw: str, corpus: Corpus, _retry: bool = True) -> Dict[str, Any]:
    """The pre-PR-6 ``match_shelfmark``, verbatim (port of ``analysis_lib.match_shelfmark``); used for numeric ranges.

    Order: exact (institution token + core is a merged id) → unique core → unique lower-cased
    core → unique compact core; PGP ``fragments.csv`` cores add pgpids; all-caps words are re-cased
    once on a miss. Kept because what a range (``T-S 12.6-7``) names is open (migration plan Q11): a range
    gets exactly what it got before PR-6.

    :param raw: Raw shelfmark string.
    :type raw: str
    :param corpus: Loaded corpus.
    :type corpus: Corpus
    :param _retry: Internal; retry once with re-cased all-caps words.
    :type _retry: bool
    :return: ``{"norm", "core", "how", "cids", "pgpids"}`` where ``how`` ∈ {junk, exact, core,
        core_lc, core_compact, ambiguous, unmatched, pgp_fragment} (``_recase`` suffix on a retry hit).
    :rtype: Dict[str, Any]
    """
    s = prep_shelfmark(raw)
    c = core_of(s)
    out: Dict[str, Any] = {"norm": s, "core": c, "how": "junk", "cids": [], "pgpids": []}
    if not c or len(c) < 4 or not re.search(r"\d", c):
        return out
    cids: Set[str] = set()
    tok = token_of(s)
    c2 = core_of(N._strip_institution(re.sub(r"^(ULC|CUL)\s+", "", s)))
    if tok and combine(tok, c) in corpus.merged:
        cids = {combine(tok, c)}
        out["how"] = "exact"
    else:
        cand = corpus.core_idx.get(c, set())
        if len(cand) == 1:
            cids = set(cand)
            out["how"] = "core"
        elif len(cand) > 1:
            out["how"] = "ambiguous"
        else:
            cand = corpus.core_idx_lc.get(c.lower(), set()) or corpus.core_idx_lc.get(c2.lower(), set())
            if len(cand) == 1:
                cids = set(cand)
                out["how"] = "core_lc"
            else:
                out["how"] = "ambiguous" if cand else "unmatched"
                if not cand:
                    ci = compact_index(corpus)
                    cc = ci.get(compact(c), set()) or ci.get(compact(c2), set())
                    if len(cc) == 1:
                        cids = set(cc)
                        out["how"] = "core_compact"
    pg: Set[str] = set()
    for cid in cids:
        pg |= set(corpus.merged[cid]["pgpids"])
    fp = corpus.frag_core_pgpids.get(c, set()) | (corpus.frag_core_pgpids.get(c2, set()) if c2 and len(c2) >= 6 else set())
    if fp and out["how"] in ("unmatched", "exact", "core", "core_lc"):
        if not pg and out["how"] == "unmatched":
            out["how"] = "pgp_fragment"
        pg |= fp
    for p in pg:
        cids |= corpus.pgp_to_cids.get(p, set())
    out["cids"] = sorted(cids)
    out["pgpids"] = sorted(pg, key=lambda x: int(x) if x.isdigit() else 0)
    if not cids and not pg and _retry and re.search(r"\b[A-Z]{4,}\b", s):
        s2 = " ".join(w.capitalize() if w.isalpha() and w.isupper() and len(w) > 3 else w for w in s.split())
        alt = _match_shelfmark_head(s2, corpus, _retry=False)
        if alt["cids"] or alt["pgpids"]:
            alt["how"] = alt["how"] + "_recase"
            alt["norm"] = s
            return alt
    return out


def match_shelfmark(raw: str, corpus: Corpus) -> Dict[str, Any]:
    """Match a raw shelfmark to merged ids and PGP ids through the facade's ``resolve`` over
    :func:`corpus_index`.

    Return shape: ``{"norm", "core", "how", "how_detail", "ambiguous", "cids", "pgpids"}``; ``cids`` are ES ids.

    * ``how`` keeps the pre-PR-6 vocabulary (``rag_l3`` counts ``match_<how>``): the facade step ``how_detail`` maps
      ``exact`` → ``exact``, ``alias`` → ``core``, ``loose`` → ``core_lc`` when the corpus' lower-cased core
      index (``core_idx_lc``, as before PR-6) names exactly the matched ids, else ``core_compact``; ``core`` →
      ``core``, ``unmatched`` → ``unmatched`` (``pgp_fragment`` when the mark's ``fragments.csv`` core names PGP
      ids and no ES id resolves); ``parent`` (new in PR-6) keeps its name. ``junk`` when the frozen
      :func:`core_of` finds no usable core (``how_detail`` = ``junk``); ``ambiguous`` (no ids, no PGP ids,
      ``ambiguous=True``) when the alias / loose / core step of a single mark or of any part of a join names
      several ids. ``_recase`` is no longer emitted outside ranges: the loose step folds the case of a mark, but
      not every all-capitals prefix (``ULC OR 1080 J 138`` keeps ``ULC`` in its id; it resolves only as a known
      spelling of a record, through the case-blind ``verbatim`` step of the index).
    * A numeric range (``T-S 10J5.17-18``, ``ff. 9-10``; ``resolve`` gives ``unmatched`` with ``note="range"``)
      gets exactly the pre-PR-6 result (:func:`_match_shelfmark_head`), with ``how_detail="range"``: migration plan
      Q11 is open.
    * PGP ids and siblings as before PR-6: the matched records' own PGP ids, plus the ``fragments.csv`` PGP ids of
      the mark's core (``frag_core_pgpids``; the institution-stripped core too when it has 6+ characters) for an
      ``exact`` / ``core`` / ``core_lc`` / ``parent`` / ``unmatched`` result; ``cids`` gain the records of those PGP
      ids from ``corpus.pgp_to_cids`` only. The index's fragment-linked PGP ids (``ShelfmarkIndex.pgpids`` /
      ``by_pgpid``) are added only when that already names several ids (a join or a multi-document fragment), so a
      mark that names one record still names exactly that one.

    :param raw: Raw shelfmark string.
    :type raw: str
    :param corpus: Loaded corpus.
    :type corpus: Corpus
    :return: The match (see above).
    :rtype: Dict[str, Any]
    """
    s = prep_shelfmark(raw)
    c = core_of(s)
    out: Dict[str, Any] = {"norm": s, "core": c, "how": "junk", "how_detail": "junk", "ambiguous": False,
                           "cids": [], "pgpids": []}
    if not c or len(c) < 4 or not re.search(r"\d", c):
        return out
    idx = corpus_index(corpus)
    res = _shelf.resolve(raw, index=idx)
    if res.note == "range":
        head = _match_shelfmark_head(raw, corpus)
        head.update(how_detail="range", ambiguous=head["how"] == "ambiguous")
        return head
    out["how_detail"] = res.how
    if res.ambiguous:
        out.update(how="ambiguous", ambiguous=True)
        return out
    how = _HOW_TO_LEGACY.get(res.how, res.how)
    cids: Set[str] = set(res.cids)
    c2 = core_of(N._strip_institution(re.sub(r"^(ULC|CUL)\s+", "", s)))
    if how == "core_compact":
        lc = getattr(corpus, "core_idx_lc", None) or {}
        if cids and set(lc.get(c.lower()) or lc.get(c2.lower()) or ()) == cids:
            how = "core_lc"  # the pre-PR-6 lower-cased core hit (it takes the fragments.csv PGP ids below)
    merged = getattr(corpus, "merged", None) or {}
    pg: Set[str] = {str(p) for cid in cids for p in (merged.get(cid) or {}).get("pgpids") or []}
    frag = getattr(corpus, "frag_core_pgpids", None) or {}
    fp = set(frag.get(c, ())) | (set(frag.get(c2, ())) if c2 and len(c2) >= 6 else set())
    if fp and how in ("unmatched", "exact", "core", "core_lc", "parent"):
        if not cids and how == "unmatched":
            how = "pgp_fragment"
        pg |= fp
    p2c = getattr(corpus, "pgp_to_cids", None) or {}
    cids |= {x for p in pg for x in p2c.get(p, ())}
    if len(cids) > 1:
        pg |= {p for cid in cids for p in idx.pgpids.get(cid, [])}
        cids |= {x for p in pg for x in idx.by_pgpid.get(p, ())}
    out["how"] = how
    out["cids"] = sorted(cids)
    out["pgpids"] = sorted(pg, key=lambda x: int(x) if x.isdigit() else 0)
    return out


def pgp_flags(pgpids: Iterable[str], cids: Iterable[str], corpus: Corpus) -> Dict[str, bool]:
    """Digital edition / digital translation / our-read flags of a document.

    :param pgpids: PGP ids.
    :param cids: Merged canonical ids.
    :param corpus: Loaded corpus.
    :returns: ``{"has_pgp_digital_edition", "has_pgp_digital_translation", "has_our_read"}``.
    """
    pgpids = list(pgpids)
    ed = any(corpus.docs.get(p, {}).get("has_transcription") == "Y" and "Digital Edition" in corpus.fn_rel.get(p, set())
             for p in pgpids)
    tr = any(corpus.docs.get(p, {}).get("has_translation") == "Y" and "Digital Translation" in corpus.fn_rel.get(p, set())
             for p in pgpids)
    rd = any(c in corpus.read_ids for c in cids)
    return {"has_pgp_digital_edition": ed, "has_pgp_digital_translation": tr, "has_our_read": rd}


# ------------------------------------------------------------------------------------ location
_TS = re.compile(r"\bT\.?\s?-\s?S\.?(?=\s|\d|[A-Z])|\bT\.\s?S\.|\bTS(?=\s|\d)")
#: Tokens ignored when matching a shelfmark inside text.
STOP_TOKENS: Set[str] = {"f", "fol", "fols", "folio", "folios", "ff", "fo", "no", "ms", "mss", "box", "cat", "recto",
                         "verso", "r", "v"}


def tokens_with_pos(text: str) -> List[Tuple[str, int, int]]:
    """Tokenise into lower-case letter runs / digit runs with character offsets (``T-S`` unified to ``ts``).

    :param text: Input text.
    :returns: ``[(token, start, end), ...]``.
    """
    t = _TS.sub(lambda m: "TS" + " " * (len(m.group(0)) - 2) if len(m.group(0)) >= 2 else "TS", text)
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+|\d+", t)]


def strip_paren_tokens(toks: List[Tuple[str, int, int]], text: str) -> List[Tuple[str, int, int]]:
    """Drop stop tokens and tokens inside short (≤ 30 chars) parentheticals.

    :param toks: Tokens with positions.
    :param text: Source text.
    :returns: Filtered tokens.
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r"\([^()\n]{0,30}\)", text)]
    return [tk for tk in toks if tk[0] not in STOP_TOKENS and not any(a <= tk[1] < b for a, b in spans)]


def locate(raw: str, text: str, toks: Optional[List[Tuple[str, int, int]]] = None,
           toks2: Optional[List[Tuple[str, int, int]]] = None) -> List[int]:
    """Character offsets where a shelfmark occurs in text (token-sequence match, port of ``analysis_lib.locate``).

    Exact token sequence first (with and without stop / parenthetical tokens), then a
    joined-skeleton fallback that tolerates ``13J`` vs ``13 J`` and requires token boundaries.

    :param raw: Raw shelfmark.
    :param text: Page text.
    :param toks: Pre-computed :func:`tokens_with_pos` of ``text``.
    :param toks2: Pre-computed :func:`strip_paren_tokens` of ``toks``.
    :returns: Sorted start offsets (empty when the shelfmark is not in the text).
    """
    s = prep_shelfmark(raw)
    k1 = [t for t, _, _ in tokens_with_pos(s)]
    if not k1 or not any(t.isdigit() for t in k1):
        return []
    toks = toks if toks is not None else tokens_with_pos(text)
    toks2 = toks2 if toks2 is not None else strip_paren_tokens(toks, text)
    k2 = [t for t, _, _ in strip_paren_tokens(tokens_with_pos(s), s)]
    hits: Set[int] = set()
    for kk, tt in ((k1, toks), (k2, toks2)):
        if not kk:
            continue
        n = len(kk)
        for i in range(len(tt) - n + 1):
            if tt[i][0] == kk[0] and all(tt[i + j][0] == kk[j] for j in range(n)):
                hits.add(tt[i][1])
        if hits:
            break
    if not hits:
        kj = "".join(k2 or k1)
        base = toks2 or toks
        sk: List[str] = []
        pos: List[int] = []
        for t, a, _ in base:
            for ci, ch in enumerate(t):
                sk.append(ch)
                pos.append(a + ci)
        skj = "".join(sk)
        starts = {a for _, a, _ in base}
        start = 0
        while True:
            i = skj.find(kj, start)
            if i < 0:
                break
            endc = i + len(kj)
            nxt = skj[endc] if endc < len(skj) else ""
            prv = skj[i - 1] if i > 0 else ""
            ok_end = not (kj[-1].isdigit() and nxt.isdigit())
            ok_start = not (kj[0].isalpha() and prv.isalpha()) and not (kj[0].isdigit() and prv.isdigit())
            if ok_end and ok_start and pos[i] in starts:
                hits.add(pos[i])
            start = i + 1
    return sorted(hits)


def is_located(raw: str, text: str) -> bool:
    """Whether a shelfmark key occurs in the page text (the located-in-text rule).

    :param raw: Raw shelfmark key.
    :param text: Page text.
    :returns: True when :func:`locate` finds it.
    """
    return bool(locate(raw, text))


def located_keys(raw_keys: Iterable[str], text: str) -> List[str]:
    """Keep only the shelfmark keys located in the text (fake / prompt-example keys drop out).

    :param raw_keys: Raw keys (e.g. a page's ``shelf_marks_mentioned``).
    :param text: Page text.
    :returns: Located raw keys, in input order, without duplicates.
    """
    toks = tokens_with_pos(text)
    toks2 = strip_paren_tokens(toks, text)
    out: List[str] = []
    for k in raw_keys:
        if k not in out and locate(k, text, toks, toks2):
            out.append(k)
    return out
