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

Public API (other text27b modules import these names):

* :data:`ShelfmarkNormalizer`, :func:`resolve_token`, :func:`combine` -- the tracked normaliser,
  re-exported.
* :data:`INSTITUTION_WORDS` -- words dropped from the front of a core when keying.
* :data:`LEAKED_RAW`, :data:`LEAKED_CORES`, :data:`LEAKED_KEYS` -- shelfmarks the structuring
  prompt used as examples (they leak into ``shelf_marks_mentioned``; count only when located).
* :func:`prep_shelfmark`, :func:`core_of`, :func:`token_of`, :func:`compact`.
* :func:`key`, :func:`keys`, :func:`canonical_to_key`, :func:`core_key`.
* :class:`Corpus`, :func:`load_corpus`, :func:`match_shelfmark`, :func:`pgp_flags`, :func:`compact_index`.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import orjson

with contextlib.redirect_stdout(io.StringIO()):
    from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as N
    from src.datasets.merging import institution_tokens as _it
    from src.datasets.merging.institution_tokens import combine, resolve_token

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
def prep_shelfmark(raw: str) -> str:
    """Clean a raw shelfmark key before normalising (underscore keys, stray prefixes, folio / side tails).

    :param raw: Raw key (from a page's ``shelf_marks_mentioned``, a PGP record, prose).
    :returns: Cleaned shelfmark string.
    """
    s = str(raw).strip()
    s = re.sub(r"^(?:Loc|Ref|Book|Doc_ref|Ms|MS_ref)_(?=[A-Z])", "", s)
    s = re.sub(r"^MS\.?\s+(?=ENA|T-S|TS\b|JTS)", "", s)
    s = re.sub(r"^(?:[uxv](?=_?[A-Z]{2}|_?T_S)|_+)", "", s).lstrip("_")
    if "_" in s and " " not in s.strip("_ "):
        s = s.replace("_", " ")
    s = re.sub(r"\s*\([^()]*\)?\s*$", "", s)
    # Folio designators become ", f. N" -- but neither a Bodleian size letter ("MS heb. f 56/12")
    # nor the Taylor-Schechter F series ("T-S F99.22"): the 2026-10-18 prototype matched "f"
    # case-insensitively, which dropped those letters and broke their keys.
    s = re.sub(r"(?<!heb\.)(?<!heb)(?<!Heb\.)(?<!Heb)(?<!arab\.)(?<!Arab\.)(?<!arab)(?<!Arab)"
               r"[\s,]+(?:f|fol|fols|folio|ff|Fol|Fols|Folio|FOL)\.?\s*(?=\d)", ", f. ", s)
    s = re.sub(r"(?<=\d)\s?[rv]$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,;:()[]{}")
    return s


def compact(core: str) -> str:
    """Compact a canonical core: lower-case, drop separators except between two digit runs.

    :param core: Canonical core (underscore-separated).
    :returns: Compact key (``T_S_NS_999_64`` → ``tsns999_64``).
    """
    c = core.lower()
    return re.sub(r"(?<=[a-z])_(?=[a-z0-9])|(?<=[0-9])_(?=[a-z])", "", c)


def core_of(sm: str) -> str:
    """Return the normaliser's canonical core for a shelfmark ('' for junk the normaliser rejects).

    :param sm: Shelfmark string.
    :returns: Canonical core.
    """
    try:
        return N.to_canonical_id(sm) or ""
    except Exception:  # the normaliser raises on some junk strings
        return ""


def token_of(sm: str) -> Optional[str]:
    """Resolve the institution token of a shelfmark.

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
    input file keeps its size and modification time.

    :param p: Build paths (default :func:`paths.load_paths`).
    :param read_files: Reads JSONL files for ``read_ids_by_model`` (default: every ``*.jsonl`` in
        ``paths.ai_reads`` except ``*.failures.jsonl``), keyed by file name.
    :param cache: Use / write the pickle cache.
    :returns: The :class:`Corpus`.
    """
    p = p or _paths.load_paths()
    inputs = _corpus_inputs(p, read_files)
    sig = [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in inputs]
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


def match_shelfmark(raw: str, corpus: Corpus, _retry: bool = True) -> Dict[str, Any]:
    """Match a raw shelfmark to merged ids and PGP ids (port of ``analysis_lib.match_shelfmark``).

    Order: exact (institution token + core is a merged id) → unique core → unique lower-cased
    core → unique compact core; PGP ``fragments.csv`` cores add pgpids; all-caps words are re-cased
    once on a miss.

    :param raw: Raw shelfmark string.
    :param corpus: Loaded corpus.
    :param _retry: Internal; retry once with re-cased all-caps words.
    :returns: ``{"norm", "core", "how", "cids", "pgpids"}`` where ``how`` ∈ {junk, exact, core,
        core_lc, core_compact, ambiguous, unmatched, pgp_fragment} (``_recase`` suffix on a retry hit).
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
        alt = match_shelfmark(s2, corpus, _retry=False)
        if alt["cids"] or alt["pgpids"]:
            alt["how"] = alt["how"] + "_recase"
            alt["norm"] = s
            return alt
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
