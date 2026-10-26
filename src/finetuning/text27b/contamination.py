# File name: contamination.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Contamination gates of the text27b build (plan §9.4): n-gram leak index, project-name scan,
literature page exclusions and the mixture re-scan.

N-gram index. The references are the eval side of the build: the targets and ``<document>``
edition inputs of every v0_1 row whose document is held out (:mod:`holdouts`, both tiers), plus the
``pgp_parallel`` editions / translations and the PGP description of every held document. Each
reference is indexed twice:

* ``w8`` -- word 8-grams (lower-cased, niqqud / harakat removed, word = run of letters / digits /
  apostrophe), the prose rule of the 2026-10-21 critic (``critic/ngram_leak.py``);
* ``c20`` -- character 20-grams over the Hebrew letters of a Hebrew-script run (final forms folded,
  everything but letters dropped, runs broken by Latin or Arabic letters), so an edition printed in
  a book still matches when brackets, line breaks or word splits differ.

An n-gram found in ``stop_df`` (5) or more distinct documents is a formula and is stop-listed
(``critic/ngram_gate_cost.py``). The document frequency counts the held documents AND, by default,
the non-held PGP documents (``background``: their pgp_parallel texts and v0_1 train rows), so a
legal or epistolary formula that only four eval documents carry is still a formula; an n-gram
that train documents already carry five times adds no contamination when a book page repeats it.
A text *leaks* a reference when it shares at least ``min_shared[kind]`` of the reference's
(non-stop) n-grams or at least ``min_share`` (2 %) of them. ``min_shared`` is 5 for ``w8`` (the
brief) and 25 for ``c20``: 25 character 20-grams is one identical run of ≈44 Hebrew letters, about
the 12 words that five shared word 8-grams need, while the 5–24 band on the 2026-10-22 corpus was
formulae (dating clauses, ketubba and oath formulae, honorific titles, Bible verses).
Hashes are stable 64-bit polynomial hashes (numpy), so an index can be saved and reloaded.

Page exclusions (:func:`build_excluded_pages`) map each literature page key to its reasons:
treatment spans of held documents, pages whose resolved mentions or located shelfmark keys name a
held document (old 10-18 page index, module E's ``rag_index.jsonl`` when present, the page's own
shelfmark keys and the shelfmarks found in its text) with a ±1 page window, an explicit local
list, the n-gram gate on the page text, the L1 dedupe against train translate / analyse targets
(decision D5, ``train_target_dup``, binding L1 rows only) and, last, the mention and n-gram gates on
the two-page chunks L1 can still form from pages not yet excluded (a shelfmark or an eval passage
can straddle a page boundary).

Public API (other text27b modules import these names):

* Tokens and hashes: :func:`word_tokens`, :func:`hebrew_runs`, :func:`gram_hashes`, :func:`text_grams`,
  :data:`KINDS`, :data:`DEFAULT_N`, :data:`DEFAULT_MIN_SHARED`, :data:`DEFAULT_MIN_SHARE`, :data:`STOP_DF`.
* :class:`NgramIndex` -- ``build(refs)``, ``ref_hits(text)``, ``page_hits(text) -> (shared, max_share)``,
  ``leak_refs(text)``, ``is_leak(text)``, ``save`` / ``load``, ``stats``.
* :func:`target_share` / :func:`block_ok` -- the per-block rule of the RAG rows (a block may hold
  < 5 % of its own target's word 8-grams).
* :func:`source_scan`.
* :func:`prose_of`, :func:`reference_text`, :func:`document_input`, :func:`reference_texts`, :func:`background_texts`,
  :func:`build_reference_index` -- the eval references of the build.
* :class:`PageResolver`, :func:`build_excluded_pages`, :func:`load_excluded_pages`, :func:`load_extra_excluded`,
  :func:`page_pairs`, :func:`pair_exclusions` (the gates on two-page L1 chunks).
* :func:`train_target_texts`, :func:`build_train_target_index`, :func:`train_dup_pages` -- the L1 dedupe
  of decision D5 (a page holding ≥ 20 % of a TRAIN translate / analyse target's word 8-grams is
  excluded with reason ``train_target_dup``).
* :data:`L1_ONLY_REASONS`, :func:`excluded_for_family` -- the dedupe reason binds L1 rows only (RAG rows
  carry pages in the prompt; the re-scan applies every other reason to every literature row).
* :func:`build_excluded_rows`, :func:`load_excluded_rows` -- D5 row exclusions: TRAIN rows whose target
  leaks an eval reference, whose text names a benchmark document, or that draw on an excluded page
  (:func:`row_page_keys`) → ``excluded_rows.json`` (``{row_id: reason}``, honoured by the mixture);
  :func:`doc_identities` -- the cross-family split check (a TRAIN row whose document another family
  file holds out is excluded too, reason ``split_conflict``).
* Shelfmark mentions in text (fixes F1, F10): :data:`MENTION_PREFIX_RE`, :func:`mention_text`, :func:`mention_spans`,
  :func:`mention_token_gaps`, :func:`mention_tokens`, :func:`compact_tokens`, :func:`mention_candidates`,
  :func:`mention_key_candidates`, :func:`mention_norm`, :class:`MentionIndex`, :func:`mention_index`,
  :func:`text_mention_reason` -- spaced and abbreviated spellings (``TS 98 J 7 6``, ``T-S 97 J 26.10``,
  ``ENA 9822 A.56``, ``Moss. IX,249``), series words (``TS Arabic Box 99.255``, ``TS G9.18``), prefixes (``NLI``,
  ``BL OR``), a line break inside a shelfmark (``T-S Misc.⏎99.24r``, ``ENA NS 7,⏎fol. 38``), ``(Cat. N)`` inserts and
  bare MIAC inventory numbers resolve to the key of the canonical form; a leaf number without its letter
  (``TS NS 999 1r.15``) counts as held when any lettered leaf is held; every text-mention rule (page exclusions,
  two-page chunks, row exclusions, re-scan) goes through :func:`text_mention_reason`.
* Parsed-entry rows (F11): :data:`NAS_PAGE_SOURCE`, :func:`unmapped_entry_row` -- a row built from a book-edition entry
  must carry its entry's page keys; one that cannot be mapped is excluded (``unmapped_entry_pages``) and counted by the
  re-scan (``unmapped_entry_rows``).
* :func:`row_source_scan` -- project-name strings anywhere in a row (messages and meta).
* Canon rows (2026-10-26): :data:`CANON_FAMILIES`, :func:`load_ktiv_benchmarks`, :func:`canon_holdout_reason` -- a canon
  row's manuscript is checked by KTIV system number, by H over pgpid / catalogue pgpids / canonical id / shelfmark key
  and by GT shingle overlap (row exclusions and re-scan).
* :func:`rescan` -- the mixture re-scan (zero H hits, zero literature mentions of held documents, zero unmapped
  parsed-entry rows, zero project-name strings, zero non-NFC rows, zero held-out canon manuscripts, n-gram report).

CLI::

    python -m src.finetuning.text27b.contamination ngram-index [--no-cache]
    python -m src.finetuning.text27b.contamination excluded-pages [--rag-index PATH] [--extra PATH] [--no-ngram] [--works W ...]
        [--no-train-dedupe] [--train-dup-share 0.2] [--families-dir DIR] [--no-pair-scan]
    python -m src.finetuning.text27b.contamination excluded-rows [--families-dir DIR] [--out DIR] [--excluded-pages auto|none|FILE]
    python -m src.finetuning.text27b.contamination rescan MIXTURE.jsonl [--out PATH]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.finetuning.text27b import holdouts as _ho
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import textnorm as _tn

LOG = logging.getLogger(__name__)

# ============================================================================== tokens and hashes
#: The two n-gram kinds: word 8-grams and Hebrew character 20-grams.
KINDS: Tuple[str, ...] = ("w8", "c20")
#: n per kind.
DEFAULT_N: Dict[str, int] = {"w8": 8, "c20": 20}
#: A text leaks a reference when it shares at least this many of its n-grams (per kind) ...
DEFAULT_MIN_SHARED: Dict[str, int] = {"w8": 5, "c20": 25}
#: ... or at least this share of them.
DEFAULT_MIN_SHARE = 0.02
#: n-grams found in this many distinct documents are formulae (stop-listed).
STOP_DF = 5
#: Minimum number of (non-stop) n-grams a reference needs for the share rule to apply.
MIN_REF_GRAMS = 1
#: Share of a target's word 8-grams a RAG literature block may hold (plan §9.2 L3).
BLOCK_MAX_SHARE = 0.05

#: Word = run of word characters (Latin, Hebrew, Arabic letters, digits) or apostrophes (critic rule).
WORD_RE = re.compile(r"[\w֐-׿؀-ۿ']+")
#: Hebrew points / cantillation and Arabic harakat, removed before tokenising.
MARKS_RE = re.compile(r"[֑-ׇً-ْٰ]")
_HE_LETTER = re.compile(r"[א-ת]")
#: A character that ends a Hebrew run: a Latin or Arabic letter.
_RUN_BREAK = re.compile(r"[A-Za-zÀ-ɏḀ-ỿؠ-يٱ-ۓ]")
_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_M64 = (1 << 64) - 1
_P = np.uint64(0x100000001B3)
_SALT: Dict[str, np.uint64] = {"w8": np.uint64(0x9E3779B97F4A7C15), "c20": np.uint64(0xC2B2AE3D27D4EB4F)}
_WORD_HASH_CACHE: Dict[str, int] = {}


def word_tokens(text: Optional[str]) -> List[str]:
    """Word tokens for the ``w8`` kind (lower-cased, points / harakat removed).

    :param text: Any text.
    :returns: Tokens in order.
    """
    if not text:
        return []
    return WORD_RE.findall(MARKS_RE.sub("", text.lower()))


def hebrew_runs(text: Optional[str], min_len: int = 1) -> List[str]:
    """Hebrew letter skeletons for the ``c20`` kind.

    The text is cut wherever a Latin or Arabic letter occurs; inside each piece only the Hebrew
    letters are kept (final forms folded), so brackets, digits, punctuation, line breaks and word
    spacing do not matter.

    :param text: Any text.
    :param min_len: Drop skeletons shorter than this.
    :returns: Letter strings.
    """
    if not text:
        return []
    out = []
    for piece in _RUN_BREAK.split(MARKS_RE.sub("", text)):
        sk = "".join(_HE_LETTER.findall(piece)).translate(_FINALS)
        if len(sk) >= min_len:
            out.append(sk)
    return out


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser over a uint64 array (wrapping arithmetic).

    :param x: uint64 values.
    :returns: Mixed uint64 values.
    """
    x = x.astype(np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _word_hash(w: str) -> int:
    """Stable 64-bit hash of a word (blake2b, memoised).

    :param w: Word.
    :returns: Unsigned 64-bit integer.
    """
    h = _WORD_HASH_CACHE.get(w)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")
        if len(_WORD_HASH_CACHE) < 4_000_000:
            _WORD_HASH_CACHE[w] = h
    return h


def _poly(units: np.ndarray, n: int) -> np.ndarray:
    """Polynomial hashes of every window of ``n`` consecutive units (wrapping uint64).

    :param units: uint64 unit hashes.
    :param n: Window length.
    :returns: One hash per window (empty when ``len(units) < n``).
    """
    L = len(units) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _P + units[j:j + L]
    return h


def gram_hashes(text: Optional[str], kind: str, n: Optional[int] = None) -> np.ndarray:
    """Distinct n-gram hashes of a text for one kind.

    :param text: Any text.
    :param kind: ``w8`` (word grams) or ``c20`` (Hebrew character grams).
    :param n: Gram length (default :data:`DEFAULT_N` of the kind).
    :returns: Sorted unique uint64 hashes (salted per kind, so kinds never collide).
    :raises ValueError: On an unknown kind.
    """
    if kind not in DEFAULT_N:
        raise ValueError(f"unknown n-gram kind {kind!r}")
    n = n or DEFAULT_N[kind]
    if kind == "w8":
        toks = word_tokens(text)
        if len(toks) < n:
            return np.zeros(0, dtype=np.uint64)
        units = np.fromiter((_word_hash(t) for t in toks), dtype=np.uint64, count=len(toks))
        parts = [_poly(units, n)]
    elif kind == "c20":
        parts = []
        for run in hebrew_runs(text, n):
            units = _mix(np.frombuffer(run.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64))
            parts.append(_poly(units, n))
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    allh = np.concatenate(parts)
    if not len(allh):
        return allh
    return np.unique(_mix(allh ^ _SALT[kind]))


def text_grams(text: Optional[str], kinds: Sequence[str] = KINDS, n: Optional[Mapping[str, int]] = None) -> Dict[str, np.ndarray]:
    """n-gram hashes of a text for several kinds.

    :param text: Any text.
    :param kinds: Kinds to compute.
    :param n: Gram length per kind (default :data:`DEFAULT_N`).
    :returns: kind → sorted unique hashes.
    """
    n = dict(DEFAULT_N, **(n or {}))
    return {k: gram_hashes(text, k, n[k]) for k in kinds}


# ============================================================================== the index
def _doc_frequency(docs: Iterable[Tuple[str, str]], kinds: Sequence[str], n: Mapping[str, int],
                   exclude: Optional[Set[str]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Document frequency of every n-gram over ``(doc, text)`` pairs (several texts per doc allowed).

    :param docs: ``(doc, text)`` pairs.
    :param kinds: Kinds to count.
    :param n: Gram length per kind.
    :param exclude: Doc ids to skip.
    :returns: ``(sorted unique hashes, number of distinct docs per hash, number of docs)``.
    """
    per_doc: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    for doc, text in docs:
        doc = str(doc)
        if (exclude and doc in exclude) or not text or not text.strip():
            continue
        for g in text_grams(text, kinds, n).values():
            if len(g):
                per_doc[doc].append(g)
    if not per_doc:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64), 0
    allg = np.concatenate([np.unique(np.concatenate(v)) for v in per_doc.values()])
    uh, df = np.unique(allg, return_counts=True)
    return uh, df, len(per_doc)


@dataclass
class RefInfo:
    """One indexed (reference, kind) pair."""

    ref_id: str
    doc: str
    kind: str
    n_grams: int


class NgramIndex:
    """Inverted n-gram index over eval references with the leak rule.

    :ivar hashes: Sorted uint64 n-gram hashes (one entry per (hash, reference-kind) pair).
    :ivar owners: Reference-kind index of each entry (aligned with :attr:`hashes`).
    :ivar refs: :class:`RefInfo` per reference-kind index.
    :ivar stop: Sorted stop-listed hashes.
    """

    def __init__(self, n: Optional[Mapping[str, int]] = None, stop_df: int = STOP_DF,
                 min_shared: Optional[Mapping[str, int]] = None, min_share: float = DEFAULT_MIN_SHARE,
                 kinds: Sequence[str] = KINDS) -> None:
        """Create an empty index.

        :param n: Gram length per kind.
        :param stop_df: Stop-list n-grams found in at least this many distinct documents.
        :param min_shared: Shared-gram threshold per kind.
        :param min_share: Share-of-one-reference threshold.
        :param kinds: Kinds indexed.
        """
        self.n = dict(DEFAULT_N, **(n or {}))
        self.stop_df = stop_df
        self.min_shared = dict(DEFAULT_MIN_SHARED, **(min_shared or {}))
        self.min_share = min_share
        self.kinds = tuple(kinds)
        self.hashes = np.zeros(0, dtype=np.uint64)
        self.owners = np.zeros(0, dtype=np.int32)
        self.refs: List[RefInfo] = []
        self.stop = np.zeros(0, dtype=np.uint64)
        self.n_input_refs = 0
        self.n_background_docs = 0

    # -------------------------------------------------------------------------------- building
    @classmethod
    def build(cls, refs: Iterable[Tuple[str, str, str]], background: Optional[Iterable[Tuple[str, str]]] = None,
              **kwargs: Any) -> "NgramIndex":
        """Index references.

        :param refs: ``(ref_id, doc, text)`` triples; ``doc`` groups references of one document for
            the stop-list document frequency. Identical texts are indexed once.
        :param background: ``(doc, text)`` pairs that only count toward the document frequency
            (documents that are not references; a background doc id equal to a reference doc id is
            ignored so no document counts twice).
        :param kwargs: Constructor arguments (``n``, ``stop_df``, ``min_shared``, ``min_share``, ``kinds``).
        :returns: The built index.
        """
        idx = cls(**kwargs)
        seen: Set[str] = set()
        ref_meta: List[Tuple[str, str, str]] = []
        gram_parts: List[np.ndarray] = []
        owner_parts: List[np.ndarray] = []
        doc_of: List[int] = []
        doc_ids: Dict[str, int] = {}
        for ref_id, doc, text in refs:
            if not text or not text.strip():
                continue
            h = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            idx.n_input_refs += 1
            d = doc_ids.setdefault(str(doc), len(doc_ids))
            for kind, g in text_grams(text, idx.kinds, idx.n).items():
                if not len(g):
                    continue
                ri = len(ref_meta)
                ref_meta.append((ref_id, str(doc), kind))
                doc_of.append(d)
                gram_parts.append(g)
                owner_parts.append(np.full(len(g), ri, dtype=np.int32))
        if not gram_parts:
            return idx
        grams = np.concatenate(gram_parts)
        owners = np.concatenate(owner_parts)
        docs = np.asarray(doc_of, dtype=np.int64)[owners]
        # document frequency over the references: distinct (hash, doc) pairs per hash
        order = np.lexsort((docs, grams))
        g_s, d_s = grams[order], docs[order]
        first = np.ones(len(g_s), dtype=bool)
        first[1:] = (g_s[1:] != g_s[:-1]) | (d_s[1:] != d_s[:-1])
        uh, df = np.unique(g_s[first], return_counts=True)
        if background is not None:
            bh, bdf, idx.n_background_docs = _doc_frequency(background, idx.kinds, idx.n, exclude=set(doc_ids))
            if len(bh):
                pos = np.minimum(np.searchsorted(bh, uh), len(bh) - 1)
                df = df + np.where(bh[pos] == uh, bdf[pos], 0)
        idx.stop = uh[df >= idx.stop_df]
        keep = ~np.isin(grams, idx.stop)
        grams, owners = grams[keep], owners[keep]
        o2 = np.argsort(grams, kind="stable")
        idx.hashes, idx.owners = grams[o2], owners[o2]
        counts = np.bincount(idx.owners, minlength=len(ref_meta))
        idx.refs = [RefInfo(r, d, k, int(c)) for (r, d, k), c in zip(ref_meta, counts)]
        return idx

    # -------------------------------------------------------------------------------- queries
    def _shared_counts(self, text: Optional[str]) -> Dict[int, int]:
        """Shared (non-stop) n-gram count per reference-kind.

        :param text: Text to test.
        :returns: reference-kind index → shared n-grams (only non-zero entries).
        """
        if not len(self.hashes) or not text:
            return {}
        q = np.concatenate([g for g in text_grams(text, self.kinds, self.n).values() if len(g)] or
                           [np.zeros(0, dtype=np.uint64)])
        if not len(q):
            return {}
        lo = np.searchsorted(self.hashes, q, side="left")
        hi = np.searchsorted(self.hashes, q, side="right")
        hit = hi > lo
        if not hit.any():
            return {}
        lo, hi = lo[hit], hi[hit]
        lens = hi - lo
        starts = np.repeat(lo, lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        own = self.owners[starts + offs]
        u, c = np.unique(own, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}

    def ref_hits(self, text: Optional[str]) -> Dict[str, Tuple[int, float]]:
        """Shared n-grams and share for every reference the text touches.

        :param text: Text to test.
        :returns: ``"<ref_id>#<kind>"`` → ``(shared n-grams, shared / reference n-grams)``.
        """
        out = {}
        for ri, n in self._shared_counts(text).items():
            r = self.refs[ri]
            out[f"{r.ref_id}#{r.kind}"] = (n, n / r.n_grams if r.n_grams else 0.0)
        return out

    def page_hits(self, text: Optional[str]) -> Tuple[int, float]:
        """Largest shared count with one reference and largest share of one reference.

        :param text: Text to test.
        :returns: ``(shared n-grams, max share of one reference)`` (``(0, 0.0)`` when clean).
        """
        hits = self.ref_hits(text)
        if not hits:
            return 0, 0.0
        return max(n for n, _ in hits.values()), max(s for _, s in hits.values())

    def _leaks(self, ri: int, n: int) -> bool:
        """The drop rule for one reference-kind.

        :param ri: Reference-kind index.
        :param n: Shared n-grams.
        :returns: True when the text leaks this reference.
        """
        r = self.refs[ri]
        return n >= self.min_shared.get(r.kind, 5) or (r.n_grams >= MIN_REF_GRAMS and n / r.n_grams >= self.min_share)

    def leak_refs(self, text: Optional[str]) -> List[Tuple[str, str, int, float]]:
        """References a text leaks, worst first.

        :param text: Text to test.
        :returns: ``[(ref_id, kind, shared, share), ...]`` sorted by share then shared, descending.
        """
        out = []
        for ri, n in self._shared_counts(text).items():
            if self._leaks(ri, n):
                r = self.refs[ri]
                out.append((r.ref_id, r.kind, n, n / r.n_grams if r.n_grams else 0.0))
        return sorted(out, key=lambda t: (-t[3], -t[2], t[0], t[1]))

    def is_leak(self, text: Optional[str]) -> bool:
        """Whether a text leaks any reference.

        :param text: Text to test.
        :returns: True on a leak.
        """
        return any(self._leaks(ri, n) for ri, n in self._shared_counts(text).items())

    def stats(self) -> Dict[str, Any]:
        """Index size and settings.

        :returns: Summary dictionary.
        """
        per_kind = collections.Counter(r.kind for r in self.refs)
        return {"references": self.n_input_refs, "reference_kinds": dict(per_kind),
                "documents": len({r.doc for r in self.refs}), "background_documents": self.n_background_docs,
                "entries": int(len(self.hashes)),
                "stop_listed": int(len(self.stop)), "n": self.n, "stop_df": self.stop_df,
                "min_shared": self.min_shared, "min_share": self.min_share}

    # -------------------------------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Path:
        """Save the index (``.npz`` with the hash arrays and a JSON header).

        :param path: Destination file.
        :returns: The written path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        header = {"n": self.n, "stop_df": self.stop_df, "min_shared": self.min_shared, "min_share": self.min_share,
                  "kinds": list(self.kinds), "n_input_refs": self.n_input_refs,
                  "n_background_docs": self.n_background_docs,
                  "refs": [[r.ref_id, r.doc, r.kind, r.n_grams] for r in self.refs]}
        fd, tmp = tempfile.mkstemp(prefix=f".{dest.stem}.", suffix=".npz", dir=str(dest.parent))
        os.close(fd)
        np.savez_compressed(tmp, hashes=self.hashes, owners=self.owners, stop=self.stop,
                            header=np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8))
        os.replace(tmp, dest)
        return dest

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NgramIndex":
        """Load an index written by :meth:`save`.

        :param path: ``.npz`` file.
        :returns: The index.
        """
        with np.load(path) as z:
            header = json.loads(bytes(z["header"]).decode("utf-8"))
            idx = cls(n=header["n"], stop_df=header["stop_df"], min_shared=header["min_shared"],
                      min_share=header["min_share"], kinds=header["kinds"])
            idx.hashes, idx.owners, idx.stop = z["hashes"], z["owners"], z["stop"]
        idx.n_input_refs = header["n_input_refs"]
        idx.n_background_docs = header.get("n_background_docs", 0)
        idx.refs = [RefInfo(*r) for r in header["refs"]]
        return idx


# ============================================================================== block rule, project names
def target_share(target: Optional[str], context: Union[str, Iterable[str]], n: int = 8) -> float:
    """Share of a target's word n-grams present in a context (the union of a RAG block's pages).

    :param target: The row's target text.
    :param context: One text or several (their n-grams are united).
    :param n: Word n-gram length.
    :returns: Shared / target n-grams (0.0 for a target shorter than ``n`` words).
    """
    tg = gram_hashes(target, "w8", n)
    if not len(tg):
        return 0.0
    texts = [context] if isinstance(context, str) else list(context)
    cg = np.unique(np.concatenate([gram_hashes(t, "w8", n) for t in texts] or [np.zeros(0, dtype=np.uint64)]))
    return float(np.isin(tg, cg, assume_unique=True).sum()) / len(tg)


def block_ok(target: Optional[str], context: Union[str, Iterable[str]], max_share: float = BLOCK_MAX_SHARE,
             n: int = 8) -> bool:
    """The L3 block rule: a literature block must hold less than ``max_share`` of its target's word n-grams.

    :param target: The row's target.
    :param context: The block's page texts.
    :param max_share: Allowed share (exclusive).
    :param n: Word n-gram length.
    :returns: True when the block may be used.
    """
    return target_share(target, context, n) < max_share


def source_scan(text: Optional[str]) -> List[str]:
    """scrubbed project names and hosts in a text (:data:`textnorm.SCRUB_RE`).

    :param text: Any text.
    :returns: The matched strings in order (empty when clean).
    """
    if not text:
        return []
    return [m.group(0) for m in _tn.SCRUB_RE.finditer(text)]


def row_source_scan(row: Mapping[str, Any]) -> List[str]:
    """project-name strings anywhere in a row: every message and every meta value (F4: meta included).

    :param row: Chat row.
    :returns: The matched strings (empty when the row is clean).
    """
    return source_scan(json.dumps(row, ensure_ascii=False))


# ============================================================================== shelfmark mentions in text (F1, F10)
#: Collection prefixes that open a shelfmark mention in prose (case-insensitive), in their spaced and
#: abbreviated spellings too: ``TS 98 J 7 6``, ``T-S 97 J 26.10``, ``T-S ns. 999.19``, ``TS Arabic Box 99.255``,
#: ``TS G9.18`` (any series letter), ``ENA 9822 A.56``, ``NLI 977.3/5``, ``BL OR 9566B.1``, ``Moss. IX,249``,
#: ``Or. 9080 J 42``, ``MIAC 99973.46``. A bare ``OR`` counts in upper case only (``or 5`` is English).
MENTION_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"T\.?\s?-?\s?S\.?(?![A-Za-z])(?=\s*[A-Za-z0-9])|"
    r"Taylor[\s-]*Schechter(?=\s*[A-Za-z0-9])|"
    r"ENA|MIAC|(?:NLI|JNUL)(?=\s*\d)|"
    r"(?:Bodl(?:eian)?\.?\s*)?MS\.?\s*Heb\.?(?=\s*[a-g]\b)|Bodl(?:eian)?\.?|"
    r"(?-i:BL\.?\s*OR|OR)(?=\s*\d)|(?:BL\.?\s*)?Or\.(?=\s*\d)|"
    r"Moss(?:eri)?\.?(?=\s*[IVXL]+\b)|"
    r"JRL|Rylands|AIU|Halper|CAJS|Westminster|Antonin|Firkovich|Mosseri)", re.I)
#: What a mention may run over after its prefix: letters, digits, blanks and ``. , / : _ -``, across at most one
#: line break (``T-S Misc.⏎99.24r``, ``ENA NS 77,⏎fol. 99``), 32 characters in all.
MENTION_TAIL_RE = re.compile(r"[A-Za-z0-9 \t\n.,/:_\-]{0,32}")
#: Catalogue-number inserts inside a shelfmark (``Bodl. MS Heb. d. 99 (Cat. 9999), fol. 9``), removed before matching.
MENTION_INSERT_RE = re.compile(
    r"[ \t]*[(\[][ \t]*(?:Cat|Neub(?:auer)?|Cowley)\.?[ \t]*(?:no\.?[ \t]*)?\d+[a-z]?[ \t]*[)\]]", re.I)
_MENTION_TOKEN_RE = re.compile(r"[a-z]+|\d+")
#: Abbreviations that name a collection whose word the shelfmark key drops (:data:`shelfmarks.INSTITUTION_WORDS`).
MENTION_BASE_ALIASES: Dict[str, str] = {"moss": "mosseri", "rylands": "jrl"}
#: Full alias set (F10): series words as the normaliser keys them (``TS Arabic Box 99.255`` → ``tsar99_255``).
MENTION_ALIASES: Dict[str, str] = {**MENTION_BASE_ALIASES, "arabic": "ar", "miscellaneous": "misc"}
#: Two-word aliases.
MENTION_PAIR_ALIASES: Dict[Tuple[str, str], str] = {("taylor", "schechter"): "ts", ("new", "series"): "ns"}
#: Words dropped anywhere inside a mention (``TS Arabic Box 99.255``).
MENTION_DROP: frozenset = frozenset({"box", "fragment", "frag"})
#: Folio words dropped between two numbers (``d 99, fol. 22``).
_FOLIO_TOKENS = frozenset({"f", "fol", "fols", "folio", "ff"})
#: Side words: a leaf number followed by one of them is still ambiguous between lettered leaves.
SIDE_TOKENS: frozenset = frozenset({"r", "v", "recto", "verso"})
#: Collections cited by bare inventory number (``inventory number: 99973.46`` on an index page, F10).
BARE_NUMBER_PREFIXES: Tuple[str, ...] = ("miac",)
_BARE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.,/])(\d{4,6})\.(\d{1,4})(?![0-9]|\.\d)")
_BARE_KEY_RE = re.compile(r"^([a-z]+)(\d+)_(\d+)$")
_LEAF_KEY_RE = re.compile(r"\d[a-z]$")


def _digits_norm(key: str) -> str:
    """A compact shelfmark key with leading zeros removed from every digit run.

    :param key: Compact key (``ts8j6_08``).
    :returns: ``ts8j6_8``.
    """
    return re.sub(r"\d+", lambda m: str(int(m.group(0))), key)


def mention_norm(key: str) -> str:
    """The form in which held keys and mention candidates are compared: leading zeros removed and a leading ``ms``
    dropped before a letter (``Bodl. Heb. d. 99`` and ``MS heb. d 99`` name one leaf).

    :param key: Compact key.
    :returns: Normalised key (``mshebd099_7`` → ``hebd99_7``; ``ms8998_7`` is kept).
    """
    k = _digits_norm(key)
    return k[2:] if k.startswith("ms") and k[2:3].isalpha() else k


def mention_text(text: Optional[str]) -> str:
    """Text prepared for mention matching: catalogue-number inserts removed (:data:`MENTION_INSERT_RE`).

    :param text: Any text.
    :returns: The text without ``(Cat. N)`` inserts.
    """
    return MENTION_INSERT_RE.sub("", text or "")


def mention_token_gaps(span: str, full: bool = True) -> List[Tuple[str, str]]:
    """Key tokens of a mention with the separator text before each: lower-case letter runs and digit runs,
    collection abbreviations aliased, leading institution words dropped (as :func:`shelfmarks.core_key` drops them),
    leading zeros removed. With ``full`` the F10 aliases apply too (``arabic`` → ``ar``, ``taylor schechter`` →
    ``ts``, ``new series`` → ``ns``) and ``box`` / ``fragment`` are dropped (their text joins the next gap).

    :param span: Mention text (prefix + tail).
    :param full: Apply the full alias / drop set (False: :data:`MENTION_BASE_ALIASES` only).
    :returns: ``[(token, gap before it)]``.
    """
    from src.finetuning.text27b import shelfmarks as _sm
    low = span.lower()
    raw: List[Tuple[str, str]] = []
    prev = 0
    for m in _MENTION_TOKEN_RE.finditer(low):
        raw.append((m.group(0), low[prev:m.start()]))
        prev = m.end()
    aliases = MENTION_ALIASES if full else MENTION_BASE_ALIASES
    out: List[Tuple[str, str]] = []
    carry = ""
    i = 0
    while i < len(raw):
        t, g = raw[i]
        g = carry + g
        carry = ""
        if full and i + 1 < len(raw) and (t, raw[i + 1][0]) in MENTION_PAIR_ALIASES:
            out.append((MENTION_PAIR_ALIASES[(t, raw[i + 1][0])], g))
            i += 2
            continue
        if full and t in MENTION_DROP:
            carry = g + t
            i += 1
            continue
        out.append((aliases.get(t, t), g))
        i += 1
    while out and out[0][0] in _sm.INSTITUTION_WORDS:
        out.pop(0)
    return [(str(int(t)) if t.isdigit() else t, g) for t, g in out]


def mention_tokens(span: str) -> List[str]:
    """Key tokens of a mention (:func:`mention_token_gaps` without the separators).

    :param span: Mention text (prefix + tail).
    :returns: Tokens (``TS 98 J 7 6, ed.`` → ``['ts', '98', 'j', '7', '6', 'ed']``).
    """
    return [t for t, _ in mention_token_gaps(span)]


def compact_tokens(tokens: Sequence[str]) -> str:
    """Join key tokens the way :func:`shelfmarks.compact` writes a key: ``_`` only between two digit runs.

    :param tokens: Tokens.
    :returns: Compact key (``['ts', 'ns', '999', '19']`` → ``tsns999_19``).
    """
    out: List[str] = []
    for i, t in enumerate(tokens):
        if i and t.isdigit() and tokens[i - 1].isdigit():
            out.append("_")
        out.append(t)
    return "".join(out)


def _drop_folio(tg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Folio words between two numbers removed (their text joins the next gap).

    :param tg: ``(token, gap)`` list.
    :returns: The list without those folio words.
    """
    out: List[Tuple[str, str]] = []
    carry = ""
    for i, (t, g) in enumerate(tg):
        if t in _FOLIO_TOKENS and 0 < i < len(tg) - 1 and tg[i - 1][0].isdigit() and tg[i + 1][0].isdigit():
            carry += g + t
            continue
        out.append((t, carry + g))
        carry = ""
    return out


def mention_candidates(span: str) -> List[Tuple[str, bool]]:
    """Candidate shelfmark keys of a mention with their openness, most tokens first.

    A candidate is every token prefix that holds a digit and does not cut a number off from the number after it
    (``TS 98 J 7 6`` → ``ts98j7_6``, never ``ts98j7``) unless a line break separates the two (an index line and the
    next line's number); folio words between two numbers are also tried without (``d 99, fol. 22`` → ``d99_22``);
    both the full and the base alias sets are tried. A candidate is *open* (F10) when it ends in a number that no
    leaf letter follows directly: nothing follows, a side word follows (``99 1r.15``), or the next token is set off
    by a separator -- such a read is ambiguous between the lettered leaves ``99.1a`` / ``99.1b``.

    :param span: Mention text.
    :returns: ``[(compact key, open)]`` (distinct keys; a key open in any variant is open).
    """
    found: Dict[str, Tuple[int, int, bool]] = {}
    order = 0
    for full in (True, False):
        tg = mention_token_gaps(span, full)
        variants = [tg]
        nofol = _drop_folio(tg)
        if nofol != tg:
            variants.append(nofol)
        for v in variants:
            toks = [t for t, _ in v]
            for n in range(len(v), 0, -1):
                last = toks[n - 1]
                nxt = v[n] if n < len(v) else None
                if last.isdigit() and nxt is not None and nxt[0].isdigit() and "\n" not in nxt[1]:
                    continue
                if not any(t.isdigit() for t in toks[:n]):
                    break
                k = compact_tokens(toks[:n])
                is_open = last.isdigit() and (nxt is None or nxt[1] != "" or nxt[0] in SIDE_TOKENS or len(nxt[0]) > 1)
                if k in found:
                    n0, o0, op0 = found[k]
                    found[k] = (n0, o0, op0 or is_open)
                else:
                    found[k] = (n, order, is_open)
                    order += 1
    return [(k, v[2]) for k, v in sorted(found.items(), key=lambda kv: (-kv[1][0], kv[1][1]))]


def mention_key_candidates(span: str) -> List[str]:
    """Candidate shelfmark keys of a mention, longest first (:func:`mention_candidates` without openness).

    :param span: Mention text.
    :returns: Distinct compact keys.
    """
    return [k for k, _ in mention_candidates(span)]


def mention_spans(text: Optional[str]) -> List[Tuple[int, int]]:
    """Spans of shelfmark-like mentions: a collection prefix (:data:`MENTION_PREFIX_RE`) and up to 32 characters
    of shelfmark-like tail (at most one line break).

    :param text: Any text (callers pass :func:`mention_text` output so ``(Cat. N)`` inserts are gone).
    :returns: ``(start, end)`` spans (they may overlap: a tail can run into the next mention).
    """
    out: List[Tuple[int, int]] = []
    text = text or ""
    for m in MENTION_PREFIX_RE.finditer(text):
        tail = MENTION_TAIL_RE.match(text, m.end())
        end = tail.end() if tail else m.end()
        first = text.find("\n", m.end(), end)
        second = text.find("\n", first + 1, end) if first >= 0 else -1
        out.append((m.start(), second if second >= 0 else end))
    return out


class MentionIndex:
    """Shelfmark keys of the hold-out set in a form spaced and abbreviated spellings resolve to (F1, F10).

    The benchmark-tier and eval-tier key maps of :class:`holdouts.Holdouts` (and the keys of their canonical ids)
    are indexed in :func:`mention_norm` form. A mention resolves to its longest candidate key
    (:func:`mention_candidates`) found in either tier, so ``TS 98 J 7 6``, ``T-S 98J7.6`` and
    ``Cambridge_CUL_T_S_8J6_8`` name the same document; an *open* candidate (a leaf number without its letter,
    ``TS NS 999 1r.15``) resolves to every held lettered leaf it may mean (``999.1a`` / ``999.1b``): an ambiguous
    read counts as held when any expansion is held. Bare inventory numbers of :data:`BARE_NUMBER_PREFIXES`
    collections resolve too (``inventory number: 99973.46``).
    """

    def __init__(self, bench: Mapping[str, Sequence[str]], evals: Mapping[str, Sequence[str]]) -> None:
        """Index two reason maps.

        :param bench: Key → reasons of the benchmark tier.
        :param evals: Key → reasons of the eval tier.
        """
        self.bench: Dict[str, Set[str]] = collections.defaultdict(set)
        self.evals: Dict[str, Set[str]] = collections.defaultdict(set)
        for src, dst in ((bench, self.bench), (evals, self.evals)):
            for k, rs in src.items():
                if k:
                    dst[mention_norm(str(k))] |= set(rs)
        self.stems: Dict[str, Set[str]] = collections.defaultdict(set)
        self.bare: Dict[str, Set[str]] = collections.defaultdict(set)
        for k in set(self.bench) | set(self.evals):
            if _LEAF_KEY_RE.search(k):
                self.stems[k[:-1]].add(k)
            m = _BARE_KEY_RE.match(k)
            if m and m.group(1) in BARE_NUMBER_PREFIXES:
                self.bare[f"{m.group(2)}_{m.group(3)}"].add(k)

    @classmethod
    def from_holdouts(cls, holdouts: _ho.Holdouts) -> "MentionIndex":
        """Build from a hold-out set (key maps plus the keys of the canonical ids).

        :param holdouts: Hold-out set.
        :returns: The index.
        """
        from src.finetuning.text27b import shelfmarks as _sm
        bench: Dict[str, Set[str]] = collections.defaultdict(set)
        evals: Dict[str, Set[str]] = collections.defaultdict(set)
        for dst, kmap, cmap in ((bench, holdouts.key_map, holdouts.canonical),
                                (evals, holdouts.eval_key_map, holdouts.eval_canonical)):
            for k, rs in kmap.items():
                dst[k] |= set(rs)
            for cid, rs in cmap.items():
                k = _sm.canonical_to_key(cid)
                if k:
                    dst[k] |= set(rs)
        return cls(bench, evals)

    def held(self, key: str) -> bool:
        """Whether a normalised key is in either tier.

        :param key: :func:`mention_norm` key.
        :returns: True when held.
        """
        return key in self.bench or key in self.evals

    def resolve_all(self, span: str) -> List[str]:
        """Held keys a mention names: its longest held candidate, or for an open candidate every held lettered
        leaf it may mean.

        :param span: Mention text.
        :returns: Normalised keys (empty when nothing held is named).
        """
        for k, is_open in mention_candidates(span):
            nk = mention_norm(k)
            if self.held(nk):
                return [nk]
            if is_open and nk in self.stems:
                return sorted(self.stems[nk])
        return []

    def resolve(self, span: str) -> Optional[str]:
        """The (first) held key a mention names, or None (:meth:`resolve_all`).

        :param span: Mention text.
        :returns: Normalised key.
        """
        keys = self.resolve_all(span)
        return keys[0] if keys else None

    def reasons(self, key: str, include_eval: bool = True) -> List[str]:
        """Hold-out reasons of a held key, with the ``key:`` prefix.

        :param key: Normalised key.
        :param include_eval: Also the eval tier's reasons.
        :returns: Sorted reasons.
        """
        rs = set(self.bench.get(key, ())) | (set(self.evals.get(key, ())) if include_eval else set())
        return sorted({_ho.derive(_ho.KEY_PREFIX, r) for r in rs})

    def find(self, text: Optional[str], include_eval: bool = True) -> List[Tuple[str, str, List[str]]]:
        """Held documents a text names (prefixed mentions and bare inventory numbers).

        :param text: Any text.
        :param include_eval: Also report eval-tier documents (False: a key held only in the eval tier is skipped).
        :returns: ``[(mention text, key, reasons)]``; reasons carry the ``key:`` prefix.
        """
        t = mention_text(text)
        out: List[Tuple[str, str, List[str]]] = []
        for a, b in mention_spans(t):
            span = t[a:b]
            for k in self.resolve_all(span):
                rs = self.reasons(k, include_eval)
                if rs:
                    out.append((span.strip(), k, rs))
        for m in _BARE_NUMBER_RE.finditer(t):
            for k in sorted(self.bare.get(f"{int(m.group(1))}_{int(m.group(2))}", ())):
                rs = self.reasons(k, include_eval)
                if rs:
                    out.append((m.group(0), k, rs))
        return out

    def reason(self, texts: Iterable[Optional[str]], include_eval: bool = True) -> Optional[str]:
        """Best hold-out reason over the documents the texts name.

        :param texts: Texts.
        :param include_eval: Also count eval-tier documents (False: benchmark tier only).
        :returns: The best reason (``key:pgp131``, ``key:test`` ...) or None.
        """
        found: List[str] = []
        for t in texts:
            if t:
                for _, _, rs in self.find(t, include_eval):
                    found += rs
        return _ho.best_reason(found)


_MENTION_CACHE: Dict[int, Tuple[Any, MentionIndex]] = {}


def mention_index(holdouts: _ho.Holdouts) -> MentionIndex:
    """The :class:`MentionIndex` of a hold-out set (built once per object).

    :param holdouts: Hold-out set.
    :returns: Its index.
    """
    hit = _MENTION_CACHE.get(id(holdouts))
    if hit is None or hit[0] is not holdouts:
        hit = (holdouts, MentionIndex.from_holdouts(holdouts))
        _MENTION_CACHE[id(holdouts)] = hit
    return hit[1]


def text_mention_reason(texts: Iterable[Optional[str]], holdouts: _ho.Holdouts, include_eval: bool = True) -> Optional[str]:
    """Best reason the texts name a held document: shelfmarks found by :func:`textnorm.find_shelfmarks` keyed by
    the normaliser (:meth:`holdouts.Holdouts.match_any`) together with the spaced / abbreviated / ambiguous / bare
    spellings of :class:`MentionIndex` (F1, F10). Used by the page exclusions, the row exclusions and the re-scan alike.

    :param texts: Texts (page text, user turn, target).
    :param holdouts: Hold-out set.
    :param include_eval: Also count eval-tier documents.
    :returns: The best reason or None.
    """
    ts = [t for t in texts if t]
    found = sorted({t[a:b] for t in ts for a, b in _tn.find_shelfmarks(t)})
    r1 = holdouts.match_any(shelfmarks=found, include_eval=include_eval) if found else None
    r2 = mention_index(holdouts).reason(ts, include_eval=include_eval)
    return _ho.best_reason(r for r in (r1, r2) if r)


# ============================================================================== references
_DOC_RE = re.compile(r"<document>\n?(.*?)(?:\n?</document>|$)", re.S)


def document_input(user: str) -> str:
    """The text between ``<document>`` tags of a user turn ('' when absent).

    :param user: User message.
    :returns: Document text.
    """
    m = _DOC_RE.search(user or "")
    return m.group(1) if m else ""


#: JSON targets (catalogue, relations) are reduced to string values at least this long for the
#: n-gram rule, so shared keys, ``null`` runs and date tuples are not counted as shared text.
JSON_PROSE_MIN_CHARS = 30


def prose_of(text: Optional[str], min_chars: int = JSON_PROSE_MIN_CHARS) -> str:
    """Text used for the n-gram rule: a JSON target becomes its long string values (one per line).

    :param text: Any text; a JSON object / array is reduced, anything else is returned unchanged.
    :param min_chars: Shortest string value kept from JSON.
    :returns: Text ('' for a JSON value without long strings).
    """
    if not text:
        return ""
    t = text.strip()
    if not t or t[0] not in "{[":
        return text
    try:
        obj = json.loads(t)
    except ValueError:
        return text
    out: List[str] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            stack.extend(reversed(list(o.values())))
        elif isinstance(o, list):
            stack.extend(reversed(o))
        elif isinstance(o, str) and len(o.strip()) >= min_chars:
            out.append(o.strip())
    return "\n".join(out)


#: Tasks whose targets are catalogue prose (bibliography clauses are stripped from these references,
#: as the v0_2 builder strips them from the targets, plan B4).
PROSE_TASKS: Tuple[str, ...] = ("describe_pgp", "catalogue_json", "analyse_document")


def reference_text(text: Optional[str], task: Optional[str] = None) -> str:
    """Normalise a reference text for the n-gram rule.

    JSON targets become their long strings (:func:`prose_of`); catalogue prose (``PROSE_TASKS``
    targets and PGP descriptions, ``task="description"``) loses its citation clauses
    (:func:`textnorm.strip_citations`), so a book title cited in a description does not match the
    bibliography of a literature page.

    :param text: Raw text.
    :param task: Task of the row the text is the target of (``description`` for PGP descriptions).
    :returns: The text to index.
    """
    t = prose_of(text)
    if t and task in PROSE_TASKS + ("description",):
        t = _tn.strip_citations(t)[0]
    return t


def _msg(row: Mapping[str, Any], role: str) -> str:
    """Content of the last message with a role.

    :param row: Chat row.
    :param role: ``user`` / ``assistant`` / ``system``.
    :returns: Content ('' when absent).
    """
    for m in reversed(row.get("messages") or []):
        if m.get("role") == role:
            return m.get("content") or ""
    return ""


def reference_texts(p: _paths.Paths, holdouts: _ho.Holdouts, include_pairs: bool = True,
                    include_descriptions: bool = True, corpus: Any = None) -> Iterator[Tuple[str, str, str]]:
    """Eval references: targets and edition inputs of held documents' v0_1 rows, their pgp_parallel
    editions / translations and their PGP descriptions.

    :param p: Build paths.
    :param holdouts: The hold-out set (both tiers are references).
    :param include_pairs: Add ``pgp_parallel`` texts of held documents.
    :param include_descriptions: Add the PGP ``description`` of held documents.
    :param corpus: Pre-loaded :class:`shelfmarks.Corpus` for the descriptions (loaded when needed).
    :returns: Iterator of ``(ref_id, doc, text)``; ``ref_id`` is ``tgt:<task>:<pgpid>`` (targets
        normalised by :func:`reference_text`), ``inp:<pgpid>``, ``pp:<field>:<pgpid>`` or ``desc:<pgpid>``; a
        second distinct text under the same id gets a ``:2`` suffix (e.g. the Hebrew-matched edition).
    """
    held = holdouts.pgpids | holdouts.eval_pgpids
    seen: Set[Tuple[str, str]] = set()
    n_ids: collections.Counter = collections.Counter()

    def new(base: str, text: str) -> Optional[str]:
        """Reference id for a text not yet emitted under ``base`` (``base``, then ``base:2`` ...).

        :param base: Base id (``tgt:<task>:<pgpid>`` / ``inp:<pgpid>``).
        :param text: Text.
        :returns: The id, or None when this text was already emitted under the base.
        """
        key = (base, hashlib.sha1(text.strip().encode("utf-8")).hexdigest())
        if not text.strip() or key in seen:
            return None
        seen.add(key)
        n_ids[base] += 1
        return base if n_ids[base] == 1 else f"{base}:{n_ids[base]}"

    for sp in ("train", "dev", "test"):
        f = p.text_ft_v0_1 / f"{sp}.jsonl"
        if not f.exists():
            LOG.warning("v0_1 file missing: %s", f.name)
            continue
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            pg = str(m.get("pgpid") or "")
            if pg not in held:
                continue
            tgt = reference_text(_msg(r, "assistant"), m.get("task"))
            rid = new(f"tgt:{m.get('task')}:{pg}", tgt)
            if rid:
                yield rid, pg, tgt
            if m.get("condition") in ("edition", "edition_rescript"):
                doc = document_input(_msg(r, "user"))
                rid = new(f"inp:{pg}", doc)
                if rid:
                    yield rid, pg, doc
    if include_pairs and p.pgp_parallel.exists():
        for r in _rows.read_jsonl(p.pgp_parallel):
            pg = str(r.get("pgpid") or "")
            if pg not in held:
                continue
            for fld in ("edition_text", "edition_text_en", "edition_text_he", "translation_en", "translation_he"):
                v = r.get(fld)
                if isinstance(v, str) and v.strip() and v.strip() != "None":
                    yield f"pp:{fld}:{pg}", pg, v
    if include_descriptions:
        if corpus is None:
            from src.finetuning.text27b import shelfmarks as _sm
            corpus = _sm.load_corpus(p)
        for pg in sorted(held):
            d = reference_text((corpus.docs.get(pg) or {}).get("description") or "", "description")
            if d.strip():
                yield f"desc:{pg}", pg, d


def background_texts(p: _paths.Paths, holdouts: _ho.Holdouts) -> Iterator[Tuple[str, str]]:
    """Texts of the documents that are NOT held (stop-list document frequency only).

    :param p: Build paths.
    :param holdouts: Hold-out set.
    :returns: Iterator of ``(doc, text)``: pgp_parallel editions / translations / descriptions and
        v0_1 train targets and edition inputs of non-held documents.
    """
    held = holdouts.pgpids | holdouts.eval_pgpids
    if p.pgp_parallel.exists():
        for r in _rows.read_jsonl(p.pgp_parallel):
            pg = str(r.get("pgpid") or "")
            if not pg or pg in held:
                continue
            for fld in ("edition_text", "edition_text_en", "edition_text_he", "translation_en", "translation_he",
                        "description"):
                v = r.get(fld)
                if isinstance(v, str) and v.strip() and v.strip() != "None":
                    yield pg, reference_text(v, "description" if fld == "description" else None)
    f = p.text_ft_v0_1 / "train.jsonl"
    if f.exists():
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            pg = str(m.get("pgpid") or "")
            if not pg or pg in held:
                continue
            yield pg, reference_text(_msg(r, "assistant"), m.get("task"))
            if m.get("condition") == "edition":
                yield pg, document_input(_msg(r, "user"))


#: Default file name stem of the cached reference index under ``out_dir/cache``.
INDEX_CACHE_STEM = "ngram_index"
#: Bumped whenever reference selection or hashing changes (part of the cache signature).
INDEX_VERSION = 4


def build_reference_index(p: Optional[_paths.Paths] = None, holdouts: Optional[_ho.Holdouts] = None,
                          cache: bool = True, background: bool = True, corpus: Any = None,
                          **kwargs: Any) -> NgramIndex:
    """Build (or load from ``out_dir/cache/ngram_index_<sig>.npz``) the eval-reference index.

    The cache is reused while the held pgpids, the v0_1 files and ``pgp_parallel`` (size and
    modification time) and the settings are unchanged.

    :param p: Build paths.
    :param holdouts: Hold-out set (default :meth:`Holdouts.load`).
    :param cache: Use / write the cache.
    :param background: Count non-held PGP documents toward the stop-list document frequency.
    :param corpus: Pre-loaded :class:`shelfmarks.Corpus` for the PGP descriptions (loaded when None).
    :param kwargs: :class:`NgramIndex` settings.
    :returns: The index.
    """
    p = p or _paths.load_paths()
    holdouts = holdouts or _ho.Holdouts.load(p=p)
    srcs = [p.text_ft_v0_1 / "train.jsonl", p.text_ft_v0_1 / "dev.jsonl", p.text_ft_v0_1 / "test.jsonl", p.pgp_parallel]
    sig_src: List[Any] = [[str(f), f.stat().st_size, f.stat().st_mtime_ns] if f.exists() else [str(f)] for f in srcs]
    held_fp = hashlib.sha1(json.dumps([sorted(holdouts.pgpids), sorted(holdouts.eval_pgpids)]).encode("utf-8")).hexdigest()
    sig_src.append([INDEX_VERSION, held_fp, background, sorted((k, str(v)) for k, v in kwargs.items())])
    sig = hashlib.sha1(json.dumps(sig_src, default=str).encode("utf-8")).hexdigest()
    cp = p.out("cache", f"{INDEX_CACHE_STEM}_{sig[:12]}.npz")
    if cache and cp.exists():
        LOG.info("n-gram index from cache %s", cp.name)
        return NgramIndex.load(cp)
    t0 = time.time()
    idx = NgramIndex.build(reference_texts(p, holdouts, corpus=corpus),
                           background=background_texts(p, holdouts) if background else None, **kwargs)
    LOG.info("n-gram index built in %.1fs: %s", time.time() - t0, idx.stats())
    if cache:
        idx.save(cp)
    return idx


# ============================================================================== train-target dedupe + row exclusions (D5)
#: Families whose TRAIN targets literature pages are deduplicated against (plan §9.4.6, decision D5).
TRAIN_DUP_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he", "analyse_document")
#: Further family files whose translate / analyse TRAIN rows count when present (L5 pairs, widened L2, the
#: public-domain edition pairs of :mod:`canon_diff`, which t27b trains as L5 translate rows, and the web translation
#: pairs of :mod:`web_pairs`, 2026-10-26).
#: The canon-comparison families are deliberately NOT part of the dedupe (decision 2026-10-26): their targets are
#: deterministic derivations of public canonical texts (an apparatus of short variant pairs, a reference with a
#: 12-word anchor), not human-written targets the model must learn to produce, and a literature page quoting a
#: Talmud or Mishneh Torah passage would otherwise be dropped for quoting the canon (a 12-word anchor has five
#: word 8-grams: one shared 8-gram is 20 %). The eval-leak gates (excluded rows, re-scan) do cover canon rows.
TRAIN_DUP_EXTRA_FILES: Tuple[str, ...] = ("book_pairs.jsonl", "analyse_document_v2.jsonl", "pd_editions.jsonl",
                                          "web_pairs.jsonl")
#: A page duplicates a train target when it holds at least this share of the target's word 8-grams.
TRAIN_DUP_SHARE = 0.20
#: Reason prefix of a page dropped by the dedupe pass.
TRAIN_DUP_REASON = "train_target_dup"
#: Page-exclusion reasons that bind the L1 completion rows only: the dedupe limits regurgitation of
#: book text as a supervised target (plan §9.4.6); a RAG row carries the page in its prompt, not its
#: target, and its block rule already keeps its own target out of the context.
L1_ONLY_REASONS: Tuple[str, ...] = (TRAIN_DUP_REASON,)
#: Families whose TRAIN rows are scanned for eval-reference leaks of their target (D5 row exclusions);
#: ``domain_reading`` is literature text, excluded page by page instead.
ROW_SCAN_FAMILIES: Tuple[str, ...] = tuple(_rows.PGP_FAMILIES) + ("rag_context",) + tuple(_rows.CANON_FAMILIES)
#: Families whose rows are built from one KTIV manuscript (``meta.sys_num``): the re-scan and the row exclusions
#: check them against the KTIV hold-outs (:func:`canon_holdout_reason`).
CANON_FAMILIES: Tuple[str, ...] = tuple(_rows.CANON_FAMILIES)
#: Reason prefix of a TRAIN canon row whose manuscript is held out (KTIV benchmark sys_num, H by id, GT overlap).
CANON_HOLDOUT_REASON = "canon_holdout"
#: Share of a fragment's 25-letter shingles found in a KTIV benchmark ground truth that holds it out (as the builder).
CANON_GT_SHARE = 0.5
EXCLUDED_ROWS_FILE = "excluded_rows.json"
EXCLUDED_ROWS_MANIFEST_FILE = "excluded_rows_manifest.json"
#: Splits that hold a document out of training in every family (cross-family split check).
NON_TRAIN_SPLITS: Tuple[str, ...] = ("dev", "test_indist", "heldout_vision", "heldout")
#: Reason prefix of a TRAIN row whose document another family file holds out.
SPLIT_CONFLICT_REASON = "split_conflict"
#: ``meta.split_reason`` values of a hold-out that concerns the SOURCE text, not the document (2026-10-26,
#: :mod:`web_pairs`): ``item_mention`` (the item-level rule, kept for commentary), ``span_mention`` (the translation
#: span or its edition names ANOTHER, held document) and ``item_eval_only`` (the item's metadata marks it eval-only).
#: The document itself is not held out, so such rows do not put the document's rows in other families into a split
#: conflict.
SOURCE_LEVEL_SPLIT_REASONS: Tuple[str, ...] = ("item_mention", "span_mention", "item_eval_only")
#: ``meta.page_keys_source`` of rows built from a parsed book-edition entry (L5 / L2, fix pass F11).
NAS_PAGE_SOURCE = "nas_entry"
#: Reason of a TRAIN parsed-entry row whose entry could not be mapped to literature pages (F11: never ungated).
UNMAPPED_ENTRY_REASON = "unmapped_entry_pages"
#: Reason prefix of a TRAIN RAG row whose literature block names an eval-tier (or benchmark) document (F10).
LITERATURE_MENTION_REASON = "literature_text_mention"


def unmapped_entry_row(meta: Mapping[str, Any]) -> Optional[str]:
    """Why a row built from a parsed book-edition entry carries no literature page keys (F11), or None.

    A parsed-entry row is one with ``page_keys_source`` = :data:`NAS_PAGE_SOURCE`, or a ``book:`` source whose
    provenance names an ``entry_no`` and no located ``work_pages``; it must carry page keys so the page rule applies.

    :param meta: Row meta.
    :returns: ``meta.page_keys_unmapped`` (``no_work`` / ``no_start_page`` / ``no_pages``), ``no_page_keys`` for a
        parsed-entry row without keys, or None (mapped, or not a parsed-entry row).
    """
    if meta.get("page_keys_unmapped"):
        return str(meta["page_keys_unmapped"])
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    entry = meta.get("page_keys_source") == NAS_PAGE_SOURCE or (
        str(meta.get("source") or "").startswith("book:") and prov.get("entry_no") not in (None, "")
        and not prov.get("work_pages"))
    return "no_page_keys" if entry and not _as_list(meta.get("page_keys")) else None


def load_ktiv_benchmarks(p: _paths.Paths) -> Tuple[Dict[str, str], Set[str]]:
    """KTIV hold-outs of the canon families: benchmark / registry system numbers and the GT letter shingles.

    The religious-140 and sealed-literary files and the registries' ``sys_nums``, read exactly as the canon builder
    does (:func:`canon_diff.ktiv_benchmark_sysnums`, :func:`canon_diff.shingle_set`).

    :param p: Build paths.
    :returns: ``({sys_num: reason}, GT shingles)``.
    """
    from src.finetuning.text27b import canon_diff as _cd

    held, gts = _cd.ktiv_benchmark_sysnums(p)
    return held, _cd.shingle_set(gts)


def canon_holdout_reason(meta: Mapping[str, Any], user: str, holdouts: _ho.Holdouts,
                         ktiv: Optional[Tuple[Mapping[str, str], Set[str]]], gt_share: float = CANON_GT_SHARE) -> Optional[str]:
    """Why a canon row's manuscript is held out, or None: by KTIV system number (religious-140, sealed, registries),
    by H over every identifier the row carries (pgpid, the catalogue's pgpids, canonical id, shelfmark key; both
    tiers, as the builder holds them out), or by the fragment's GT shingle overlap.

    :param meta: Row meta (``sys_num`` / ``provenance.ktiv_sys_num``, ``pgpid``, ``provenance.catalogue_pgpids``,
        ``canonical_id``, ``shelfmark``).
    :param user: The row's user turn (its ``<document>`` is the fragment).
    :param holdouts: Hold-out set.
    :param ktiv: :func:`load_ktiv_benchmarks` output (None skips the sys_num and GT routes).
    :param gt_share: Shingle share that holds a fragment out.
    :returns: ``sys_num:<reason>``, ``id:<hold-out reason>``, ``gt_overlap`` or None.
    """
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    held, shingles = ktiv if ktiv is not None else ({}, set())
    sn = str(meta.get("sys_num") or prov.get("ktiv_sys_num") or "")
    if sn and sn in held:
        return f"sys_num:{held[sn]}"
    pgpids = [x for x in [meta.get("pgpid")] + list(prov.get("catalogue_pgpids") or []) + list(meta.get("pgpids") or [])
              if x not in (None, "", "None")]
    r = holdouts.match_any(pgpids=pgpids, canonicals=[c for c in [meta.get("canonical_id")] if c],
                           shelfmarks=[s for s in [meta.get("shelfmark")] if s], include_eval=True)
    if r:
        return f"id:{r}"
    if shingles:
        from src.finetuning.text27b import canon_diff as _cd

        doc = document_input(user)
        if doc and _cd.shingle_share(doc, shingles) >= gt_share:
            return "gt_overlap"
    return None


def doc_identities(meta: Mapping[str, Any]) -> Set[str]:
    """Document identities of a family row for the cross-family split check.

    :param meta: Row meta.
    :returns: ``pgpid:<id>`` for ``pgpid`` and every ``pgpids`` entry, plus ``book:<source>:<doc>`` for a
        new-book document (a ``doc_id`` written as ``book:…``, or ``provenance.book`` + ``provenance.entry_no``) and the
        ``web:…`` doc id of a web-pair document.
    """
    out: Set[str] = set()
    for pg in [meta.get("pgpid")] + list(meta.get("pgpids") or []):
        if pg not in (None, "", "None"):
            out.add(f"pgpid:{pg}")
    did = str(meta.get("doc_id") or "")
    if did.startswith(("book:", "web:")):
        out.add(did)
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    if prov.get("book") and prov.get("entry_no") not in (None, ""):
        out.add(f"book:{prov['book']}:{prov['entry_no']}")
    return out


def excluded_for_family(reasons: Sequence[str], family: str) -> bool:
    """Whether an excluded page's reasons bar it from rows of a family.

    :param reasons: The page's exclusion reasons.
    :param family: Row family (``domain_reading`` is barred by every reason; other families ignore
        :data:`L1_ONLY_REASONS`).
    :returns: True when the page may not be used.
    """
    if family == "domain_reading":
        return bool(reasons)
    return any(str(r).split(":")[0] not in L1_ONLY_REASONS for r in reasons)


def train_target_texts(families_dir: Union[str, Path], families: Sequence[str] = TRAIN_DUP_FAMILIES,
                       extra_files: Sequence[str] = TRAIN_DUP_EXTRA_FILES) -> Iterator[Tuple[str, str, str]]:
    """TRAIN targets of the translate / analyse families, as n-gram references for the L1 dedupe.

    Reads ``<family>.jsonl`` for each family plus every ``extra_files`` entry that exists; only rows
    with ``split == "train"`` and a family in ``families`` count. Targets go through
    :func:`reference_text` (analysis prose loses its citations, as the reference index does).

    :param families_dir: Directory of the family files.
    :param families: Families whose targets count.
    :param extra_files: Further family files read when present.
    :returns: Iterator of ``(ref_id, doc, text)`` with ``ref_id`` = ``<family>:<doc>``.
    """
    fdir = Path(families_dir)
    files = [fdir / f"{f}.jsonl" for f in families] + [fdir / f for f in extra_files]
    for f in files:
        if not f.exists():
            continue
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            fam = str(m.get("family") or m.get("task") or "")
            if m.get("split") != "train" or fam not in families:
                continue
            doc = str(m.get("pgpid") or m.get("doc_id") or m.get("row_id") or "")
            text = reference_text(_msg(r, "assistant"), fam)
            if text.strip():
                yield f"{fam}:{doc}", doc, text


def build_train_target_index(texts: Iterable[Tuple[str, str, str]], share: float = TRAIN_DUP_SHARE,
                             stop_df: int = STOP_DF) -> NgramIndex:
    """Word 8-gram index over train targets with the dedupe rule only (share ≥ ``share`` of one target).

    The shared-count rule is switched off (a page quoting five 8-grams of a long translation is not a
    copy of it); n-grams in ``stop_df`` or more targets' documents are formulae, as in the eval index.

    :param texts: ``(ref_id, doc, text)`` triples (:func:`train_target_texts`).
    :param share: Share of one target's (non-stop) 8-grams that makes a page a duplicate.
    :param stop_df: Stop-list document frequency.
    :returns: The index.
    """
    return NgramIndex.build(texts, kinds=("w8",), min_shared={"w8": sys.maxsize}, min_share=share, stop_df=stop_df)


def train_dup_pages(pages_list: Sequence[Any], index: NgramIndex) -> Dict[str, Set[str]]:
    """Pages that hold at least the dedupe share of some train target's 8-grams.

    :param pages_list: :class:`pages.PageRec` records.
    :param index: :func:`build_train_target_index` output.
    :returns: page key → ``{"train_target_dup:<family>:<doc>", ...}`` (at most ``MAX_NGRAM_REASONS``).
    """
    out: Dict[str, Set[str]] = {}
    for pg in pages_list:
        leaks = index.leak_refs(_page_text(pg))
        if leaks:
            out[pg.page_key] = {f"{TRAIN_DUP_REASON}:{rid}" for rid, _, _, _ in leaks[:MAX_NGRAM_REASONS]}
    return out


def build_excluded_rows(families_dir: Union[str, Path], index: NgramIndex,
                        families: Sequence[str] = ROW_SCAN_FAMILIES,
                        holdouts: Optional[_ho.Holdouts] = None,
                        split_conflicts: bool = True,
                        excluded_pages: Optional[Mapping[str, Sequence[str]]] = None,
                        ktiv: Optional[Tuple[Mapping[str, str], Set[str]]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """D5 row exclusions: TRAIN rows whose target leaks an eval reference (the n-gram rule of the index)
    and, with ``holdouts``, TRAIN rows whose text (input or target) names a benchmark-tier document (the
    re-scan's ``benchmark_text_mentions`` rule, :func:`text_mention_reason`: spaced spellings included, F1;
    e.g. an edition field that holds a scholar's list of shelfmarks, a description naming ``Moss. IX,249``).
    With ``excluded_pages``, a TRAIN row drawing on an excluded literature page (:func:`row_page_keys`; reasons in
    :data:`L1_ONLY_REASONS` do not bind non-L1 rows) is excluded (F2: analyse / book-pair / RAG rows).
    With ``holdouts``, a TRAIN RAG row whose literature block names a held document of either tier is excluded too
    (``literature_text_mention:<r>``, F10: the literature rule of the page exclusions applied to the rendered block).
    A TRAIN row built from a parsed book-edition entry that could not be mapped to literature pages is excluded
    (``unmapped_entry_pages:<why>``, F11: :func:`unmapped_entry_row`).
    With ``split_conflicts``, a TRAIN row not excluded otherwise is excluded when another family file
    holds the same document out (:func:`doc_identities` match against a row in :data:`NON_TRAIN_SPLITS`;
    e.g. L5 and L2 hashing one new-book document into different splits); source-level hold-outs
    (:data:`SOURCE_LEVEL_SPLIT_REASONS`) hold out their source, not the document, and are not counted.

    Every ``*.jsonl`` family file under ``families_dir`` is read (builder families, the noise slice,
    L5 / L2 files and ``rag_context`` when present); a row counts when its split is ``train`` and its
    family is in ``families``. The target is normalised with :func:`reference_text` (JSON → long
    strings, catalogue prose without citations), exactly as :func:`rescan` reads it.

    :param families_dir: Directory of the family files.
    :param index: Eval-reference index (:func:`build_reference_index`).
    :param families: Families scanned.
    :param holdouts: Hold-out set for the mention rule (None skips it).
    :param split_conflicts: Apply the cross-family split check.
    :param excluded_pages: Page exclusions (``excluded_pages.json``) for the page rule (None skips it).
    :param ktiv: KTIV hold-outs (:func:`load_ktiv_benchmarks`); with ``holdouts``, a TRAIN canon row whose manuscript
        is held out (:func:`canon_holdout_reason`) is excluded (``canon_holdout:<why>``; the builder already puts such
        rows in ``heldout``, so this is the second line).
    :returns: ``(row_id → reason, manifest)``; the reason is ``ngram_target:<ref>#<kind>`` (worst reference),
        ``benchmark_text_mention:<hold-out reason>``, ``literature_text_mention:<hold-out reason>``,
        ``excluded_page:<page key>``, ``unmapped_entry_pages:<why>`` or ``split_conflict:<split>:<identity>``.
    """
    t0 = time.time()
    out: Dict[str, str] = {}
    scanned: collections.Counter = collections.Counter()
    hit: collections.Counter = collections.Counter()
    by_kind: collections.Counter = collections.Counter()
    by_ref_tier: collections.Counter = collections.Counter()
    files: Dict[str, int] = {}
    cache: Dict[str, List[Tuple[str, str, int, float]]] = {}
    examples: List[Dict[str, Any]] = []
    non_train: Dict[str, str] = {}
    pending: List[Tuple[str, str, str, Set[str], Any]] = []
    for f in sorted(Path(families_dir).glob("*.jsonl")):
        n = 0
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            fam = str(m.get("family") or m.get("task") or "")
            if split_conflicts and m.get("split") in NON_TRAIN_SPLITS and \
                    m.get("split_reason") not in SOURCE_LEVEL_SPLIT_REASONS:
                for ident in doc_identities(m):
                    non_train.setdefault(ident, str(m.get("split")))
            if m.get("split") != "train" or fam not in families:
                continue
            n += 1
            cond = str(m.get("condition"))
            scanned[f"{fam}|{cond}"] += 1
            tgt = reference_text(_msg(r, "assistant"), fam)
            key = hashlib.sha1(tgt.encode("utf-8")).hexdigest()
            if key not in cache:
                cache[key] = index.leak_refs(tgt)
            lk = cache[key]
            rid = str(m.get("row_id") or "")
            if split_conflicts:
                pending.append((rid, fam, cond, doc_identities(m), m.get("pgpid")))
            if not lk and holdouts is not None:
                rb = text_mention_reason([_msg(r, "user"), _msg(r, "assistant")], holdouts, include_eval=False)
                if rb:
                    out[rid] = f"benchmark_text_mention:{rb}"
                    hit[f"{fam}|{cond}"] += 1
                    by_kind["benchmark_text_mention"] += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"row_id": rid, "family": fam, "condition": cond, "pgpid": m.get("pgpid"),
                                         "mention": rb})
                    continue
            if not lk and holdouts is not None and fam in CANON_FAMILIES:
                rc = canon_holdout_reason(m, _msg(r, "user"), holdouts, ktiv)
                if rc:
                    out[rid] = f"{CANON_HOLDOUT_REASON}:{rc}"
                    hit[f"{fam}|{cond}"] += 1
                    by_kind[CANON_HOLDOUT_REASON] += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"row_id": rid, "family": fam, "condition": cond, "sys_num": m.get("sys_num"),
                                         "canon_holdout": rc})
                    continue
            if not lk and holdouts is not None and fam in LITERATURE_FAMILIES:
                rl = text_mention_reason([_literature_part(r, fam)], holdouts)
                if rl:
                    out[rid] = f"{LITERATURE_MENTION_REASON}:{rl}"
                    hit[f"{fam}|{cond}"] += 1
                    by_kind[LITERATURE_MENTION_REASON] += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"row_id": rid, "family": fam, "condition": cond, "pgpid": m.get("pgpid"),
                                         "mention": rl})
                    continue
            if not lk and excluded_pages is not None:
                keys, _ = row_page_keys(m, None)
                bad = [k for k in keys if k in excluded_pages and excluded_for_family(excluded_pages[k], fam)]
                if bad:
                    out[rid] = f"excluded_page:{bad[0]}"
                    hit[f"{fam}|{cond}"] += 1
                    by_kind["excluded_page"] += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"row_id": rid, "family": fam, "condition": cond, "pgpid": m.get("pgpid"),
                                         "pages": bad[:3]})
                    continue
            why = unmapped_entry_row(m)
            if not lk and why:
                out[rid] = f"{UNMAPPED_ENTRY_REASON}:{why}"
                hit[f"{fam}|{cond}"] += 1
                by_kind[UNMAPPED_ENTRY_REASON] += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"row_id": rid, "family": fam, "condition": cond, "doc_id": m.get("doc_id"),
                                     "unmapped": why})
                continue
            if not lk:
                continue
            out[rid] = f"ngram_target:{lk[0][0]}#{lk[0][1]}"
            hit[f"{fam}|{cond}"] += 1
            by_kind[lk[0][1]] += 1
            by_ref_tier[lk[0][0].split(":")[0]] += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append({"row_id": rid, "family": fam, "condition": cond, "pgpid": m.get("pgpid"),
                                 "ref": lk[0][0], "kind": lk[0][1], "shared": lk[0][2], "share": round(lk[0][3], 4)})
        files[f.name] = n
    for rid, fam, cond, idents, pg in pending:
        clash = sorted(i for i in idents if i in non_train)
        if rid in out or not clash:
            continue
        out[rid] = f"{SPLIT_CONFLICT_REASON}:{non_train[clash[0]]}:{clash[0]}"
        hit[f"{fam}|{cond}"] += 1
        by_kind[SPLIT_CONFLICT_REASON] += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append({"row_id": rid, "family": fam, "condition": cond, "pgpid": pg, "conflict": out[rid]})
    docs = {(k.split("|")[0]) for k in hit}
    manifest = {"build": _rows.BUILD, "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "files_scanned": files, "train_rows_scanned": dict(sorted(scanned.items())),
                "excluded_rows": len(out), "excluded_by_family_condition": dict(sorted(hit.items())),
                "excluded_by_kind": dict(by_kind), "excluded_by_reference_type": dict(by_ref_tier),
                "families_with_exclusions": sorted(docs), "distinct_targets_scanned": len(cache),
                "ngram_index": index.stats(), "examples": examples, "seconds": round(time.time() - t0, 1)}
    return dict(sorted(out.items())), manifest


def load_excluded_rows(path: Optional[Union[str, Path]]) -> Dict[str, str]:
    """Read ``excluded_rows.json`` (``{row_id: reason}``; a list of row ids is also accepted).

    :param path: File (None or missing → empty mapping).
    :returns: row_id → reason.
    """
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {str(k): "listed" for k in data}
    return {str(k): str(v) for k, v in data.items()}


# ============================================================================== page exclusions
#: File names under ``paths.out_dir``.
EXCLUDED_PAGES_FILE = "excluded_pages.json"
EXCLUDED_MANIFEST_FILE = "excluded_pages_manifest.json"
RAG_INDEX_FILE = "rag_index.jsonl"
RESCAN_FILE = "rescan_report.json"
#: Local list of extra page exclusions (``logs/text_finetune_27b/``, git-ignored).
EXTRA_EXCLUDED_FILE = "extra_excluded_pages.json"
#: Most n-gram references named per excluded page.
MAX_NGRAM_REASONS = 3
_PAGE_NO_RE = re.compile(r"page_(\d+)")
_TRAILING_BLANKS = re.compile(r"[ \t]+(?=\n)")


class PageResolver:
    """Map ``(work, part, page file)`` triples of older indexes onto :func:`pages.page_key` keys.

    Join order: exact ``(work, part, page number)``; then ``(work, last part component, page number)``
    when that is unique (older indexes keep only the last component of nested parts).
    """

    def __init__(self, page_keys: Iterable[str]) -> None:
        """Index page keys.

        :param page_keys: Keys of the literature pages (``work/part/page_file``).
        """
        from src.finetuning.text27b import pages as _pages
        self._pages = _pages
        self.keys: Set[str] = set()
        self.by_full: Dict[Tuple[str, str, int], str] = {}
        self.by_last: Dict[Tuple[str, str, int], Set[str]] = collections.defaultdict(set)
        self.by_part: Dict[Tuple[str, str], Dict[int, str]] = collections.defaultdict(dict)
        for k in page_keys:
            work, part, pf = _pages.parse_page_key(k)
            n = self.page_no(pf)
            self.keys.add(k)
            self.by_full[(work, part, n)] = k
            self.by_last[(work, part.split("/")[-1], n)].add(k)
            self.by_part[(work, part)][n] = k

    @staticmethod
    def page_no(page_file: str) -> int:
        """Page number of a page file name (-1 when absent).

        :param page_file: ``page_038_structured.json`` or similar.
        :returns: The number.
        """
        m = _PAGE_NO_RE.search(str(page_file))
        return int(m.group(1)) if m else -1

    def resolve(self, work: str, part: str, page_file: str) -> Optional[str]:
        """Page key of a triple.

        :param work: Top-level work directory.
        :param part: Part (full or last component).
        :param page_file: Page file name.
        :returns: The key, or None when unknown or ambiguous.
        """
        n = self.page_no(page_file)
        k = self.by_full.get((work, part, n))
        if k:
            return k
        cand = self.by_last.get((work, str(part).split("/")[-1], n), set())
        return next(iter(cand)) if len(cand) == 1 else None

    def resolve_key(self, key: str) -> Optional[str]:
        """Normalise a page key written by another module (exact, else re-joined through :meth:`resolve`).

        :param key: Page key.
        :returns: The known key or None.
        """
        if key in self.keys:
            return key
        try:
            work, part, pf = self._pages.parse_page_key(key)
        except ValueError:
            return None
        return self.resolve(work, part, pf)

    def neighbours(self, key: str, window: int = 1) -> List[str]:
        """Keys of the pages within ``window`` of a page in the same part (the page itself excluded).

        :param key: Page key.
        :param window: Distance in page numbers.
        :returns: Neighbour keys that exist.
        """
        work, part, pf = self._pages.parse_page_key(key)
        n = self.page_no(pf)
        part_pages = self.by_part.get((work, part), {})
        return [part_pages[n + d] for d in range(-window, window + 1) if d and (n + d) in part_pages]


def _as_list(value: Any) -> List[str]:
    """Coerce a JSON value (list, string, None) to a list of non-empty strings.

    :param value: Raw value.
    :returns: Strings.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if str(value).strip() else []


def _row_ids(row: Mapping[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """PGP ids, canonical ids and shelfmark strings of an index row (lenient field names).

    :param row: Old page-index or ``rag_index.jsonl`` row.
    :returns: ``(pgpids, canonical ids, shelfmarks)``.
    """
    pg = _as_list(row.get("pgpids")) + _as_list(row.get("pgpid"))
    cids: List[str] = []
    for f in ("merged_doc_ids", "merged_ids", "cids", "canonical_ids", "canonical_id"):
        cids += _as_list(row.get(f))
    sms: List[str] = []
    for f in ("shelfmark_raw", "shelfmark", "shelfmarks"):
        sms += _as_list(row.get(f))
    return pg, cids, sms


def _index_mentions(path: Path, resolver: PageResolver, holdouts: _ho.Holdouts, source: str,
                    from_triple: bool) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Pages of a mention index that name a held document.

    A raw shelfmark of a row counts only when it is located in the page text (``located_in_text``)
    or is not one of the structuring prompt's example shelfmarks.

    :param path: JSONL index.
    :param resolver: Page-key resolver.
    :param holdouts: Hold-out set.
    :param source: Source label for the reasons (``index`` / ``rag``).
    :param from_triple: Rows carry ``book`` / ``part`` / ``page_file`` (old index) instead of ``page_key``.
    :returns: ``(page key → reasons, counts)``.
    """
    from src.finetuning.text27b import shelfmarks as _sm
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    cnt: collections.Counter = collections.Counter()
    for r in _rows.read_jsonl(path):
        cnt["rows"] += 1
        if from_triple:
            key = resolver.resolve(str(r.get("book") or r.get("work") or ""), str(r.get("part") or ""),
                                   str(r.get("page_file") or ""))
        else:
            key = resolver.resolve_key(str(r.get("page_key") or ""))
        if key is None:
            cnt["rows_unjoined"] += 1
            continue
        pg, cids, sms = _row_ids(r)
        located = _ho._truthy(r.get("located_in_text")) if "located_in_text" in r else True
        sms = [s for s in sms if located or _sm.key(s) not in _sm.LEAKED_KEYS]
        reason = holdouts.match_any(pg, cids, sms)
        if reason:
            out[key].add(f"mention:{source}:{reason}")
            cnt["rows_held"] += 1
    cnt["pages_held"] = len(out)
    return out, dict(cnt)


def _page_text(page: Any) -> str:
    """OCR text and structured text of a page, joined (what L1 / L3 may draw on).

    :param page: :class:`pages.PageRec`.
    :returns: Text.
    """
    return (page.ocr_text or "") + "\n" + (page.full_main_text or "")


def _page_mentions(pages_list: Sequence[Any], holdouts: _ho.Holdouts) -> Dict[str, Set[str]]:
    """Pages whose own located shelfmark keys, or shelfmarks found in their text, name a held document.

    :param pages_list: :class:`pages.PageRec` records.
    :param holdouts: Hold-out set.
    :returns: page key → reasons (``mention:page_keys:<r>`` / ``mention:text:<r>``).
    """
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    for pg in pages_list:
        text = _page_text(pg)
        # trailing blanks before a line break hide a shelfmark split over two lines (``T-S ⏎12.34``);
        # the L1 rows strip them, so the scan sees both forms
        text = text + "\n" + _TRAILING_BLANKS.sub("", text)
        located = pg.located_shelf_marks(use_ocr=False) + [k for k in pg.located_shelf_marks(use_ocr=True)]
        r1 = holdouts.match_any(shelfmarks=sorted(set(located)))
        if r1:
            out[pg.page_key].add(f"mention:page_keys:{r1}")
        r2 = text_mention_reason([text], holdouts)
        if r2:
            out[pg.page_key].add(f"mention:text:{r2}")
    return out


def page_pairs(pages_list: Sequence[Any]) -> Iterator[Tuple[str, str, str]]:
    """Consecutive pages of one part with the text an L1 two-page chunk of them would carry.

    The L1 builder joins two consecutive kept pages of a part (furniture-stripped OCR text,
    :func:`literature_l1.training_text`, running heads per part) with a blank line; a shelfmark or an
    eval passage can straddle that boundary, so the page gates alone miss it.

    :param pages_list: :class:`pages.PageRec` records.
    :returns: Iterator of ``(page key a, page key b, joined text)`` for page numbers ``n`` and ``n + 1``.
    """
    from src.finetuning.text27b import literature_l1 as _l1  # sibling module (furniture rules of the L1 rows)
    heads = _l1.running_heads_by_part(pages_list)
    by_part: Dict[Tuple[str, str], List[Any]] = collections.defaultdict(list)
    for pg in pages_list:
        by_part[(pg.work, pg.part)].append(pg)
    for wp in sorted(by_part):
        pgs = sorted(by_part[wp], key=lambda x: x.page_no)
        for a, b in zip(pgs, pgs[1:]):
            if b.page_no != a.page_no + 1:
                continue
            ta = _l1.training_text(a, heads.get(wp, ()))[0]
            tb = _l1.training_text(b, heads.get(wp, ()))[0]
            if ta and tb:
                yield a.page_key, b.page_key, ta + "\n\n" + tb


def pair_exclusions(pages_list: Sequence[Any], holdouts: _ho.Holdouts, index: Optional[NgramIndex],
                    skip: Set[str], resolver: PageResolver, window: int = 1) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Gates on two-page chunks (:func:`page_pairs`) whose pages are not excluded yet.

    A pair whose joined text names a held document (both tiers, as the page mention rule) excludes
    both pages (``mention:pair:<r>``) and their ±``window`` neighbours (``window:mention:pair:<r>``);
    a pair whose joined text leaks an eval reference excludes both pages (``ngram_pair:<ref>#<kind>``).

    :param pages_list: :class:`pages.PageRec` records.
    :param holdouts: Hold-out set.
    :param index: Eval-reference n-gram index (None skips the n-gram part).
    :param skip: Page keys already excluded (a pair touching one is never formed by L1).
    :param resolver: Page-key resolver (for the window).
    :param window: Neighbour window around a pair mention.
    :returns: ``(page key → reasons, counts)``.
    """
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    counts: collections.Counter = collections.Counter()
    for ka, kb, text in page_pairs(pages_list):
        if ka in skip or kb in skip:
            continue
        counts["pairs_checked"] += 1
        r = text_mention_reason([text], holdouts)
        if r:
            counts["pairs_mention"] += 1
            for k in (ka, kb):
                out[k].add(f"mention:pair:{r}")
                for nb in resolver.neighbours(k, window):
                    if nb not in (ka, kb):
                        out[nb].add(f"window:mention:pair:{r}")
        if index is not None:
            leaks = index.leak_refs(text)
            if leaks:
                counts["pairs_ngram"] += 1
                for k in (ka, kb):
                    out[k] |= {f"ngram_pair:{rid}#{kind}" for rid, kind, _, _ in leaks[:MAX_NGRAM_REASONS]}
    return out, dict(counts)


def load_extra_excluded(path: Optional[Path], resolver: PageResolver) -> Tuple[Dict[str, Set[str]], int]:
    """Read the local extra-exclusion list (a list of page keys / globs, or ``{key or glob: reason(s)}``).

    :param path: JSON file (missing → nothing, logged).
    :param resolver: Page-key resolver (globs are expanded over its keys).
    :returns: ``(page key → reasons, number of entries that matched no page)``.
    """
    import fnmatch
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    if path is None or not Path(path).exists():
        LOG.info("no extra exclusion list (%s)", Path(path).name if path else None)
        return out, 0
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.items() if isinstance(data, dict) else [(k, "list") for k in data]
    unmatched = 0
    for pat, reasons in items:
        rs = _as_list(reasons) or ["list"]
        keys = [k for k in resolver.keys if fnmatch.fnmatchcase(k, pat)] if any(ch in pat for ch in "*?[") \
            else [k for k in (resolver.resolve_key(pat),) if k]
        if not keys:
            unmatched += 1
        for k in keys:
            out[k] |= {f"extra:{r}" for r in rs}
    return out, unmatched


def build_excluded_pages(p: Optional[_paths.Paths] = None, holdouts: Optional[_ho.Holdouts] = None,
                         pages_list: Optional[Sequence[Any]] = None, index: Optional[NgramIndex] = None,
                         rag_index: Optional[Path] = None, extra: Optional[Path] = None, window: int = 1,
                         use_ngram: bool = True, use_old_index: bool = True,
                         use_page_scan: bool = True,
                         train_dup_index: Optional[NgramIndex] = None,
                         use_pair_scan: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Literature pages that must not enter training text, with reasons (plan §9.4.2 / §9.4.3).

    Reasons: ``treatment:<r>`` (page inside a treatment span of a held document),
    ``mention:<source>:<r>`` (the 10-18 page index, ``rag_index.jsonl``, the page's own located keys,
    shelfmarks found in its text) and ``window:<...>`` for the ±``window`` neighbours of a mention
    page, ``extra:<reason>`` (local list), ``ngram:<ref>#<kind>`` (n-gram gate on the page text).
    ``<r>`` is the hold-out reason (``pgp131``, ``test``, ``key:dev`` ...). With ``train_dup_index``
    the L1 dedupe pass of decision D5 adds ``train_target_dup:<family>:<doc>`` for pages holding
    ≥ 20 % of a train translate / analyse target's word 8-grams (no window).

    :param p: Build paths.
    :param holdouts: Hold-out set (default :meth:`Holdouts.load`).
    :param pages_list: Literature pages (default :func:`pages.iter_pages` over ``paths.academic_literature``).
    :param index: n-gram index (default :func:`build_reference_index` when ``use_ngram``).
    :param rag_index: Module E's mention index (default ``out_dir/rag_index.jsonl``; missing → logged and skipped).
    :param extra: Local extra list (default ``logs/text_finetune_27b/extra_excluded_pages.json``).
    :param window: Neighbour window around mention pages.
    :param use_ngram: Apply the n-gram gate.
    :param use_old_index: Use the 10-18 page index (``paths.analysis_pages_index``).
    :param use_page_scan: Use the pages' own located keys and the shelfmarks in their text.
    :param train_dup_index: Train-target index of the dedupe pass (:func:`build_train_target_index`); None skips it.
    :param use_pair_scan: Run the mention and n-gram gates on two-page chunks of pages not yet excluded
        (:func:`pair_exclusions`; an L1 chunk joins two consecutive pages).
    :returns: ``(page key → sorted reasons, manifest)``.
    """
    from src.finetuning.text27b import pages as _pages
    p = p or _paths.load_paths()
    holdouts = holdouts or _ho.Holdouts.load(p=p)
    t0 = time.time()
    pages_list = list(pages_list) if pages_list is not None else _pages.iter_pages(p.academic_literature)
    resolver = PageResolver(pg.page_key for pg in pages_list)
    reasons: Dict[str, Set[str]] = collections.defaultdict(set)
    inputs: Dict[str, Any] = {}
    counts: Dict[str, Any] = {"pages": len(pages_list)}

    # treatment spans of held documents
    tr_cnt: collections.Counter = collections.Counter()
    if p.document_treatments.exists():
        for t in _rows.read_jsonl(p.document_treatments):
            tr_cnt["spans"] += 1
            pg, cids, sms = _row_ids(t)
            r = holdouts.match_any(pg, cids, _as_list(t.get("shelfmarks")))
            if not r:
                continue
            tr_cnt["spans_held"] += 1
            for pf in _as_list(t.get("pages")):
                k = resolver.resolve(str(t.get("book") or ""), str(t.get("part") or ""), pf)
                if k is None:
                    tr_cnt["pages_unjoined"] += 1
                    continue
                reasons[k].add(f"treatment:{r}")
        inputs["document_treatments"] = True
    else:
        LOG.warning("document_treatments missing: %s", p.document_treatments.name)
        inputs["document_treatments"] = False
    counts["treatments"] = dict(tr_cnt)

    # mentions (with a window)
    mentions: Dict[str, Set[str]] = collections.defaultdict(set)
    if use_old_index and p.analysis_pages_index.exists():
        m, c = _index_mentions(p.analysis_pages_index, resolver, holdouts, "index", from_triple=True)
        for k, v in m.items():
            mentions[k] |= v
        counts["old_index"] = c
        inputs["analysis_pages_index"] = True
    else:
        inputs["analysis_pages_index"] = False
    rag_path = Path(rag_index) if rag_index is not None else p.out(RAG_INDEX_FILE)
    if rag_path.exists():
        m, c = _index_mentions(rag_path, resolver, holdouts, "rag", from_triple=False)
        for k, v in m.items():
            mentions[k] |= v
        counts["rag_index"] = c
        inputs["rag_index"] = True
    else:
        LOG.warning("rag_index.jsonl not found (%s); mention exclusions use the other sources only -- "
                    "re-run after rag_l3 build_index", rag_path.name)
        inputs["rag_index"] = False
    if use_page_scan:
        for k, v in _page_mentions(pages_list, holdouts).items():
            mentions[k] |= v
    for k, v in mentions.items():
        reasons[k] |= v
        for nb in resolver.neighbours(k, window):
            reasons[nb] |= {"window:" + r for r in v}
    counts["mention_pages"] = len(mentions)

    # local extra list
    extra_path = Path(extra) if extra is not None else p.out_dir.parent / EXTRA_EXCLUDED_FILE
    ex, unmatched = load_extra_excluded(extra_path, resolver)
    for k, v in ex.items():
        reasons[k] |= v
    inputs["extra_excluded_pages"] = extra_path.exists()
    counts["extra"] = {"pages": len(ex), "unmatched_entries": unmatched}

    # n-gram gate
    ng_pages = 0
    if use_ngram:
        index = index if index is not None else build_reference_index(p, holdouts)
        for pg in pages_list:
            leaks = index.leak_refs(_page_text(pg))
            if leaks:
                ng_pages += 1
                reasons[pg.page_key] |= {f"ngram:{rid}#{kind}" for rid, kind, _, _ in leaks[:MAX_NGRAM_REASONS]}
        counts["ngram_index"] = index.stats()
    counts["ngram_pages"] = ng_pages

    # L1 dedupe against train translate / analyse targets (D5, plan §9.4.6)
    if train_dup_index is not None:
        dup = train_dup_pages(pages_list, train_dup_index)
        for k, v in dup.items():
            reasons[k] |= v
        counts["train_dup_pages"] = len(dup)
        counts["train_dup_index"] = train_dup_index.stats()
    inputs["train_dup"] = train_dup_index is not None

    # two-page chunks (after every page-level reason, so only pairs L1 could still form are checked)
    if use_pair_scan:
        pr, pc = pair_exclusions(pages_list, holdouts, index if use_ngram else None, set(reasons), resolver, window)
        for k, v in pr.items():
            reasons[k] |= v
        counts["pair_scan"] = {**pc, "pages_added": len(pr)}

    out = {k: sorted(v) for k, v in sorted(reasons.items())}
    counts.update(_exclusion_counts(out, pages_list))
    manifest = {"build": _rows.BUILD, "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "inputs": inputs, "counts": counts, "seconds": round(time.time() - t0, 1),
                "build_args": {"window": window, "use_ngram": use_ngram, "use_old_index": use_old_index,
                               "use_page_scan": use_page_scan, "rag_index": rag_path.name, "extra": extra_path.name,
                               "train_dup": train_dup_index is not None, "pair_scan": use_pair_scan,
                               "train_dup_share": train_dup_index.min_share if train_dup_index is not None else None}}
    return out, manifest


def _reason_class(reason: str) -> str:
    """Class of a page reason for the counts (``treatment``, ``mention:index``, ``window``, ``ngram`` ...).

    :param reason: Reason string.
    :returns: Class label.
    """
    head = reason.split(":")
    if head[0] == "mention":
        return ":".join(head[:2])
    return head[0]


def _exclusion_counts(excluded: Mapping[str, List[str]], pages_list: Sequence[Any]) -> Dict[str, Any]:
    """Pages and characters excluded, per reason class, per hold-out tier and per work.

    :param excluded: page key → reasons.
    :param pages_list: Pages (for the character counts).
    :returns: Count blocks.
    """
    chars = {pg.page_key: len(pg.ocr_text or pg.full_main_text or "") for pg in pages_list}
    by_class: collections.Counter = collections.Counter()
    only_class: collections.Counter = collections.Counter()
    by_tier: collections.Counter = collections.Counter()
    by_work: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    bench = set(_ho.BENCHMARK_REASONS)
    for k, rs in excluded.items():
        cls = {_reason_class(r) for r in rs}
        for c in cls:
            by_class[c] += 1
        if len(cls) == 1:
            only_class[next(iter(cls))] += 1
        tiers = {"benchmark" if any(b in r.split(":") for b in bench) else "eval" for r in rs if not r.startswith("ngram")}
        for t in tiers:
            by_tier[t] += 1
        w = k.split("/")[0]
        by_work[w]["pages"] += 1
        by_work[w]["chars"] += chars.get(k, 0)
    return {"excluded_pages": len(excluded), "excluded_chars": sum(chars.get(k, 0) for k in excluded),
            "pages_by_reason_class": dict(sorted(by_class.items())),
            "pages_by_only_reason_class": dict(sorted(only_class.items())),
            "pages_by_tier": dict(by_tier),
            "by_work": {w: dict(c) for w, c in sorted(by_work.items(), key=lambda kv: -kv[1]["chars"])}}


def load_excluded_pages(path: Optional[Union[str, Path]] = None, p: Optional[_paths.Paths] = None) -> Dict[str, List[str]]:
    """Read ``excluded_pages.json``.

    :param path: File (default ``out_dir/excluded_pages.json``).
    :param p: Build paths for the default.
    :returns: page key → reasons.
    """
    path = Path(path) if path is not None else (p or _paths.load_paths()).out(EXCLUDED_PAGES_FILE)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: Any) -> Path:
    """Write JSON atomically (UTF-8, indented).

    :param path: Destination.
    :param obj: JSON-serialisable object.
    :returns: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# ============================================================================== mixture re-scan
#: Families whose row text is literature (an n-gram leak there is a failure, not a report).
LITERATURE_FAMILIES: Tuple[str, ...] = tuple(_rows.LITERATURE_FAMILIES)
_SOURCE_MARK = "[SOURCE "
#: Example rows kept per report section.
MAX_EXAMPLES = 25


def row_page_keys(meta: Mapping[str, Any], resolver: Optional[PageResolver]) -> Tuple[List[str], bool]:
    """Literature page keys a row draws on (``page_keys`` / ``page_key`` / ``provenance.page_keys`` /
    ``lit_sources`` / ``work`` + ``page_range``; for a row without ``meta.page_keys``, every page of
    ``provenance.work_pages``, the L5 located spans: intro / edition / translation pages).

    :param meta: Row meta.
    :param resolver: Page-key resolver (None: keys are taken as written).
    :returns: ``(keys, checked)`` where ``checked`` is False when the row names literature pages in a
        form that could not be turned into keys.
    """
    keys: List[str] = []
    unresolved = False
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    own = _as_list(meta.get("page_keys"))
    cand: List[str] = (own + _as_list(meta.get("page_key"))
                       + [k for k in _as_list(prov.get("page_keys")) if k not in own])
    work_pages = prov.get("work_pages") if isinstance(prov.get("work_pages"), Mapping) else {}
    if "page_keys" not in meta:
        for v in work_pages.values():
            cand += [k for k in _as_list(v) if k not in cand]
    for src in meta.get("lit_sources") or []:
        if not isinstance(src, Mapping):
            continue
        k = src.get("page_key") or (src.get("page") if "/" in str(src.get("page") or "") else None)
        if k:
            cand.append(str(k))
        else:
            unresolved = True
    for k in cand:
        rk = resolver.resolve_key(k) if resolver is not None else k
        if rk:
            keys.append(rk)
        else:
            unresolved = True
    if not cand and meta.get("work") and meta.get("page_range") and meta.get("family") == "domain_reading":
        unresolved = True
    return keys, not unresolved


def _literature_part(row: Mapping[str, Any], family: str) -> str:
    """The literature text of a row (whole row for domain reading; the ``[SOURCE n]`` block for RAG rows).

    :param row: Chat row.
    :param family: Row family.
    :returns: Text ('' when the row has none).
    """
    user = _msg(row, "user")
    if family == "domain_reading":
        return user + "\n" + _msg(row, "assistant")
    i = user.find(_SOURCE_MARK)
    return user[i:] if i >= 0 else ""


def rescan(mixture: Union[str, Path], p: Optional[_paths.Paths] = None, holdouts: Optional[_ho.Holdouts] = None,
           index: Optional[NgramIndex] = None, excluded: Optional[Mapping[str, List[str]]] = None,
           resolver: Optional[PageResolver] = None, use_ngram: bool = True,
           ktiv: Optional[Tuple[Mapping[str, str], Set[str]]] = None) -> Dict[str, Any]:
    """Re-scan a materialised mixture: zero H hits, zero project-name strings, n-gram report (plan §9.4.7).

    Checks (each must be 0): ``benchmark_meta_hits`` (row pgpid / shelfmark in the benchmark tier),
    ``eval_split_meta_hits`` (row document in v0_1 dev / test), ``excluded_page_hits`` (any row -- literature,
    RAG, analyse, book pair -- drawing on an excluded page, :func:`row_page_keys`; L1-only reasons bind
    ``domain_reading`` alone), ``benchmark_text_mentions`` (a benchmark shelfmark named in a row's user or
    assistant text, spaced spellings included: :func:`text_mention_reason`), ``literature_text_mentions`` (the
    literature text of an L1 / RAG row -- whole row / ``[SOURCE n]`` block -- naming a held document of EITHER tier,
    F10), ``unmapped_entry_rows`` (a parsed-entry row without literature page keys, F11: :func:`unmapped_entry_row`),
    ``scrub_rows`` (any project-name string anywhere in the row, meta included: :func:`row_source_scan`), ``non_nfc_rows`` (a
    message content not in NFC), ``literature_ngram_leaks`` (literature text leaking an eval reference),
    ``canon_holdout_hits`` (a canon-family row -- or any row naming a KTIV ``sys_num`` -- whose manuscript is held out:
    KTIV system number of religious-140 / sealed / the registries, H by pgpid / catalogue pgpids / canonical id /
    shelfmark key in either tier, or ≥ 50 % GT shingle overlap; :func:`canon_holdout_reason`). Reported,
    not checked: n-gram leaks of PGP-family targets / inputs, eval-tier text mentions (whole row text), extra-benchmark
    meta hits, literature rows whose pages could not be checked, rows with page keys per family. ``distinct_rows`` gives each check's count
    over distinct row ids (a ×2 row counts once there).

    :param mixture: Mixture JSONL.
    :param p: Build paths.
    :param holdouts: Hold-out set (default :meth:`Holdouts.load`).
    :param index: n-gram index (default :func:`build_reference_index` when ``use_ngram``).
    :param excluded: Page exclusions (default ``out_dir/excluded_pages.json`` when present).
    :param resolver: Page-key resolver for literature meta (default: keys as written).
    :param use_ngram: Run the n-gram scan.
    :param ktiv: KTIV hold-outs (default: :func:`load_ktiv_benchmarks`, loaded at the first canon row).
    :returns: The report (also what the CLI writes to ``rescan_report.json``).
    """
    p = p or _paths.load_paths()
    holdouts = holdouts or _ho.Holdouts.load(p=p)
    if excluded is None:
        ep = p.out(EXCLUDED_PAGES_FILE)
        excluded = load_excluded_pages(ep) if ep.exists() else None
        if excluded is None:
            LOG.warning("excluded_pages.json not found; literature page check skipped")
    if use_ngram and index is None:
        index = build_reference_index(p, holdouts)
    t0 = time.time()
    checks: collections.Counter = collections.Counter()
    report: collections.Counter = collections.Counter()
    fam_rows: collections.Counter = collections.Counter()
    ngram_by_fam: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    scrub_by_fam: collections.Counter = collections.Counter()
    examples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

    def ex(section: str, item: Dict[str, Any]) -> None:
        """Keep an example for a report section.

        :param section: Section name.
        :param item: Example (row ids and numbers only, no text).
        :returns: None.
        """
        if len(examples[section]) < MAX_EXAMPLES:
            examples[section].append(item)

    n = 0
    distinct: Dict[str, Set[str]] = collections.defaultdict(set)
    paged_rows: collections.Counter = collections.Counter()
    for row in _rows.read_jsonl(mixture):
        n += 1
        meta = row.get("meta") or {}
        fam = str(meta.get("family") or meta.get("task") or "")
        rid = meta.get("row_id")
        fam_rows[fam] += 1
        pg, sm = meta.get("pgpid"), meta.get("shelfmark")
        rb = holdouts.is_benchmark(pg, sm) if (pg or sm) else None
        if rb:
            checks["benchmark_meta_hits"] += 1
            distinct["benchmark_meta_hits"].add(str(rid))
            ex("benchmark_meta_hits", {"row_id": rid, "family": fam, "pgpid": pg, "reason": rb})
        re_ = holdouts.is_eval_doc(pg, sm) if (pg or sm) else None
        if re_:
            base = re_.split(":")[-1]
            if base in _ho.EVAL_REASONS:
                checks["eval_split_meta_hits"] += 1
                distinct["eval_split_meta_hits"].add(str(rid))
                ex("eval_split_meta_hits", {"row_id": rid, "family": fam, "pgpid": pg, "reason": re_})
            else:
                report["extra_benchmark_meta_hits"] += 1
        texts = [m.get("content") or "" for m in row.get("messages") or []]
        if row_source_scan(row):
            checks["scrub_rows"] += 1
            distinct["scrub_rows"].add(str(rid))
            scrub_by_fam[fam] += 1
            ex("scrub_rows", {"row_id": rid, "family": fam,
                            "where": "messages" if any(source_scan(t) for t in texts) else "meta"})
        bad_nfc = _rows.non_nfc_roles(row)
        if bad_nfc:
            checks["non_nfc_rows"] += 1
            distinct["non_nfc_rows"].add(str(rid))
            ex("non_nfc_rows", {"row_id": rid, "family": fam, "roles": bad_nfc})
        rb2 = text_mention_reason(texts[1:], holdouts, include_eval=False)
        if rb2:
            checks["benchmark_text_mentions"] += 1
            distinct["benchmark_text_mentions"].add(str(rid))
            ex("benchmark_text_mentions", {"row_id": rid, "family": fam, "reason": rb2,
                                           "mentions": [(m_, k_) for m_, k_, _ in mention_index(holdouts).find(
                                               "\n".join(texts[1:]))][:3]})
        elif text_mention_reason(texts[1:], holdouts):
            report["eval_text_mentions"] += 1
            distinct["eval_text_mentions"].add(str(rid))
            ex("eval_text_mentions", {"row_id": rid, "family": fam,
                                      "mentions": [(m_, k_) for m_, k_, _ in mention_index(holdouts).find(
                                          "\n".join(texts[1:]))][:3]})
        if fam in LITERATURE_FAMILIES:
            lt_text = _literature_part(row, fam)
            rl = text_mention_reason([lt_text], holdouts) if lt_text else None
            if rl:
                checks["literature_text_mentions"] += 1
                distinct["literature_text_mentions"].add(str(rid))
                ex("literature_text_mentions", {"row_id": rid, "family": fam, "reason": rl,
                                                "mentions": [(m_, k_) for m_, k_, _ in
                                                             mention_index(holdouts).find(lt_text)][:3]})
        if fam in CANON_FAMILIES or meta.get("sys_num"):
            if ktiv is None:
                ktiv = load_ktiv_benchmarks(p)
            report["canon_rows_checked"] += 1
            rc = canon_holdout_reason(meta, texts[1] if len(texts) > 1 else "", holdouts, ktiv)
            if rc:
                checks["canon_holdout_hits"] += 1
                distinct["canon_holdout_hits"].add(str(rid))
                ex("canon_holdout_hits", {"row_id": rid, "family": fam, "sys_num": meta.get("sys_num"), "reason": rc})
        why = unmapped_entry_row(meta)
        if why:
            checks["unmapped_entry_rows"] += 1
            distinct["unmapped_entry_rows"].add(str(rid))
            ex("unmapped_entry_rows", {"row_id": rid, "family": fam, "doc_id": meta.get("doc_id"), "why": why})
        keys, ok = row_page_keys(meta, resolver)
        if keys:
            paged_rows[fam] += 1
        if fam in LITERATURE_FAMILIES and (not ok or not keys):
            report["literature_rows_pages_unchecked"] += 1
        if excluded is not None and keys:
            bad = [k for k in keys if k in excluded and excluded_for_family(excluded[k], fam)]
            if bad:
                checks["excluded_page_hits"] += 1
                distinct["excluded_page_hits"].add(str(rid))
                ex("excluded_page_hits", {"row_id": rid, "family": fam, "pages": bad[:5]})
        if use_ngram and index is not None:
            if fam in LITERATURE_FAMILIES:
                lt = _literature_part(row, fam)
                lk = index.leak_refs(lt) if lt else []
                if lk:
                    checks["literature_ngram_leaks"] += 1
                    distinct["literature_ngram_leaks"].add(str(rid))
                    ex("literature_ngram_leaks", {"row_id": rid, "family": fam, "ref": lk[0][0], "kind": lk[0][1],
                                                  "shared": lk[0][2], "share": round(lk[0][3], 4)})
            for part, text in (("target", reference_text(_msg(row, "assistant"), fam)), ("input", _msg(row, "user"))):
                if fam == "domain_reading":
                    continue
                if fam == "rag_context" and part == "input":
                    i = text.find(_SOURCE_MARK)
                    text = text[:i] if i >= 0 else text
                lk = index.leak_refs(text)
                if lk:
                    ngram_by_fam[fam][part] += 1
                    ex(f"ngram_{part}", {"row_id": rid, "family": fam, "pgpid": pg, "ref": lk[0][0], "kind": lk[0][1],
                                         "shared": lk[0][2], "share": round(lk[0][3], 4)})
    names = ("benchmark_meta_hits", "eval_split_meta_hits", "excluded_page_hits", "benchmark_text_mentions",
             "literature_text_mentions", "unmapped_entry_rows", "scrub_rows", "non_nfc_rows", "literature_ngram_leaks",
             "canon_holdout_hits")
    check_block = {k: {"value": int(checks.get(k, 0)), "ok": checks.get(k, 0) == 0,
                       "skipped": (k == "excluded_page_hits" and excluded is None) or
                                  (k == "literature_ngram_leaks" and not use_ngram)} for k in names}
    return {"build": _rows.BUILD, "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "mixture": Path(mixture).name, "rows": n, "rows_by_family": dict(sorted(fam_rows.items())),
            "ok": all(v["ok"] for v in check_block.values()), "checks": check_block,
            "distinct_rows": {k: len(v) for k, v in sorted(distinct.items())},
            "rows_with_page_keys_by_family": dict(sorted(paged_rows.items())),
            "reported": dict(report), "scrub_rows_by_family": dict(scrub_by_fam),
            "ngram_leak_rows_by_family": {f: dict(c) for f, c in sorted(ngram_by_fam.items())},
            "ngram_index": index.stats() if (use_ngram and index is not None) else None,
            "examples": dict(examples), "seconds": round(time.time() - t0, 1)}


# ============================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``ngram-index`` / ``excluded-pages`` / ``rescan``.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit status (``rescan`` returns 1 when a check fails).
    """
    ap = argparse.ArgumentParser(prog="python -m src.finetuning.text27b.contamination", description=__doc__.split("\n")[0])
    ap.add_argument("--paths", default=None, help="paths.json (default $TEXT27B_PATHS or logs/text_finetune_27b/paths.json)")
    ap.add_argument("--holdouts", default=None, help="holdouts.json (default <out_dir>/holdouts.json)")
    ap.add_argument("--no-background", action="store_true", help="stop-list document frequency over held documents only")
    ap.add_argument("--no-cache", action="store_true", help="rebuild the n-gram index")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ngram-index", help="build the eval-reference n-gram index (cached) and its manifest")
    e = sub.add_parser("excluded-pages", help="write excluded_pages.json + excluded_pages_manifest.json")
    e.add_argument("--rag-index", default=None, help="module E mention index (default <out_dir>/rag_index.jsonl)")
    e.add_argument("--extra", default=None, help="local extra list (default logs/text_finetune_27b/extra_excluded_pages.json)")
    e.add_argument("--no-ngram", action="store_true")
    e.add_argument("--window", type=int, default=1)
    e.add_argument("--works", nargs="*", default=None, help="restrict to these work directories (development)")
    e.add_argument("--out", default=None, help="output directory (default <out_dir>)")
    e.add_argument("--no-train-dedupe", action="store_true", help="skip the D5 L1 dedupe against train targets")
    e.add_argument("--train-dup-share", type=float, default=TRAIN_DUP_SHARE)
    e.add_argument("--families-dir", default=None, help="family files for the dedupe (default <out_dir>/families)")
    e.add_argument("--no-pair-scan", action="store_true", help="skip the two-page chunk gates")
    xr = sub.add_parser("excluded-rows", help="write excluded_rows.json (D5: train rows whose target leaks an eval reference)")
    xr.add_argument("--families-dir", default=None, help="default <out_dir>/families")
    xr.add_argument("--out", default=None, help="output directory (default <out_dir>)")
    xr.add_argument("--excluded-pages", default="auto",
                    help="page exclusions for the page rule: auto (<out_dir>/excluded_pages.json when present), none, or FILE")
    r = sub.add_parser("rescan", help="re-scan a materialised mixture -> rescan_report.json")
    r.add_argument("mixture")
    r.add_argument("--out", default=None, help="report path (default <out_dir>/rescan_report.json)")
    r.add_argument("--no-ngram", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = _paths.load_paths(a.paths)
    h = _ho.Holdouts.load(a.holdouts, p)
    idx_kwargs = {"cache": not a.no_cache, "background": not a.no_background}
    if a.cmd == "ngram-index":
        idx = build_reference_index(p, h, **idx_kwargs)
        dest = _write_json(p.out("ngram_index_manifest.json"),
                           {"build": _rows.BUILD, "stats": idx.stats(), "build_args": idx_kwargs})
        print(json.dumps(idx.stats(), indent=1))
        print(f"wrote {dest}")
        return 0
    if a.cmd == "excluded-pages":
        from src.finetuning.text27b import pages as _pages
        pl = _pages.iter_pages(p.academic_literature, works=a.works)
        idx = None if a.no_ngram else build_reference_index(p, h, **idx_kwargs)
        dup_idx = None
        if not a.no_train_dedupe:
            fdir = Path(a.families_dir) if a.families_dir else p.out("families")
            dup_idx = build_train_target_index(train_target_texts(fdir), share=a.train_dup_share)
        ex_pages, manifest = build_excluded_pages(p, h, pl, idx, a.rag_index, a.extra, a.window, not a.no_ngram,
                                                  train_dup_index=dup_idx, use_pair_scan=not a.no_pair_scan)
        manifest["build_args"].update({"works": a.works, **idx_kwargs})
        out_dir = Path(a.out) if a.out else p.out_dir
        _write_json(out_dir / EXCLUDED_PAGES_FILE, ex_pages)
        _write_json(out_dir / EXCLUDED_MANIFEST_FILE, manifest)
        c = manifest["counts"]
        print(json.dumps({k: c[k] for k in ("pages", "excluded_pages", "excluded_chars", "pages_by_reason_class",
                                             "pages_by_only_reason_class", "pages_by_tier") if k in c}, indent=1))
        print(f"wrote {out_dir / EXCLUDED_PAGES_FILE}")
        return 0
    if a.cmd == "excluded-rows":
        idx = build_reference_index(p, h, **idx_kwargs)
        fdir = Path(a.families_dir) if a.families_dir else p.out("families")
        ep = None if a.excluded_pages == "none" else (p.out(EXCLUDED_PAGES_FILE) if a.excluded_pages == "auto"
                                                      else Path(a.excluded_pages))
        ex_pages = load_excluded_pages(ep) if ep is not None and ep.exists() else None
        ex_rows, manifest = build_excluded_rows(fdir, idx, holdouts=h, excluded_pages=ex_pages,
                                                ktiv=load_ktiv_benchmarks(p))
        manifest["build_args"] = {"families_dir": fdir.name, "excluded_pages": ep.name if ex_pages is not None else None,
                                  **idx_kwargs}
        out_dir = Path(a.out) if a.out else p.out_dir
        _write_json(out_dir / EXCLUDED_ROWS_FILE, ex_rows)
        _write_json(out_dir / EXCLUDED_ROWS_MANIFEST_FILE, manifest)
        print(json.dumps({k: manifest[k] for k in ("train_rows_scanned", "excluded_rows", "excluded_by_family_condition",
                                                    "excluded_by_kind", "excluded_by_reference_type")}, indent=1))
        print(f"wrote {out_dir / EXCLUDED_ROWS_FILE}")
        return 0
    idx = None if a.no_ngram else build_reference_index(p, h, **idx_kwargs)
    rep = rescan(a.mixture, p, h, idx, use_ngram=not a.no_ngram)
    rep["build_args"] = {"mixture": Path(a.mixture).name, "use_ngram": not a.no_ngram, **idx_kwargs}
    dest = _write_json(Path(a.out) if a.out else p.out(RESCAN_FILE), rep)
    print(json.dumps({"ok": rep["ok"], "checks": rep["checks"], "reported": rep["reported"]}, indent=1))
    print(f"wrote {dest}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
