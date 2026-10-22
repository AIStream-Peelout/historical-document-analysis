# File name: builder_v0.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""The 2026-10-18 builder of the multi-task text fine-tune dataset (``text_ft_v0`` / ``v0_1``), moved into the package.

A faithful move of the research prototype ``build_text_ft_v0.py``: functions, constants, gates,
seeds and the row layout are unchanged, so :func:`run_v0` on the v1.1 pairs and the PGP export
named in ``paths.json`` reproduces ``text_ft_v0_1`` (prompts, targets, inputs, splits). Only the
plumbing changed:

* inputs and outputs come from :class:`V0Config` (built from :func:`paths.load_paths`) instead of
  module globals. Holdings-specific settings (licence strings per book, books excluded from the
  analysis family, extra scholar surnames, reader file names) come from a git-ignored local JSON
  next to ``paths.json`` (``builder_local.json``; see :func:`load_local_config`);
* the ``build_pairs`` / ``probe_common`` helpers the prototype used are ported here
  (:func:`load_csv`, :func:`split_multi`, :func:`split_of`, :func:`build_merged_map`,
  :func:`load_reads`, :func:`classify_label`, :func:`clean_text`); CER uses
  :func:`textnorm.normalize_for_cer` with ``v11=False`` (the v0 builder's own recomputation);
* shared text helpers are imported from :mod:`~src.finetuning.text27b.textnorm` (``clean_read``,
  ``strip_latin_lines``, ``edition_as_plain``, ``strip_marks``, ``grounding`` ...). Known deviation:
  textnorm fixes the prototype's ``skeleton_rtl`` (a word-final ט / צ / ע no longer takes its geresh
  value), which can move a few analysis grounding shares; no PGP-field family depends on it;
* the prompts (``SYSTEM``, ``INSTRUCTION``, ``SCRIPT_NAME``) are those of :mod:`rows`, and
  ``COND_NOTE`` here is ``rows.COND_NOTE_V0`` (the v0 notes, verbatim).

``analyse_document`` needs the span detector of the research module ``build_analysis`` (literature
pages, book-specific heading rules, git-ignored); :func:`load_full_spans` imports it from
``paths.probe_dir`` at call time and only then. The v0_2 builder (:mod:`builder`) does not need it.

Public API: :class:`V0Config`, :func:`load_local_config`, :func:`run_v0`, :func:`write_v0`,
:func:`main`, and every prototype function and class (``Builder``, ``Noiser``, ``PeopleIndex``,
``TokenCounter``, ``clean_description``, ``catalogue_record``, ``relations_from_catalogue``,
``read_variants``, ``read_only_pool``, ``build_analysis_rows``, ``sanity_checks`` ...).

Run (writes ``<out_dir>/v0_1_repro`` unless ``--out-dir`` is given)::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.builder_v0 [--limit N] [--no-analysis]
"""
from __future__ import annotations

import argparse
import codecs
import collections
import csv
import dataclasses
import hashlib
import html
import importlib
import inspect
import json
import logging
import math  # noqa: F401  (kept from the prototype's imports)
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import Levenshtein

from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import textnorm as tn
from src.finetuning.text27b.textnorm import (clean_read, edition_as_plain, entity_candidates, grounding,  # noqa: F401
                                             overlap60, rtl_skeletons, skeleton_latin, skeleton_rtl,
                                             strip_latin_lines, strip_marks)

csv.field_size_limit(sys.maxsize)
LOG = logging.getLogger(__name__)
T0 = time.time()
SEED = 20261018
MAX_TOKENS = 6000
CHAT_OVERHEAD = 12  # role markers / special tokens per message (approximate)
NOISE_CER_RANGE = (0.08, 0.32)
READ_GATE = {"cer_max": 0.45, "coverage_min": 0.6, "coverage_max": 1.5}
NOISE_PAIR_GATE = {"cer_max": 0.5, "coverage_min": 0.8, "coverage_max": 1.25}
READ_ONLY_GATE = {"agreed_share_min": 0.25, "chars_min": 150}
GROUNDING_PASS = 0.6
#: Readers whose outputs the v0 builder uses (file names under ``paths.ai_reads``; a local config may override).
READ_FILES: Dict[str, str] = {"a50": "ai_reads_qwen3-vl-8b-heb-v22i-a50.jsonl",
                              "v21b": "ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl"}
STATS: collections.Counter = collections.Counter()
PGP_LICENCE = "pgp_export_cc_by_4.0 (underlying edition/translation by the cited scholar)"
#: Filled from the local config by :func:`apply_local_config` (book work-dir -> licence string).
LICENCE_BY_BOOK: Dict[str, str] = {}
#: Filled from the local config (books whose OCR makes analysis targets unusable).
ANALYSIS_EXCLUDED_BOOKS: Set[str] = set()
#: Place names that are never letter parties; :func:`load_place_names` adds PGP ``places.csv``.
PLACE_EXTRAS: Set[str] = {"Egypt", "Palestine", "Syria", "India", "Yemen", "Sicily", "Ifrīqiya", "Maghrib", "the Maghrib"}
PLACE_NAMES: Set[str] = set(PLACE_EXTRAS)

SYSTEM = _rows.SYSTEM
COND_NOTE = dict(_rows.COND_NOTE_V0)
SCRIPT_NAME = dict(_rows.SCRIPT_NAME)
INSTRUCTION = dict(_rows.INSTRUCTION)
FAMILIES = tuple(INSTRUCTION)


# ============================================================================ configuration
@dataclasses.dataclass
class V0Config:
    """Inputs and outputs of one v0 build (what the prototype kept in module globals)."""

    pairs: Path
    pgp_dir: Path
    merged_map: Path
    merged_jsonl: Path
    benchmark_ids: Path
    samples: Path
    treatments: Path
    probe_dir: Path
    read_files: Dict[str, Path]
    tokenizer: Path
    out_dir: Path
    noise_model_path: Path
    v0_1_dir: Optional[Path] = None
    scholar_extra: List[str] = dataclasses.field(default_factory=list)
    registry_fn: Callable[[], Tuple[Set[str], Set[str]]] = registered_benchmark_documents

    @classmethod
    def from_paths(cls, p: Optional[_paths.Paths] = None, out_dir: Optional[Path] = None,
                   local: Optional[Dict[str, Any]] = None) -> "V0Config":
        """Build a config from ``paths.json`` (and the local builder config).

        :param p: Loaded paths (default :func:`paths.load_paths`).
        :param out_dir: Output directory (default ``<paths.out_dir>/v0_1_repro``).
        :param local: Local config dict (default :func:`load_local_config`).
        :returns: The config.
        """
        p = p or _paths.load_paths()
        local = load_local_config(p) if local is None else local
        read_names = {**READ_FILES, **(local.get("read_files") or {})}
        out = Path(out_dir) if out_dir is not None else p.out("v0_1_repro")
        return cls(pairs=p.pgp_parallel, pgp_dir=p.pgp_export, merged_map=p.merged_pgpid_map,
                   merged_jsonl=p.merged_shelfmarks, benchmark_ids=p.benchmark_ids, samples=p.samples,
                   treatments=p.document_treatments, probe_dir=p.probe_dir,
                   read_files={m: p.ai_reads / f for m, f in read_names.items()}, tokenizer=p.tokenizer_json,
                   out_dir=out, noise_model_path=out / "noise_model.json", v0_1_dir=p.text_ft_v0_1,
                   scholar_extra=list(local.get("scholar_extra") or []))


def load_local_config(p: Optional[_paths.Paths] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the git-ignored local builder config (holdings-specific settings kept out of committed code).

    Keys: ``licence_by_book`` {work dir: licence string}, ``analysis_excluded_books`` [work dirs],
    ``scholar_extra`` [surnames], ``read_files`` {reader: file name under ``paths.ai_reads``}.

    :param p: Loaded paths; the file defaults to ``builder_local.json`` next to ``paths.json``.
    :param path: Explicit file (overrides ``p``).
    :returns: The config dict ({} when the file does not exist).
    """
    if path is None:
        p = p or _paths.load_paths()
        explicit = p.get("builder_local")
        path = Path(explicit) if explicit else (Path(p.source_file).parent if p.source_file else p.repo) / "builder_local.json"
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def apply_local_config(local: Dict[str, Any]) -> None:
    """Install the local licence map and excluded-book list into the module globals the prototype reads.

    :param local: Output of :func:`load_local_config`.
    :returns: None.
    """
    LICENCE_BY_BOOK.clear()
    LICENCE_BY_BOOK.update(local.get("licence_by_book") or {})
    ANALYSIS_EXCLUDED_BOOKS.clear()
    ANALYSIS_EXCLUDED_BOOKS.update(local.get("analysis_excluded_books") or [])


def load_place_names(pgp_dir: Path) -> Set[str]:
    """Set :data:`PLACE_NAMES` to the PGP ``places.csv`` names plus :data:`PLACE_EXTRAS`.

    :param pgp_dir: PGP export directory.
    :returns: The new place-name set.
    """
    names = {r["name"] for r in load_csv(pgp_dir / "places.csv")} | PLACE_EXTRAS
    PLACE_NAMES.clear()
    PLACE_NAMES.update(names)
    return PLACE_NAMES


def _norm(text: Optional[str]) -> str:
    """The v0 builder's CER normalisation (``probe_common.normalize_for_cer`` with ``v11=False``).

    :param text: Raw text.
    :returns: Normalised text.
    """
    return tn.normalize_for_cer(text, v11=False)


def _cer(hyp: Optional[str], ref: Optional[str]) -> float:
    """The v0 builder's CER (``probe_common.cer`` with ``v11=False``).

    :param hyp: Hypothesis.
    :param ref: Reference.
    :returns: CER.
    """
    return tn.cer(hyp, ref, v11=False)


# ============================================================================ ported build_pairs helpers
def load_csv(path: Path) -> List[Dict[str, str]]:
    """Read a PGP CSV, stripping the BOM from the first header (port of ``build_pairs.load_csv``).

    :param path: CSV path.
    :returns: Rows as dicts.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [name.lstrip("﻿") for name in (reader.fieldnames or [])]
        return list(reader)


def split_multi(value: str) -> List[str]:
    """Split a PGP multi-value field on ',' or ';' outside parentheses (port of ``build_pairs.split_multi``).

    :param value: Raw field.
    :returns: Non-empty stripped parts.
    """
    if not value:
        return []
    return [p.strip() for p in re.split(r"[;,](?![^()]*\))", value) if p.strip()]


def split_of(pgpid: str) -> str:
    """Deterministic split: first byte of sha1(pgpid) mod 100 → <80 train, <90 dev, else test.

    :param pgpid: PGP document id.
    :returns: ``train``, ``dev`` or ``test``.
    """
    bucket = hashlib.sha1(pgpid.encode("utf-8")).digest()[0] % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def _collect_image_urls(node: Any, acc: List[str]) -> None:
    """Collect image URL strings recursively (helper of :func:`build_merged_map`).

    :param node: JSON node.
    :param acc: Accumulator (appended in place).
    :returns: None.
    """
    if isinstance(node, str):
        if node.startswith("http") and re.search(r"\.(jpe?g|png|tiff?)(\?|$)", node, re.I):
            acc.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_image_urls(value, acc)
    elif isinstance(node, list):
        for value in node:
            _collect_image_urls(value, acc)


def build_merged_map(cache: Path, merged_jsonl: Optional[Path] = None, rebuild: bool = False
                     ) -> Dict[str, Dict[str, List[str]]]:
    """Canonical id → pgpids and image URLs for every merged record linked to PGP (port of ``build_pairs``).

    :param cache: The cached map (``merged_pgpid_map.json``); read when it exists and ``rebuild`` is False.
    :param merged_jsonl: The merged corpus JSONL, scanned when the cache is missing or ``rebuild`` is set.
    :param rebuild: Ignore the cache and rescan (the cache is then rewritten).
    :returns: ``{canonical_id: {"pgpids": [...], "image_urls": [...]}}``.
    :raises FileNotFoundError: When neither the cache nor the merged file is available.
    """
    if cache.exists() and not rebuild:
        with open(cache, encoding="utf-8") as fh:
            out = json.load(fh)
        log(f"merged map from cache: {len(out):,} canonical ids")
        return out
    if merged_jsonl is None or not merged_jsonl.exists():
        raise FileNotFoundError(f"no merged map cache at {cache} and no merged corpus to rebuild it from")
    out = {}
    n_lines = 0
    with open(merged_jsonl, encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if '"pgpids": ["' not in line and '"pgp": {' not in line:
                continue
            rec = json.loads(line)
            raw_ids = list(rec.get("pgpids") or [])
            pgp_src = (rec.get("sources") or {}).get("pgp") or {}
            frag = pgp_src.get("fragment") or {} if isinstance(pgp_src, dict) else {}
            raw_ids.append(str(frag.get("pgpids") or ""))
            pgpids = sorted({p.strip() for raw in raw_ids for p in str(raw).split(";") if p.strip()})
            if not pgpids:
                continue
            acc: List[str] = []
            _collect_image_urls(rec.get("images"), acc)
            out[rec["canonical_id"]] = {"pgpids": pgpids, "image_urls": sorted(set(acc))}
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    log(f"merged map built: {len(out):,} PGP-linked canonical ids of {n_lines:,} records")
    return out


_HTML_TAG = re.compile(r"</?(?:p|br|span|div|em|i|b|strong|sup|sub|u|a|ol|ul|li|font|hr)\b[^>]*>", re.I)
_LABEL_ANCHORS = {"recto", "verso", "margin", "margins", "address", "column", "columns", "upside",
                  "side", "page", "fol", "folio", "main", "vertical", "vertically", "sideways",
                  "perpendicular", "colophon", "endorsement", "reverse", "obverse"}
_LABEL_WORDS = _LABEL_ANCHORS | {
    "right", "left", "top", "upper", "lower", "bottom", "down", "text", "continued", "cont",
    "written", "in", "on", "the", "of", "and", "at", "between", "lines", "line", "a", "b", "r",
    "v", "arabic", "hebrew", "judaeo-arabic", "aramaic", "script", "outer", "inner", "back",
    "front", "end", "beginning", "part", "first", "second", "third", "sheet", "panel",
    "fragment", "frag", "diagonal", "diagonally", "corner", "header", "heading", "signature",
    "signatures", "witnesses", "fold", "middle", "center", "centre", "space", "blank", "verse",
    "recto/verso", "lefthand", "righthand", "hand", "another", "different", "small", "large",
    "letters", "ink", "red", "same", "under", "above", "below", "upright", "inverted"}
_SIDE_ONLY_WORDS = {"recto", "verso", "side", "page", "fol", "folio", "a", "b", "r", "v",
                    "reverse", "obverse", "1", "2", "first", "second"}
_HE_SIDE = re.compile(r"^[\(\[]?\s*(?:ע\s*[\"״׳'`]{1,2}\s*[אב]|עמ(?:וד|׳)?\s*[אב][׳']?|צד\s*[אב][׳']?|"
                      r"דף\s*[אב]?[׳']?)\s*[\)\]]?\s*[:.]?\s*$")
_HE_MARGIN = re.compile(r"^[\(\[]?\s*(?:ב?ה?שולי(?:ים|יים|י)?.{0,20}|כתובת|ב?צד\s+(?:ימין|שמאל).{0,15}|"
                        r"בגב.{0,15}|הפוך|בין השיטין)\s*[\)\]]?\s*:\s*$")
_SHELFMARK_HEAD = re.compile(r"^(?:T-?S|ENA|Bodl|CUL|Or\b|Mosseri|AIU|BL\b|JTS|Halper|Moss\.|"
                             r"Westminster|ULC|MS\b|Heb\b|PER\b|P\.\s*Heid|Cambridge)", re.I)


def classify_label(line: str) -> Optional[str]:
    """Decide whether a line is a side or margin label (port of ``build_pairs.classify_label``).

    :param line: One stripped line.
    :returns: ``"side"`` (dropped), ``"margin"`` (kept as ``§ label``) or None (text).
    """
    if not line or len(line) > 60:
        return None
    if _HE_SIDE.match(line):
        return "side"
    if _HE_MARGIN.match(line):
        return "margin"
    if re.search(r"[֐-׿؀-ۿ]", line):
        return None
    words = [w for w in re.split(r"[\s,.:;()\[\]\-–—_/]+", line.lower()) if w]
    if not words or not all(w in _LABEL_WORDS or w.isdigit() for w in words):
        return None
    if not any(w in _LABEL_ANCHORS for w in words):
        if line.rstrip().endswith(":") and words[0] in {"arabic", "hebrew", "aramaic"}:
            return "margin"
        return None
    return "side" if all(w in _SIDE_ONLY_WORDS for w in words) else "margin"


def clean_text(raw: str, shelfmarks: List[str], counter: collections.Counter) -> str:
    """Clean a footnote's content (port of ``build_pairs.clean_text``).

    Strips HTML tags (``<br>`` → newline) and entities, turns NBSP into spaces, drops side-label
    lines, keeps margin / address / column labels as ``§ <label>``, drops a heading line that is just
    the shelfmark, collapses runs of blank lines and trims.

    :param raw: Footnote content.
    :param shelfmarks: The document's shelfmarks (for the heading check).
    :param counter: Counter of cleaning actions (updated in place).
    :returns: Cleaned text.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if _HTML_TAG.search(text):
        counter["html_tags"] += 1
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = _HTML_TAG.sub("", text)
    if re.search(r"&(?:[a-z]+|#\d+);", text):
        counter["html_entities"] += 1
        text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")
    sm_keys = {re.sub(r"[\W_]+", "", s).lower() for s in shelfmarks if s}
    out: List[str] = []
    first_content = True
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        kind = classify_label(stripped)
        if kind == "side":
            counter["side_labels_dropped"] += 1
            continue
        if kind == "margin":
            counter["margin_labels_kept"] += 1
            out.append("§ " + stripped.rstrip(":").strip())
            continue
        if first_content:
            first_content = False
            key = re.sub(r"[\W_]+", "", stripped).lower()
            latin_only = not re.search(r"[֐-׿؀-ۿ]", stripped)
            if latin_only and len(stripped) < 45 and (
                    key in sm_keys or (_SHELFMARK_HEAD.match(stripped) and re.search(r"\d", stripped))):
                counter["shelfmark_heading_dropped"] += 1
                continue
        out.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _better_read(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether read record ``a`` beats ``b`` (parsed, then newest, then most agreed lines).

    :param a: Candidate ai_reads record.
    :param b: Current best record.
    :returns: True when ``a`` is preferred.
    """
    ra, rb = a["ai_read"], b["ai_read"]
    ka = (bool(ra.get("parsed")), ra.get("decoded_at") or "", ra.get("n_agreed") or 0)
    kb = (bool(rb.get("parsed")), rb.get("decoded_at") or "", rb.get("n_agreed") or 0)
    return ka > kb


def load_reads(path: Path, known_ids: Set[str], url_to_id: Dict[str, str], model: str = ""
               ) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Best read record per (canonical id, image_index) of one reader's file (port of ``build_pairs.load_reads``).

    :param path: The reader's ``ai_reads_*.jsonl``.
    :param known_ids: PGP-linked canonical ids of the merged corpus.
    :param url_to_id: Image URL → canonical id fallback for ids renamed between index versions.
    :param model: Reader name (for the log line and stats).
    :returns: ``{canonical_id: {image_index: page dict}}``.
    """
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    n = n_direct = n_url = n_miss = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            doc_id = rec.get("doc_id")
            if doc_id in known_ids:
                n_direct += 1
            elif rec.get("image_url") in url_to_id:
                doc_id = url_to_id[rec["image_url"]]
                n_url += 1
            else:
                n_miss += 1
                continue
            k = (doc_id, int(rec.get("image_index") or 0))
            if k not in best or _better_read(rec, best[k]):
                best[k] = rec
    out: Dict[str, Dict[int, Dict[str, Any]]] = collections.defaultdict(dict)
    for (doc_id, idx), rec in best.items():
        read = rec["ai_read"]
        lines = sorted(read.get("lines") or [], key=lambda ln: ln.get("index", 0))
        out[doc_id][idx] = {
            "doc_id": doc_id, "image_index": idx, "image_url": rec.get("image_url"),
            "image_sha256": rec.get("image_sha256"), "n_lines": read.get("n_lines", len(lines)),
            "n_agreed": read.get("n_agreed"), "decode_policy": bool(read.get("decode_policy")),
            "decoded_at": read.get("decoded_at"), "parsed": bool(read.get("parsed")),
            "text": "\n".join((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip()),
        }
    log(f"reads {model}: {n:,} records, {n_direct:,} by doc_id, {n_url:,} by image_url, "
        f"{n_miss:,} not PGP-linked; {len(best):,} pages over {len(out):,} canonical ids")
    return out



# ============================================================================ utilities
def log(msg: str) -> None:
    """Print a timestamped progress line.

    :param msg: Message.
    """
    LOG.info(f"[{time.time() - T0:7.1f}s] {msg}")


def stable_int(*parts: Any) -> int:
    """Deterministic integer hash of the given parts (for seeding per-row RNGs).

    :param parts: Values joined with ``|``.
    :returns: A 64-bit integer.
    """
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File path.
    :returns: Parsed rows.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TokenCounter:
    """Qwen3.8-27B token counts from the local ``tokenizer.json`` (falls back to a char heuristic)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        """Load the tokenizer if it exists locally.

        :param path: Path to ``tokenizer.json`` (default ``paths.tokenizer_json``).
        """
        path = Path(path) if path is not None else _paths.load_paths().tokenizer_json
        self.tok = _rows.get_tokenizer(path) if path.exists() else None
        self.exact = self.tok is not None
        self.cache: Dict[str, int] = {}

    def count(self, text: str) -> int:
        """Token count of ``text``.

        :param text: Text.
        :returns: Number of tokens.
        """
        if text in self.cache:
            return self.cache[text]
        if self.tok is not None:
            n = len(self.tok.encode(text, add_special_tokens=False).ids)
        else:
            rtl = len(re.findall(r"[֐-׿؀-ۿ]", text))
            n = int(rtl / 2.5 + (len(text) - rtl) / 3.2) + 1
        if len(text) > 200:
            self.cache[text] = n
        return n

    def messages(self, msgs: List[Dict[str, str]]) -> int:
        """Token count of a chat row.

        :param msgs: Messages with ``content``.
        :returns: Tokens including a per-message overhead.
        """
        return sum(self.count(m["content"]) + CHAT_OVERHEAD for m in msgs)


# ============================================================================ groups and hold-outs
class UnionFind:
    """Minimal union-find over string keys."""

    def __init__(self) -> None:
        """Create an empty structure."""
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        """Root of ``x`` (path halving).

        :param x: Key.
        :returns: Root key.
        """
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Join the sets of ``a`` and ``b``.

        :param a: Key.
        :param b: Key.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=int)] = min(ra, rb, key=int)


def build_groups(merged: Dict[str, Dict[str, List[str]]], all_pgpids: Iterable[str], fragments_csv: Path
                 ) -> Dict[str, str]:
    """Shelfmark groups: pgpids connected through a shared fragment (fragments.csv) or merged id.

    :param merged: canonical id -> ``{"pgpids": [...]}`` map.
    :param all_pgpids: Every pgpid that needs a group.
    :param fragments_csv: PGP ``fragments.csv``.
    :returns: pgpid -> group id (``g<min pgpid>``).
    """
    uf = UnionFind()
    for p in all_pgpids:
        if p.isdigit():
            uf.find(p)
    for fr in load_csv(fragments_csv):
        ids = [x for x in split_multi(fr.get("pgpids", "")) if x.isdigit()]
        for x in ids[1:]:
            uf.union(ids[0], x)
    for info in merged.values():
        ids = [x for x in info["pgpids"] if x.isdigit()]
        for x in ids[1:]:
            uf.union(ids[0], x)
    return {p: "g" + uf.find(p) for p in uf.parent}


def held_out_pgpids(merged: Dict[str, Dict[str, List[str]]], bench_ids: Path,
                    registry_fn: Callable[[], Tuple[Set[str], Set[str]]] = registered_benchmark_documents
                    ) -> Tuple[Set[str], Set[str]]:
    """PGP ids and canonical ids held out by the vision benchmarks (registries + frozen PGP-131).

    The pgpids come from the merged map only (the v0 behaviour; the v0_2 builder adds shelfmark-key matching).

    :param merged: canonical id -> pgpids map.
    :param bench_ids: The frozen PGP-131 canonical id list (``benchmark_ids.json``).
    :param registry_fn: Returns the registered benchmarks' ``(canonical ids, pgpids)``.
    :returns: ``(pgpids, canonical ids)``.
    """
    ids, pgpids = registry_fn()
    pgpids = set(pgpids)
    with open(bench_ids, encoding="utf-8") as fh:
        b131 = set(json.load(fh))
    ids = set(ids) | b131
    for cid in ids:
        pgpids |= set(merged.get(cid, {}).get("pgpids", []))
    STATS["holdout_canonical_ids"] = len(ids)
    STATS["holdout_pgpids"] = len(pgpids)
    STATS["holdout_pgp131_canonical_ids"] = len(b131)
    return set(pgpids), ids


def probe_pgpids(samples: Path) -> Set[str]:
    """Every pgpid used by the zero-shot probe samples (kept out of train).

    :param samples: The probe ``samples.json``.
    :returns: pgpids.
    """
    out: Set[str] = set()

    def walk(o: Any) -> None:
        """Collect ``pgpid`` values recursively.

        :param o: JSON value.
        """
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "pgpid":
                    out.add(str(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    with open(samples, encoding="utf-8") as fh:
        walk(json.load(fh).get("sets", {}))
    return out


def group_splits(groups: Dict[str, str], forced_test: Set[str]) -> Dict[str, str]:
    """Split per group: the most held-out member split of :func:`split_of` (train < dev < test), with
    groups containing a forced pgpid moved to test.

    Keeping the member-level split means every document that is dev/test in ``pgp_parallel_v1``
    (and so in the probe sets) stays dev/test here.

    :param groups: pgpid -> group id.
    :param forced_test: pgpids whose group must not be in train (benchmarks, probe samples).
    :returns: group id -> split.
    """
    order = {"train": 0, "dev": 1, "test": 2}
    best: Dict[str, str] = {}
    for p, g in groups.items():
        s = split_of(p)
        if p in forced_test:
            s = "test"
        if g not in best or order[s] > order[best[g]]:
            best[g] = s
    return best


# ============================================================================ cleaning
_URL = re.compile(r"https?://\S+|www\.\S+")
_CITE_PAREN = re.compile(r"\s*\((?:[^()]*?(?:[Ii]nformation (?:in part )?from|index card|Med\.?\s?Soc|Mediterranean Society|"
                         r"CUDL|PGPID|\bcf\.|\bsee\b|\bSee\b|\bpp?\.\s*\d|\bvol\.|Goitein's|[Ee]dited|[Tt]ranslated|"
                         r"Description (?:by|from)|India Book)[^()]*)\)")
_IB_PREFIX = re.compile(r"^\s*(?:India Book\s*(?:\(IB\))?|IB)\s*[IVX]+\s*,?\s*\d+[a-z]?\s*[:.]\s*")
_SIDE_SCAFFOLD = re.compile(r"(?:(?<=^)|(?<=[.;]\s))(?:Recto|Verso|Side [AB12]|Fol\.\s*\d+[rv]?)(?:\s*\([^)]{0,40}\))?\s*:\s*")
_INITIALS_END = re.compile(r"[\s.;,(—–-]+\(?([A-Z]{2,4})\)?\.?\s*$")
_INITIALS_PAREN = re.compile(r"\s*\((?:[A-Z]{2,4})(?:[,/ ]+[A-Z]{2,4})*\)")
_NOT_INITIALS = {"CE", "AM", "AH", "BCE", "AD", "BC", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII"}
_REF_SENT = re.compile(r"^(?:See |See also|Cf\.|For (?:more|further|the)|Join|Joins|This is a join|Joined|Same (?:hand|scribe) as|"
                       r"Edited|Translated|Described|Discussed|Mentioned in|Cited|Listed|Information(?: in part)? from|Images? |"
                       r"Also edited|Edition|Transcription|Translation|Bibliography|Image)", re.I)
_REF_ANYWHERE = re.compile(r"\b(?:edited|translated|published|discussed|described|transcribed)\s+(?:by|in)\b|\bEdition in\b|"
                           r"\bindex cards?\b|\bnote cards?\b|\bGoitein notes\b", re.I)
_ABBREV = re.compile(r"\b(b|bt|R|ca|Mr|St|al|ibn)\.(?=\s)")


def clean_description(desc: str) -> Tuple[str, collections.Counter]:
    """Clean a PGP description into a target summary.

    Strips URLs, India Book prefixes (``India Book III, 46:``), ``Recto:`` / ``Verso:`` scaffolding,
    citation and provenance parentheticals (``(Information from Goitein's index cards)``,
    ``(PGPID 1277)``, ``(Med. Soc. II, ...)``), editor initials (``ASE``, ``VMR``, ``(MR)``), and whole
    sentences that are cross-references (``See ...``, ``Join ...``, sentences naming another PGPID).

    :param desc: Raw description.
    :returns: ``(cleaned text, counter of actions)``.
    """
    c: collections.Counter = collections.Counter()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", (desc or "").replace("\xa0", " ")).strip()
    if _URL.search(text):
        c["url"] += 1
        text = _URL.sub("", text)
    if _IB_PREFIX.search(text):
        c["india_book_prefix"] += 1
        text = _IB_PREFIX.sub("", text)
    n = len(_CITE_PAREN.findall(text))
    if n:
        c["citation_paren"] += n
        text = _CITE_PAREN.sub("", text)
    if _SIDE_SCAFFOLD.search(text):
        c["side_scaffold"] += 1
        text = _SIDE_SCAFFOLD.sub(lambda m: "\x00", text)
        text = re.sub(r"\x00\s*(\w)", lambda m: m.group(1).upper(), text).replace("\x00", "")
    for m in list(_INITIALS_PAREN.finditer(text)):
        if m.group(0).strip("() ") not in _NOT_INITIALS:
            c["initials_paren"] += 1
            text = text.replace(m.group(0), "")
    for _ in range(2):
        m = _INITIALS_END.search(text)
        if m and m.group(1) not in _NOT_INITIALS:
            c["initials_end"] += 1
            text = text[:m.start()].rstrip() + ("." if not text[:m.start()].rstrip().endswith(".") else "")
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"“])", text)
    kept = []
    for s in sents:
        if "PGPID" in s or re.search(r"\bjoin(?:s|ed)?\b", s, re.I) or _REF_SENT.match(s.strip()) or _REF_ANYWHERE.search(s):
            c["reference_sentence"] += 1
            continue
        kept.append(s.strip())
    text = " ".join(k for k in kept if k)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ;,")
    text = re.sub(r"\.{2}(?!\.)", ".", text)
    return text, c


_RTL = re.compile(r"[֐-׿؀-ۿ]")


def coarse_lang(lang_class: str, ja_score: float) -> str:
    """Coarse language class; unknown-language Hebrew-script rows re-classified by ``ja_score``.

    :param lang_class: ``pgp_parallel`` lang_class (``JA``, ``JA?``, ``HE?`` ...).
    :param ja_score: Share of Judaeo-Arabic-looking tokens.
    :returns: ``JA``, ``AR``, ``HE``, ``ARAM`` or ``OTHER``.
    """
    base = lang_class.rstrip("?")
    if lang_class.endswith("?") and base in ("JA", "HE"):
        return "JA" if (ja_score or 0) >= 0.10 else "HE"
    return base if base in ("JA", "AR", "HE", "ARAM") else "OTHER"


def lang_from_pgp(primary: str) -> str:
    """Coarse language class from ``languages_primary`` (read-only pool).

    :param primary: PGP languages_primary string.
    :returns: Coarse class.
    """
    p = primary or ""
    if "Judaeo-Arabic" in p:
        return "JA"
    if "Arabic" in p:
        return "AR"
    if "Aramaic" in p and "Hebrew" not in p:
        return "ARAM"
    if "Hebrew" in p:
        return "HE"
    return "OTHER" if p else "UNKNOWN"


# ============================================================================ catalogue targets
_YEAR = re.compile(r"(?<!\d)(\d{3,4})(?:-\d{2}){0,2}")


def parse_date(doc: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Catalogue date block from PGP date fields (conversion already done by PGP in ``*_standard``).

    :param doc: documents.csv row.
    :returns: Date dict or None when PGP has no date.
    """
    std = (doc.get("doc_date_standard") or "").strip()
    inferred = False
    if not std:
        std = (doc.get("inferred_date_standard") or "").strip()
        inferred = bool(std)
    if not std:
        return None
    years = [int(y) for y in _YEAR.findall(std)]
    return {
        "original": (doc.get("doc_date_original") or "").strip() or None,
        "calendar": (doc.get("doc_date_calendar") or "").strip() or None,
        "standard": std,
        "year_ce_start": min(years) if years else None,
        "year_ce_end": max(years) if years else None,
        "inferred": inferred,
    }


TITLES = {"nasi", "gaon", "gaʾon", "nagid", "judge", "rayyis", "ḥaver", "haver", "cantor", "community", "rav", "sultan", "amir",
          "amīr", "caliph", "vizier", "qāḍī", "qadi"}
_NAME_STOP = re.compile(r"\s*(?:,|\bin\b|\bat\b|\bfrom\b|\bto\b|\baddressed\b|\bof\b(?=\s+[A-Z]\w+(?:\s|$))|\bwho\b|\bwhich\b|\bconcerning\b|\babout\b|\bregarding\b|\bwith\b|\bon\b|"
                        r"\bdated\b|\basking\b|\brequesting\b|\bprobably\b|\bperhaps\b|\bincluding\b|\band\b(?=\s+[a-z]))")
_KIN_PREFIX = re.compile(r"^(?:(?:his|her|their)\s+(?:\w+\s+){0,2}?(?:son|daughter|brother|sister|father|mother|wife|husband|"
                         r"uncle|nephew|cousin|partner|friend|colleague|teacher|student|son-in-law|father-in-law|"
                         r"brother-in-law|in-law)s?\s*,?\s*)", re.I)
_LETTER_BOTH = re.compile(r"\b[Ll]etter\b[^.;:]{0,60}?\b(?:from|sent by|written by)\s+(?P<s>[^.;:]+?)\s+(?:addressed\s+)?to\s+"
                          r"(?P<r>[^.;:]+?)(?=[.;:(]|$)")
_LETTER_FROM = re.compile(r"\b[Ll]etter\b[^.;:]{0,60}?\b(?:from|sent by|written by)\s+(?P<s>[^.;:]+?)(?=[.;:(]|$)")
_LETTER_TO = re.compile(r"\b[Ll]etter\b[^.;:]{0,60}?\b(?:addressed\s+)?to\s+(?P<r>[^.;:]+?)(?=[.;:(]|$)")


def _clean_party(raw: str) -> Optional[str]:
    """Trim a letter party string to a personal name.

    :param raw: Text captured between ``from``/``to`` and the next delimiter.
    :returns: Name or None when it does not look like a name.
    """
    s = re.sub(r"^(?:the|an?|one|a certain)\s+(?=[A-Z])", "", raw.strip())
    s = _KIN_PREFIX.sub("", re.sub(r"\s*\([^)]*\)", "", s))
    m = _NAME_STOP.search(s)
    if m:
        s = s[:m.start()]
    s = re.sub(r"\s*\([^)]*$", "", s).strip(" ,")
    if s.lower() in TITLES:
        return None
    core = s.lstrip("ʿʾ'‘’")
    if not core or not core[0].isupper() or len(s) > 70 or len(s) < 3 or s in PLACE_NAMES:
        return None
    for tok in s.replace("\u2024", ". ").split():
        t = tok.lstrip("ʿʾ'‘’[")
        if t and not t[0].isupper() and not re.match(r"^(?:b|bt|ibn|ben|bar|bat|de|di|el|al|l|ha|ha-|w|wa)\b|^(?:al|l|ha|el|ibn)-|^\.|^\d", t):
            return None
    if re.match(r"^(?:Unknown|Anonymous|An?\b|The\b|Someone|Somebody|Fustat|Cairo|Alexandria)", s):
        return None
    return s


def letter_parties(desc: str) -> Tuple[Optional[str], Optional[str]]:
    """Sender and recipient named in a PGP letter description (``Letter from X, in P, to Y``).

    :param desc: Description (cleaned or raw).
    :returns: ``(sender, recipient)``; either may be None.
    """
    protected = _ABBREV.sub(lambda m: m.group(1) + "\u2024", desc or "")[:300]
    m = _LETTER_BOTH.search(protected)
    if m:
        s, r = _clean_party(m.group("s")), _clean_party(m.group("r"))
    else:
        mf, mt = _LETTER_FROM.search(protected), _LETTER_TO.search(protected)
        s = _clean_party(mf.group("s")) if mf else None
        r = _clean_party(mt.group("r")) if mt else None
    return tuple(x.replace("\u2024", ".") if x else None for x in (s, r))


class PeopleIndex:
    """people.csv names (and their parenthesis-free variants) indexed by first token."""

    def __init__(self, path: Path) -> None:
        """Load people.csv.

        :param path: people.csv path.
        """
        self.by_first: Dict[str, List[Tuple[str, str, List[str]]]] = collections.defaultdict(list)
        n = 0
        for r in load_csv(path):
            roles = [x.strip() for x in (r.get("social_roles") or "").split(",") if x.strip()]
            variants = {r["name"], re.sub(r"\([^)]*\)", "", r["name"])}
            variants |= {v.strip() for v in (r.get("name_variants") or "").split(",")}
            for v in variants:
                v = re.sub(r"\s+", " ", v).strip()
                if len(v) < 8 or len(v.split()) < 2 or not re.search(r"[A-Za-z]", v):
                    continue
                self.by_first[v.split()[0]].append((v, r["name"], roles))
                n += 1
        STATS["people_csv_variants"] = n

    def find(self, text: str) -> List[Tuple[str, List[str]]]:
        """People whose name occurs verbatim in ``text`` (longest match wins per start token).

        :param text: Description.
        :returns: ``[(canonical name, social roles)]``.
        """
        found: Dict[str, List[str]] = {}
        toks = list(re.finditer(r"\S+", text))
        for t in toks:
            for v, name, roles in sorted(self.by_first.get(t.group(0).strip(",.;:()"), []), key=lambda x: -len(x[0])):
                seg = text[t.start():t.start() + len(v) + 1]
                if seg.startswith(v) and (len(seg) == len(v) or not seg[len(v)].isalnum()):
                    found.setdefault(name, roles)
                    break
        return list(found.items())


#: PGP editorial meta tags; the scrubbed project's abbreviation variant is decoded at import (rot13, fix pass F12).
META_TAG = re.compile(r"^(?:ib-|IB\d|DIMME$|to edit$|" + codecs.decode("STC", "rot_13")
                      + r"|PGP|India Book|unedited|Mediterranean People$|stub$)", re.I)


def catalogue_record(doc: Dict[str, str], row: Dict[str, Any], summary: str, people_ix: PeopleIndex
                     ) -> Dict[str, Any]:
    """Build the catalogue JSON target from PGP fields (all human-entered).

    :param doc: documents.csv row.
    :param row: Normalised document info (type, tags, places, languages).
    :param summary: Cleaned description.
    :param people_ix: people.csv index.
    :returns: Catalogue dict.
    """
    sender, recipient = letter_parties(summary) if "etter" in (row["type"] or "") or "etter" in summary[:80] else (None, None)
    people: List[Dict[str, Any]] = []
    seen = set()
    for name, role in ((sender, "sender"), (recipient, "recipient")):
        if name:
            people.append({"name": name, "role": role})
            seen.add(strip_marks(name))
    for name, roles in people_ix.find(summary):
        key = strip_marks(name)
        if key in seen or any(key in s or s in key for s in seen):
            for p in people:
                if strip_marks(p["name"]) in key or key in strip_marks(p["name"]):
                    p.setdefault("social_roles", roles)
            continue
        seen.add(key)
        people.append({"name": name, "role": "mentioned", **({"social_roles": roles} if roles else {})})
    return {
        "document_type": row["type"] or None,
        "languages": {"primary": row["languages_primary"], "secondary": row["languages_secondary"]},
        "date": parse_date(doc),
        "origin": row["origin"] or None,
        "destination": row["destination"] or None,
        "places_mentioned": row["mentioned"],
        "people": people,
        "tags": [t for t in row["tags"] if not META_TAG.match(t)],
        "summary": summary or None,
    }


def relations_from_catalogue(cat: Dict[str, Any], row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Gold triplets from the catalogue record.

    :param cat: Catalogue dict from :func:`catalogue_record`.
    :param row: Normalised document info (for possibly-mentioned places).
    :returns: Relation dicts.
    """
    rel: List[Dict[str, str]] = []
    if cat["origin"]:
        rel.append({"subject": "THIS_DOCUMENT", "predicate": "origin", "object": cat["origin"]})
    if cat["destination"]:
        rel.append({"subject": "THIS_DOCUMENT", "predicate": "destination", "object": cat["destination"]})
    for p in cat["places_mentioned"]:
        rel.append({"subject": "THIS_DOCUMENT", "predicate": "mentions_place", "object": p})
    for p in row.get("possibly_mentioned", []):
        rel.append({"subject": "THIS_DOCUMENT", "predicate": "possibly_mentions_place", "object": p})
    for p in cat["people"]:
        pred = {"sender": "sender", "recipient": "recipient"}.get(p["role"], "mentions_person")
        rel.append({"subject": "THIS_DOCUMENT", "predicate": pred, "object": p["name"]})
        for r in p.get("social_roles", []):
            rel.append({"subject": p["name"], "predicate": "has_social_role", "object": r})
    return rel


# ============================================================================ noise model
def _alignment_pairs(rows: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, str]]]:
    """(edition, read) pairs per script that pass the noise-pair gate (after read loop cleaning).

    :param rows: pgp_parallel rows.
    :returns: ``{script: [(edition_norm, read_norm)]}``.
    """
    out: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for r in rows:
        if r["edition_script"] not in ("hebrew", "arabic"):
            continue
        ed = _norm(r["edition_text"])
        if len(ed) < 200:
            continue
        best = None
        for rd in r.get("reads", []):
            txt, _ = clean_read(rd["text"])
            rn = _norm(txt)
            cov = len(rn) / len(ed)
            cer = Levenshtein.distance(rn, ed) / len(ed)
            g = NOISE_PAIR_GATE
            if g["coverage_min"] <= cov <= g["coverage_max"] and cer <= g["cer_max"]:
                if best is None or cer < best[0]:
                    best = (cer, rn)
        if best:
            out[r["edition_script"]].append((ed, best[1]))
    return out


def build_noise_model(rows: List[Dict[str, Any]], read_texts: List[str], out_path: Optional[Path] = None
                      ) -> Dict[str, Any]:
    """Learn character-level substitution / deletion / insertion rates per script from read↔edition
    Levenshtein editops, plus the rate of ``[...]`` gap markers per read line.

    :param rows: pgp_parallel rows.
    :param read_texts: Loop-cleaned read texts (for the gap-marker rate).
    :param out_path: Where to write the model as JSON (``noise_model.json``); None skips writing.
    :returns: Noise model dict (also written to ``noise_model.json``).
    """
    pairs = _alignment_pairs(rows)
    model: Dict[str, Any] = {"built": time.strftime("%Y-%m-%d"), "gate": NOISE_PAIR_GATE,
                             "normalisation": "probe_common.normalize_for_cer (editorial signs, § lines, line numbers "
                                              "dropped; whitespace collapsed)", "scripts": {}}
    n_lines = sum(t.count("\n") + 1 for t in read_texts)
    n_gap = sum(t.count("[...]") for t in read_texts)
    model["gap_marker_per_line"] = round(n_gap / max(n_lines, 1), 5)
    for script, prs in pairs.items():
        cnt: collections.Counter = collections.Counter()
        sub: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        dele: collections.Counter = collections.Counter()
        ins_after: collections.Counter = collections.Counter()
        ins_char: collections.Counter = collections.Counter()
        tot_sub = tot_del = tot_ins = tot = 0
        for ed, rd in prs:
            cnt.update(ed)
            tot += len(ed)
            for op, i, j in Levenshtein.editops(ed, rd):
                if op == "replace":
                    sub[ed[i]][rd[j]] += 1
                    tot_sub += 1
                elif op == "delete":
                    dele[ed[i]] += 1
                    tot_del += 1
                else:
                    ins_after[ed[i - 1] if i > 0 else "^"] += 1
                    ins_char[rd[j]] += 1
                    tot_ins += 1
        glob_sub, glob_del, glob_ins = tot_sub / tot, tot_del / tot, tot_ins / tot
        prior = 200.0  # pseudo-count toward the global rates for rare characters
        chars = {}
        for ch, n in cnt.most_common():
            s_n = sum(sub[ch].values())
            chars[ch] = {
                "n": n,
                "p_sub": round((s_n + prior * glob_sub) / (n + prior), 5),
                "p_del": round((dele[ch] + prior * glob_del) / (n + prior), 5),
                "p_ins_after": round((ins_after[ch] + prior * glob_ins) / (n + prior), 5),
                "sub_to": {d: round(k / s_n, 4) for d, k in sub[ch].most_common(12)} if s_n else {},
            }
        conf = collections.Counter({(c, d): k for c, dd in sub.items() for d, k in dd.items()})
        model["scripts"][script] = {
            "n_pairs": len(prs), "edition_chars": tot,
            "rate_sub": round(glob_sub, 5), "rate_del": round(glob_del, 5), "rate_ins": round(glob_ins, 5),
            "cer_empirical": round((tot_sub + tot_del + tot_ins) / tot, 5),
            "insert_chars": {d: round(k / max(tot_ins, 1), 4) for d, k in ins_char.most_common(30)},
            "top_confusions": [{"edition": c, "read": d, "count": k} for (c, d), k in conf.most_common(40)],
            "top_deletions": [{"edition": c, "count": k, "rate": round(k / cnt[c], 4)} for c, k in dele.most_common(15)],
            "chars": chars,
        }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(model, fh, ensure_ascii=False, indent=1)
    return model


class Noiser:
    """Applies the learned confusion model to an edition at a sampled target CER."""

    def __init__(self, model: Dict[str, Any]) -> None:
        """Prepare per-script sampling tables.

        :param model: Output of :func:`build_noise_model`.
        """
        self.model = model
        self.tables: Dict[str, Dict[str, Any]] = {}
        for script, m in model["scripts"].items():
            ins = list(m["insert_chars"].items())
            self.tables[script] = {
                "chars": m["chars"], "cer": m["cer_empirical"],
                "glob": (m["rate_sub"], m["rate_del"], m["rate_ins"]),
                "ins_chars": [c for c, _ in ins], "ins_w": [w for _, w in ins],
                "sub_alpha": [c for c, v in m["chars"].items() if v["n"] > 500 and c != " "],
            }
        self.gap_rate = model.get("gap_marker_per_line", 0.0)
        self.calib: Dict[str, float] = {k: 1.0 for k in self.tables}

    def calibrate(self, editions: Dict[str, List[str]], n: int = 300) -> Dict[str, float]:
        """Rescale so the measured CER (``probe_common.cer``) matches the sampled target on average.

        :param editions: script -> edition texts (train documents).
        :param n: Editions per script used for calibration.
        :returns: script -> calibration factor (measured / target before rescaling).
        """
        for script, texts in editions.items():
            if script not in self.tables or not texts:
                continue
            rng = random.Random(stable_int(SEED, "calibrate", script))
            ratios = []
            for t in rng.sample(texts, min(n, len(texts))):
                noised, target = self.apply(t, script, rng)
                ratios.append(_cer(noised, t) / target)
            self.calib[script] = statistics.median(ratios)
        return dict(self.calib)

    def available(self, script: str) -> bool:
        """Whether a model exists for the script.

        :param script: ``hebrew`` / ``arabic``.
        :returns: True when the script was learned.
        """
        return script in self.tables

    def apply(self, text: str, script: str, rng: random.Random) -> Tuple[str, float]:
        """Noise ``text`` (an edition) so it looks like one of our reads.

        :param text: Cleaned edition text.
        :param script: Edition script.
        :param rng: Seeded RNG.
        :returns: ``(noised text, target CER that was sampled)``.
        """
        tb = self.tables[script]
        target = rng.uniform(*NOISE_CER_RANGE)
        k = target / tb["cer"] / self.calib.get(script, 1.0)
        g_sub, g_del, g_ins = tb["glob"]
        out_lines = []
        for line in edition_as_plain(text).split("\n"):
            buf: List[str] = []
            for ch in line:
                info = tb["chars"].get(ch)
                p_sub, p_del, p_ins = (info["p_sub"], info["p_del"], info["p_ins_after"]) if info else (g_sub, g_del, g_ins)
                u = rng.random()
                if u < min(p_del * k, 0.9):
                    pass
                elif u < min((p_del + p_sub) * k, 0.95):
                    tos = info["sub_to"] if info and info["sub_to"] else None
                    if tos:
                        buf.append(rng.choices(list(tos), weights=list(tos.values()))[0])
                    else:
                        buf.append(rng.choice(tb["sub_alpha"]))
                else:
                    buf.append(ch)
                if rng.random() < min(p_ins * k, 0.5):
                    buf.append(rng.choices(tb["ins_chars"], weights=tb["ins_w"])[0])
            ln = re.sub(r" {2,}", " ", "".join(buf)).strip()
            if ln and rng.random() < self.gap_rate * k:
                words = ln.split(" ")
                cut = rng.randrange(len(words))
                ln = " ".join(words[:cut] + ["[...]"])
            out_lines.append(ln)
        return "\n".join(out_lines).strip(), target


# ============================================================================ sides
def _side_key(label: str) -> str:
    """Normalise a side label (``Recto``, ``ע"א``, ``fol. 2 verso``) to a comparison key.

    :param label: Label line.
    :returns: Key such as ``recto``, ``verso``, ``2v``.
    """
    s = label.strip().strip("()[]:. ").lower()
    if re.match(r"^ע\s*[\"״׳'`]{1,2}\s*א|^עמ(?:וד|׳)?\s*א|^צד\s*א", s):
        return "recto"
    if re.match(r"^ע\s*[\"״׳'`]{1,2}\s*ב|^עמ(?:וד|׳)?\s*ב|^צד\s*ב", s):
        return "verso"
    num = "".join(re.findall(r"\d+", s))
    side = "r" if re.search(r"recto|\br\b|obverse|\ba\b", s) else "v" if re.search(r"verso|\bv\b|reverse|\bb\b", s) else ""
    return (num + side) or s


def split_sides(raw: str) -> List[Tuple[str, str]]:
    """Cut a raw footnote into ``(side key, cleaned text)`` segments at side-label lines.

    :param raw: Raw footnote content (side labels present).
    :returns: Segments in order; empty when there are fewer than two labelled sides.
    """
    segs: List[Tuple[str, List[str]]] = []
    cur_key, cur_lines = "", []
    for line in raw.replace("\r\n", "\n").split("\n"):
        st = line.strip()
        if st and classify_label(st) == "side":
            if cur_lines and any(x.strip() for x in cur_lines):
                segs.append((cur_key, cur_lines))
            cur_key, cur_lines = _side_key(st), []
            continue
        cur_lines.append(line)
    if cur_lines and any(x.strip() for x in cur_lines):
        segs.append((cur_key, cur_lines))
    labelled = [(k, clean_text("\n".join(v), [], collections.Counter())) for k, v in segs if k]
    if len(labelled) < 2 or len({k for k, _ in labelled}) != len(labelled):
        return []
    return [(k, t) for k, t in labelled if t]


# ============================================================================ analysis targets
def load_full_spans(treatments: Path, probe_dir: Path) -> List[Dict[str, Any]]:
    """Re-run the span detector of ``build_analysis.py`` with the 1,500-char sample cap lifted, and
    return the document_treatments rows with the full section line lists attached.

    The detector is the git-ignored research module in ``probe_dir`` (book-specific heading rules);
    it is imported here, at call time, and only for the analysis family of a v0 reproduction.

    :param treatments: ``document_treatments.jsonl``.
    :param probe_dir: Directory holding ``build_analysis.py`` / ``analysis_lib.py``.
    :returns: Treatment rows with ``sections`` = ``{intro|translation|notes|edition: [lines]}``.
    """
    if str(probe_dir) not in sys.path:
        sys.path.insert(0, str(probe_dir))
    ba = importlib.import_module("build_analysis")  # loads the corpus pickle
    src = inspect.getsource(ba.classify_lines)
    assert src.count("< 1500") == 2, "build_analysis.classify_lines changed; update the patch"
    exec(compile(src.replace("< 1500", "< 10**12"), "classify_lines_full", "exec"), ba.__dict__)
    corpus = ba.load_corpus()
    spans = ba.detect_spans(ba.iter_pages(), corpus)
    rows = load_jsonl(treatments)
    assert len(rows) == len(spans)
    out = []
    for sp, r in zip(spans, rows):
        assert (sp.book, sp.part, sp.label) == (r["book"], r["part"], r["doc_label"])
        full = {k: sum(len(x) for x in v) for k, v in sp.samples.items()}
        assert all(full.get(k, 0) == r[f"{k}_chars"] for k in ("intro", "notes", "translation", "edition"))
        out.append({**r, "sections": {k: list(v) for k, v in sp.samples.items()}})
    return out


_RUNNING_HEAD = re.compile(r"^[A-Z0-9ḤṢṬẒĀĪŪ\s,.;:'’\-–—&]+$")
_CITATION_LINE = re.compile(r"(?:\bpp?\.\s*\d|\bn\.\s*\d|\bvol\.|\bibid\b|\bop\. cit|Med\.?\s*Soc|\bsee\b|\bcf\.)", re.I)
_CITE_PAREN_C = re.compile(r"\s*\((?:[^()]{0,160}?(?:\bpp?\.\s*\d|\bn\.\s*\d|\bvol\.|Med\.?\s*Soc|\bibid|\bsee\b|\bcf\.|"
                           r"\b(?:1[89]|20)\d{2}\b)[^()]{0,160}?)\)", re.I)


def clean_commentary(lines: List[str]) -> Tuple[str, collections.Counter]:
    """Turn a span's intro / notes lines into target prose.

    Drops running heads (all-caps lines, bare page numbers), pure citation lines and parenthetical
    bibliographic citations; rejoins hyphenated line breaks; starts a new paragraph at a footnote
    number or after a short line ending a sentence.

    :param lines: Section lines in page order.
    :returns: ``(prose, counter of actions)``.
    """
    c: collections.Counter = collections.Counter()
    kept: List[str] = []
    widths = [len(x) for x in lines if len(x) > 20] or [60]
    med = statistics.median(widths)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[\dxivlXIVL{}\[\]()\s.]+", s) or (len(s.split()) >= 2 and _RUNNING_HEAD.match(s) and len(s) < 80):
            c["running_head"] += 1
            continue
        if len(s) < 140 and _CITATION_LINE.search(s) and len(re.findall(r"\b[a-z]{4,}\b", s)) <= 4:
            c["citation_line"] += 1
            continue
        kept.append(s)
    paras: List[str] = []
    cur = ""
    for s in kept:
        new_para = bool(re.match(r"^\d{1,3}\s+\S", s)) and bool(cur)
        if new_para:
            paras.append(cur)
            cur = s
            continue
        if cur.endswith("-") and s[:1].islower():
            cur = cur[:-1] + s
        else:
            cur = (cur + " " + s).strip()
        if len(s) < 0.6 * med and re.search(r"[.:;!?”\"]$", s):
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    text = "\n\n".join(p.strip() for p in paras if p.strip())
    n = len(_CITE_PAREN_C.findall(text))
    if n:
        c["citation_paren"] += n
        text = _CITE_PAREN_C.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), c


def context_block(cat: Dict[str, Any]) -> str:
    """The 'known catalogue context' block shown in analysis prompts (explicit design exception).

    :param cat: Catalogue record.
    :returns: Block text.
    """
    d = cat.get("date") or {}
    lines = [f"Type: {cat['document_type']}" if cat.get("document_type") else "",
             f"Date: {d.get('original') or ''} {('(' + d['standard'] + ')') if d.get('standard') else ''}".strip()
             if d else "",
             f"Origin: {cat['origin']}" if cat.get("origin") else "",
             f"Destination: {cat['destination']}" if cat.get("destination") else "",
             f"Places mentioned: {', '.join(cat['places_mentioned'])}" if cat.get("places_mentioned") else "",
             f"Catalogue description: {cat['summary']}" if cat.get("summary") else ""]
    return "\n".join(x for x in lines if x and x != "Date:")


# ============================================================================ rows
def user_prompt(family: str, condition: str, script: str, text: str, extra: str = "") -> str:
    """User message for a family / condition.

    :param family: Task family.
    :param condition: Input condition.
    :param script: Script of the input.
    :param text: Document text.
    :param extra: Additional blocks (context, translation) placed after the document.
    :returns: Prompt.
    """
    head = INSTRUCTION[family].format(script=SCRIPT_NAME.get(script, "Hebrew script"))
    body = f"{head}\n\n{COND_NOTE[condition]}\n\n<document>\n{text}\n</document>"
    return body + (f"\n\n{extra}" if extra else "")


def truncate_lines(text: str, max_tokens: int, tc: TokenCounter) -> Tuple[str, bool]:
    """Keep leading lines of ``text`` within ``max_tokens``.

    :param text: Text.
    :param max_tokens: Budget.
    :param tc: Token counter.
    :returns: ``(text, truncated?)``.
    """
    if tc.count(text) <= max_tokens:
        return text, False
    lines = text.split("\n")
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tc.count("\n".join(lines[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return "\n".join(lines[:lo]).rstrip() + "\n[...]", True


def make_row(family: str, condition: str, doc: Dict[str, Any], user: str, target: str, meta_extra: Dict[str, Any],
             tc: TokenCounter) -> Dict[str, Any]:
    """Assemble one chat row with its meta block.

    :param family: Task family.
    :param condition: Input condition.
    :param doc: Document info (pgpid, group, split, lang).
    :param user: User prompt.
    :param target: Assistant target.
    :param meta_extra: Extra meta fields (source, licence, provenance, input_cer ...).
    :param tc: Token counter.
    :returns: Row dict.
    """
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
            {"role": "assistant", "content": target}]
    meta = {"task": family, "lang_class": doc["lang"], "condition": condition, "pgpid": doc["pgpid"],
            "shelfmark": doc["shelfmark"], "shelfmark_group": doc["group"], "split": doc["split"],
            "held_out_vision_benchmark": doc["held_out"], "target_chars": len(target), "input_cer": None,
            "tokens": tc.messages(msgs), "target_tokens": tc.count(target)}
    meta.update(meta_extra)
    return {"messages": msgs, "meta": meta}


class Builder:
    """Holds shared resources and emits rows per family."""

    def __init__(self, tc: TokenCounter, noiser: Noiser, people_ix: PeopleIndex, docs_csv: Dict[str, Dict[str, str]]
                 ) -> None:
        """Create the builder.

        :param tc: Token counter.
        :param noiser: Noise sampler.
        :param people_ix: people.csv index.
        :param docs_csv: pgpid -> documents.csv row.
        """
        self.tc, self.noiser, self.people_ix, self.docs_csv = tc, noiser, people_ix, docs_csv
        self.rows: List[Dict[str, Any]] = []
        self.drops: collections.Counter = collections.Counter()
        self.desc_actions: collections.Counter = collections.Counter()

    # ------------------------------------------------------------------ inputs per doc
    def inputs(self, doc: Dict[str, Any], family: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Input variants for a document: edition, read(s), and a noised edition in train.

        :param doc: Document info.
        :param family: Task family (seeds the noise draw).
        :returns: ``[(condition, text, meta extras)]``.
        """
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        if doc.get("edition"):
            out.append(("edition", doc["edition"], {"input_source": f"pgp_digital_edition:{doc['edition_slug']}"}))
            if doc["split"] == "train" and self.noiser.available(doc["script"]):
                rng = random.Random(stable_int(SEED, doc["pgpid"], family, doc.get("side", "")))
                noised, target_cer = self.noiser.apply(doc["edition"], doc["script"], rng)
                out.append(("noised_edition", noised, {
                    "input_source": f"noised:{doc['edition_slug']}", "noise_target_cer": round(target_cer, 3),
                    "input_cer": round(_cer(noised, doc["edition"]), 4)}))
        reads = doc.get("reads_ok", [])
        if doc["split"] != "train":
            reads = reads[:1]
        for rd in reads:
            out.append(("read", rd["text"], {"input_source": f"read:{rd['model']}", "read_model": rd["model"],
                                             "input_cer": rd.get("cer"), "read_coverage": rd.get("coverage"),
                                             "read_agreed_share": rd.get("agreed_share"),
                                             "edition_available": bool(doc.get("edition"))}))
        return out

    def add(self, family: str, doc: Dict[str, Any], target: str, meta_extra: Dict[str, Any],
            extra_block: str = "", allow_input_truncation: bool = True,
            inputs: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None) -> int:
        """Emit rows of one family for every input variant of ``doc`` within the token cap.

        :param family: Task family.
        :param doc: Document info.
        :param target: Assistant target.
        :param meta_extra: Meta fields shared by the variants.
        :param extra_block: Context block appended to the prompt.
        :param allow_input_truncation: Truncate long inputs (catalogue-type families) instead of dropping.
        :param inputs: Precomputed input variants (default: :meth:`inputs`).
        :returns: Rows added.
        """
        n = 0
        for cond, text, mx in (inputs if inputs is not None else self.inputs(doc, family)):
            fixed = self.tc.count(SYSTEM) + self.tc.count(target) + self.tc.count(extra_block) + 260 + 3 * CHAT_OVERHEAD
            budget = MAX_TOKENS - fixed
            truncated = False
            if self.tc.count(text) > budget:
                if not allow_input_truncation or budget < 300:
                    self.drops[f"{family}:{cond}:over_cap"] += 1
                    continue
                text, truncated = truncate_lines(text, budget, self.tc)
            row = make_row(family, cond, doc, user_prompt(family, cond, doc["script"], text, extra_block), target,
                           {**meta_extra, **mx, "input_truncated": truncated}, self.tc)
            if row["meta"]["tokens"] > MAX_TOKENS:
                self.drops[f"{family}:{cond}:over_cap_after_build"] += 1
                continue
            self.rows.append(row)
            n += 1
        return n

    # ------------------------------------------------------------------ families
    def translate(self, doc: Dict[str, Any], prow: Dict[str, Any]) -> None:
        """translate_en / translate_he rows (whole document, or per side when over the cap).

        :param doc: Document info.
        :param prow: pgp_parallel row.
        """
        if doc["script"] not in ("hebrew", "arabic"):
            self.drops["translate:latin_or_mixed_edition"] += 1
            return
        for lang, family in (("en", "translate_en"), ("he", "translate_he")):
            tgt = prow.get(f"translation_{lang}")
            if not tgt:
                continue
            tr_entry = next((t for t in prow["translations"] if t.get("language") == lang and t.get("text") == tgt), None)
            if overlap60(tgt, doc["edition"]):
                self.drops[f"{family}:target_overlaps_edition"] += 1
                continue
            ed_n, tr_n = len(_norm(doc["edition"])), len(_norm(tgt))
            ratio = tr_n / max(ed_n, 1)
            if ratio < 0.3 or ratio > 5:
                self.drops[f"{family}:length_ratio"] += 1
                continue
            meta = {"source": f"pgp_digital_translation:{tr_entry['source_slug'] if tr_entry else '?'}",
                    "licence": PGP_LICENCE,
                    "provenance": {"edition_slug": doc["edition_slug"],
                                   "translation_slug": tr_entry["source_slug"] if tr_entry else None,
                                   "translation_location": tr_entry.get("location") if tr_entry else None},
                    "target_lang": lang, "side": None}
            whole = self.tc.count(doc["edition"]) + self.tc.count(tgt) + self.tc.count(SYSTEM) + 300
            if whole <= MAX_TOKENS:
                self.add(family, doc, tgt, meta, allow_input_truncation=False)
                continue
            # over the cap: split by side labels present in both raw footnotes
            ed_entry = next((e for e in prow["editions"] if e["source_slug"] == doc["edition_slug"]), None)
            ed_sides = split_sides(ed_entry["raw"]) if ed_entry else []
            tr_sides = split_sides(tr_entry["raw"]) if tr_entry else []
            if not ed_sides or [k for k, _ in ed_sides] != [k for k, _ in tr_sides]:
                if ed_sides and tr_sides and sorted(k for k, _ in ed_sides) == sorted(k for k, _ in tr_sides):
                    tr_map = dict(tr_sides)
                    tr_sides = [(k, tr_map[k]) for k, _ in ed_sides]
                else:
                    self.drops[f"{family}:over_cap_no_side_alignment"] += 1
                    continue
            added = 0
            for (k, ed_t), (_, tr_t) in zip(ed_sides, tr_sides):
                ed_t = strip_latin_lines(ed_t)[0]
                sdoc = {**doc, "edition": ed_t, "side": k, "reads_ok": []}  # reads are not side-aligned
                n_ed, n_tr = len(_norm(ed_t)), len(_norm(tr_t))
                if n_ed < 40 or n_tr < 40 or not (0.3 <= n_tr / n_ed <= 5):
                    self.drops[f"{family}:side_ratio"] += 1
                    continue
                added += self.add(family, sdoc, tr_t, {**meta, "side": k}, allow_input_truncation=False)
            self.drops[f"{family}:side_split_docs"] += 1
            if not added:
                self.drops[f"{family}:side_split_no_rows"] += 1

    def catalogue_families(self, doc: Dict[str, Any], prow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """catalogue_json, describe_pgp and extract_relations rows.

        :param doc: Document info.
        :param prow: Normalised document fields (type, tags, places, languages, description).
        :returns: The catalogue record (for analysis context), or None when the description is unusable.
        """
        if doc.get("edition") and doc["script"] not in ("hebrew", "arabic"):
            self.drops["catalogue:latin_or_mixed_edition"] += 1
            return None
        summary, acts = clean_description(prow["description"])
        self.desc_actions.update(acts)
        if len(summary) < 40:
            self.drops["catalogue:description_too_short"] += 1
            return None
        dcsv = self.docs_csv.get(doc["pgpid"], {})
        cat = catalogue_record(dcsv, prow, summary, self.people_ix)
        base = {"licence": PGP_LICENCE, "provenance": {"pgp_document": doc["pgpid"],
                                                       "edition_slug": doc.get("edition_slug")}}
        variants = self.inputs(doc, "catalogue")
        if not variants:
            return cat
        self.add("catalogue_json", doc, json.dumps(cat, ensure_ascii=False),
                 {**base, "source": "pgp_catalogue_fields"}, inputs=variants)
        if len(summary) >= 80:
            self.add("describe_pgp", doc, summary, {**base, "source": "pgp_description_cleaned"},
                     inputs=self.inputs(doc, "describe_pgp"))
        rel = relations_from_catalogue(cat, prow)
        if rel:
            self.add("extract_relations", doc, json.dumps({"relations": rel}, ensure_ascii=False),
                     {**base, "source": "pgp_gold_places+description_people", "n_relations": len(rel)},
                     inputs=self.inputs(doc, "extract_relations"))
        return cat


# ============================================================================ document assembly
def read_variants(prow: Dict[str, Any], edition: str) -> List[Dict[str, Any]]:
    """Reads of an edition document that pass the read gate (after loop cleaning), best CER first.

    :param prow: pgp_parallel row.
    :param edition: The cleaned edition text the CER is measured against.
    :returns: ``[{model, text, cer, coverage, agreed_share}]``.
    """
    ed = _norm(edition)
    out = []
    for rd in prow.get("reads", []):
        txt, removed = clean_read(rd["text"])
        rn = _norm(txt)
        if not ed or not rn:
            continue
        cov = len(rn) / len(ed)
        cer = Levenshtein.distance(rn, ed) / len(ed)
        STATS[f"read_{rd['model']}_candidates"] += 1
        g = READ_GATE
        if not (g["coverage_min"] <= cov <= g["coverage_max"] and cer <= g["cer_max"]):
            STATS[f"read_{rd['model']}_gate_fail"] += 1
            continue
        n_lines = sum(p.get("n_lines") or 0 for p in rd["pages"])
        n_agr = sum(p.get("n_agreed") or 0 for p in rd["pages"])
        out.append({"model": rd["model"], "text": txt, "cer": round(cer, 4), "coverage": round(cov, 3),
                    "agreed_share": round(n_agr / n_lines, 3) if n_lines else None, "loop_lines_removed": removed})
    out.sort(key=lambda r: r["cer"])
    if len(out) == 2 and Levenshtein.ratio(out[0]["text"], out[1]["text"]) > 0.9:
        out = out[:1]
    return out


def normalise_fields_from_parallel(prow: Dict[str, Any]) -> Dict[str, Any]:
    """Catalogue-relevant fields of a pgp_parallel row.

    :param prow: pgp_parallel row.
    :returns: Field dict.
    """
    return {"type": prow["type"], "tags": prow["tags"], "origin": prow["origin"], "destination": prow["destination"],
            "mentioned": prow["mentioned"], "possibly_mentioned": prow["possibly_mentioned"],
            "languages_primary": [x.strip() for x in (prow["languages_primary"] or "").split(",") if x.strip()],
            "languages_secondary": [x.strip() for x in (prow["languages_secondary"] or "").split(",") if x.strip()],
            "description": prow["description"]}


def normalise_fields_from_csv(d: Dict[str, str]) -> Dict[str, Any]:
    """Catalogue-relevant fields of a documents.csv row (read-only pool).

    :param d: documents.csv row.
    :returns: Field dict.
    """
    def lst(v: str) -> List[str]:
        """Split a comma-list.

        :param v: Value.
        :returns: Items.
        """
        return split_multi(v or "")
    return {"type": d["type"], "tags": lst(d["tags"]), "origin": d["origin"].strip(), "destination": d["destination"].strip(),
            "mentioned": lst(d["mentioned"]), "possibly_mentioned": lst(d["possibly_mentioned"]),
            "languages_primary": lst(d["languages_primary"]), "languages_secondary": lst(d["languages_secondary"]),
            "description": d["description"]}


def read_only_pool(merged: Dict[str, Dict[str, List[str]]], edition_pgpids: Set[str], docs_csv: Dict[str, Dict[str, str]],
                   read_files: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """PGP documents without a digital edition whose fragment(s) carry only that document and have a read.

    :param merged: canonical id -> pgpids / image urls.
    :param edition_pgpids: pgpids that have a digital edition (excluded here).
    :param docs_csv: documents.csv rows.
    :param read_files: reader -> ``ai_reads_*.jsonl`` (``a50`` and ``v21b``).
    :returns: pgpid -> ``{"read": {...}, "cids": [...]}``.
    """
    pgp_to_canon: Dict[str, List[str]] = collections.defaultdict(list)
    url_to_id: Dict[str, str] = {}
    for cid, info in merged.items():
        for p in info["pgpids"]:
            pgp_to_canon[p].append(cid)
        for url in info["image_urls"]:
            url_to_id.setdefault(url, cid)
    reads = {m: load_reads(read_files[m], set(merged), url_to_id, m) for m in ("a50", "v21b")}
    out: Dict[str, Dict[str, Any]] = {}
    for p, d in docs_csv.items():
        if p in edition_pgpids or len(d.get("description") or "") < 80:
            continue
        cids = pgp_to_canon.get(p, [])
        if not cids or any(merged[c]["pgpids"] != [p] for c in cids):
            continue
        best = None
        for model in ("a50", "v21b"):
            pages, seen = [], set()
            for cid in cids:
                for idx in sorted(reads[model].get(cid, {})):
                    pg = reads[model][cid][idx]
                    if pg["image_sha256"] in seen or not pg["text"]:
                        continue
                    seen.add(pg["image_sha256"])
                    pages.append(pg)
            if not pages:
                continue
            txt, removed = clean_read("\n\n".join(pg["text"] for pg in pages))
            n_lines = sum(pg.get("n_lines") or 0 for pg in pages)
            share = sum(pg.get("n_agreed") or 0 for pg in pages) / n_lines if n_lines else 0.0
            if len(txt) < READ_ONLY_GATE["chars_min"] or share < READ_ONLY_GATE["agreed_share_min"]:
                continue
            cand = {"model": model, "text": txt, "cer": None, "coverage": None, "agreed_share": round(share, 3),
                    "loop_lines_removed": removed}
            if best is None or share > best["agreed_share"]:
                best = cand
        if best:
            out[p] = {"read": best, "cids": cids}
    return out


def script_of(text: str) -> str:
    """Dominant script of an input text.

    :param text: Text.
    :returns: ``hebrew`` / ``arabic`` / ``latin`` / ``mixed``.
    """
    he = len(re.findall(r"[א-ת]", text))
    ar = len(re.findall(r"[ء-ي]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    tot = he + ar + la
    if not tot:
        return "mixed"
    for name, n in (("hebrew", he), ("arabic", ar), ("latin", la)):
        if n / tot >= 0.6:
            return name
    return "mixed"


# ============================================================================ analysis family
def build_analysis_rows(b: Builder, spans: List[Dict[str, Any]], docinfo: Dict[str, Dict[str, Any]],
                        cats: Dict[str, Dict[str, Any]], scholar_names: Set[str]) -> Dict[str, Any]:
    """analyse_document rows from the document-treatment spans that pass the grounding test.

    Tier A: input + catalogue context; tier B (widened): + the book's own translation of the document.

    :param b: Builder.
    :param spans: Treatment rows with full sections.
    :param docinfo: pgpid -> document info (edition docs only).
    :param cats: pgpid -> catalogue record.
    :param scholar_names: Folded scholar surnames.
    :returns: Report dict (counts, grounding distribution, per-book pass counts).
    """
    rep: Dict[str, Any] = collections.defaultdict(int)
    per_book: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    shares: List[float] = []
    seen_pgpids: Set[str] = set()
    for sp in spans:
        book = sp["book"]
        if not sp["boundary_clean"] or sp["intro_chars"] + sp["notes_chars"] < 300:
            rep["not_a_target"] += 1
            continue
        if book in ANALYSIS_EXCLUDED_BOOKS:
            rep[f"excluded_book:{book}"] += 1
            per_book[book]["excluded_book"] += 1
            continue
        per_book[book]["targets"] += 1
        with_ed = [p for p in sp["pgpids"] if p in docinfo]
        if len(with_ed) != 1:
            rep["no_edition" if not with_ed else "several_pgpids_with_edition"] += 1
            per_book[book]["no_single_edition"] += 1
            continue
        pgpid = with_ed[0]
        if pgpid in seen_pgpids:
            rep["duplicate_pgpid"] += 1
            continue
        doc = docinfo[pgpid]
        ed_grams = {doc["edition"][i:i + 12] for i in range(0, max(len(doc["edition"]) - 12, 0))}

        def not_copied(lines: List[str]) -> List[str]:
            """Lines that do not copy the edition (12-gram share < 0.3).

            :param lines: Section lines.
            :returns: Kept lines.
            """
            keep = []
            for ln in lines:
                grams = [ln[i:i + 12] for i in range(0, max(len(ln) - 12, 0))]
                if grams and sum(g in ed_grams for g in grams) / len(grams) >= 0.3:
                    rep["target_lines_copying_edition"] += 1
                    continue
                keep.append(ln)
            return keep
        intro, c1 = clean_commentary(not_copied(sp["sections"].get("intro", [])))
        notes, c2 = clean_commentary(not_copied(sp["sections"].get("notes", [])))
        rep["clean_actions"] = rep.get("clean_actions", 0) + sum((c1 + c2).values())
        target = (intro + ("\n\nNotes\n\n" + notes if notes else "")).strip()
        if len(target) < 300:
            rep["target_short_after_cleaning"] += 1
            continue
        cat = cats.get(pgpid)
        ctx = ("Known catalogue context:\n" + context_block(cat)) if cat else ""
        if ctx and overlap60(target, ctx):
            # the PGP description copies the book's commentary: keep only the structured fields
            rep["context_description_withheld_overlap"] += 1
            ctx = "Known catalogue context:\n" + context_block({**cat, "summary": None})
        book_tr, _ = clean_commentary(sp["sections"].get("translation", []))
        src_a = doc["edition"] + "\n" + ctx
        share_a, n_c, miss_a = grounding(target, src_a, scholar_names)
        tier, extra, share = None, ctx, share_a
        if share_a >= GROUNDING_PASS:
            tier = "A"
        elif len(book_tr) >= 150:
            share_b, _, miss_b = grounding(target, src_a + "\n" + book_tr, scholar_names)
            share = share_b
            if share_b >= GROUNDING_PASS and not overlap60(target, book_tr):
                tier = "B"
                extra = (ctx + "\n\n" if ctx else "") + "Translation (from the edition):\n" + book_tr
        shares.append(share)
        if not tier:
            rep["grounding_fail"] += 1
            per_book[book]["grounding_fail"] += 1
            if per_book[book]["grounding_fail"] <= 2:
                rep.setdefault("fail_examples", []).append({"book": book, "label": sp["doc_label"], "pgpid": pgpid,
                                                            "share": round(share, 2), "missing": miss_a[:8]})
            continue
        rep[f"pass_tier_{tier}"] += 1
        per_book[book][f"pass_{tier}"] += 1
        # truncate notes from the end when over the cap
        t_target, truncated = target, False
        budget = MAX_TOKENS - b.tc.count(SYSTEM) - b.tc.count(doc["edition"]) - b.tc.count(extra) - 300
        if b.tc.count(t_target) > budget:
            paras = target.split("\n\n")
            while len(paras) > 1 and b.tc.count("\n\n".join(paras)) > budget:
                paras.pop()
                truncated = True
            t_target = "\n\n".join(paras)
            if b.tc.count(t_target) > budget:
                rep["over_cap_dropped"] += 1
                continue
        meta = {"source": f"book:{book}", "licence": LICENCE_BY_BOOK.get(book, "unknown"),
                "provenance": {"book": book, "part": sp["part"], "doc_label": sp["doc_label"], "pages": sp["pages"],
                               "printed_pages": sp["printed_pages"], "edition_slug": doc["edition_slug"]},
                "grounding_share": round(share, 3), "grounding_candidates": n_c, "grounding_tier": tier,
                "target_truncated": truncated, "target_lang": "he" if script_of(t_target) == "hebrew" else "en"}
        n = b.add("analyse_document", doc, t_target, meta, extra_block=extra)
        per_book[book]["rows"] += n
        seen_pgpids.add(pgpid)
    rep["grounding_share_quantiles"] = ([round(q, 2) for q in statistics.quantiles(shares, n=10)]
                                        if len(shares) >= 10 else shares)
    rep["grounding_share_ge"] = {str(t): sum(s >= t for s in shares) for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    rep["evaluated"] = len(shares)
    rep["per_book"] = {k: dict(v) for k, v in per_book.items()}
    return dict(rep)


# ============================================================================ checks and stats
def sanity_checks(rows: List[Dict[str, Any]], held_pgpids: Set[str], held_cids: Set[str],
                  cids_by_pgpid: Dict[str, List[str]]) -> Dict[str, Any]:
    """Split leakage, hold-out, target-in-input and parse checks.

    :param rows: All rows.
    :param held_pgpids: Benchmark pgpids.
    :param held_cids: Benchmark canonical ids.
    :param cids_by_pgpid: pgpid -> canonical ids.
    :returns: Check results (each ``*_violations`` should be 0).
    """
    split_of_pgpid: Dict[str, Set[str]] = collections.defaultdict(set)
    split_of_group: Dict[str, Set[str]] = collections.defaultdict(set)
    bench_in_train = target_in_input = parse_fail = quoted = 0
    examples: List[str] = []
    for r in rows:
        m = r["meta"]
        split_of_pgpid[m["pgpid"]].add(m["split"])
        split_of_group[m["shelfmark_group"]].add(m["split"])
        if m["split"] == "train" and (m["pgpid"] in held_pgpids or any(c in held_cids for c in cids_by_pgpid.get(m["pgpid"], []))):
            bench_in_train += 1
        line = json.dumps(r, ensure_ascii=False)
        back = json.loads(line)
        if [x["role"] for x in back["messages"]] != ["system", "user", "assistant"] or not back["messages"][2]["content"]:
            parse_fail += 1
        if m["task"] in ("catalogue_json", "extract_relations"):
            json.loads(back["messages"][2]["content"])
        tgt = re.sub(r"\s+", " ", r["messages"][2]["content"])
        inp = re.sub(r"\s+", " ", r["messages"][1]["content"])
        if m["task"] in ("catalogue_json", "extract_relations"):
            tgt = re.sub(r"\s+", " ", json.loads(r["messages"][2]["content"]).get("summary") or "") if m["task"] == "catalogue_json" else ""
        hit = None
        if len(tgt) >= 60:
            for i in range(0, len(tgt) - 60, 30):
                if tgt[i:i + 60] in inp:
                    hit = tgt[i:i + 60]
                    break
        if hit is not None:
            rtl_share = len(_RTL.findall(hit)) / max(len(re.findall(r"\w", hit)), 1)
            if rtl_share >= 0.5:
                quoted += 1  # the human target quotes the document (a description citing a word or phrase)
            else:
                target_in_input += 1
                if len(examples) < 5:
                    examples.append(f"{m['task']}:{m['pgpid']}:{m['condition']}:{hit[:40]}")
    return {
        "pgpid_in_two_splits_violations": sum(1 for v in split_of_pgpid.values() if len(v) > 1),
        "group_in_two_splits_violations": sum(1 for v in split_of_group.values() if len(v) > 1),
        "benchmark_in_train_violations": bench_in_train,
        "target_window60_in_input_violations": target_in_input,
        "target_in_input_examples": examples,
        "target_quotes_document_rtl_window_in_input (allowed)": quoted,
        "parse_failures": parse_fail,
        "rows_checked": len(rows),
    }


def write_stats(rows: List[Dict[str, Any]], extra: Dict[str, Any], tc: TokenCounter, out_dir: Path) -> Dict[str, Any]:
    """Write STATS.md (volume table, length distribution, examples) and return headline numbers.

    :param rows: All rows.
    :param extra: Extra report blocks.
    :param tc: Token counter (for the exactness note).
    :param out_dir: Directory of ``STATS.md``.
    :returns: Headline numbers.
    """
    vol: Dict[Tuple[str, str, str], Dict[str, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        m = r["meta"]
        vol[(m["task"], m["lang_class"], m["condition"])][m["split"]].append(m["tokens"])
    fam_tot: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: collections.Counter())
    for (f, _l, _c), d in vol.items():
        for s, toks in d.items():
            fam_tot[f][f"{s}_rows"] += len(toks)
            fam_tot[f][f"{s}_tokens"] += sum(toks)
    L = ["# text_ft_v0: volume and examples", "",
         f"Built {time.strftime('%Y-%m-%d %H:%M')} by `build_text_ft_v0.py` (seed {SEED}). Token counts: "
         f"{'exact, Qwen3.8-27B tokenizer.json (local), + ' + str(CHAT_OVERHEAD) + ' per message' if tc.exact else 'estimated chars/3.2 Latin, chars/2.5 Hebrew/Arabic'}. "
         f"Cap {MAX_TOKENS} tokens per row (system + user + assistant).", "",
         "## Totals per family", "",
         "| family | train rows | train tokens | dev rows | dev tokens | test rows | test tokens |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FAMILIES:
        t = fam_tot.get(f, {})
        L.append(f"| {f} | {t.get('train_rows', 0):,} | {t.get('train_tokens', 0):,} | {t.get('dev_rows', 0):,} | "
                 f"{t.get('dev_tokens', 0):,} | {t.get('test_rows', 0):,} | {t.get('test_tokens', 0):,} |")
    tot = collections.Counter()
    for t in fam_tot.values():
        tot.update(t)
    L.append(f"| **all** | {tot['train_rows']:,} | {tot['train_tokens']:,} | {tot['dev_rows']:,} | {tot['dev_tokens']:,} | "
             f"{tot['test_rows']:,} | {tot['test_tokens']:,} |")
    L += ["", "## Rows per family × language × condition", "",
          "Tokens are train tokens. `read` rows on edition documents carry the measured CER in `meta.input_cer`; "
          "read-only-pool rows (no edition) have `input_cer = null`.", "",
          "| family | lang | condition | train | dev | test | train tokens | median tokens/row |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(vol, key=lambda k: (FAMILIES.index(k[0]), k[1], k[2])):
        d = vol[key]
        allt = [x for v in d.values() for x in v]
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(d.get('train', [])):,} | {len(d.get('dev', [])):,} | "
                 f"{len(d.get('test', [])):,} | {sum(d.get('train', [])):,} | {int(statistics.median(allt)) if allt else 0:,} |")
    toks = sorted(r["meta"]["tokens"] for r in rows)
    qs = [toks[int(len(toks) * q) - 1] for q in (0.5, 0.9, 0.95, 0.99)] + [toks[-1]]
    L += ["", "## Sequence length (all rows, tokens)", "",
          f"p50 {qs[0]:,} · p90 {qs[1]:,} · p95 {qs[2]:,} · p99 {qs[3]:,} · max {qs[4]:,}.", ""]
    L += ["| family | p50 | p90 | p99 | max | rows with input truncated | analysis targets truncated |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FAMILIES:
        ft = sorted(r["meta"]["tokens"] for r in rows if r["meta"]["task"] == f)
        if not ft:
            continue
        L.append(f"| {f} | {ft[len(ft) // 2]:,} | {ft[int(len(ft) * 0.9) - 1]:,} | {ft[int(len(ft) * 0.99) - 1]:,} | {ft[-1]:,} | "
                 f"{sum(1 for r in rows if r['meta']['task'] == f and r['meta'].get('input_truncated'))} | "
                 f"{sum(1 for r in rows if r['meta']['task'] == f and r['meta'].get('target_truncated'))} |")
    rc = [r["meta"]["input_cer"] for r in rows if r["meta"]["condition"] == "read" and r["meta"]["input_cer"] is not None]
    nc = [r["meta"]["input_cer"] for r in rows if r["meta"]["condition"] == "noised_edition"]
    if rc and nc:
        L += ["", "## Input CER of the noisy conditions", "",
              f"- read (edition documents, after loop cleaning and the gate CER ≤ {READ_GATE['cer_max']}, coverage "
              f"{READ_GATE['coverage_min']}–{READ_GATE['coverage_max']}): n = {len(rc):,}, quartiles "
              f"{', '.join(f'{q:.3f}' for q in statistics.quantiles(rc, n=4))}",
              f"- noised_edition (target CER sampled U{NOISE_CER_RANGE}): n = {len(nc):,}, measured quartiles "
              f"{', '.join(f'{q:.3f}' for q in statistics.quantiles(nc, n=4))}"]
    L += ["", "## Build report", "", "```json", json.dumps(extra, ensure_ascii=False, indent=1, default=str)[:12000], "```", "",
          "## Examples (2 per family, each message truncated to 500 characters)", ""]
    rng = random.Random(SEED)
    for f in FAMILIES:
        cand = [r for r in rows if r["meta"]["task"] == f and r["meta"]["split"] == "train"] or \
               [r for r in rows if r["meta"]["task"] == f]
        if not cand:
            continue
        L.append(f"### {f}")
        picks = rng.sample(cand, min(2, len(cand)))
        for r in picks:
            m = r["meta"]
            L += ["", f"`{m['condition']}` · pgpid {m['pgpid']} · {m['lang_class']} · {m['split']} · source `{m['source']}` · "
                      f"tokens {m['tokens']} · input_cer {m['input_cer']}", "",
                  "**user**", "", "```", r["messages"][1]["content"][:500], "```", "",
                  "**assistant**", "", "```", r["messages"][2]["content"][:500], "```", ""]
    with open(out_dir / "STATS.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return {"families": {f: dict(v) for f, v in fam_tot.items()}, "total": dict(tot),
            "length_quantiles_p50_p90_p95_p99_max": qs}


# ============================================================================ main
def select_rows(items: Sequence[Any], key: Callable[[Any], str], limit: Optional[int] = None,
                pgpids: Optional[Set[str]] = None) -> List[Any]:
    """Restrict a document list for a sample build (order kept).

    :param items: Pairs rows or ``(pgpid, info)`` pool items, in build order.
    :param key: pgpid of an item.
    :param limit: Keep at most this many items (after the ``pgpids`` filter); None keeps all.
    :param pgpids: Keep only these documents; None keeps all.
    :returns: The selected items.
    """
    out = [it for it in items if pgpids is None or key(it) in pgpids]
    return out if limit is None else out[:limit]


def run_v0(cfg: V0Config, limit: Optional[int] = None, analysis: bool = True,
           pgpids: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Build every v0 row in memory (the prototype's ``main`` up to the write step).

    Groups, splits, the noise model and its calibration are always computed over the whole input, so a
    sample build (``limit`` / ``pgpids``) yields exactly the rows the full build gives those documents.

    :param cfg: Build configuration.
    :param limit: Process at most this many edition documents and this many read-only-pool documents.
    :param analysis: Build ``analyse_document`` (imports the research span detector, see :func:`load_full_spans`).
    :param pgpids: Process only these documents.
    :returns: ``{"rows", "builder", "noise_model", "tc", "docinfo", "cats", "groups", "gsplit", "held_pgpids",
        "held_cids", "cids_by_pgpid", "analysis", "pool_docs"}``.
    """
    STATS.clear()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    load_place_names(cfg.pgp_dir)
    tc = TokenCounter(cfg.tokenizer)
    log(f"tokenizer exact={tc.exact}")
    prows = load_jsonl(cfg.pairs)
    log(f"pgp_parallel rows {len(prows):,}")
    docs_csv = {r["pgpid"]: r for r in load_csv(cfg.pgp_dir / "documents.csv")}
    merged = build_merged_map(cfg.merged_map, cfg.merged_jsonl)
    held_pgpids, held_cids = held_out_pgpids(merged, cfg.benchmark_ids, cfg.registry_fn)
    probes = probe_pgpids(cfg.samples)
    STATS["probe_pgpids"] = len(probes)
    groups = build_groups(merged, list(docs_csv) + [r["pgpid"] for r in prows], cfg.pgp_dir / "fragments.csv")
    gsplit = group_splits(groups, held_pgpids | probes | {r["pgpid"] for r in prows if r["held_out_vision_benchmark"]})
    cids_by_pgpid: Dict[str, List[str]] = collections.defaultdict(list)
    for cid, info in merged.items():
        for p in info["pgpids"]:
            cids_by_pgpid[p].append(cid)

    # ---- noise model
    read_texts = [clean_read(rd["text"])[0] for r in prows for rd in r.get("reads", [])]
    nm = build_noise_model(prows, read_texts, cfg.noise_model_path)
    log("noise model: " + json.dumps({s: {k: v[k] for k in ("n_pairs", "cer_empirical", "rate_sub", "rate_del", "rate_ins")}
                                      for s, v in nm["scripts"].items()}))
    noiser = Noiser(nm)
    calib_pool: Dict[str, List[str]] = collections.defaultdict(list)
    for r in prows:
        if r["edition_script"] in ("hebrew", "arabic") and split_of(r["pgpid"]) == "train" and len(r["edition_text"]) >= 300:
            calib_pool[r["edition_script"]].append(r["edition_text"])
    STATS["noise_calibration_measured_over_target"] = {k: round(v, 3) for k, v in noiser.calibrate(calib_pool).items()}
    log(f"noise calibration {STATS['noise_calibration_measured_over_target']}")
    people_ix = PeopleIndex(cfg.pgp_dir / "people.csv")
    b = Builder(tc, noiser, people_ix, docs_csv)

    # ---- edition documents
    docinfo: Dict[str, Dict[str, Any]] = {}
    cats: Dict[str, Dict[str, Any]] = {}
    for prow in select_rows(prows, lambda r: r["pgpid"], limit, pgpids):
        p = prow["pgpid"]
        held = prow["held_out_vision_benchmark"] or p in held_pgpids or any(c in held_cids for c in prow["merged_doc_ids"])
        g = groups.get(p, "g" + p)
        split = gsplit.get(g, "test")
        if held and split == "train":
            raise AssertionError(f"held-out {p} in train")
        edition, n_lat = strip_latin_lines(prow["edition_text"])
        STATS["edition_latin_lines_dropped"] += n_lat
        STATS["editions_with_latin_lines_dropped"] += bool(n_lat)
        doc = {"pgpid": p, "shelfmark": prow["shelfmark"], "group": g, "split": split, "held_out": bool(held),
               "lang": coarse_lang(prow["lang_class"], prow.get("ja_score") or 0), "lang_fine": prow["lang_class"],
               "edition": edition, "edition_slug": prow["edition_source_slug"],
               "script": prow["edition_script"], "reads_ok": read_variants(prow, edition)}
        docinfo[p] = doc
        b.translate(doc, prow)
        cat = b.catalogue_families(doc, normalise_fields_from_parallel(prow))
        if cat:
            cats[p] = cat
    log(f"edition docs done: rows {len(b.rows):,}")

    # ---- read-only pool (catalogue families only)
    pool = read_only_pool(merged, {r["pgpid"] for r in prows}, docs_csv, cfg.read_files)
    STATS["read_only_pool_docs"] = len(pool)
    pool_items = select_rows(sorted(pool.items(), key=lambda kv: int(kv[0])), lambda kv: kv[0], limit, pgpids)
    for p, info in pool_items:
        d = docs_csv[p]
        held = p in held_pgpids or any(c in held_cids for c in info["cids"])
        g = groups.get(p, "g" + p)
        split = gsplit.get(g, "test")
        if held and split == "train":
            raise AssertionError(f"held-out {p} in train")
        doc = {"pgpid": p, "shelfmark": d["shelfmark"], "group": g, "split": split, "held_out": bool(held),
               "lang": lang_from_pgp(d["languages_primary"]), "lang_fine": d["languages_primary"], "edition": None,
               "edition_slug": None, "script": script_of(info["read"]["text"]), "reads_ok": [info["read"]]}
        b.catalogue_families(doc, normalise_fields_from_csv(d))
    log(f"read-only pool {len(pool):,} docs ({len(pool_items):,} processed); rows {len(b.rows):,}")

    # ---- analysis
    arep: Dict[str, Any] = {"skipped": True}
    if analysis:
        spans = load_full_spans(cfg.treatments, cfg.probe_dir)
        scholar_names = tn.scholar_surnames(cfg.pgp_dir / "sources.csv", cfg.scholar_extra)
        arep = build_analysis_rows(b, spans, docinfo, cats, scholar_names)
        log(f"analysis: {json.dumps({k: v for k, v in arep.items() if k.startswith('pass') or k in ('grounding_fail', 'evaluated')})}")
    return {"rows": b.rows, "builder": b, "noise_model": nm, "tc": tc, "docinfo": docinfo, "cats": cats, "groups": groups,
            "gsplit": gsplit, "held_pgpids": held_pgpids, "held_cids": held_cids, "cids_by_pgpid": cids_by_pgpid,
            "analysis": arep, "pool_docs": len(pool_items)}


def write_v0(result: Dict[str, Any], cfg: V0Config) -> Dict[str, Any]:
    """Write ``{train,dev,test}.jsonl``, ``STATS.md`` and ``build_report.json`` of a :func:`run_v0` result.

    :param result: Output of :func:`run_v0`.
    :param cfg: Build configuration (``out_dir``).
    :returns: The build report.
    """
    out_dir = cfg.out_dir
    rows, b, tc = result["rows"], result["builder"], result["tc"]
    order = {f: i for i, f in enumerate(FAMILIES)}
    rows.sort(key=lambda r: (r["meta"]["split"], order[r["meta"]["task"]], int(r["meta"]["pgpid"]), r["meta"]["condition"],
                             str(r["meta"].get("side")), r["meta"].get("read_model") or ""))
    for split in ("train", "dev", "test"):
        part = [r for r in rows if r["meta"]["split"] == split]
        random.Random(stable_int(SEED, split)).shuffle(part)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    reread = [r for s in ("train", "dev", "test") for r in load_jsonl(out_dir / f"{s}.jsonl")]
    checks = sanity_checks(reread, result["held_pgpids"], result["held_cids"], result["cids_by_pgpid"])
    log("checks " + json.dumps(checks))
    people_stats: collections.Counter = collections.Counter()
    for c in result["cats"].values():
        for pp in c["people"]:
            people_stats[pp["role"]] += 1
        people_stats["docs_with_people"] += bool(c["people"])
        people_stats["docs_with_date"] += bool(c["date"])
    nm = result["noise_model"]
    report = {"stats": dict(STATS), "drops": dict(b.drops), "description_cleaning": dict(b.desc_actions),
              "people_in_catalogue_records_edition_docs": dict(people_stats), "analysis": result["analysis"], "checks": checks,
              "groups": {"n_groups": len(set(result["groups"].values())),
                         "split_counts": dict(collections.Counter(result["gsplit"].values()))},
              "noise_model_summary": {s: {k: v[k] for k in ("n_pairs", "edition_chars", "cer_empirical", "rate_sub", "rate_del", "rate_ins")}
                                      for s, v in nm["scripts"].items()}}
    head = write_stats(reread, report, tc, out_dir)
    report["headline"] = head
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    log("headline " + json.dumps(head))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: reproduce the v0 build (``{train,dev,test}.jsonl``, ``noise_model.json``, ``STATS.md``, ``build_report.json``).

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: None.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", type=Path, default=None, help="paths.json (default: $TEXT27B_PATHS or the repo default)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default <out_dir>/v0_1_repro)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N edition docs and N pool docs")
    ap.add_argument("--no-analysis", action="store_true", help="skip analyse_document (no research span detector)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = _paths.load_paths(args.paths)
    local = load_local_config(p)
    apply_local_config(local)
    cfg = V0Config.from_paths(p, args.out_dir, local)
    result = run_v0(cfg, limit=args.limit, analysis=not args.no_analysis)
    write_v0(result, cfg)


if __name__ == "__main__":
    main()
