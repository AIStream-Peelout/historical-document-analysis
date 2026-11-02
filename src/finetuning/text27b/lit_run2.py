# File name: lit_run2.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Run-2 literature mining: every training signal in the scholarship ingested on 2026-10-27, per pipeline.

Reads the literature pages (:func:`pages.iter_pages`, OCR text, never the structurer's model-written fields as
targets), cuts documents with the segmenters of :mod:`lit_segment` and the agent-located spans files (D15), and
writes CANDIDATES -- nothing here touches a training mixture, an eval manifest, the Hub, LM Studio or a paid API.

Pipelines (one JSONL each under the run directory; rows in the :mod:`rows` schema where a builder exists):

``translate_en`` / ``translate_he``
    Built by the existing L5 code path (:func:`book_pairs.build` + :func:`book_pairs.write_outputs` into the run
    directory, never ``paths.out_dir``) from three span sources: the numbered-anthology translations (input = the
    PGP digital edition, ``input: pgp_edition``), the reconstructed-formulary pairs of a doc-headed dissertation
    (both sides in the work) and the agent-located spans files named in the local config. The L5 gates apply
    unchanged: F1 length band, pairing check, independence screen against PGP translations, ``dup_pgp_translation``,
    hold-outs (benchmark / eval tier → ``heldout``).
``vlm_pages_hebrew_ja`` / ``vlm_pages_arabic``
    Human edition text per document side, in the ``book_editions`` entry layout the page-row builder
    (``qwen_hebrew.build_book_edition_rows``) reads (``transcription_lines`` of ``{n, side, part, text}``), with the
    resolved canonical ids and the served index's image count / URLs (read-only lookup). Images are not copied.
``catalogue_descriptions``
    Catalogue descriptions tied to a shelfmark (doc-headed catalogue part, numbered catalogues), with parsed
    document type / date / stated line count, and a ``describe_pgp``-schema row when an edition input exists.
``analysis_commentary``
    Scholar commentary tied to a shelfmark: anthology introductions, web posts about a fragment (anchored by the
    October spans file's ``commentary_docs``, else the title shelfmark), and the located discussion passages of the
    literature index (:func:`rag_l3.run_build_index` over the new works). An ``analyse_document`` row is rendered
    when PGP holds an edition of the anchor document; L5 analysis rows of the anthology are written separately.

Every candidate carries ``provenance`` (work, part, page keys and printed pages, extraction method, licence),
``resolution`` (shelfmark raw → variant → canonical ids → pgpids, :func:`resolve`), ``held_out`` / ``held_out_reason``
(:class:`holdouts.Holdouts`: benchmark + eval tiers by id / key, plus held documents NAMED in the candidate's texts,
:func:`contamination.text_mention_reason`), ``split`` and ``quality`` (score in [0, 1] plus flags).

No holdings, hosts or index names are written here: works, page ranges, licences and the image index come from the
local, git-ignored config (``paths.json`` key ``run2_config``, default
``logs/run2_data_2026-10-27/run2_local.json``). Elasticsearch credentials are read from the environment (loaded
from the repository ``.env`` inside the process) and never printed.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.lit_run2 [--config FILE] [--no-es] [--no-book-pairs]
        [--rebuild-index]

Public API: :func:`load_config`, :class:`Resolver`, :func:`neubauer_crosswalk`, :class:`ImageIndex`,
:func:`quality_flags_edition`, :func:`doc_headed_records`, :func:`formulary_source`, :func:`anthology_source`,
:func:`span_vlm_entries`, :func:`web_edition_entries`, :func:`generic_edition_entries`, :func:`catalogue_records`,
:func:`web_commentary`, :func:`index_passages`, :func:`render_describe_row`, :func:`render_analyse_row`,
:func:`run`, :func:`main`, :func:`doc_numbers`, :func:`pgp_doc_crosswalk`, :func:`pgp_fragment_cids`, :func:`doc_hints`,
:func:`text_layer_dir`, :func:`layer_page_text`, :func:`apply_text_layers`.

Resolution order of one printed shelfmark (:meth:`Resolver.match_one`): legacy spellings (:func:`lit_segment.legacy_variants`)
against the merged corpus → PGP's historic shelfmarks → the lettered pieces of the named box / folio (a numbered piece
``no. N`` is anchored to them, never mapped by position). PGP-id hints per document number (PGP footnotes citing the work
with ``Doc. N``; a reviewed hints file) apply where no shelfmark resolves. Per-work config keys: ``text_layer`` (works
entry: read ``<stem>_structured_<layer>`` instead of the OCR text), ``pgp_source_slug``, ``shelfmark_fixes`` (printed
shelfmarks the OCR misread, with the evidence) and ``doc_hints_file`` (doc-headed entries).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import book_pairs as _bp
from src.finetuning.text27b import builder as _bd
from src.finetuning.text27b import lit_segment as ls
from src.finetuning.text27b import pages as _pg
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("logs") / "run2_data_2026-10-27" / "run2_local.json"
PIPELINE_FILES: Tuple[str, ...] = ("translate_en", "translate_he", "vlm_pages_hebrew_ja", "vlm_pages_arabic",
                                   "catalogue_descriptions", "analysis_commentary")
#: Matches with more PGP ids than this are too broad to anchor a row (a whole box, a bare volume number).
BROAD_PGPIDS = 6
#: Commentary targets shorter than this are not kept.
COMMENTARY_MIN_CHARS = 300
#: Longest commentary / description target kept (characters).
TARGET_MAX_CHARS = 9000


# ===================================================================================== config
def load_config(path: Optional[Path] = None, p: Optional[_paths.Paths] = None) -> Dict[str, Any]:
    """Read the local run-2 config.

    :param path: Explicit config file.
    :param p: Build paths (``run2_config`` entry, else :data:`DEFAULT_CONFIG` under the repository).
    :returns: The config dict (``_path`` holds the file it came from).
    """
    p = p or _paths.load_paths()
    cfg_path = Path(path) if path else Path(p.get("run2_config") or (p.repo / DEFAULT_CONFIG))
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_path"] = str(cfg_path)
    return cfg


def _sha16(*parts: Any) -> str:
    """Short stable hash of the parts.

    :param parts: Values.
    :returns: 16 hex characters.
    """
    return hashlib.sha1("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()[:16]


# ===================================================================================== resolution
def neubauer_crosswalk(texts: Iterable[str]) -> Dict[str, str]:
    """Neubauer catalogue number → ``MS Heb. <letter> <number>`` from headings that print both.

    :param texts: Texts with headings such as ``Bodl. MS Heb. b 12 (Cat. 2875), f. 10``.
    :returns: ``{"2875": "MS Heb. b 12"}`` (a number seen with two different volumes is dropped).
    """
    seen: Dict[str, Set[str]] = collections.defaultdict(set)
    for t in texts:
        for m in re.finditer(r"MS\.?\s*Heb\.?\s*([a-f])\.?\s*(\d{1,3})\s*\(Cat\.?\s*(\d{4})\)", t or ""):
            seen[m.group(3)].add(f"MS Heb. {m.group(1)} {m.group(2)}")
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def doc_numbers(location: str) -> List[str]:
    """Document numbers a PGP footnote location names (``Doc. 14``, ``Docs. 90, 91``, ``doc. 159-160``,
    ``docs. 67 and 136``; page references after the numbers are ignored).

    :param location: Footnote ``location`` text.
    :returns: Numbers as strings (ranges expanded, at most 20 per range).
    """
    m = re.match(r"^\s*docs?\.?\s*((?:\d+\s*(?:[–-]\s*\d+)?\s*(?:,|and|&)?\s*)+)", location or "", re.I)
    if not m:
        return []
    out: List[str] = []
    for a, b in re.findall(r"(\d+)(?:\s*[–-]\s*(\d+))?", m.group(1)):
        lo, hi = int(a), int(b) if b else int(a)
        if lo <= hi <= lo + 20:
            out += [str(x) for x in range(lo, hi + 1)]
    return out


def pgp_doc_crosswalk(fn_rows: Iterable[Mapping[str, Any]], slug: str) -> Dict[str, List[str]]:
    """Document number of a work → PGP ids, from PGP's footnotes that cite the work with a ``Doc. N`` location.

    :param fn_rows: PGP footnote rows (``Corpus.fn_rows``: ``pgpid``, ``slug``, ``location``).
    :param slug: PGP source slug of the work.
    :returns: ``{"241": ["26455"], ...}``.
    """
    out: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in fn_rows:
        if r.get("slug") != slug:
            continue
        for n in doc_numbers(str(r.get("location") or "")):
            out[n].add(str(r["pgpid"]))
    return {k: sorted(v, key=lambda x: int(x) if x.isdigit() else 0) for k, v in out.items()}


def pgp_fragment_cids(corpus: Any, pgpids: Sequence[str]) -> List[str]:
    """Canonical ids synthesised from the PGP fragment shelfmarks of PGP ids that have no merged record (for an
    image lookup in the served index by id; the index carries no PGP-id field).

    :param corpus: :class:`shelfmarks.Corpus` (``fragments``).
    :param pgpids: PGP ids.
    :returns: Ids (institution token + normalised core), possibly empty.
    """
    want = {str(p) for p in pgpids}
    out: Set[str] = set()
    for fr in getattr(corpus, "fragments", None) or []:
        if want & set(fr.get("pgpids") or []):
            s = _sm.prep_shelfmark(fr["shelfmark"])
            tok, core = _sm.token_of(s), _sm.core_of(s)
            if tok and core:
                out.add(_sm.combine(tok, core))
    return sorted(out)


class Resolver:
    """Shelfmark → canonical ids → PGP ids, hold-outs and splits for the run-2 candidates."""

    def __init__(self, corpus: Any, holdouts: Any, ix: Any, neubauer: Optional[Mapping[str, str]] = None,
                 mention_reason: Optional[Callable[..., Optional[str]]] = None) -> None:
        """Hold the lookups.

        :param corpus: :class:`shelfmarks.Corpus`.
        :param holdouts: :class:`holdouts.Holdouts`.
        :param ix: :class:`book_pairs.PgpIndex` (editions, translations, v0_1 placement).
        :param neubauer: Neubauer crosswalk (:func:`neubauer_crosswalk`).
        :param mention_reason: ``(texts, holdouts) -> reason`` (default :func:`contamination.text_mention_reason`).
        """
        self.corpus = corpus
        self.holdouts = holdouts
        self.ix = ix
        self.neubauer = dict(neubauer or {})
        if mention_reason is None:
            from src.finetuning.text27b import contamination as _ct
            mention_reason = _ct.text_mention_reason
        self.mention_reason = mention_reason
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._historic: Optional[Dict[str, Set[str]]] = None
        self._lettered: Optional[Dict[str, Set[str]]] = None

    # ------------------------------------------------------------------ fallbacks
    def historic_index(self) -> Dict[str, Set[str]]:
        """Key of every historic spelling PGP records for a fragment → current fragment shelfmarks (built once from
        the corpus' ``fragments`` rows, ``shelfmarks_historic`` split on ``;``).

        :returns: ``{key: {current shelfmark}}``.
        """
        if self._historic is None:
            idx: Dict[str, Set[str]] = collections.defaultdict(set)
            for fr in getattr(self.corpus, "fragments", None) or []:
                for h in str(fr.get("historic") or "").split(";"):
                    for k in _sm.keys(h.strip()) if h.strip() else []:
                        idx[k].add(fr["shelfmark"])
            self._historic = dict(idx)
        return self._historic

    def lettered_index(self) -> Dict[str, Set[str]]:
        """Corpus core of a box / folio → cores of its lettered pieces (``T_S_NS_J185`` → ``T_S_NS_J185a`` ...).

        :returns: ``{base core: {piece core}}``.
        """
        if self._lettered is None:
            idx: Dict[str, Set[str]] = collections.defaultdict(set)
            for core in getattr(self.corpus, "core_idx", None) or {}:
                m = re.match(r"^(.*\d)[A-Za-z]$", core)
                if m:
                    idx[m.group(1)].add(core)
            self._lettered = dict(idx)
        return self._lettered

    def historic_match(self, raw: str, variants: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Match a printed shelfmark through PGP's historic shelfmarks (``TS NS J 401, no. 7`` → historic
        ``T-S NS J401.7`` → current ``T-S NS J401a``). A key naming more than one current fragment, or one of another
        institution, is not used.

        :param raw: Printed shelfmark.
        :param variants: Its legacy variants.
        :returns: Match dict (``how`` = ``historic:<how>``, ``historic_current``) or None.
        """
        idx = self.historic_index()
        for i, v in enumerate(variants):
            tok_v = _sm.token_of(_sm.prep_shelfmark(v))
            for k in _sm.keys(v):
                cur = sorted(idx.get(k) or ())
                if len(cur) != 1:
                    continue
                tok_c = _sm.token_of(_sm.prep_shelfmark(cur[0]))
                if tok_v and tok_c and tok_v != tok_c:
                    continue
                m = _sm.match_shelfmark(cur[0], self.corpus)
                if m["cids"] or m["pgpids"]:
                    return {"raw": raw, "variant": v, "how": f"historic:{m['how']}", "cids": list(m["cids"]),
                            "pgpids": list(m["pgpids"]), "variant_index": i, "historic_current": cur[0]}
        return None

    def piece_match(self, raw: str) -> Optional[Dict[str, Any]]:
        """Lettered pieces of the mark a printed shelfmark names.

        A bare box / folio whose pieces carry letters in the corpus (``TS NS J 303`` → ``T-S NS J303a``–``c``)
        resolves to all of them (``how = "pieces"``, ``ambiguous`` when they belong to more than one PGP document). A
        numbered piece (``TS NS J 185, no. 7``) is never mapped onto a letter by position (the letters do not follow
        the old numbers): it stays unresolved but ANCHORED to the base mark's pieces (``how = "piece_anchor"``,
        ``piece``, ``anchor_cids``, ``anchor_pgpids``; ``cids`` / ``pgpids`` empty).

        :param raw: Printed shelfmark.
        :returns: Match dict or None.
        """
        po = ls.piece_of(raw)
        base_raw, piece = po if po else (raw, None)
        lettered = self.lettered_index()
        for i, v in enumerate(ls.legacy_variants(base_raw, self.neubauer)):
            core = _sm.core_of(_sm.prep_shelfmark(v))
            pieces = sorted(lettered.get(core) or ())
            cids: Set[str] = {c for pc in pieces for c in self.corpus.core_idx.get(pc, ())}
            if piece:
                base = _sm.match_shelfmark(v, self.corpus)
                cids |= set(base["cids"])
            if not cids:
                continue
            pgp = sorted({p for c in cids for p in (self.corpus.merged.get(c) or {}).get("pgpids", [])},
                         key=lambda x: int(x) if str(x).isdigit() else 0)
            out = {"raw": raw, "variant": v, "variant_index": i, "pieces": [p.split("_")[-1] for p in pieces]}
            if piece:
                out.update({"how": "piece_anchor", "piece": piece, "cids": [], "pgpids": [],
                            "anchor_cids": sorted(cids), "anchor_pgpids": pgp})
            else:
                out.update({"how": "pieces", "cids": sorted(cids), "pgpids": pgp, "ambiguous": len(pgp) > 1})
            return out
        return None

    def match_one(self, raw: str) -> Dict[str, Any]:
        """Resolve one printed shelfmark: its legacy variants against the corpus, then PGP's historic shelfmarks
        (:meth:`historic_match`), then the lettered pieces of the named mark (:meth:`piece_match`).

        :param raw: Shelfmark as printed.
        :returns: ``{"raw", "variant", "how", "cids", "pgpids", "variant_index"}`` (empty ids when unmatched; extra
            keys ``historic_current`` / ``pieces``, ``ambiguous`` / ``piece``, ``anchor_cids``, ``anchor_pgpids``).
        """
        if raw in self._cache:
            return self._cache[raw]
        best: Dict[str, Any] = {"raw": raw, "variant": None, "how": "unmatched", "cids": [], "pgpids": [],
                                "variant_index": None}
        variants = ls.legacy_variants(raw, self.neubauer)
        for i, v in enumerate(variants):
            m = _sm.match_shelfmark(v, self.corpus)
            if m["cids"] or m["pgpids"]:
                best = {"raw": raw, "variant": v, "how": m["how"], "cids": list(m["cids"]), "pgpids": list(m["pgpids"]),
                        "variant_index": i}
                break
        else:
            best = self.historic_match(raw, variants) or self.piece_match(raw) or best
        self._cache[raw] = best
        return best

    def resolve(self, shelfmarks: Sequence[str], pgpid_hints: Sequence[str] = (),
                hint_source: str = "agent") -> Dict[str, Any]:
        """Resolve every shelfmark of a document.

        PGP-id hints (an agent's record, or a scholarship crosswalk such as PGP's ``Doc. N`` footnote locations of the
        work) are used when no shelfmark resolves to a specific document (unmatched or only anchored to a box's
        pieces), and narrow an ambiguous lettered-piece match to the hinted documents; a hint that contradicts a
        resolved shelfmark is recorded (``hint_disagrees``) but not applied.

        :param shelfmarks: Printed shelfmarks (joins as separate entries).
        :param pgpid_hints: PGP ids recorded for the document elsewhere.
        :param hint_source: Where the hints come from (recorded as ``hint_source``).
        :returns: ``{"shelfmarks_raw", "matched", "canonical_ids", "pgpids", "broad", "resolved", "via_hint",
            "ambiguous", "anchors", "hint_source", "hint_disagrees"}``.
        """
        matched = [self.match_one(part) for s in shelfmarks if s for part in ls.split_joins(s)]
        cids = sorted({c for m in matched for c in m["cids"]})
        pgpids = sorted({x for m in matched for x in m["pgpids"]}, key=lambda x: int(x) if str(x).isdigit() else 0)
        anchors = [{"raw": m["raw"], "piece": m.get("piece"), "anchor_cids": m.get("anchor_cids") or [],
                    "anchor_pgpids": m.get("anchor_pgpids") or []} for m in matched if m["how"] == "piece_anchor"]
        ambiguous = any(m.get("ambiguous") for m in matched)
        hints = sorted({str(x) for x in pgpid_hints or ()}, key=lambda x: int(x) if x.isdigit() else 0)
        via_hint, disagrees = False, False
        if hints and not (cids or pgpids):
            pgpids = hints
            cids = sorted({c for x in pgpids for c in self.corpus.pgp_to_cids.get(x, set())})
            via_hint = True
        elif hints and ambiguous and set(hints) & set(pgpids):
            keep = set(hints) & set(pgpids)
            pgpids = sorted(keep, key=lambda x: int(x) if x.isdigit() else 0)
            cids = sorted({c for x in pgpids for c in self.corpus.pgp_to_cids.get(x, set())})
            via_hint, ambiguous = True, False
        elif hints and not set(hints) & set(pgpids):
            disagrees = True
        anchor_pg = {p for a in anchors for p in a["anchor_pgpids"]}
        return {"shelfmarks_raw": [s for s in shelfmarks if s], "matched": matched, "canonical_ids": cids,
                "pgpids": pgpids, "broad": len(pgpids) > BROAD_PGPIDS, "resolved": bool(cids or pgpids),
                "via_hint": via_hint, "ambiguous": ambiguous, "anchors": anchors,
                "hint_source": hint_source if hints else None, "hint_disagrees": disagrees,
                "hint_within_anchor": bool(via_hint and anchor_pg and set(pgpids) <= anchor_pg)}

    def holdout(self, res: Mapping[str, Any], texts: Sequence[Optional[str]] = ()) -> Tuple[Optional[str], Optional[str]]:
        """Hold-out reason of a resolved document and of the held documents its texts name.

        :param res: Output of :meth:`resolve`.
        :param texts: Texts that would enter a row (target, input).
        :returns: ``(document reason, mention reason)``; either may be None.
        """
        shelf = list(res.get("shelfmarks_raw") or []) + [m["variant"] for m in res.get("matched") or [] if m.get("variant")]
        doc = self.holdouts.match_any(pgpids=[] if res.get("broad") else res.get("pgpids") or [],
                                      canonicals=[] if res.get("broad") else res.get("canonical_ids") or [],
                                      shelfmarks=shelf, include_eval=True)
        mention = self.mention_reason([t for t in texts if t], self.holdouts) if any(texts) else None
        return doc, mention

    def split(self, res: Mapping[str, Any], doc_reason: Optional[str], mention: Optional[str], key: str) -> Tuple[str, str]:
        """Split of a candidate: ``heldout`` for any hold-out reason, ``train`` when v0_1 trains the PGP id, else the
        D14 hash split of the document key.

        :param res: Output of :meth:`resolve`.
        :param doc_reason: Document hold-out reason.
        :param mention: Mention hold-out reason.
        :param key: Document key for the hash split.
        :returns: ``(split, split_reason)``.
        """
        if doc_reason:
            return "heldout", "document"
        if mention:
            return "heldout", "span_mention"
        if not res.get("broad"):
            for pid in res.get("pgpids") or []:
                if self.ix is not None and self.ix.v01_split(pid) == "train":
                    return "train", "v0_1_train_group"
        keys = [k for s in res.get("shelfmarks_raw") or [] for k in _sm.keys(s)]
        return _bp.hash_split(keys[0] if keys else key), "hash"

    def annotate(self, rec: Dict[str, Any], shelfmarks: Sequence[str], texts: Sequence[Optional[str]], key: str,
                 pgpid_hints: Sequence[str] = (), hint_source: str = "agent") -> Dict[str, Any]:
        """Add ``resolution``, ``held_out``, ``held_out_reason``, ``split`` and ``split_reason`` to a candidate.

        :param rec: Candidate record (updated in place).
        :param shelfmarks: Printed shelfmarks.
        :param texts: Texts entering a row.
        :param key: Fallback document key.
        :param pgpid_hints: PGP ids recorded for the document elsewhere (:meth:`resolve`).
        :param hint_source: Where the hints come from.
        :returns: The record.
        """
        res = self.resolve(shelfmarks, pgpid_hints, hint_source)
        if not res["canonical_ids"] and res["pgpids"] and not res["broad"]:
            res["canonical_ids_synth"] = pgp_fragment_cids(self.corpus, res["pgpids"])
        doc_r, men_r = self.holdout(res, texts)
        split, why = self.split(res, doc_r, men_r, key)
        rec.update({"resolution": res, "held_out": bool(doc_r or men_r),
                    "held_out_reason": (f"document:{doc_r}" if doc_r else (f"span_mention:{men_r}" if men_r else None)),
                    "split": split, "split_reason": why})
        return rec

    def pgp_edition(self, pgpids: Sequence[str]) -> Optional[Tuple[str, str, str]]:
        """PGP digital edition of the first of the PGP ids that has one.

        :param pgpids: PGP ids.
        :returns: ``(edition text, edition slug, pgpid)`` or None.
        """
        for pid in pgpids:
            d = self.ix.docs.get(str(pid)) if self.ix is not None else None
            if d and d.get("edition_text"):
                return tn.strip_latin_lines(d["edition_text"])[0], str(d.get("edition_source_slug")), str(pid)
        return None


# ===================================================================================== images
class ImageIndex:
    """Read-only image lookup in the served fragment index (``_id`` = canonical id), cached on disk, with an optional
    map of locally downloaded images (source ``cudl_local``) for ids whose served record has none."""

    FIELDS = ("image_urls", "has_images", "image_preferred_source")
    LOCAL_SOURCE = "cudl_local"

    def __init__(self, index: Optional[str], cache: Path, enabled: bool = True,
                 local_images: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None) -> None:
        """Set up the lookup.

        :param index: Index name (from the local config).
        :param cache: JSON cache file (``{cid: {"found", "n_images", "source", "image_urls"}}``).
        :param enabled: False answers from the cache only.
        :param local_images: ``{cid: [{"path", "canvas_label", "manifest_id"}, ...]}`` of downloaded images
            (:meth:`load_local`); used only for ids whose served record has no images.
        """
        self.index = index
        self.cache_path = Path(cache)
        self.enabled = enabled and bool(index)
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            self.data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.local: Dict[str, List[Dict[str, Any]]] = {c: [dict(x) for x in v] for c, v in (local_images or {}).items() if v}

    @staticmethod
    def load_local(path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
        """Read a local-images map (``cache/local_images.json`` written by the image download).

        :param path: JSON file, or None.
        :returns: ``{cid: [{"path", "canvas_label", "manifest_id"}, ...]}`` (empty when ``path`` is None).
        """
        if path is None:
            return {}
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def record(self, cid: str) -> Optional[Dict[str, Any]]:
        """Effective image record of one id: the served-index record, unless it is missing or has no images and
        local images exist (then a ``cudl_local`` record listing them).

        :param cid: Canonical id.
        :returns: ``{"found", "n_images", "source", "image_urls"[, "local_images"]}`` or None (never looked up).
        """
        v = self.data.get(cid)
        loc = self.local.get(cid)
        if loc and not (v and v.get("n_images")):
            return {"found": True, "n_images": len(loc), "source": self.LOCAL_SOURCE, "image_urls": [],
                    "local_images": loc}
        return v

    def n_images(self, cid: str) -> int:
        """Number of images of one id (served index, else local).

        :param cid: Canonical id.
        :returns: Count (0 when unknown).
        """
        return int((self.record(cid) or {}).get("n_images") or 0)

    def _client(self) -> Any:
        """Elasticsearch client from the environment (``.env`` loaded inside the process; nothing printed).

        :returns: Client.
        """
        from dotenv import load_dotenv
        from elasticsearch import Elasticsearch
        from src.datasets.indexing.elastic_index_genizah import es_config_from_env
        load_dotenv(_paths.repo_root() / ".env")
        return Elasticsearch(**es_config_from_env(), request_timeout=120)

    def fetch(self, cids: Iterable[str], batch: int = 400) -> int:
        """Look up the ids not yet cached (``mget``; reads only).

        :param cids: Canonical ids.
        :param batch: Ids per request.
        :returns: Number of ids looked up.
        """
        todo = sorted({c for c in cids if c and c not in self.data})
        if not todo or not self.enabled:
            return 0
        es = self._client()
        for i in range(0, len(todo), batch):
            got = es.mget(index=self.index, ids=todo[i:i + batch], source_includes=list(self.FIELDS))
            for d in got["docs"]:
                src = d.get("_source") or {}
                urls = src.get("image_urls") or []
                urls = urls if isinstance(urls, list) else [urls]
                self.data[d["_id"]] = {"found": bool(d.get("found")), "n_images": len(urls),
                                       "source": src.get("image_preferred_source"), "image_urls": urls}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        return len(todo)

    def summary(self, cids: Sequence[str]) -> Dict[str, Any]:
        """Image availability of a document's canonical ids.

        An id with no served images but with local images counts as found / with images under source ``cudl_local``;
        its ``per_id`` entry lists the local files (``local_images``: path, canvas label, manifest id).

        :param cids: Canonical ids.
        :returns: ``{"looked_up", "found", "with_images", "n_images", "sources", "per_id"}``.
        """
        per = {c: self.record(c) for c in cids}
        known = {c: v for c, v in per.items() if v is not None}
        out_per: Dict[str, Dict[str, Any]] = {}
        for c, v in known.items():
            if v["found"]:
                out_per[c] = {"n_images": v["n_images"], "source": v.get("source"), "image_urls": v["image_urls"]}
                if v.get("local_images"):
                    out_per[c]["local_images"] = v["local_images"]
        return {"looked_up": len(known), "found": sum(1 for v in known.values() if v["found"]),
                "with_images": sum(1 for v in known.values() if v["n_images"]),
                "n_images": sum(v["n_images"] for v in known.values()),
                "sources": sorted({str(v["source"]) for v in known.values() if v.get("source")}),
                "per_id": out_per}


# ===================================================================================== page access
def text_layer_dir(page: Any, layer: str) -> Path:
    """Directory of an alternative text layer next to a page's structuring directory
    (``<part dir>/<stem>_structured_<layer>``).

    :param page: PageRec.
    :param layer: Layer name (``gemini_lines``).
    :returns: The directory path (it may not exist).
    """
    comps = page.structuring_dir.split("/")
    base = Path(page.path).parents[len(comps)]
    stem = re.sub(r"_structured.*$", "", comps[0])
    return base / f"{stem}_structured_{layer}"


def layer_page_text(data: Mapping[str, Any]) -> Optional[str]:
    """Text of one page of an alternative text layer: ``full_main_text``, else ``lines`` (strings or dicts with
    ``text``) joined by newlines.

    :param data: Page JSON of the layer.
    :returns: Text or None when the page carries none.
    """
    t = data.get("full_main_text")
    if isinstance(t, str) and t.strip():
        return t
    lines = data.get("lines")
    if isinstance(lines, list) and lines:
        out = [x.get("text", "") if isinstance(x, dict) else str(x) for x in lines]
        return "\n".join(out) if any(x.strip() for x in out) else None
    return None


def apply_text_layers(pages: Sequence[Any], cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Replace the OCR text of a work part's pages by an alternative text layer when the config asks for one
    (``works[<work>/<part>]["text_layer"] = "<layer>"``) and ``<stem>_structured_<layer>`` exists; otherwise the
    pages keep the current OCR text. A replaced page keeps its original text in ``ocr_text_raw`` and gets
    ``ocr_source = "text_layer:<layer>"``; its page JSON (footnotes, printed page) is the one already loaded.

    :param pages: PageRecs (changed in place).
    :param cfg: Run config.
    :returns: ``{work_part: {"layer", "dir", "exists", "pages_replaced"}}`` for every part that names a layer.
    """
    want = {wp: w["text_layer"] for wp, w in (cfg.get("works") or {}).items() if isinstance(w, dict) and w.get("text_layer")}
    out: Dict[str, Dict[str, Any]] = {}
    if not want:
        return out
    by_part: Dict[str, List[Any]] = collections.defaultdict(list)
    for pg in pages:
        by_part[f"{pg.work}/{pg.part.split('/')[0]}"].append(pg)
    for wp, layer in want.items():
        pgs = by_part.get(wp) or []
        d = text_layer_dir(pgs[0], layer) if pgs else None
        info = {"layer": layer, "dir": str(d) if d else None, "exists": bool(d and d.is_dir()), "pages_replaced": 0}
        if info["exists"]:
            texts: Dict[int, str] = {}
            for f in sorted(d.glob("page_*_structured.json")):
                data = json.loads(f.read_text(encoding="utf-8"))
                md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                try:
                    no = int(md.get("page_number"))
                except (TypeError, ValueError):
                    no = _pg.page_no_of(f.name)
                t = layer_page_text(data)
                if t is not None:
                    texts[no] = t
            for pg in pgs:
                if pg.page_no in texts:
                    pg.ocr_text_raw, pg.ocr_text = pg.ocr_text_raw or pg.ocr_text, texts[pg.page_no]
                    pg.ocr_source = f"text_layer:{layer}"
                    info["pages_replaced"] += 1
        else:
            LOG.info("text layer %s of %s not found; current OCR text kept", layer, wp)
        out[wp] = info
    return out


def page_inputs(pages: Sequence[Any], lo: int, hi: int) -> List[ls.PageInput]:
    """Segmenter inputs for the pages of one part within a PDF page range.

    :param pages: :class:`pages.PageRec` of one part, in order.
    :param lo: First PDF page.
    :param hi: Last PDF page (inclusive).
    :returns: Page inputs (OCR text, footnote start line and footnote lines from the page JSON).
    """
    out = []
    for p in pages:
        if not (lo <= p.page_no <= hi) or not p.ocr_text:
            continue
        t = p.ocr_text
        fs = _pg.footnote_region_start(t, p.data)
        fl = t[:fs].count("\n") if fs is not None else None
        notes = frozenset(ls.footnote_line_set(t.split("\n"), _pg.footnote_texts(p.data)))
        out.append(ls.PageInput(p.page_key, p.page_no, t, fl, notes))
    return out


def printed_pages(pages_by_key: Mapping[str, Any], keys: Sequence[str]) -> List[Any]:
    """Printed page numbers of page keys (PDF page number when the printed one is unknown).

    :param pages_by_key: page key → PageRec.
    :param keys: Page keys.
    :returns: Printed page labels in order.
    """
    out = []
    for k in keys:
        p = pages_by_key.get(k)
        if p is None:
            continue
        out.append(p.printed_page_num if p.printed_page_num is not None else f"pdf{p.page_no}")
    return out


# ===================================================================================== quality
def quality_flags_edition(lines: Sequence[str], stats: Mapping[str, int], lines_stated: Optional[int] = None
                          ) -> Tuple[float, List[str]]:
    """Quality score and flags of an edition text extracted from OCR.

    :param lines: Kept edition lines.
    :param stats: :func:`lit_segment.clean_edition_lines` stats.
    :param lines_stated: Line count the scholar states for the document, if known.
    :returns: ``(score in [0, 1], flags)``.
    """
    flags: List[str] = []
    score = 1.0
    text = "\n".join(lines)
    he, ar, lat = ls.letters(text)
    if he + ar < 80:
        flags.append("short_edition")
        score -= 0.3
    rev = ls.reversed_hebrew_share(text)
    if rev is not None and rev > 0.3:
        flags.append(f"reversed_hebrew:{rev}")
        score -= 0.6
    raw = max(1, stats.get("raw", 0))
    debris = (stats.get("debris", 0) + stats.get("numeric", 0)) / raw
    if debris > 0.4:
        flags.append(f"ocr_debris_share:{debris:.2f}")
        score -= 0.2
    short = stats.get("short", 0) / max(1, stats.get("kept", 0))
    if short > 0.15:
        flags.append(f"fragmented_lines:{short:.2f}")
        score -= 0.2
    paren_edges = sum(1 for ln in lines if re.match(r"^[)\]]", ln) or re.search(r"[(\[]$", ln))
    if lines and paren_edges / len(lines) > 0.3:
        flags.append("bracket_mirroring")
        score -= 0.1
    if lines_stated:
        ratio = len(lines) / lines_stated
        if not 0.75 <= ratio <= 1.35:
            flags.append(f"line_count_ratio:{ratio:.2f}")
            score -= 0.3
    return round(max(0.0, score), 2), flags


def script_of(text: str) -> str:
    """``hebrew`` / ``arabic`` / ``mixed`` by letter counts.

    :param text: Text.
    :returns: Script name.
    """
    return tn.dominant_script(text)


# ===================================================================================== doc-headed works
def doc_headed_records(pages: Sequence[Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Catalogue and edition sections of a doc-headed work, joined by document number.

    :param pages: PageRecs of the work's part, in order.
    :param spec: Local config entry (``catalogue_pages``, ``edition_pages``).
    :returns: ``{"catalogue": [Section], "edition": [Section], "by_number": {no: {"catalogue", "edition"}}}``.
    """
    lo, hi = spec["catalogue_pages"]
    cat = ls.segment_doc_headed(page_inputs(pages, lo, hi), header_scan=12, whole_page=True)
    lo, hi = spec["edition_pages"]
    ed = ls.segment_doc_headed(page_inputs(pages, lo, hi), header_scan=25, whole_page=True, footnotes=False)
    by_no: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    for role, secs in (("catalogue", cat), ("edition", ed)):
        for s in secs:
            for no in s.label.replace("RF", "RF:").split(","):
                no = no if not no.startswith("RF:") else "RF" + no[3:]
                by_no[no].setdefault(role, s)
    return {"catalogue": cat, "edition": ed, "by_number": dict(by_no)}


def description_text(sec: ls.Section) -> Tuple[str, List[str]]:
    """Catalogue description of a section: stamp, page-number and heading lines removed, wrapped lines joined.

    :param sec: Catalogue section.
    :returns: ``(text, ops)``.
    """
    ops: List[str] = []
    lines = []
    for ln in sec.text.split("\n"):
        s = ln.strip()
        if not s or ls.is_stamp_line(s):
            continue
        if re.fullmatch(r"[\divxlc]{1,4}", s):
            ops.append("drop_page_number")
            continue
        lines.append(s)
    text, joins = _bp.prose_paragraphs("\n".join(lines))
    if joins:
        ops.append("prose_paragraphs")
    text2 = re.sub(r"%3B|%(?=[;,.\s])", "", text)
    if text2 != text:
        ops.append("drop_percent_artifacts")
    return text2.strip(), sorted(set(ops))


def formulary_source(pages_by_key: Mapping[str, Any], recs: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Spans-file source (book_pairs format) for the reconstructed formularies of a doc-headed work: edition from
    the edition part, English translation from the catalogue part, both cut by line range (verbatim pieces).

    :param pages_by_key: page key → PageRec.
    :param recs: :func:`doc_headed_records` output.
    :param spec: Local config entry (``slug``, ``work``, licence fields).
    :returns: Source dict.
    """
    docs = []
    for no, pair in sorted(recs["by_number"].items()):
        if not no.startswith("RF") or "catalogue" not in pair or "edition" not in pair:
            continue

        def pieces(sec: ls.Section, rtl_only: bool = False) -> List[Dict[str, Any]]:
            """Pieces of a section's ranges (edition: split around lines without Hebrew / Arabic letters, which are
            OCR debris and line-number columns on the typescript pages).

            :param sec: Section.
            :param rtl_only: Keep only lines with at least two Hebrew / Arabic letters.
            :returns: Piece dicts.
            """
            out = []
            for k, a, b in sec.ranges:
                text = pages_by_key[k].ocr_text
                lines = text.split("\n")
                if not rtl_only:
                    out.append(_bp.piece_from_lines(k, text, a, b).to_dict())
                    continue
                skip = [i for i in range(a, b + 1) if sum(ls.letters(lines[i])[:2]) < 2]
                for x, y in ls._ranges_without(lines, a, b, skip):
                    out.append(_bp.piece_from_lines(k, text, x, y).to_dict())
            return out

        docs.append({"doc": no, "shelfmarks": [], "pgpids": [], "lang_class": "JA",
                     "spans": {"edition": {"pieces": pieces(pair["edition"], rtl_only=True), "normalise": ["drop_line_numbers",
                                                                                            "drop_digit_remnants"]},
                               "translation": {"pieces": pieces(pair["catalogue"]),
                                               "normalise": ["drop_bare_numbers", "drop_footnote_refs",
                                                             "prose_paragraphs"]}},
                     "note": "reconstructed formulary (composite of several deeds; no single shelfmark)"})
    return {"slug": f"{spec['slug']}_formularies", "work": spec["work"], "target_lang": "en", "input": "span",
            "licence": spec.get("licence"), "licence_class": spec.get("licence_class"), "pgp_slugs": [],
            "located": "programmatic: Rec. Form. headers of the edition and catalogue parts (lit_segment)", "docs": docs}


# ===================================================================================== anthology works
def page_note_numbers(page: Any) -> Set[str]:
    """Footnote numbers of a page: digits in the page JSON's footnote keys and the numbers that open lines in the
    lower part of the OCR text.

    :param page: PageRec.
    :returns: Numbers as strings.
    """
    out = {re.sub(r"\D", "", str(k)) for k in (page.data.get("footnotes") or {}) if re.search(r"\d", str(k))}
    lines = (page.ocr_text or "").split("\n")
    for ln in lines[int(len(lines) * 0.4):]:
        m = re.match(r"^\s*(\d{1,3})\s+\S", ln)
        if m:
            out.add(m.group(1))
    return {x for x in out if x}


def strip_note_digits(printed: str, notes: Set[str], resolver: "Resolver") -> Tuple[str, Optional[str]]:
    """Remove a footnote number the OCR glued to a shelfmark (``TS 13 J 20.457`` with note 57 on the page).

    The printed form is kept when it resolves (to a document; a piece anchor does not count); otherwise the last one
    or two digits (after an optional side letter) are dropped only when they are a footnote number of the same page
    and the remainder resolves. Last resort: one or two trailing digits that are the first or last digits of a longer
    note number of the page (the OCR kept one digit of a two-digit superscript: ``18 (1).336`` with note 66).

    :param printed: Shelfmark as printed.
    :param notes: Footnote numbers of the page (:func:`page_note_numbers`).
    :param resolver: Resolver.
    :returns: ``(shelfmark to use, flag or None)``.
    """
    def ok(sm: str) -> bool:
        """Whether a spelling resolves to a document.

        :param sm: Shelfmark.
        :returns: True when matched (not unmatched, not only anchored).
        """
        return resolver.match_one(sm)["how"] not in ("unmatched", "piece_anchor")

    if ok(printed):
        return printed, None
    m = re.match(r"^(.*?\d)([rv]?)(\d{1,3})$", printed)
    for k in (2, 1, 3):
        if len(printed) <= k or not printed[-k:].isdigit() or printed[-k:] not in notes:
            continue
        cand = printed[:-k].rstrip()
        if cand and ok(cand):
            return cand, f"footnote_digits_stripped:{printed[-k:]}"
    if m and m.group(2) and m.group(3) in notes:
        cand = m.group(1)
        if ok(cand):
            return cand, f"side_and_footnote_stripped:{m.group(2)}{m.group(3)}"
    for k in (1, 2):
        tail = printed[-k:] if len(printed) > k else ""
        if not tail.isdigit():
            continue
        longer = sorted(n for n in notes if len(n) > k and (n.startswith(tail) or n.endswith(tail)))
        cand = printed[:-k].rstrip()
        if longer and cand and re.search(r"\d$", cand) and ok(cand):
            return cand, f"footnote_digits_stripped_partial:{tail}~{longer[0]}"
    return printed, None



def anthology_source(pages_by_key: Mapping[str, Any], secs: Sequence[ls.Section], spec: Mapping[str, Any],
                     resolver: Resolver) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Spans-file source (``input: pgp_edition``) for a numbered anthology of translations, plus one record per
    entry for the inventory / commentary candidates.

    :param pages_by_key: page key → PageRec.
    :param secs: :func:`lit_segment.anthology_sections` output.
    :param spec: Local config entry (``slug``, ``work``, ``target_lang``, licence fields).
    :param resolver: Shelfmark resolver.
    :returns: ``(source, entry records)``.
    """
    docs, records = [], []
    for s in secs:
        for j, d in enumerate(s.extra.get("docs") or []):
            doc_id = s.label if j == 0 else f"{s.label}.{j + 1}"
            lk = d["line"][0]
            printed, note_flag = strip_note_digits(d["shelfmark"], page_note_numbers(pages_by_key[lk]), resolver)
            m = resolver.match_one(printed)
            if m["how"] in ("unmatched", "piece_anchor") and d.get("alt"):
                ma = resolver.match_one(d["alt"])
                if ma["how"] not in ("unmatched", "piece_anchor"):
                    printed, m = d["alt"], ma
                    note_flag = "printed_alternate_classmark" + (f";{note_flag}" if note_flag else "")
            d["shelfmark_used"], d["note_flag"] = printed, note_flag
            listy = bool(s.extra.get("title_is_shelfmark")) or ls.numbered_share(d["text"].split("\n")) > 0.4
            shelf = [m["variant"] or printed]
            pg_hint = [] if len(m["pgpids"]) > BROAD_PGPIDS else list(m["pgpids"])
            spans: Dict[str, Any] = {}
            if d["translation"]:
                spans["translation"] = {
                    "pieces": [_bp.piece_from_lines(k, pages_by_key[k].ocr_text, a, b).to_dict()
                               for k, a, b in d["translation"]],
                    "normalise": ["drop_footnote_refs"] + ([] if listy else ["prose_paragraphs"])}
            if j == 0 and s.extra.get("intro"):
                spans["intro"] = {"pieces": [_bp.piece_from_lines(k, pages_by_key[k].ocr_text, a, b).to_dict()
                                             for k, a, b in s.extra["intro"]],
                                  "normalise": ["drop_footnote_refs", "prose_paragraphs"]}
            entry = {"doc": doc_id, "shelfmarks": shelf, "pgpids": pg_hint, "lang_class": _lang_class(d.get("language")),
                     "spans": spans, "note": f"{s.extra.get('title', '')[:80]} | printed: {d['shelfmark']}"}
            if not spans:
                entry["not_placed"] = "no_translation_lines"
            docs.append(entry)
            records.append({"doc": doc_id, "title": s.extra.get("title"), "shelfmark_printed": printed,
                            "shelfmark_ocr": d["shelfmark"], "shelfmark_alt": d.get("alt"), "shelfmark_flag": note_flag,
                            "language": d.get("language"), "list_document": listy,
                            "intro_text": s.extra.get("intro_text") if j == 0 else "",
                            "translation_chars": len(d["text"]), "page_keys": list(dict.fromkeys(
                                [k for k, _, _ in (s.extra.get("intro") or [])] + [k for k, _, _ in d["translation"]]))})
    src = {"slug": spec["slug"], "work": spec["work"], "target_lang": spec.get("target_lang", "en"),
           "input": spec.get("input", "pgp_edition"), "licence": spec.get("licence"),
           "licence_class": spec.get("licence_class"), "pgp_slugs": list(spec.get("pgp_slugs") or []),
           "located": "programmatic: numbered entry headings, <shelfmark> (<language>) lines, running heads and "
                      "footnote lines removed (lit_segment.anthology_sections)", "docs": docs}
    return src, records


def _lang_class(language: Optional[str]) -> str:
    """Language class from an anthology language note.

    :param language: e.g. ``Judaeo-Arabic``, ``Hebrew``, ``Arabic``, ``Aramaic``.
    :returns: ``JA`` / ``HE`` / ``AR`` / ``ARAM`` (``JA`` when unknown).
    """
    s = (language or "").lower()
    if "judaeo" in s or "judeo" in s:
        return "JA"
    if "aramaic" in s:
        return "ARAM"
    if "hebrew" in s:
        return "HE"
    if "arabic" in s:
        return "AR"
    return "JA"


# ===================================================================================== VLM entries
def _entry(source: str, doc: str, shelfmarks: Sequence[str], sides: Sequence[Tuple[str, Sequence[str]]],
           method: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """A ``book_editions``-layout entry (what the page-row builder reads) for one document.

    :param source: Source slug.
    :param doc: Document label.
    :param shelfmarks: Printed shelfmarks.
    :param sides: ``[(side, lines)]``.
    :param method: Extraction method.
    :param provenance: Provenance fields.
    :returns: The entry dict (no resolution yet).
    """
    tl = []
    for side, lines in sides:
        for n, t in enumerate(lines, 1):
            tl.append({"n": n, "side": side, "part": None, "text": t})
    text = "\n".join(t["text"] for t in tl)
    return {"source": source, "doc": doc, "shelfmark": " + ".join(shelfmarks), "shelfmarks": list(shelfmarks),
            "side": sides[0][0] if len(sides) == 1 else "both", "script": script_of(text),
            "transcription_lines": tl, "extraction_method": method, "provenance": dict(provenance)}


def span_vlm_entries(spans: Mapping[str, Any], page_texts: Mapping[str, Optional[str]],
                     pages_by_key: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Edition entries from the ``edition_sides`` of agent-located spans files (D15 pieces, verified on cut).

    :param spans: Spans file content.
    :param page_texts: page key → OCR text.
    :param pages_by_key: page key → PageRec (printed pages).
    :returns: ``(entries, drop counter)``.
    """
    out: List[Dict[str, Any]] = []
    drops: collections.Counter = collections.Counter()
    for src in spans.get("sources") or []:
        for d in src.get("docs") or []:
            sides = []
            pages: List[str] = []
            for es in d.get("edition_sides") or []:
                raw, info = _bp.cut_span(es, page_texts)
                if raw is None:
                    drops[f"{src['slug']}:cut_failed"] += 1
                    continue
                text, _ = _bp.normalise(raw, es.get("normalise") or [])
                lines, _ = ls.clean_edition_lines(text.split("\n"))
                if lines:
                    sides.append((es.get("side") or "unknown", lines))
                    pages += info.get("pages") or []
            if not sides:
                drops[f"{src['slug']}:no_edition_sides"] += 1
                continue
            e = _entry(src["slug"], str(d["doc"]), d.get("shelfmarks") or [], sides, "agent_located_spans_D15",
                       {"work": src.get("work"), "work_slug": src.get("work_slug"), "page_keys": list(dict.fromkeys(pages)),
                        "printed_pages": printed_pages(pages_by_key, list(dict.fromkeys(pages))),
                        "licence": src.get("licence"), "licence_class": src.get("licence_class"),
                        "ocr_quality": d.get("ocr_quality"), "note": d.get("note")})
            e["pgpid_hints"] = [str(x) for x in d.get("pgpids") or []]
            out.append(e)
    return out, drops


def web_edition_entries(web_spans: Mapping[str, Any], root: Path, dirs_licence: Mapping[str, Any]
                        ) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Edition entries from the web items' verified edition spans (the October web-pairs spans file).

    :param web_spans: Web-pairs spans file content (``items``).
    :param root: Literature root (item texts live at ``<dir>/text/<slug>.txt``).
    :param dirs_licence: dir → licence string.
    :returns: ``(entries, drop counter)``.
    """
    out: List[Dict[str, Any]] = []
    drops: collections.Counter = collections.Counter()
    for it in web_spans.get("items") or []:
        key = f"{it['dir']}/text/{it['slug']}.txt"
        path = root / key
        if not path.exists():
            drops["item_text_missing"] += 1
            continue
        page_texts = {key: path.read_text(encoding="utf-8")}
        for d in it.get("docs") or []:
            span = (d.get("spans") or {}).get("edition")
            if not span:
                continue
            raw, info = _bp.cut_span(span, page_texts)
            if raw is None:
                drops["cut_failed"] += 1
                continue
            text, _ = _bp.normalise(raw, [op for op in span.get("normalise") or [] if op in _bp.NORMALISERS])
            lines, _ = ls.clean_edition_lines(text.split("\n"))
            if not lines:
                drops["no_lines"] += 1
                continue
            out.append(_entry(f"web:{it['dir']}", f"{it['slug']}:{d.get('doc')}", d.get("shelfmarks") or [],
                              [("unknown", lines)], "web_pairs_span_D15",
                              {"work": it["dir"], "work_slug": it["slug"], "page_keys": [key], "printed_pages": [],
                               "licence": dirs_licence.get(it["dir"]), "licence_class": None,
                               "note": d.get("note"), "coverage": "unknown (web transcriptions are often partial)"}))
    return out, drops


def generic_edition_entries(pages: Sequence[Any], index_rows: Sequence[Mapping[str, Any]], works: Sequence[str],
                            min_numbered: float = 0.3, max_dist: int = 12) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Low-confidence edition entries: numbered Hebrew / Arabic-script runs on pages where the literature index
    LOCATES a shelfmark at most ``max_dist`` lines above the run (the nearest one anchors it).

    :param pages: PageRecs (any works; filtered by ``works``).
    :param index_rows: Literature index rows (:func:`rag_l3.build_index`) of the same pages.
    :param works: ``<work>/<part>`` prefixes to scan.
    :param min_numbered: Minimum share of numbered lines in a run (edition lines carry line numbers).
    :param max_dist: Maximum line distance between the anchoring mention and the run start.
    :returns: ``(entries, drop counter)``.
    """
    located: Dict[str, List[str]] = collections.defaultdict(list)
    for r in index_rows:
        if r.get("located_in_text") and (r.get("merged_doc_ids") or r.get("pgpids")):
            located[r["page_key"]].append(r["shelfmark_raw"])
    out: List[Dict[str, Any]] = []
    drops: collections.Counter = collections.Counter()
    for p in pages:
        wp = f"{p.work}/{p.part.split('/')[0]}"
        if not any(wp.startswith(w) for w in works):
            continue
        text = p.ocr_text or p.full_main_text or ""
        lines = text.split("\n")
        runs = ls.rtl_runs(lines)
        if not runs:
            continue
        marks = []
        for raw in located.get(p.page_key, []):
            for i in _sm.locate(raw, text):
                marks.append((text[:i].count("\n"), raw))
        for a, b in runs:
            seg = lines[a:b + 1]
            if ls.numbered_share(seg) < min_numbered:
                drops["run_not_numbered"] += 1
                continue
            above = [(a - ln, raw) for ln, raw in marks if 0 <= a - ln <= max_dist]
            if not above:
                drops["run_unanchored"] += 1
                continue
            raw = min(above)[1]
            clean, st = ls.clean_edition_lines(seg)
            if len(clean) < 4:
                drops["run_short"] += 1
                continue
            e = _entry(f"generic:{p.work}", f"{p.part.split('/')[0]}:p{p.page_no}:{a}", [raw], [("unknown", clean)],
                       "generic_numbered_rtl_run",
                       {"work": p.work, "work_slug": p.part.split("/")[0], "page_keys": [p.page_key],
                        "printed_pages": printed_pages({p.page_key: p}, [p.page_key]), "lines": [a, b],
                        "anchor_distance": min(above)[0], "clean_stats": st})
            out.append(e)
    return out, drops


# ===================================================================================== catalogue
def catalogue_records(pages: Sequence[Any], spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Entries of a numbered catalogue (entry number → shelfmark by the config's template).

    :param pages: PageRecs of the catalogue part.
    :param spec: Local config entry (``pages``, ``shelfmark_template`` such as ``"<prefix> {n}"``, ``slug``).
    :returns: Records ``{"doc", "shelfmarks", "text", "page_keys", "ops", "flags"}``.
    """
    lo, hi = spec["pages"]
    secs = ls.catalogue_entries(page_inputs(pages, lo, hi), min_no=int(spec.get("first_no", 1)))
    out = []
    for s in secs:
        lines, flags = [], []
        garbled = 0
        for ln in s.text.split("\n"):
            t = ln.strip()
            if not t or ls.is_stamp_line(t):
                continue
            # Hebrew printed in a legacy font comes out as Latin garbage: drop lines dominated by it
            words = re.findall(r"[A-Za-z']+", t)
            odd = sum(1 for w in words if re.search(r"[A-Z].*[A-Z]|^[a-z]{1,2}$", w) and w.lower() not in ("ms", "mss"))
            if words and odd / len(words) > 0.45:
                garbled += 1
                continue
            lines.append(t)
        if garbled:
            flags.append(f"garbled_lines_dropped:{garbled}")
        text, _ = _bp.prose_paragraphs("\n".join(lines))
        out.append({"doc": s.label, "shelfmarks": [spec["shelfmark_template"].format(n=s.label)], "text": text.strip(),
                    "page_keys": s.pages, "ops": ["drop_stamp_lines", "drop_garbled_lines", "prose_paragraphs"],
                    "flags": flags})
    return out


# ===================================================================================== commentary
def _item_text(pages: Sequence[Any], hebrew: bool = False) -> str:
    """Prose of a web item without stamps: for an English item the Hebrew / Arabic-script lines (editions) are
    left out; for a Hebrew item (``hebrew``) the Latin-script lines (site navigation, captions) are left out instead.

    :param pages: PageRecs of the item, in order.
    :param hebrew: The item's prose is Hebrew.
    :returns: Text.
    """
    keep = []
    for p in pages:
        for ln in (p.full_main_text or "").split("\n"):
            if ls.is_stamp_line(ln):
                continue
            if hebrew:
                he, ar, lat = ls.letters(ln)
                if ln.strip() and he + ar < max(3, lat):
                    continue
            elif _pg.is_rtl_line(ln):
                continue
            keep.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def strip_byline(text: str, title: Optional[str], max_lines: int = 4) -> Tuple[str, int]:
    """Drop a post's title and byline lines from the start of its text.

    :param text: Item prose.
    :param title: Item title.
    :param max_lines: Leading non-blank lines examined.
    :returns: ``(text, lines dropped)``.
    """
    lines = text.split("\n")
    tkey = re.sub(r"\W+", "", (title or "").lower())
    dropped, seen, i = 0, 0, 0
    while i < len(lines) and seen < max_lines:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        seen += 1
        k = re.sub(r"\W+", "", s.lower())
        byline = len(s.split()) <= 8 and not re.search(r"[.!?]$", s) and not re.search(r"\d", s)
        if (tkey and k and (k in tkey or tkey in k)) or byline:
            lines[i] = ""
            dropped += 1
            i += 1
            continue
        break
    return "\n".join(lines).strip(), dropped


def web_commentary(parts: Mapping[str, Sequence[Any]], web_spans: Mapping[str, Any], dirs: Sequence[str],
                   titles: Mapping[str, Optional[str]], skip: Mapping[str, Sequence[str]],
                   hebrew_dirs: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Commentary candidates from web items: one per item, anchored to the documents the item discusses.

    Anchors: the October spans file's ``commentary_docs`` (agent-read), else shelfmarks in the item title.

    :param parts: ``<dir>/<slug>`` → the item's PageRecs.
    :param web_spans: Web-pairs spans file content.
    :param dirs: Web directories to use.
    :param titles: ``<dir>/<slug>`` → item title.
    :param skip: dir → skipped slugs (not Genizah / low value).
    :param hebrew_dirs: Directories whose posts are written in Hebrew (:func:`_item_text`).
    :returns: Records ``{"source", "doc", "shelfmarks", "text", "page_keys", "anchor_source", "flags"}``.
    """
    anchors: Dict[str, List[str]] = {}
    for it in web_spans.get("items") or []:
        if it.get("commentary_docs"):
            anchors[f"{it['dir']}/{it['slug']}"] = list(it["commentary_docs"])
    out = []
    for key, pgs in sorted(parts.items()):
        d, slug = key.split("/", 1)
        if d not in dirs or slug in set(skip.get(d) or []):
            continue
        flags = []
        sms = anchors.get(key)
        how = "spans_file_commentary_docs"
        if not sms:
            title = titles.get(key) or ""
            sms = [title[a:b] for a, b in tn.find_shelfmarks(title)]
            how = "title_shelfmark"
        if not sms:
            continue
        text, nb = strip_byline(_item_text(pgs, hebrew=d in set(hebrew_dirs)), titles.get(key))
        if nb:
            flags.append(f"byline_lines_dropped:{nb}")
        if len(text) < COMMENTARY_MIN_CHARS:
            continue
        if len(text) > TARGET_MAX_CHARS:
            flags.append(f"long_target_chars:{len(text)}")
        if len(sms) > 4:
            flags.append(f"many_anchors:{len(sms)}")
        out.append({"source": f"web:{d}", "doc": slug, "shelfmarks": sms, "text": text,
                    "page_keys": [p.page_key for p in pgs], "anchor_source": how, "flags": flags,
                    "title": titles.get(key)})
    return out


def passage_around(text: str, offset: int, max_chars: int = 2400) -> Tuple[int, int]:
    """Paragraph(s) around a character offset: the blank-line-bounded paragraph, widened by neighbours while short.

    :param text: Page text.
    :param offset: Mention offset.
    :param max_chars: Stop widening beyond this length.
    :returns: ``(start, end)`` character offsets.
    """
    paras = [(m.start(), m.end()) for m in re.finditer(r"(?:[^\n]+\n?)+", text)]
    i = next((k for k, (a, b) in enumerate(paras) if a <= offset < b), None)
    if i is None:
        return max(0, offset - max_chars // 2), min(len(text), offset + max_chars // 2)
    a, b = paras[i]
    lo, hi = i, i
    while b - a < 600 and (lo > 0 or hi < len(paras) - 1):
        if hi < len(paras) - 1 and paras[hi + 1][1] - a <= max_chars:
            hi += 1
            b = paras[hi][1]
        elif lo > 0 and b - paras[lo - 1][0] <= max_chars:
            lo -= 1
            a = paras[lo][0]
        else:
            break
    if b - a > max_chars:
        a, b = max(a, offset - max_chars // 2), min(b, offset + max_chars // 2)
    return a, b


def index_passages(pages_by_key: Mapping[str, Any], index_rows: Sequence[Mapping[str, Any]], works: Sequence[str]
                   ) -> List[Dict[str, Any]]:
    """Discussion passages of located, resolved shelfmark mentions (literature index ``discuss`` tier).

    :param pages_by_key: page key → PageRec.
    :param index_rows: Literature index rows.
    :param works: ``<work>/<part>`` prefixes to use.
    :returns: Records ``{"source", "doc", "shelfmarks", "text", "page_keys", "anchor_source", "flags", "lang"}``.
    """
    out = []
    for r in index_rows:
        wp = f"{r['book']}/{r['part'].split('/')[0]}"
        if not any(wp.startswith(w) for w in works):
            continue
        if r.get("tier") != "discuss" or r.get("evidence") != "located":
            continue
        if not (r.get("merged_doc_ids") or r.get("pgpids")):
            continue
        p = pages_by_key.get(r["page_key"])
        if p is None:
            continue
        text = p.ocr_text or p.full_main_text or ""
        fs = _pg.footnote_region_start(text, p.data)
        body = text[:fs] if fs else text
        occ = _sm.locate(r["shelfmark_raw"], body)
        if not occ:
            continue
        a, b = passage_around(body, occ[0])
        passage, _ = ls.scrub_stamps(body[a:b].strip())
        if len(passage) < COMMENTARY_MIN_CHARS:
            continue
        flags = ["context_passage_not_full_commentary"]
        if r.get("passage_other_shelfmarks", 0) > 2:
            flags.append(f"other_shelfmarks:{r['passage_other_shelfmarks']}")
        out.append({"source": f"lit:{r['book']}", "doc": f"{r['part'].split('/')[0]}:{r['page_file']}:{r['shelfmark_core']}",
                    "shelfmarks": [r["shelfmark_raw"]], "text": passage, "page_keys": [r["page_key"]],
                    "anchor_source": "literature_index_located_discuss", "flags": flags,
                    "lang": _pg.lang_of(passage), "kinds": r.get("kinds")})
    return out


# ===================================================================================== row rendering
def render_describe_row(rec: Mapping[str, Any], input_text: str, input_source: str) -> Dict[str, Any]:
    """A ``describe_pgp``-schema row for a catalogue candidate (condition ``edition``).

    :param rec: Catalogue candidate (``target``, ``split``, ``resolution``, provenance ...).
    :param input_text: Edition text (the document input).
    :param input_source: Where the input comes from.
    :returns: Row ``{"messages", "meta"}``.
    """
    script = script_of(input_text)
    user = _rows.user_prompt("describe_pgp", "edition", script, input_text)
    res = rec["resolution"]
    meta = _rows.new_meta("describe_pgp", "edition", rec["split"], rec["doc_id"],
                          lang_class=rec.get("lang_class") or ("AR" if script == "arabic" else "JA"),
                          pgpid=(res["pgpids"][0] if res["pgpids"] and not res["broad"] else None),
                          shelfmark=rec["shelfmark"], source=rec["source"], input_source=input_source,
                          licence=rec["provenance"].get("licence"), licence_class=rec["provenance"].get("licence_class"),
                          target_raw=rec.get("target_raw"), strip_ops=rec.get("strip_ops") or [],
                          held_out_vision_benchmark=False, run2=True, doc_id=rec["doc_id"],
                          page_keys=rec["provenance"].get("page_keys") or [])
    return {"messages": _rows.make_messages(user, rec["target"]), "meta": meta}


def render_analyse_row(rec: Mapping[str, Any], input_text: str, input_source: str) -> Dict[str, Any]:
    """An ``analyse_document``-schema row for a commentary candidate (condition ``edition``).

    :param rec: Commentary candidate.
    :param input_text: Edition text of the anchor document.
    :param input_source: Where the input comes from.
    :returns: Row ``{"messages", "meta"}``.
    """
    script = script_of(input_text)
    user = _rows.user_prompt("analyse_document", "edition", script, input_text)
    res = rec["resolution"]
    meta = _rows.new_meta("analyse_document", "edition", rec["split"], rec["doc_id"],
                          lang_class="AR" if script == "arabic" else "JA",
                          pgpid=(res["pgpids"][0] if res["pgpids"] and not res["broad"] else None),
                          shelfmark=rec["shelfmark"], source=rec["source"], input_source=input_source,
                          licence=rec["provenance"].get("licence"), licence_class=rec["provenance"].get("licence_class"),
                          target_raw=rec.get("target_raw"), strip_ops=rec.get("strip_ops") or [],
                          held_out_vision_benchmark=False, run2=True, doc_id=rec["doc_id"],
                          page_keys=rec["provenance"].get("page_keys") or [])
    return {"messages": _rows.make_messages(user, rec["target"]), "meta": meta}


# ===================================================================================== run
@dataclass
class Run2Context:
    """Everything one run shares."""

    p: _paths.Paths
    cfg: Dict[str, Any]
    out: Path
    pages: List[Any]
    pages_by_key: Dict[str, Any]
    parts: Dict[str, List[Any]]
    resolver: Resolver
    images: ImageIndex
    index_rows: List[Dict[str, Any]] = field(default_factory=list)
    drops: collections.Counter = field(default_factory=collections.Counter)
    health: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def reversed_layer(self, work_part: str) -> Optional[str]:
        """Flag naming a reversed Hebrew text layer of a work part (word order or letters), or None.

        :param work_part: ``<work>/<part>``.
        :returns: ``hebrew_text_layer_reversed:<kind>:<share>`` or None.
        """
        h = self.health.get(work_part) or {}
        for kind in ("word_order", "letters"):
            v = h.get(kind)
            if v is not None and v > 0.3:
                return f"hebrew_text_layer_reversed:{kind}:{v}"
        return None


def text_layer_health(parts: Mapping[str, Sequence[Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Reversed-Hebrew diagnostics per work part (:func:`lit_segment.word_order_reversal`,
    :func:`lit_segment.reversed_hebrew_share`).

    :param parts: ``<work>/<part>`` → pages.
    :returns: ``{work_part: {"word_order", "letters"}}``.
    """
    out = {}
    for wp, pgs in parts.items():
        text = "\n".join(p.ocr_text or p.full_main_text or "" for p in pgs)
        out[wp] = {"word_order": ls.word_order_reversal(text), "letters": ls.reversed_hebrew_share(text)}
    return out


def _parts(pages: Sequence[Any]) -> Dict[str, List[Any]]:
    """Pages grouped by ``<work>/<first part component>``.

    :param pages: PageRecs.
    :returns: key → pages in order.
    """
    out: Dict[str, List[Any]] = collections.defaultdict(list)
    for p in pages:
        out[f"{p.work}/{p.part.split('/')[0]}"].append(p)
    return dict(out)


def _licence_of(cfg: Mapping[str, Any], work_part: str) -> Tuple[Optional[str], Optional[str]]:
    """Licence and class of a work from the config (``works`` entries, else the work directory default).

    :param cfg: Config.
    :param work_part: ``<work>/<part>``.
    :returns: ``(licence, licence_class)``.
    """
    w = (cfg.get("works") or {}).get(work_part) or {}
    d = (cfg.get("dir_licence") or {}).get(work_part.split("/")[0]) or {}
    return w.get("licence") or d.get("licence"), w.get("licence_class") or d.get("licence_class")


def _finish(ctx: Run2Context, rec: Dict[str, Any], shelfmarks: Sequence[str], texts: Sequence[Optional[str]]) -> Dict[str, Any]:
    """Resolve, hold out and attach image availability to a candidate.

    :param ctx: Run context.
    :param rec: Candidate.
    :param shelfmarks: Printed shelfmarks.
    :param texts: Texts entering a row.
    :returns: The record.
    """
    ctx.resolver.annotate(rec, shelfmarks, texts, rec.get("doc_id") or rec.get("doc") or "",
                          rec.get("pgpid_hints") or (), rec.get("pgpid_hint_source") or "agent")
    return rec


def doc_hints(ctx: Run2Context, spec: Mapping[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """PGP-id hints per document number of a doc-headed work: PGP's ``Doc. N`` footnote locations for the work's
    PGP source slug (``pgp_source_slug``) and an optional reviewed hints file (``doc_hints_file``, JSON
    ``{"docs": {"<no>": {"pgpids": [...], "evidence": "..."}}}``; applied only when the config names it).

    :param ctx: Run context.
    :param spec: Doc-headed config entry.
    :returns: ``(crosswalk hints, reviewed-file hints)``.
    """
    xw = pgp_doc_crosswalk(getattr(ctx.resolver.corpus, "fn_rows", None) or [], spec["pgp_source_slug"]) \
        if spec.get("pgp_source_slug") else {}
    cur: Dict[str, List[str]] = {}
    if spec.get("doc_hints_file"):
        path = Path(spec["doc_hints_file"])
        path = path if path.is_absolute() else ctx.p.repo / path
        cur = {str(k): [str(x) for x in v.get("pgpids") or []]
               for k, v in (json.loads(path.read_text(encoding="utf-8")).get("docs") or {}).items()}
    return xw, cur


def build_doc_headed(ctx: Run2Context) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Catalogue candidates, edition (VLM) entries and formulary span sources of the doc-headed works.

    :param ctx: Run context.
    :returns: ``(catalogue candidates, vlm entries, formulary sources)``.
    """
    cats, vlm, sources = [], [], []
    for spec in ctx.cfg.get("doc_headed") or []:
        wp = f"{spec['work']}/{spec['part']}"
        pgs = ctx.parts.get(wp) or []
        recs = doc_headed_records(pgs, spec)
        lic, lcl = _licence_of(ctx.cfg, wp)
        spec = {**spec, "licence": spec.get("licence") or lic, "licence_class": spec.get("licence_class") or lcl}
        sources.append(formulary_source(ctx.pages_by_key, recs, spec))
        ed_secs = {s.label: s for s in recs["edition"]}
        xw, curated = doc_hints(ctx, spec)
        fixes = spec.get("shelfmark_fixes") or {}
        for csec in recs["catalogue"]:
            if csec.kind == "rec_form":
                continue
            nos = csec.label.split(",")
            esecs = [ed_secs[n] for n in nos if n in ed_secs] or ([ed_secs[csec.label]] if csec.label in ed_secs else [])
            sm = csec.shelfmark_raw or next((e.shelfmark_raw for e in esecs if e.shelfmark_raw), None)
            sm_ocr = None
            if csec.label in fixes:
                sm_ocr, sm = sm, fixes[csec.label]["shelfmark"]
            hints = sorted({p for n in nos for p in xw.get(n, [])}, key=lambda x: int(x) if x.isdigit() else 0)
            hint_src = "pgp_footnote_doc_number" if hints else None
            cur_h = sorted({p for n in nos for p in curated.get(n, [])}, key=lambda x: int(x) if x.isdigit() else 0)
            if cur_h:
                hints = sorted(set(hints) | set(cur_h), key=lambda x: int(x) if x.isdigit() else 0)
                hint_src = "pgp_footnote_doc_number+reviewed_hints" if hint_src else "reviewed_hints"
            desc, ops = description_text(csec)
            fields = ls.parse_description(desc)
            ed_lines: List[str] = []
            st_all: collections.Counter = collections.Counter()
            for e in esecs:
                lines, st = ls.clean_edition_lines(e.text.split("\n"))
                ed_lines += lines
                st_all.update(st)
            q, qflags = quality_flags_edition(ed_lines, st_all, fields.get("lines_stated") if len(nos) == 1 else None)
            doc_id = f"lit:{spec['slug']}:{csec.label}"
            prov = {"work": spec["work"], "work_slug": spec["part"], "page_keys": csec.pages,
                    "printed_pages": printed_pages(ctx.pages_by_key, csec.pages), "licence": spec["licence"],
                    "licence_class": spec["licence_class"], "extraction_method": "doc_headed_catalogue_part"}
            shelf = [sm] if sm else []
            cat = {"pipeline": "catalogue_descriptions", "source": f"lit:{spec['slug']}", "doc": csec.label,
                   "doc_id": doc_id, "shelfmark": sm, "description": desc, "description_ops": ops,
                   "fields": fields, "provenance": prov, "edition_lines": len(ed_lines),
                   "flags": (["no_shelfmark_line"] if not sm else []) + (["column_scramble_risk"])
                   + (["shelfmark_corrected_from_page_image"] if sm_ocr is not None else []),
                   "quality": {"score": 0.7 if sm else 0.3,
                               "note": "typewritten catalogue read by Vision OCR; wrapped lines can be out of order"},
                   "pgpid_hints": hints, "pgpid_hint_source": hint_src}
            if sm_ocr is not None:
                cat["shelfmark_ocr"] = sm_ocr
                cat["shelfmark_fix_evidence"] = fixes[csec.label].get("evidence")
            _finish(ctx, cat, shelf, [desc])
            cats.append(cat)
            if esecs and ed_lines:
                epages = [k for e in esecs for k in e.pages]
                e = _entry(f"lit:{spec['slug']}", csec.label, shelf, [("unknown", ed_lines)], "doc_headed_edition_part",
                           {"work": spec["work"], "work_slug": spec["part"], "page_keys": epages,
                            "printed_pages": printed_pages(ctx.pages_by_key, epages), "licence": spec["licence"],
                            "licence_class": spec["licence_class"], "clean_stats": dict(st_all),
                            "lines_stated": fields.get("lines_stated")})
                e["quality"] = {"score": q, "flags": qflags}
                e["catalogue_doc_id"] = doc_id
                e["pgpid_hints"], e["pgpid_hint_source"] = hints, hint_src
                if sm_ocr is not None:
                    e["shelfmark_ocr"] = sm_ocr
                vlm.append(e)
    return cats, vlm, sources


def build_anthologies(ctx: Run2Context) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Span sources and entry records of the numbered anthologies.

    :param ctx: Run context.
    :returns: ``(sources, entry records with resolution)``.
    """
    sources, records = [], []
    for spec in ctx.cfg.get("anthology") or []:
        wp = f"{spec['work']}/{spec['part']}"
        lic, lcl = _licence_of(ctx.cfg, wp)
        spec = {**spec, "licence": spec.get("licence") or lic, "licence_class": spec.get("licence_class") or lcl}
        lo, hi = spec["pages"]
        secs = ls.anthology_sections(page_inputs(ctx.parts.get(wp) or [], lo, hi),
                                     stop=re.compile(spec["stop"]) if spec.get("stop") else None)
        src, recs = anthology_source(ctx.pages_by_key, secs, spec, ctx.resolver)
        sources.append(src)
        for r in recs:
            r.update({"source": f"book:{spec['slug']}", "doc_id": f"book:{spec['slug']}:{r['doc']}",
                      "provenance": {"work": spec["work"], "work_slug": spec["part"], "page_keys": r["page_keys"],
                                     "printed_pages": printed_pages(ctx.pages_by_key, r["page_keys"]),
                                     "licence": spec["licence"], "licence_class": spec["licence_class"],
                                     "extraction_method": "anthology_sections"}})
            _finish(ctx, r, [r["shelfmark_printed"]], [r.get("intro_text")])
            records.append(r)
    return sources, records


def run_book_pairs(ctx: Run2Context, sources: Sequence[Mapping[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Run the L5 builder on the run-2 span sources and write its outputs into the run directory.

    :param ctx: Run context.
    :param sources: Span sources (book_pairs spans format).
    :param out_dir: Output root for :func:`book_pairs.write_outputs` (never ``paths.out_dir``).
    :returns: ``{"result", "manifest"}``.
    """
    if Path(out_dir).resolve() == ctx.p.out_dir.resolve():
        raise ValueError("run-2 book_pairs output must not go to the t27b build directory")
    works = sorted({str(s["work"]) for s in sources if s.get("work")})
    page_texts = {pk: p.ocr_text for pk, p in ctx.pages_by_key.items() if pk.split("/", 1)[0] in works}
    res = _bp.build(ctx.p, {"nas_books": [], "priority": []}, {"sources": list(sources)},
                    ix=ctx.resolver.ix, page_texts=page_texts)
    man = _bp.write_outputs(res, ctx.p, out_dir=out_dir, build_args={"run2": True, "sources": [s["slug"] for s in sources]})
    return {"result": res, "manifest": man}


def run(config: Optional[Path] = None, use_es: bool = True, with_book_pairs: bool = True,
        rebuild_index: bool = False) -> Dict[str, Any]:
    """Mine every configured work and write the run-2 candidate files.

    :param config: Local config file.
    :param use_es: Look up images in the served index (else cache only).
    :param with_book_pairs: Run the L5 builder on the span sources.
    :param rebuild_index: Rebuild the literature index of the scanned works first (:func:`rag_l3.run_build_index`
        into ``<run dir>/rag_index``; needed after new works arrive).
    :returns: Summary dict (also written as ``run2_summary.json``).
    """
    t0 = time.time()
    p = _paths.load_paths()
    cfg = load_config(config, p)
    out = (p.repo / cfg.get("out_dir", "logs/run2_data_2026-10-27")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    works = sorted(set(cfg.get("scan_dirs") or []) | {w.split("/")[0] for w in (cfg.get("works") or {})})
    if rebuild_index or not (out / "rag_index" / "rag_index.jsonl").exists():
        from src.finetuning.text27b import rag_l3 as _rag
        _rag.run_build_index(p, works=works, test_docs=False, out_dir=out / "rag_index")
    pages = _pg.iter_pages(p.academic_literature, works=works)
    layers = apply_text_layers(pages, cfg)
    pages_by_key = {pg.page_key: pg for pg in pages}
    parts = _parts(pages)
    from src.finetuning.text27b import holdouts as _ho
    from src.finetuning.text27b import web_pairs as _wp
    corpus = _sm.load_corpus(p)
    ix = _wp._index(p, with_grams=True)
    hold = _ho.Holdouts.load(p=p)
    neu = neubauer_crosswalk(pg.ocr_text or "" for pg in pages if pg.work in set(cfg.get("neubauer_from_works") or []))
    resolver = Resolver(corpus, hold, ix, neu)
    local_file = cfg.get("local_images_file")
    images = ImageIndex(cfg.get("es_index"), out / "cache" / "es_images.json", enabled=use_es,
                        local_images=ImageIndex.load_local(p.repo / local_file if local_file else None))
    LOG.info("local images: %d ids (%s)", len(images.local), local_file)
    index_rows = [json.loads(ln) for ln in open(out / "rag_index" / "rag_index.jsonl", encoding="utf-8")] \
        if (out / "rag_index" / "rag_index.jsonl").exists() else []
    ctx = Run2Context(p, cfg, out, pages, pages_by_key, parts, resolver, images, index_rows,
                      health=text_layer_health(parts))
    LOG.info("[%5.1fs] %d pages, %d parts, %d index rows", time.time() - t0, len(pages), len(parts), len(index_rows))

    from src.finetuning.text27b import lit_run2_build as _b
    summary = _b.build_all(ctx, with_book_pairs=with_book_pairs)
    summary["text_layers"] = layers
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (out / "run2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments.
    :returns: None.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-es", action="store_true", help="answer image lookups from the cache only")
    ap.add_argument("--no-book-pairs", action="store_true", help="skip the L5 builder (translation / analysis rows)")
    ap.add_argument("--rebuild-index", action="store_true", help="rebuild the literature index of the scanned works first")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    s = run(args.config, use_es=not args.no_es, with_book_pairs=not args.no_book_pairs, rebuild_index=args.rebuild_index)
    print(json.dumps(s.get("counts", {}), indent=1))


if __name__ == "__main__":
    main()
