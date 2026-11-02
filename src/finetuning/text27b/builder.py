# File name: builder.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""v0_2 builder of the PGP families (``translate_en``, ``translate_he``, ``catalogue_json``,
``describe_pgp``, ``extract_relations``, ``analyse_document``) on top of :mod:`builder_v0`.

The v0 builder's document assembly, read gate, noise model, side splitting, catalogue records and
group splits are reused unchanged. On top of them (plan 2026-10-21 §3, amendment §9.0, brief F):

* **B3** line markers (``(n–m)`` / U+2212 range markers, line references, leading line numbers) are
  stripped from every prose target (translations, descriptions, catalogue summaries, analyses) to
  a fixpoint; the pre-strip target is kept in ``meta.target_raw`` and the strips in ``meta.strip_ops``.
* **B4** citations, cross-references, dimensions, shelfmark mentions and project-name strings are stripped
  from the description (so from ``describe_pgp``, the ``catalogue_json`` summary and the people /
  relations derived from it) and from ``analyse_document`` targets, which also lose material /
  dimension / Hebrew citation clauses (:func:`strip_analysis_extras`) and are dropped under 200
  characters. project-name strings are stripped from every target. ``audit/strip_sample_50.jsonl`` holds
  a stratified before/after sample for the hand audit.
* **B5** ``meta.pgp_description_dup`` marks ``describe_pgp`` rows whose target is the catalogue
  summary of a ``catalogue_json`` row of the same document (one per-document budget downstream).
* **B7** ``translate_he`` rows use ``edition_text_he`` where it differs from the English-matched edition.
* **B8** no ``noised_edition`` rows for Arabic-script / Arabic-language documents.
* **B9** ``edition_rescript`` rows: ``translate_en`` of Arabic documents with the edition re-written
  in Hebrew letters (:func:`textnorm.transliterate_ar_to_ja_plain`), same human target.
* **F1** translation / edition length band 0.5–2.5 (en) and 0.5–2.0 (he) on
  :func:`textnorm.normalize_for_cer` lengths (v0: 0.3–5).
* **B10 / §9.0** hold-outs H are matched by pgpid, canonical id *and* normalised shelfmark key
  (:class:`HoldoutIndex`; the holdouts module, when present, is consulted too), at shelfmark-group
  level. Splits: ``train`` and ``dev`` as in v0; test groups become ``heldout`` (PGP-131, probe
  docs, other H), ``heldout_vision`` (benchmark registries / vision flag) or ``test_indist``.
* exposure tags from the exposure module (``exposure.jsonl``) when it exists, else
  ``reader_exposed=None``; exposed train reads at CER < 0.05 are dropped (B2 near-duplicates).
* project-name strings in inputs (e.g. a side label that names the project's image) are removed line by
  line (:func:`strip_source_names_input`) before the noise draw.
* v0_2 row schema: the v0_2 read note for every read-like condition, ``family``, ``build``,
  ``row_id``, exact rendered token counts (``enable_thinking=False``; rows over 6,144 dropped).

Analysis targets are carried forward from ``text_ft_v0_1`` (the v0 span detector depends on
book-specific research code); their context block is rebuilt from the v0_2 catalogue record.

Outputs under ``--out-dir`` (default ``paths.out_dir``): ``families/<family>.jsonl`` (all splits),
``families/build_report.json`` (counts per family × condition × lang_class × split, checks,
build args), ``noise_model.json``, ``audit/strip_sample_50.jsonl``.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.builder                 # full build
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.builder --limit 300 --out-dir DIR   # sample

Public API: :func:`clean_target`, :func:`strip_markers`, :func:`strip_analysis_extras`,
:func:`strip_source_names_input`, :func:`band_ok`, :func:`noise_band`, :class:`HoldoutIndex`, :func:`holdouts_module_adapter`,
:func:`assign_split`, :class:`BuilderV2`, :func:`make_row_v2`, :func:`load_v01_analysis`,
:func:`carry_forward_analysis`, :func:`apply_exposure`, :func:`finalize_tokens`, :func:`count_table`,
:func:`checks_v02`, :func:`audit_sample`, :class:`V2Config`, :func:`build_v02`, :func:`write_v02`, :func:`main`.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import builder_v0 as bv
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import target_hygiene as _th
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)
BUILD = _rows.BUILD
#: F1: allowed normalised translation / edition length ratio per target language (v0: 0.3–5).
F1_BAND: Dict[str, Tuple[float, float]] = {"en": (0.5, 2.5), "he": (0.5, 2.0)}
#: Families whose targets get the B4 citation strip (catalogue_json via its summary).
B4_FAMILIES: Tuple[str, ...] = ("describe_pgp", "catalogue_json", "analyse_document")
ANALYSIS_MIN_CHARS = 200
#: Prep pass 2026-10-26: an analysis target shorter than this after the evidence strip is dropped.
ANALYSIS_EVIDENCE_MIN_CHARS = _th.MIN_CHARS_AFTER
MAX_ROW_TOKENS = 6144
EXPOSED_NEAR_DUP_CER = 0.05
AUDIT_SIZE = 50
#: CER bands of the noise eval (E7); the last band is open-ended.
NOISE_BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9))
#: Hold-out reasons of :class:`HoldoutIndex` (the holdouts module uses the same names, with ``key:`` / ``group:`` prefixes).
KNOWN_REASONS = frozenset({"pgp131", "registry", "vision_flag", "probe"})
#: Hold-out reasons that put a test group into ``heldout_vision``.
VISION_REASONS = frozenset({"registry", "vision_flag"})
#: Hold-out reasons whose documents count as vision-benchmark hold-outs (``meta.held_out_vision_benchmark``).
BENCH_REASONS = frozenset({"pgp131", "registry", "vision_flag"})
FAMILY_ORDER = {f: i for i, f in enumerate(_rows.PGP_FAMILIES)}
COND_ORDER = {c: i for i, c in enumerate(_rows.CONDITIONS)}
SPLIT_ORDER = {s: i for i, s in enumerate(_rows.SPLITS)}


# ============================================================================== target strips
def _tidy(text: str) -> str:
    """Tidy whitespace and separators left behind by deletions in prose (paragraph breaks kept).

    Collapses doubled spaces and doubled separators, removes a space before punctuation that ends a
    word, removes separators (``,`` ``;`` ``:`` or a lone ``.``) left at the start of a line and
    ``,`` / ``;`` at its end, and drops lines that hold nothing but separators. Ellipses and
    ``[...]`` lines are kept.

    :param text: Text after one or more strips.
    :returns: The tidied text.
    """
    paras = []
    for para in re.split(r"\n[ \t]*\n", text):
        lines = []
        for ln in para.split("\n"):
            ln = re.sub(r"[ \t]{2,}", " ", ln).strip()
            ln = re.sub(r"[ \t]+([.,;:!?])(?=\s|$)", r"\1", ln)
            ln = re.sub(r"(?<!\.)\.\.(?!\.)", ".", ln)
            ln = re.sub(r"([,;:])(?:[ \t]*[,;:])+", r"\1", ln)
            ln = re.sub(r"^(?:[ \t]*(?:[,;:]|\.(?!\.)))+[ \t]*", "", ln).rstrip(" \t,;")
            if ln.strip() and not re.fullmatch(r"[\s,;:.]+", ln):
                lines.append(ln)
        if lines:
            paras.append("\n".join(lines))
    return "\n\n".join(paras).strip()


def strip_markers(text: str, max_iter: int = 3) -> Tuple[str, List[str]]:
    """B3: :func:`textnorm.strip_line_markers` applied until nothing changes.

    :param text: Target text.
    :param max_iter: Safety bound on the passes.
    :returns: ``(text, ops)`` with ops prefixed ``B3:``.
    """
    ops: List[str] = []
    for _ in range(max_iter):
        new, step = tn.strip_line_markers(text)
        if not step or new == text:
            break
        ops.extend(f"B3:{o}" for o in step if f"B3:{o}" not in ops)
        text = new
    return text, ops


_DIM_NUM = r"\d+(?:[.,]\d+)?[%⅛⅜⅝⅞½¼¾]?"
_DIMS = re.compile(r"(?:\b|(?<=\s))" + _DIM_NUM + r"\s*(?:cm\.?|mm\.?)?\s*[x×X]\s*" + _DIM_NUM +
                   r"\s*(?:cm\b\.?|mm\b\.?|inches\b\.?|in\.|מ[\"״׳']?מ|ס[\"״׳']?מ)\s*[;,]?")
_MATERIAL = re.compile(r"(?:(?<=^)|(?<=[.;:)\]}]\s)|(?<=\n))(?:Paper|Parchment|Vellum|Cloth|Leather|נייר|קלף)"
                       r"(?:\s*,\s*(?:one|two|three|a)\s+(?:double\s+)?(?:pages?|leaf|leaves|sheets?))?\s*[.;](?=\s|$)")
_HE_CITE = re.compile(r"(?:(?<=^)|(?<=[.;]\s)|(?<=\n))(?:ו?נזכר(?:ה|ו)?\s+(?:אצל\b|ב(?=\S))|"
                      r"ו?(?:ראה|ראו|השווה|השוו|עיין)\b(?!\s+את\b)|(?:פורס(?:ם|מה)|נדפס(?:ה)?)\b)[^.\n]*\.?")
_GENIZAH_DOCS_NO = re.compile(r"כתבי\s+גניזה\s+מס['׳]?\s*[\d\-–]+")
_LEAD_FOLIO = re.compile(r"(?:(?<=^)|(?<=\n))[\s,;]*(?:f|fol|fols)\.\s*\d+[a-z]?[\s,;.]*")
_FOLIO_AFTER_BRACKET = re.compile(r"(?<=[^\w\s])[ \t]*,[ \t]*f(?:ol|ols)?\.[ \t]*\d+[a-z]?\b")
_LEAD_IB = re.compile(r"(?:(?<=^)|(?<=\n))\s*[IVX]{1,4},\s*\d+[a-z]?\b\s*")


def strip_analysis_extras(text: str) -> Tuple[str, List[str]]:
    """B4 extras for ``analyse_document`` headnotes (after :func:`textnorm.strip_citations`).

    Removes physical-description clauses the document text cannot support and citation residue the
    prose strip does not cover: dimensions (``25.5 cm. x 8.8 cm.``, ``190x140 מ"מ``,
    ``83 × 5 inches``), bare material sentences (``Paper.``, ``Parchment.``), Hebrew citation
    sentences (``נזכר אצל גויטיין ...``, ``וראה ...``), Genizah-documents numbers, a folio residue left
    after a removed shelfmark (``, f. 20`` at a paragraph start or after a bracket) and a leading India
    Book number (``I, 43b``).

    :param text: Analysis target (already citation-stripped).
    :returns: ``(text, ops)`` with ops prefixed ``B4:``.
    """
    ops: List[str] = []
    for name, pat in (("dimensions", _DIMS), ("genizah_docs_no", _GENIZAH_DOCS_NO), ("material", _MATERIAL),
                      ("he_citation", _HE_CITE), ("folio_residue", _LEAD_FOLIO), ("folio_residue", _FOLIO_AFTER_BRACKET),
                      ("ib_number", _LEAD_IB)):
        new = pat.sub(" ", text) if name not in ("folio_residue", "ib_number") else pat.sub("", text)
        if new != text:
            if f"B4:{name}" not in ops:
                ops.append(f"B4:{name}")
            text = re.sub(r"[ \t]{2,}", " ", new)
    return (_tidy(text), ops) if ops else (text, ops)


def analysis_evidence_strip(target: str, min_chars: int = ANALYSIS_EVIDENCE_MIN_CHARS
                            ) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    """Prep-pass strip of an ``analyse_document`` target: sentences resting on evidence the input lacks.

    Applied after :func:`clean_target` and the :data:`ANALYSIS_MIN_CHARS` floor (both unchanged): physical description,
    cross-references to numbered documents / plates and bibliographic apparatus go
    (:func:`target_hygiene.strip_unavailable_evidence`); a target left shorter than ``min_chars`` is dropped.

    :param target: Analysis target after the v0_2 strips.
    :param min_chars: Shortest target kept after this strip.
    :returns: ``(target or None when dropped, ops, info)``; ``info`` = ``{"removed_by_reason": {...},
        "chars_before": n, "chars_after": m}`` (counts only -- the removed sentences stay out of ``meta``; the pre-strip
        text is ``meta.target_raw``).
    """
    out, ops, removed = _th.strip_unavailable_evidence(target)
    info: Dict[str, Any] = {"removed_by_reason": dict(collections.Counter(w for w, _ in removed)),
                            "chars_before": len(target), "chars_after": len(out)}
    if len(out) < min_chars:
        return None, ops, info
    return out, ops, info


def clean_target(text: str, family: str) -> Tuple[str, List[str]]:
    """All v0_2 target strips for one family (B4 citations + project names, analysis extras, B3 markers).

    For ``catalogue_json`` pass the summary (prose), not the JSON string.

    :param text: Target prose.
    :param family: Row family (decides which strips apply).
    :returns: ``(text, ops)``; ops are ``B4:<rule>`` / ``B3:<marker kind>`` in application order.
    """
    if not text:
        return text, []
    ops: List[str] = []
    out = text
    if family in B4_FAMILIES:
        out, cops = tn.strip_citations(out)
        ops += [f"B4:{o}" for o in cops]
    out, n_scrub = tn.strip_source_names(out)
    if n_scrub:
        ops.append(f"B4:{tn.SCRUB_OP}")
    if family == "analyse_document":
        out, eops = strip_analysis_extras(out)
        ops += eops
    b4_fired = bool(ops)
    out, mops = strip_markers(out)
    ops += mops
    if not ops:
        return text, []
    return (_tidy(out) if b4_fired and family in B4_FAMILIES else out.strip()), ops


def strip_source_names_input(text: Optional[str]) -> Tuple[Optional[str], int]:
    """project-name strip for row inputs (editions, reads), bounded to the line that carries the hit.

    :func:`textnorm.strip_source_names` deletes a hit's parenthetical, clause or sentence; an edition has no
    sentence punctuation, so a sentence could run over many lines. Here each line with a hit is
    stripped on its own (``§ Verso (Image 4 in <project name>)`` → ``§ Verso``) and a line left empty is dropped.

    :param text: Input text (or None).
    :returns: ``(text, hits removed)``.
    """
    if not text or not tn.has_source_names(text):
        return text, 0
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        if tn.has_source_names(ln):
            new, k = tn.strip_source_names(ln)
            n += k
            new = new.rstrip()
            if not new.strip():
                continue
            ln = new
        out.append(ln)
    return "\n".join(out), n


def band_ok(edition: str, target: str, lang: str) -> Tuple[bool, float]:
    """F1: whether a translation's length ratio to its edition lies in :data:`F1_BAND`.

    :param edition: Edition text.
    :param target: Translation (after strips).
    :param lang: ``en`` or ``he``.
    :returns: ``(inside band, ratio)``.
    """
    ed_n = len(tn.normalize_for_cer(edition))
    ratio = len(tn.normalize_for_cer(target)) / max(ed_n, 1)
    lo, hi = F1_BAND[lang]
    return lo <= ratio <= hi, ratio


def noise_band(cer: Optional[float]) -> Optional[str]:
    """E7 CER band label of an input (``0-0.15`` ... ``0.6-0.9``; ``>0.9`` beyond).

    :param cer: Input CER or None.
    :returns: Band label, or None when the CER is unknown.
    """
    if cer is None:
        return None
    for lo, hi in NOISE_BANDS:
        if lo <= cer < hi:
            return f"{lo:g}-{hi:g}"
    return ">0.9" if cer >= NOISE_BANDS[-1][1] else None


# ============================================================================== hold-outs
ExternalHoldout = Callable[[str, Optional[str], Sequence[str]], Optional[str]]


@dataclasses.dataclass
class HoldoutIndex:
    """Held-out documents H, matched by pgpid, canonical id or normalised shelfmark key (brief rule 8)."""

    pgp131_ids: Set[str]
    pgp131_pgpids: Set[str]
    pgp131_keys: Set[str]
    registry_ids: Set[str]
    registry_pgpids: Set[str]
    registry_keys: Set[str]
    probe_pgpids: Set[str]
    external: Optional[ExternalHoldout] = None

    @classmethod
    def build(cls, merged: Dict[str, Dict[str, List[str]]], pgp131_ids: Iterable[str],
              registry: Tuple[Iterable[str], Iterable[str]], probe_pgpids: Iterable[str],
              external: Optional[ExternalHoldout] = None) -> "HoldoutIndex":
        """Assemble H from its sources.

        :param merged: canonical id -> ``{"pgpids": [...]}`` (pgpids reachable through the merge).
        :param pgp131_ids: The frozen PGP-131 canonical ids.
        :param registry: ``(canonical ids, pgpids)`` of the registered benchmarks.
        :param probe_pgpids: pgpids of the zero-shot probe samples.
        :param external: Optional extra matcher (the holdouts module), see :func:`holdouts_module_adapter`.
        :returns: The index.
        """
        b131 = set(pgp131_ids)
        reg_ids, reg_pg = (set(registry[0]), {str(x) for x in registry[1]})

        def via_map(ids: Set[str]) -> Set[str]:
            """pgpids of canonical ids through the merged map.

            :param ids: Canonical ids.
            :returns: pgpids.
            """
            return {p for c in ids for p in merged.get(c, {}).get("pgpids", [])}

        return cls(pgp131_ids=b131, pgp131_pgpids=via_map(b131),
                   pgp131_keys={_sm.canonical_to_key(c) for c in b131} - {""},
                   registry_ids=reg_ids, registry_pgpids=reg_pg | via_map(reg_ids),
                   registry_keys={_sm.canonical_to_key(c) for c in reg_ids} - {""},
                   probe_pgpids={str(p) for p in probe_pgpids}, external=external)

    @staticmethod
    def doc_keys(shelfmark: Optional[str], cids: Sequence[str]) -> Set[str]:
        """Shelfmark keys of a document: every part of its shelfmark plus every canonical id.

        :param shelfmark: PGP shelfmark (joins split on ``+``).
        :param cids: Merged canonical ids of the document.
        :returns: Non-empty keys.
        """
        return (set(_sm.keys(shelfmark)) | {_sm.canonical_to_key(c) for c in cids}) - {""}

    def match(self, pgpid: str, shelfmark: Optional[str], cids: Sequence[str], vision_flag: bool = False,
              keys: Optional[Set[str]] = None) -> Dict[str, str]:
        """Why a document is held out.

        :param pgpid: PGP document id.
        :param shelfmark: Its shelfmark.
        :param cids: Its merged canonical ids.
        :param vision_flag: The pairs file's ``held_out_vision_benchmark`` flag.
        :param keys: Precomputed :meth:`doc_keys` (computed when None).
        :returns: ``{reason: how}`` with reason ∈ {pgp131, registry, vision_flag, probe, holdouts_module} and
            how ∈ {pgpid, canonical_id, shelfmark_key, flag, external}; empty when not held out. A reason
            only the external matcher gives (prefixes ``key:`` / ``group:`` dropped) is recorded as ``external``.
        """
        ks = self.doc_keys(shelfmark, cids) if keys is None else keys
        cs = set(cids)
        out: Dict[str, str] = {}
        for reason, ids, pgs, kk in (("pgp131", self.pgp131_ids, self.pgp131_pgpids, self.pgp131_keys),
                                     ("registry", self.registry_ids, self.registry_pgpids, self.registry_keys)):
            if pgpid in pgs:
                out[reason] = "pgpid"
            elif cs & ids:
                out[reason] = "canonical_id"
            elif ks & kk:
                out[reason] = "shelfmark_key"
        if vision_flag:
            out["vision_flag"] = "flag"
        if pgpid in self.probe_pgpids:
            out["probe"] = "pgpid"
        if self.external is not None:
            ext = self.external(pgpid, shelfmark, list(cids))
            if ext:
                base = ext.split(":")[-1]
                out.setdefault(base if base in KNOWN_REASONS else "holdouts_module", "external")
        return out


def holdouts_module_adapter(p: Optional[_paths.Paths] = None, path: Optional[Path] = None) -> Optional[ExternalHoldout]:
    """Wrap the holdouts module's ``Holdouts.load().is_benchmark`` as an extra matcher, when it is available.

    ``holdouts.json`` is an optional input of this build (phase 2 builds it after this module);
    when it is missing, or the API does not take ``(pgpid, shelfmark, canonical)``, the builder relies
    on :class:`HoldoutIndex` alone and says so in the log.

    :param p: Build paths (default location ``paths.out_dir / holdouts.json``).
    :param path: Explicit ``holdouts.json``.
    :returns: ``f(pgpid, shelfmark, cids) -> reason | None`` or None.
    """
    try:
        from src.finetuning.text27b import holdouts as _ho  # optional sibling module
        inst = _ho.Holdouts.load(path, p)
    except (ImportError, AttributeError, FileNotFoundError) as exc:
        LOG.info(f"holdouts module not used ({type(exc).__name__}: {exc})")
        return None

    def call(pgpid: str, shelfmark: Optional[str], cids: Sequence[str]) -> Optional[str]:
        """Ask the holdouts module once per canonical id (or once without one).

        :param pgpid: PGP document id.
        :param shelfmark: Shelfmark.
        :param cids: Canonical ids.
        :returns: The first reason given, or None.
        """
        for cid in (list(cids) or [None]):
            reason = inst.is_benchmark(pgpid, shelfmark, cid)
            if reason:
                return str(reason)
        return None

    try:
        call("0", None, [])
    except TypeError as exc:
        LOG.warning(f"holdouts module API mismatch, not used: {exc}")
        return None
    return call


def assign_split(v0_split: str, group_reasons: Iterable[str]) -> str:
    """v0_2 split of a document from its v0 group split and its group's hold-out reasons (B10).

    :param v0_split: ``train`` / ``dev`` / ``test`` from :func:`builder_v0.group_splits`.
    :param group_reasons: Union of the hold-out reasons of the document's shelfmark group.
    :returns: ``train``, ``dev``, ``heldout`` (PGP-131, probe, other H), ``heldout_vision`` or ``test_indist``.
    :raises AssertionError: When a held-out group is not a test group.
    """
    rs = set(group_reasons)
    if rs and v0_split != "test":
        raise AssertionError(f"held-out group in {v0_split}: {sorted(rs)}")
    if v0_split in ("train", "dev"):
        return v0_split
    if "pgp131" in rs:
        return "heldout"
    if rs & VISION_REASONS:
        return "heldout_vision"
    if rs:
        return "heldout"
    return "test_indist"


# ============================================================================== rows
def make_row_v2(family: str, condition: str, doc: Dict[str, Any], user: str, target: str,
                meta_extra: Dict[str, Any], tc: bv.TokenCounter) -> Dict[str, Any]:
    """Assemble one v0_2 row (v0 meta + the v0_2 fields of brief rule 5).

    ``tokens`` / ``target_tokens`` are the v0 estimates here; :func:`finalize_tokens` replaces them
    with exact rendered counts.

    :param family: Row family.
    :param condition: Input condition.
    :param doc: Document info (pgpid, shelfmark, group, split, split_v2, held_out, lang, holdout_reasons).
    :param user: User prompt.
    :param target: Assistant target.
    :param meta_extra: Extra meta fields.
    :param tc: Token counter.
    :returns: Row dict.
    """
    msgs = _rows.make_messages(user, target)
    meta: Dict[str, Any] = {
        "task": family, "family": family, "lang_class": doc["lang"], "condition": condition, "pgpid": doc["pgpid"],
        "shelfmark": doc["shelfmark"], "shelfmark_group": doc["group"], "split": doc["split_v2"],
        "split_v0": doc["split"], "held_out_vision_benchmark": doc["held_out"],
        "holdout_reasons": doc.get("holdout_reasons", []), "target_chars": len(msgs[2]["content"]), "input_cer": None,
        "tokens": tc.messages(msgs), "target_tokens": tc.count(msgs[2]["content"]), "read_model": None,
        "target_raw": None, "strip_ops": [], "pgp_description_dup": False, "reader_exposed": None,
        "exposure_level": None, "noise_band": None, "noise_class": None, "build": BUILD}
    meta.update(meta_extra)
    side = meta.get("side")
    meta["row_id"] = _rows.row_id(family, doc["pgpid"], condition, meta.get("read_model"), *([side] if side else []))
    if condition in _rows.READ_LIKE_CONDITIONS:
        meta["noise_band"] = noise_band(meta.get("input_cer"))
        meta["noise_class"] = ("synthetic" if condition == "noised_edition"
                               else "read_gate" if meta.get("input_cer") is not None else "read_only_pool")
    return {"messages": msgs, "meta": meta}


class BuilderV2(bv.Builder):
    """The v0 builder with the v0_2 fixes (prompts, strips, B7/B8/B9, F1 band, v0_2 meta)."""

    def __init__(self, tc: bv.TokenCounter, noiser: bv.Noiser, people_ix: bv.PeopleIndex,
                 docs_csv: Dict[str, Dict[str, str]]) -> None:
        """Create the builder.

        :param tc: Token counter.
        :param noiser: Noise sampler.
        :param people_ix: people.csv index.
        :param docs_csv: pgpid -> documents.csv row.
        """
        super().__init__(tc, noiser, people_ix, docs_csv)
        self.fixes: collections.Counter = collections.Counter()
        self.strip_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ inputs and rows
    def inputs(self, doc: Dict[str, Any], family: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """v0 input variants without noised editions of Arabic documents (B8).

        :param doc: Document info.
        :param family: Task family (seeds the noise draw as in v0).
        :returns: ``[(condition, text, meta extras)]``.
        """
        clean_ed, n_scrub = strip_source_names_input(doc.get("edition"))
        if n_scrub:  # strip before the noise draw so a noised copy cannot keep a mangled project-name string
            self.fixes[f"scrub_stripped_edition_inputs:{family}"] += 1
            doc = {**doc, "edition": clean_ed}
        out = super().inputs(doc, family)
        if doc.get("script") == "arabic" or doc.get("lang") == "AR":
            n = sum(1 for cond, _, _ in out if cond == "noised_edition")
            if n:
                self.fixes[f"B8_noised_ar_dropped:{family}"] += n
                out = [x for x in out if x[0] != "noised_edition"]
        return out

    def add(self, family: str, doc: Dict[str, Any], target: str, meta_extra: Dict[str, Any],
            extra_block: str = "", allow_input_truncation: bool = True,
            inputs: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None) -> int:
        """Emit v0_2 rows of one family for every input variant of ``doc`` (v0 cap and truncation logic).

        :param family: Task family.
        :param doc: Document info.
        :param target: Assistant target.
        :param meta_extra: Meta fields shared by the variants.
        :param extra_block: Context block appended to the prompt.
        :param allow_input_truncation: Truncate long inputs instead of dropping the row.
        :param inputs: Precomputed input variants (default :meth:`inputs`); an ``input_script`` key in a
            variant's extras overrides the document script in the prompt.
        :returns: Rows added.
        """
        n = 0
        for cond, text, mx in (inputs if inputs is not None else self.inputs(doc, family)):
            mx = dict(mx)
            text, n_scrub = strip_source_names_input(text)
            if n_scrub:
                self.fixes[f"scrub_stripped_inputs:{family}:{cond}"] += 1
            script = mx.pop("input_script", None) or doc["script"]
            fixed = (self.tc.count(bv.SYSTEM) + self.tc.count(target) + self.tc.count(extra_block) + 260
                     + 3 * bv.CHAT_OVERHEAD)
            budget = bv.MAX_TOKENS - fixed
            truncated = False
            if self.tc.count(text) > budget:
                if not allow_input_truncation or budget < 300:
                    self.drops[f"{family}:{cond}:over_cap"] += 1
                    continue
                text, truncated = bv.truncate_lines(text, budget, self.tc)
            user = _rows.user_prompt(family, cond, script, text, extra_block)
            row = make_row_v2(family, cond, doc, user, target, {**meta_extra, **mx, "input_truncated": truncated}, self.tc)
            if script != doc["script"]:
                row["meta"]["input_script"] = script
            if row["meta"]["tokens"] > bv.MAX_TOKENS:
                self.drops[f"{family}:{cond}:over_cap_after_build"] += 1
                continue
            self.rows.append(row)
            n += 1
        return n

    def add_rescript(self, doc: Dict[str, Any], target: str, meta: Dict[str, Any]) -> int:
        """B9: an ``edition_rescript`` translate_en row for an Arabic-script Arabic document.

        :param doc: Document (or side) info with the Arabic-script edition.
        :param target: The human English translation (unchanged).
        :param meta: Meta of the edition row.
        :returns: Rows added (0 when the document is not Arabic).
        """
        if not (doc.get("script") == "arabic" and doc.get("lang") == "AR" and doc.get("edition")):
            return 0
        text = tn.transliterate_ar_to_ja_plain(doc["edition"])
        mx = {"input_source": f"rescript_ar_to_ja_plain:{doc['edition_slug']}", "input_script": "hebrew",
              "rescript": "transliterate_ar_to_ja_plain"}
        n = self.add("translate_en", doc, target, meta, allow_input_truncation=False,
                     inputs=[("edition_rescript", text, mx)])
        self.fixes["B9_rescript_rows"] += n
        return n

    # ------------------------------------------------------------------ families
    def he_edition_doc(self, doc: Dict[str, Any], prow: Dict[str, Any]) -> Dict[str, Any]:
        """B7: the document info with the edition matched to the Hebrew translation, when it differs.

        :param doc: Document info built on ``edition_text``.
        :param prow: pgp_parallel v1.1 row (``edition_text_he`` / ``edition_source_slug_he``).
        :returns: ``doc`` itself, or a copy with the Hebrew-matched edition, its script and its reads.
        """
        he = prow.get("edition_text_he")
        if not he or he == prow.get("edition_text"):
            return doc
        edition, _ = bv.strip_latin_lines(he)
        if not edition or edition == doc["edition"]:
            return doc
        slug = prow.get("edition_source_slug_he") or doc["edition_slug"]
        entry = next((e for e in prow.get("editions", []) if e.get("source_slug") == slug), None)
        script = (entry or {}).get("script") or tn.dominant_script(edition)
        if script not in ("hebrew", "arabic"):
            self.fixes["B7_he_edition_not_semitic_kept_default"] += 1
            return doc
        self.fixes["B7_he_edition_docs"] += 1
        return {**doc, "edition": edition, "edition_slug": slug, "script": script,
                "reads_ok": bv.read_variants(prow, edition), "edition_variant": "he"}

    def translate(self, doc: Dict[str, Any], prow: Dict[str, Any]) -> None:
        """translate_en / translate_he rows with B3/B4 project-name strips, the F1 band, B7 and B9 (else as v0).

        :param doc: Document info.
        :param prow: pgp_parallel row.
        """
        if doc["script"] not in ("hebrew", "arabic"):
            self.drops["translate:latin_or_mixed_edition"] += 1
            return
        for lang, family in (("en", "translate_en"), ("he", "translate_he")):
            raw = prow.get(f"translation_{lang}")
            if not raw:
                continue
            tr_entry = next((t for t in prow["translations"] if t.get("language") == lang and t.get("text") == raw), None)
            fdoc = self.he_edition_doc(doc, prow) if lang == "he" else doc
            tgt, ops = clean_target(raw, family)
            if not tgt:
                self.drops[f"{family}:target_empty_after_strip"] += 1
                continue
            if bv.overlap60(tgt, fdoc["edition"]):
                self.drops[f"{family}:target_overlaps_edition"] += 1
                continue
            ok, ratio = band_ok(fdoc["edition"], tgt, lang)
            if not ok:
                self.drops[f"{family}:length_ratio_F1"] += 1
                continue
            meta = {"source": f"pgp_digital_translation:{tr_entry['source_slug'] if tr_entry else '?'}",
                    "licence": bv.PGP_LICENCE,
                    "provenance": {"edition_slug": fdoc["edition_slug"],
                                   "translation_slug": tr_entry["source_slug"] if tr_entry else None,
                                   "translation_location": tr_entry.get("location") if tr_entry else None},
                    "target_lang": lang, "side": None, "target_raw": raw if ops else None, "strip_ops": ops,
                    "length_ratio": round(ratio, 3), "edition_matched_to": "he" if fdoc is not doc else "default"}
            if ops:
                self.fixes[f"B3B4_stripped_targets:{family}"] += 1
            whole = self.tc.count(fdoc["edition"]) + self.tc.count(tgt) + self.tc.count(bv.SYSTEM) + 300
            if whole <= bv.MAX_TOKENS:
                self.add(family, fdoc, tgt, meta, allow_input_truncation=False)
                if family == "translate_en":
                    self.add_rescript(fdoc, tgt, meta)
                continue
            self.translate_sides(family, lang, fdoc, prow, tr_entry, meta)

    def translate_sides(self, family: str, lang: str, doc: Dict[str, Any], prow: Dict[str, Any],
                        tr_entry: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> int:
        """Over-cap translations split by side labels present in both raw footnotes (v0 logic, v0_2 strips / band).

        :param family: ``translate_en`` / ``translate_he``.
        :param lang: ``en`` / ``he``.
        :param doc: Document info (edition already chosen for the language).
        :param prow: pgp_parallel row.
        :param tr_entry: The translation entry of the row.
        :param meta: Meta of the whole-document row.
        :returns: Rows added.
        """
        ed_entry = next((e for e in prow["editions"] if e["source_slug"] == doc["edition_slug"]), None)
        ed_sides = bv.split_sides(ed_entry["raw"]) if ed_entry else []
        tr_sides = bv.split_sides(tr_entry["raw"]) if tr_entry else []
        if not ed_sides or [k for k, _ in ed_sides] != [k for k, _ in tr_sides]:
            if ed_sides and tr_sides and sorted(k for k, _ in ed_sides) == sorted(k for k, _ in tr_sides):
                tr_map = dict(tr_sides)
                tr_sides = [(k, tr_map[k]) for k, _ in ed_sides]
            else:
                self.drops[f"{family}:over_cap_no_side_alignment"] += 1
                return 0
        added = 0
        for (k, ed_t), (_, tr_raw) in zip(ed_sides, tr_sides):
            ed_t = bv.strip_latin_lines(ed_t)[0]
            sdoc = {**doc, "edition": ed_t, "side": k, "reads_ok": []}  # reads are not side-aligned
            tr_t, sops = clean_target(tr_raw, family)
            n_ed, n_tr = len(tn.normalize_for_cer(ed_t)), len(tn.normalize_for_cer(tr_t))
            ok, ratio = band_ok(ed_t, tr_t, lang)
            if n_ed < 40 or n_tr < 40 or not ok:
                self.drops[f"{family}:side_ratio"] += 1
                continue
            smeta = {**meta, "side": k, "target_raw": tr_raw if sops else None, "strip_ops": sops,
                     "length_ratio": round(ratio, 3)}
            added += self.add(family, sdoc, tr_t, smeta, allow_input_truncation=False)
            if family == "translate_en":
                added += self.add_rescript(sdoc, tr_t, smeta)
        self.drops[f"{family}:side_split_docs"] += 1
        if not added:
            self.drops[f"{family}:side_split_no_rows"] += 1
        return added

    def catalogue_families(self, doc: Dict[str, Any], prow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """catalogue_json, describe_pgp and extract_relations rows from the B4-stripped description.

        The v0 cleaned description is stripped (:func:`clean_target`), then the catalogue record,
        the description target and the relations are all derived from the stripped text; the v0
        versions are kept as ``target_raw`` where they differ.

        :param doc: Document info.
        :param prow: Normalised document fields (type, tags, places, languages, description).
        :returns: The catalogue record (for the analysis context), or None when the description is unusable.
        """
        if doc.get("edition") and doc["script"] not in ("hebrew", "arabic"):
            self.drops["catalogue:latin_or_mixed_edition"] += 1
            return None
        raw_summary, acts = bv.clean_description(prow["description"])
        self.desc_actions.update(acts)
        summary, ops = clean_target(raw_summary, "describe_pgp")
        if len(summary) < 40:
            self.drops["catalogue:description_too_short" + ("_after_strip" if len(raw_summary) >= 40 else "")] += 1
            return None
        dcsv = self.docs_csv.get(doc["pgpid"], {})
        cat = bv.catalogue_record(dcsv, prow, summary, self.people_ix)
        cat_raw = bv.catalogue_record(dcsv, prow, raw_summary, self.people_ix) if ops else cat
        base = {"licence": bv.PGP_LICENCE, "provenance": {"pgp_document": doc["pgpid"], "edition_slug": doc.get("edition_slug")}}
        if ops:
            self.fixes["B4_descriptions_stripped"] += 1
            self.strip_records.append({"family": "describe_pgp", "pgpid": doc["pgpid"], "split": doc["split_v2"],
                                       "ops": ops, "before": raw_summary, "after": summary})
        variants = self.inputs(doc, "catalogue")
        if not variants:
            return cat
        cat_json = json.dumps(cat, ensure_ascii=False)
        n_cat = self.add("catalogue_json", doc, cat_json,
                         {**base, "source": "pgp_catalogue_fields", "strip_ops": ops,
                          "target_raw": json.dumps(cat_raw, ensure_ascii=False) if ops else None}, inputs=variants)
        if len(summary) >= 80:
            self.add("describe_pgp", doc, summary,
                     {**base, "source": "pgp_description_cleaned", "strip_ops": ops,
                      "target_raw": raw_summary if ops else None, "pgp_description_dup": bool(n_cat)},
                     inputs=self.inputs(doc, "describe_pgp"))
        elif len(raw_summary) >= 80:
            self.drops["describe_pgp:too_short_after_strip"] += 1
        rel = bv.relations_from_catalogue(cat, prow)
        rel_raw = bv.relations_from_catalogue(cat_raw, prow)
        if rel:
            changed = rel != rel_raw
            self.fixes["relations_changed_by_strip"] += changed
            self.add("extract_relations", doc, json.dumps({"relations": rel}, ensure_ascii=False),
                     {**base, "source": "pgp_gold_places+description_people", "n_relations": len(rel),
                      "strip_ops": ops if changed else [],
                      "target_raw": json.dumps({"relations": rel_raw}, ensure_ascii=False) if changed else None},
                     inputs=self.inputs(doc, "extract_relations"))
        return cat


# ============================================================================== analysis (carried forward)
_DOC_END = "</document>\n\n"
_TR_MARK = "Translation (from the edition):\n"


def load_v01_analysis(v0_1_dir: Path) -> Dict[str, Dict[str, Any]]:
    """The v0_1 ``analyse_document`` targets, one per document (from its edition-condition row).

    :param v0_1_dir: ``text_ft_v0_1`` directory (``{train,dev,test}.jsonl``).
    :returns: pgpid -> ``{"target", "extra", "meta"}`` (``extra`` = the block after the document).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        path = Path(v0_1_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for r in _rows.read_jsonl(path):
            m = r["meta"]
            if m.get("task") != "analyse_document" or m.get("condition") != "edition":
                continue
            user = r["messages"][1]["content"]
            i = user.find(_DOC_END)
            out[str(m["pgpid"])] = {"target": r["messages"][2]["content"], "meta": m,
                                    "extra": user[i + len(_DOC_END):] if i >= 0 else ""}
    return out


def carry_forward_analysis(b: BuilderV2, v01: Dict[str, Dict[str, Any]], docinfo: Dict[str, Dict[str, Any]],
                           cats: Dict[str, Dict[str, Any]], scholar_names: Set[str]) -> Dict[str, Any]:
    """analyse_document rows from the v0_1 targets, re-stripped (B4/B3) and with a v0_2 context block.

    The context block is rebuilt from the v0_2 catalogue record (summary withheld when the target
    copies it, as in v0); a tier-B translation block is kept from v0_1. ``grounding_share_v2`` is
    recomputed on the stripped target (reported, not gated).

    :param b: Builder.
    :param v01: Output of :func:`load_v01_analysis`.
    :param docinfo: pgpid -> document info of the edition documents in this build.
    :param cats: pgpid -> v0_2 catalogue record.
    :param scholar_names: Folded scholar surnames for the grounding test.
    :returns: Report counts.
    """
    rep: collections.Counter = collections.Counter()
    for pgpid in sorted(v01, key=lambda x: int(x) if x.isdigit() else 0):
        if pgpid not in docinfo:
            rep["doc_not_in_build"] += 1
            continue
        src, doc = v01[pgpid], docinfo[pgpid]
        target, ops = clean_target(src["target"], "analyse_document")
        if len(target) < ANALYSIS_MIN_CHARS:
            rep["dropped_short_after_strip"] += 1
            continue
        target, eops, _ev = analysis_evidence_strip(target)
        if target is None:
            rep["dropped_short_after_evidence_strip"] += 1
            continue
        ops = ops + eops
        if ops:
            b.strip_records.append({"family": "analyse_document", "pgpid": pgpid, "split": doc["split_v2"], "ops": ops,
                                    "before": src["target"], "after": target})
        cat = cats.get(pgpid)
        ctx = ("Known catalogue context:\n" + bv.context_block(cat)) if cat else ""
        if ctx and bv.overlap60(target, ctx):
            rep["context_description_withheld_overlap"] += 1
            ctx = "Known catalogue context:\n" + bv.context_block({**cat, "summary": None})
        j = src["extra"].find(_TR_MARK)
        tr_part = src["extra"][j:] if j >= 0 else ""
        extra = (ctx + "\n\n" if ctx and tr_part else ctx) + tr_part
        share, n_c, _ = tn.grounding(target, doc["edition"] + "\n" + extra, scholar_names)
        m0 = src["meta"]
        meta = {k: m0.get(k) for k in ("source", "licence", "provenance", "grounding_share", "grounding_candidates",
                                       "grounding_tier", "target_truncated", "target_lang")}
        meta.update({"grounding_share_v2": round(share, 3), "grounding_candidates_v2": n_c, "analysis_source": "v0_1",
                     "target_raw": src["target"] if ops else None, "strip_ops": ops})
        n = b.add("analyse_document", doc, target, meta, extra_block=extra)
        rep["docs_with_rows"] += bool(n)
        rep["rows"] += n
        rep["grounding_v2_below_0.6"] += share < bv.GROUNDING_PASS
    return dict(rep)


# ============================================================================== exposure, tokens, checks
def apply_exposure(rows: List[Dict[str, Any]], path: Optional[Path],
                   drop_below_cer: Optional[float] = EXPOSED_NEAR_DUP_CER) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Tag rows with reader exposure via the exposure module (when ``exposure.jsonl`` exists) and drop
    exposed train reads that are near-copies of the edition (B2).

    :param rows: Rows (meta ``reader_exposed`` / ``exposure_level`` default None).
    :param path: ``exposure.jsonl``; None or missing leaves the tags at None.
    :param drop_below_cer: Drop train ``read`` rows with ``reader_exposed`` and ``input_cer`` below this; None keeps all.
    :returns: ``(rows, report)``.
    """
    if path is None or not Path(path).exists():
        return rows, {"applied": False, "path": str(path) if path else None}
    from src.finetuning.text27b import exposure as _exp  # sibling module that writes exposure.jsonl

    stats: collections.Counter = collections.Counter()
    rows = _exp.tag_rows(rows, Path(path), stats)
    rep: Dict[str, Any] = {"applied": True, "path": str(path), "levels": dict(stats), "dropped_exposed_near_dup": 0}
    if drop_below_cer is not None:
        keep = []
        for r in rows:
            if r["meta"]["split"] == "train" and _exp.near_duplicate(r, drop_below_cer):
                rep["dropped_exposed_near_dup"] += 1
                continue
            keep.append(r)
        rows = keep
    return rows, rep


def finalize_tokens(rows: List[Dict[str, Any]], max_tokens: int = MAX_ROW_TOKENS, batch: int = 256,
                    tokenizer_path: Optional[Path] = None, template_path: Optional[Path] = None
                    ) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Exact token counts: render each row with ``enable_thinking=False`` (render contract checked) and tokenize.

    Sets ``meta.tokens`` (full rendered row), ``meta.target_tokens`` and ``meta.target_chars``; rows
    over ``max_tokens`` are dropped.

    :param rows: Rows.
    :param max_tokens: Cap on the rendered length.
    :param batch: Rows per tokenizer batch.
    :param tokenizer_path: Explicit ``tokenizer.json``.
    :param template_path: Explicit ``chat_template.jinja``.
    :returns: ``(kept rows, drop counter per family)``.
    """
    kept: List[Dict[str, Any]] = []
    drops: collections.Counter = collections.Counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        rendered = [_rows.render_chat(r["messages"], template_path=template_path) for r in chunk]
        n_full = _rows.count_tokens_batch(rendered, tokenizer_path)
        n_tgt = _rows.count_tokens_batch([r["messages"][2]["content"] for r in chunk], tokenizer_path)
        for r, nf, nt in zip(chunk, n_full, n_tgt):
            r["meta"]["tokens"], r["meta"]["target_tokens"] = nf, nt
            r["meta"]["target_chars"] = len(r["messages"][2]["content"])
            if nf > max_tokens:
                drops[r["meta"]["family"]] += 1
                continue
            kept.append(r)
    return kept, drops


def count_table(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Rows and tokens per family × condition × lang_class × split, plus per family × split totals.

    :param rows: Rows.
    :returns: ``{"by_cell": {family: {condition: {lang: {split: {"rows", "tokens"}}}}}, "by_family_split": ...,
        "total": {"rows", "tokens"}}``.
    """
    cell: Dict[str, Any] = {}
    fam: Dict[str, Any] = {}
    tot = {"rows": 0, "tokens": 0}
    for r in rows:
        m = r["meta"]
        c = cell.setdefault(m["family"], {}).setdefault(m["condition"], {}).setdefault(m["lang_class"], {}) \
            .setdefault(m["split"], {"rows": 0, "tokens": 0})
        f = fam.setdefault(m["family"], {}).setdefault(m["split"], {"rows": 0, "tokens": 0})
        for d in (c, f, tot):
            d["rows"] += 1
            d["tokens"] += int(m.get("tokens") or 0)
    return {"by_cell": cell, "by_family_split": fam, "total": tot}


def _target_prose(row: Dict[str, Any]) -> str:
    """Prose of a target for marker checks (the summary for catalogue rows, '' for relations).

    :param row: Row.
    :returns: Prose text.
    """
    fam, tgt = row["meta"]["family"], row["messages"][2]["content"]
    if fam == "catalogue_json":
        return json.loads(tgt).get("summary") or ""
    return "" if fam == "extract_relations" else tgt


def checks_v02(rows: List[Dict[str, Any]], held_pgpids: Set[str], held_cids: Set[str],
               cids_by_pgpid: Dict[str, List[str]], pgp131_pgpids: Set[str],
               bodleian: Dict[str, List[str]]) -> Dict[str, Any]:
    """v0 sanity checks plus the v0_2 ones (every ``*_violations`` must be 0).

    :param rows: Final rows.
    :param held_pgpids: pgpids of every held-out document (any reason).
    :param held_cids: Held-out canonical ids (PGP-131 + registries).
    :param cids_by_pgpid: pgpid -> canonical ids.
    :param pgp131_pgpids: pgpids of documents matched to PGP-131 (any route).
    :param bodleian: PGP-131 ids without a pgpid in the merged map -> pgpids matched by shelfmark key.
    :returns: Check results.
    """
    out = bv.sanity_checks(rows, held_pgpids, held_cids, cids_by_pgpid)
    marker = scrub_t = scrub_i = ar_noised = rescript_bad = schema = 0
    ids: collections.Counter = collections.Counter()
    pgp131_split: collections.Counter = collections.Counter()
    examples: Dict[str, List[str]] = collections.defaultdict(list)
    for r in rows:
        m = r["meta"]
        ids[m["row_id"]] += 1
        prose = _target_prose(r)
        if prose and strip_markers(prose)[1]:
            marker += 1
            if len(examples["marker"]) < 3:
                examples["marker"].append(m["row_id"])
        if tn.has_source_names(r["messages"][2]["content"]):
            scrub_t += 1
        if tn.has_source_names(r["messages"][1]["content"]):
            scrub_i += 1
            if len(examples["scrub_input"]) < 3:
                examples["scrub_input"].append(m["row_id"])
        if m["condition"] == "noised_edition" and m["lang_class"] == "AR":
            ar_noised += 1
        if m["condition"] == "edition_rescript" and m["family"] != "translate_en":
            rescript_bad += 1
        if _rows.validate_row(r):
            schema += 1
            if len(examples["schema"]) < 3:
                examples["schema"].append(f"{m.get('row_id')}: {_rows.validate_row(r)[:2]}")
        if m["pgpid"] in pgp131_pgpids:
            pgp131_split[m["split"]] += 1
    bod_rows: Dict[str, Dict[str, int]] = {}
    bod_bad = 0
    for cid, pgs in bodleian.items():
        c = collections.Counter(r["meta"]["split"] for r in rows if r["meta"]["pgpid"] in set(pgs))
        bod_rows[cid] = dict(c)
        bod_bad += sum(v for k, v in c.items() if k != "heldout")
    out.update({
        "line_marker_rows_violations": marker,
        "scrub_target_rows_violations": scrub_t,
        "scrub_input_rows_violations": scrub_i,
        "ar_noised_rows_violations": ar_noised,
        "rescript_outside_translate_en_violations": rescript_bad,
        "duplicate_row_id_violations": sum(n - 1 for n in ids.values() if n > 1),
        "schema_violations": schema,
        "pgp131_rows_not_heldout_violations": sum(v for k, v in pgp131_split.items() if k != "heldout"),
        "pgp131_rows_by_split": dict(pgp131_split),
        "bodleian_unmapped_rows_not_heldout_violations": bod_bad,
        "bodleian_unmapped_rows_by_split": bod_rows,
        "violation_examples": dict(examples),
    })
    return out


def audit_sample(records: List[Dict[str, Any]], n: int = AUDIT_SIZE, seed: int = bv.SEED) -> List[Dict[str, Any]]:
    """Stratified before/after sample of B4 strips for the hand audit (round-robin over the first B4 rule).

    :param records: Strip records ``{family, pgpid, split, ops, before, after}``.
    :param n: Sample size.
    :param seed: Shuffle seed.
    :returns: Up to ``n`` records with ``chars_removed`` added, analysis records first in each round.
    """
    b4 = [r for r in records if any(o.startswith("B4:") for o in r["ops"])]
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(b4, key=lambda x: (x["family"], int(x["pgpid"]) if str(x["pgpid"]).isdigit() else 0)):
        first = next(o for o in r["ops"] if o.startswith("B4:"))
        strata[(r["family"], first)].append(r)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata, key=lambda k: (k[0] != "analyse_document", k))
    out: List[Dict[str, Any]] = []
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                r = strata[k].pop()
                out.append({**r, "chars_removed": len(r["before"]) - len(r["after"])})
    return out


# ============================================================================== orchestration
@dataclasses.dataclass
class V2Config:
    """Inputs, outputs and switches of one v0_2 build."""

    v0: bv.V0Config
    out_dir: Path
    analysis_source: str = "v0_1"
    exposure_path: Optional[Path] = None
    use_holdouts_module: bool = True
    drop_exposed_below_cer: Optional[float] = EXPOSED_NEAR_DUP_CER
    max_row_tokens: int = MAX_ROW_TOKENS
    chat_template: Optional[Path] = None
    holdouts_path: Optional[Path] = None

    @classmethod
    def from_paths(cls, p: Optional[_paths.Paths] = None, out_dir: Optional[Path] = None,
                   local: Optional[Dict[str, Any]] = None, **kw: Any) -> "V2Config":
        """Build a config from ``paths.json``.

        :param p: Loaded paths (default :func:`paths.load_paths`).
        :param out_dir: Output root (default ``paths.out_dir``).
        :param local: Local builder config (default :func:`builder_v0.load_local_config`).
        :param kw: Further :class:`V2Config` fields.
        :returns: The config.
        """
        p = p or _paths.load_paths()
        out = Path(out_dir) if out_dir is not None else p.out_dir
        v0 = bv.V0Config.from_paths(p, out, local)
        v0.noise_model_path = out / "noise_model.json"
        kw.setdefault("exposure_path", out / "exposure.jsonl")
        kw.setdefault("holdouts_path", p.out("holdouts.json"))
        kw.setdefault("chat_template", p.chat_template)
        return cls(v0=v0, out_dir=out, **kw)


def select_docs(items: Sequence[Any], key: Callable[[Any], str], limit: Optional[int],
                include: Optional[Set[str]]) -> List[Any]:
    """Sample selection: the first ``limit`` items plus every item whose pgpid is in ``include`` (order kept).

    :param items: Items in build order.
    :param key: pgpid of an item.
    :param limit: Leading items to keep (None = all).
    :param include: pgpids always kept.
    :returns: Selected items.
    """
    if limit is None and not include:
        return list(items)
    head = set(range(limit)) if limit is not None else set()
    inc = include or set()
    return [it for i, it in enumerate(items) if i in head or key(it) in inc]


def build_v02(cfg: V2Config, limit: Optional[int] = None, include: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Build the v0_2 PGP-family rows in memory.

    Hold-out reasons, groups, splits and the noise model are computed over every document even for
    a sample build, so sampled documents get the rows the full build gives them.

    :param cfg: Build configuration.
    :param limit: Process the first ``limit`` edition documents and the first ``limit`` pool documents.
    :param include: pgpids processed in addition to the first ``limit``.
    :returns: Result dict (rows, reports, hold-out sets) for :func:`write_v02`.
    """
    t0 = time.time()
    v0 = cfg.v0
    bv.STATS.clear()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    bv.load_place_names(v0.pgp_dir)
    tc = bv.TokenCounter(v0.tokenizer)
    prows = bv.load_jsonl(v0.pairs)
    docs_csv = {r["pgpid"]: r for r in bv.load_csv(v0.pgp_dir / "documents.csv")}
    merged = bv.build_merged_map(v0.merged_map, v0.merged_jsonl)
    with open(v0.benchmark_ids, encoding="utf-8") as fh:
        b131 = json.load(fh)
    registry = v0.registry_fn()
    probes = bv.probe_pgpids(v0.samples)
    hold = HoldoutIndex.build(merged, b131, registry, probes,
                              external=holdouts_module_adapter(path=cfg.holdouts_path) if cfg.use_holdouts_module else None)
    cids_by_pgpid: Dict[str, List[str]] = collections.defaultdict(list)
    for cid, info in merged.items():
        for p in info["pgpids"]:
            cids_by_pgpid[p].append(cid)
    prow_by = {r["pgpid"]: r for r in prows}
    all_pgpids = list(docs_csv) + [p for p in prow_by if p not in docs_csv]

    # ---- hold-out reasons for every document, then group splits (B10 / §9.0)
    reasons: Dict[str, Dict[str, str]] = {}
    doc_keys: Dict[str, Set[str]] = {}
    for p in all_pgpids:
        pr = prow_by.get(p)
        shelf = pr["shelfmark"] if pr else docs_csv[p].get("shelfmark")
        cids = sorted(set(cids_by_pgpid.get(p, [])) | set(pr["merged_doc_ids"] if pr else []))
        doc_keys[p] = hold.doc_keys(shelf, cids)
        r = hold.match(p, shelf, cids, bool(pr and pr["held_out_vision_benchmark"]), doc_keys[p])
        if r:
            reasons[p] = r
    groups = bv.build_groups(merged, all_pgpids, v0.pgp_dir / "fragments.csv")
    gsplit = bv.group_splits(groups, set(reasons))
    group_reasons: Dict[str, Set[str]] = collections.defaultdict(set)
    for p, r in reasons.items():
        group_reasons[groups.get(p, "g" + p)] |= set(r)
    unmapped = sorted(c for c in hold.pgp131_ids if not merged.get(c, {}).get("pgpids"))
    bodleian = {c: sorted((p for p, ks in doc_keys.items() if _sm.canonical_to_key(c) in ks), key=int) for c in unmapped}
    pgp131_pgpids = {p for p, r in reasons.items() if "pgp131" in r}

    def split_info(p: str) -> Dict[str, Any]:
        """Split fields of a document.

        :param p: pgpid.
        :returns: ``group``, ``split`` (v0), ``split_v2``, ``held_out``, ``holdout_reasons``.
        """
        g = groups.get(p, "g" + p)
        s0 = gsplit.get(g, "test")
        own = set(reasons.get(p, {}))
        return {"group": g, "split": s0, "split_v2": assign_split(s0, group_reasons.get(g, set())),
                "held_out": bool(own & BENCH_REASONS), "holdout_reasons": sorted(group_reasons.get(g, set()))}

    # ---- noise model (as v0)
    read_texts = [bv.clean_read(rd["text"])[0] for r in prows for rd in r.get("reads", [])]
    nm = bv.build_noise_model(prows, read_texts, v0.noise_model_path)
    noiser = bv.Noiser(nm)
    calib_pool: Dict[str, List[str]] = collections.defaultdict(list)
    for r in prows:
        if r["edition_script"] in ("hebrew", "arabic") and bv.split_of(r["pgpid"]) == "train" and len(r["edition_text"]) >= 300:
            calib_pool[r["edition_script"]].append(r["edition_text"])
    calib = noiser.calibrate(calib_pool)
    b = BuilderV2(tc, noiser, bv.PeopleIndex(v0.pgp_dir / "people.csv"), docs_csv)
    LOG.info(f"[{time.time() - t0:6.1f}s] setup done: {len(prows):,} pairs rows, {len(reasons):,} held-out docs")

    # ---- edition documents
    docinfo: Dict[str, Dict[str, Any]] = {}
    cats: Dict[str, Dict[str, Any]] = {}
    for prow in select_docs(prows, lambda r: r["pgpid"], limit, include):
        p = prow["pgpid"]
        edition, n_lat = bv.strip_latin_lines(prow["edition_text"])
        doc = {"pgpid": p, "shelfmark": prow["shelfmark"], **split_info(p),
               "lang": bv.coarse_lang(prow["lang_class"], prow.get("ja_score") or 0), "lang_fine": prow["lang_class"],
               "edition": edition, "edition_slug": prow["edition_source_slug"],
               "script": prow["edition_script"], "reads_ok": bv.read_variants(prow, edition)}
        docinfo[p] = doc
        b.translate(doc, prow)
        cat = b.catalogue_families(doc, bv.normalise_fields_from_parallel(prow))
        if cat:
            cats[p] = cat
    LOG.info(f"[{time.time() - t0:6.1f}s] edition docs {len(docinfo):,}: rows {len(b.rows):,}")

    # ---- read-only pool
    pool = bv.read_only_pool(merged, set(prow_by), docs_csv, v0.read_files)
    pool_items = select_docs(sorted(pool.items(), key=lambda kv: int(kv[0])), lambda kv: kv[0], limit, include)
    for p, info in pool_items:
        d = docs_csv[p]
        doc = {"pgpid": p, "shelfmark": d["shelfmark"], **split_info(p),
               "lang": bv.lang_from_pgp(d["languages_primary"]), "lang_fine": d["languages_primary"], "edition": None,
               "edition_slug": None, "script": bv.script_of(info["read"]["text"]), "reads_ok": [info["read"]]}
        b.catalogue_families(doc, bv.normalise_fields_from_csv(d))
    LOG.info(f"[{time.time() - t0:6.1f}s] pool docs {len(pool_items):,} of {len(pool):,}: rows {len(b.rows):,}")

    # ---- analysis (v0_1 targets carried forward)
    arep: Dict[str, Any] = {"source": cfg.analysis_source}
    if cfg.analysis_source == "v0_1" and v0.v0_1_dir is not None:
        names = tn.scholar_surnames(v0.pgp_dir / "sources.csv", v0.scholar_extra)
        arep.update(carry_forward_analysis(b, load_v01_analysis(v0.v0_1_dir), docinfo, cats, names))
    rows = b.rows

    # ---- exposure, exact tokens
    rows, exp_rep = apply_exposure(rows, cfg.exposure_path, cfg.drop_exposed_below_cer)
    rows, tok_drops = finalize_tokens(rows, cfg.max_row_tokens, tokenizer_path=v0.tokenizer, template_path=cfg.chat_template)
    LOG.info(f"[{time.time() - t0:6.1f}s] rows {len(rows):,} after exposure and the token cap")
    how = collections.Counter(f"{r}:{h}" for rs in reasons.values() for r, h in rs.items())
    return {"rows": rows, "builder": b, "noise_model": nm, "noise_calibration": calib, "docinfo": docinfo, "cats": cats,
            "groups": groups, "gsplit": gsplit, "reasons": reasons, "bodleian": bodleian, "pgp131_pgpids": pgp131_pgpids,
            "held_pgpids": set(reasons), "held_cids": hold.pgp131_ids | hold.registry_ids, "cids_by_pgpid": cids_by_pgpid,
            "analysis": arep, "exposure": exp_rep, "token_drops": dict(tok_drops), "holdout_how": dict(how),
            "holdouts_module_used": hold.external is not None, "n_pool_docs": len(pool_items), "seconds": round(time.time() - t0, 1)}


def _row_sort_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    """Deterministic order inside a family file.

    :param r: Row.
    :returns: Sort key (split, pgpid, condition, side, read model).
    """
    m = r["meta"]
    return (SPLIT_ORDER.get(m["split"], 99), int(m["pgpid"]) if str(m["pgpid"]).isdigit() else 0,
            COND_ORDER.get(m["condition"], 99), str(m.get("side") or ""), m.get("read_model") or "")


def write_v02(result: Dict[str, Any], cfg: V2Config, build_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``families/<family>.jsonl``, ``families/build_report.json`` and ``audit/strip_sample_50.jsonl``.

    :param result: Output of :func:`build_v02`.
    :param cfg: Build configuration.
    :param build_args: CLI arguments recorded in the report.
    :returns: The build report.
    """
    rows, b = result["rows"], result["builder"]
    fam_dir = cfg.out_dir / "families"
    by_fam: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_fam[r["meta"]["family"]].append(r)
    written: Dict[str, Any] = {}
    for fam in sorted(by_fam, key=lambda f: FAMILY_ORDER.get(f, 99)):
        part = sorted(by_fam[fam], key=_row_sort_key)
        n = _rows.write_jsonl(fam_dir / f"{fam}.jsonl", part)
        written[fam] = {"rows": n, "fingerprint": _rows.fingerprint(part)}
    reread = [r for fam in written for r in _rows.read_jsonl(fam_dir / f"{fam}.jsonl")]
    checks = checks_v02(reread, result["held_pgpids"], result["held_cids"], result["cids_by_pgpid"],
                        result["pgp131_pgpids"], result["bodleian"])
    audit = audit_sample(b.strip_records)
    _rows.write_jsonl(cfg.out_dir / "audit" / f"strip_sample_{AUDIT_SIZE}.jsonl", audit)
    fix_rows = collections.Counter()
    for r in reread:
        m = r["meta"]
        if m.get("strip_ops"):
            fix_rows[f"stripped:{m['family']}"] += 1
            for o in m["strip_ops"]:
                fix_rows[f"op:{m['family']}:{o}"] += 1
        fix_rows[f"split:{m['split']}"] += 1
        if m["condition"] == "edition_rescript":
            fix_rows["B9_rescript_rows"] += 1
        if m.get("edition_matched_to") == "he":
            fix_rows["B7_rows_he_edition"] += 1
        if m.get("pgp_description_dup"):
            fix_rows["B5_describe_dup_rows"] += 1
    nm = result["noise_model"]
    report = {
        "build": BUILD, "built": time.strftime("%Y-%m-%d %H:%M"), "build_args": build_args or {},
        "counts": count_table(reread), "families": written, "checks": checks,
        "fixes": {"builder": dict(b.fixes), "rows": dict(sorted(fix_rows.items()))},
        "drops": dict(b.drops), "description_cleaning": dict(b.desc_actions), "stats_v0": dict(bv.STATS),
        "holdouts": {"documents": len(result["reasons"]), "how": result["holdout_how"],
                     "module_used": result["holdouts_module_used"],
                     "bodleian_unmapped_pgp131": result["bodleian"],
                     "group_split_counts": dict(collections.Counter(result["gsplit"].values()))},
        "analysis": result["analysis"], "exposure": result["exposure"], "token_cap_drops": result["token_drops"],
        "audit_sample": {"file": f"audit/strip_sample_{AUDIT_SIZE}.jsonl", "rows": len(audit),
                         "b4_records": sum(1 for r in b.strip_records if any(o.startswith("B4:") for o in r["ops"]))},
        "noise_model_summary": {s: {k: v[k] for k in ("n_pairs", "edition_chars", "cer_empirical", "rate_sub", "rate_del", "rate_ins")}
                                for s, v in nm["scripts"].items()},
        "noise_calibration": {k: round(v, 3) for k, v in result["noise_calibration"].items()},
        "documents": {"edition": len(result["docinfo"]), "pool": result["n_pool_docs"]},
        "seconds": result["seconds"]}
    with open(fam_dir / "build_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: build the v0_2 PGP families.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: None.
    :raises SystemExit: When a sample build (``--limit`` / ``--include``) does not name its own ``--out-dir``.
    """
    ap = argparse.ArgumentParser(description="text27b v0_2 builder of the PGP families")
    ap.add_argument("--paths", type=Path, default=None, help="paths.json (default: $TEXT27B_PATHS or the repo default)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output root (default paths.out_dir)")
    ap.add_argument("--limit", type=int, default=None, help="sample: first N edition docs and first N pool docs")
    ap.add_argument("--include", type=str, default=None,
                    help="sample: extra pgpids (comma list, or @file with one pgpid per line)")
    ap.add_argument("--analysis-source", choices=("v0_1", "none"), default="v0_1")
    ap.add_argument("--exposure", type=Path, default=None, help="exposure.jsonl (default <out-dir>/exposure.jsonl)")
    ap.add_argument("--holdouts", type=Path, default=None, help="holdouts.json (default <paths.out_dir>/holdouts.json)")
    ap.add_argument("--no-holdouts-module", action="store_true", help="do not consult the holdouts module")
    ap.add_argument("--drop-exposed-cer", type=float, default=EXPOSED_NEAR_DUP_CER,
                    help="drop exposed train reads below this CER (negative = keep all)")
    ap.add_argument("--max-row-tokens", type=int, default=MAX_ROW_TOKENS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    include: Optional[Set[str]] = None
    if args.include:
        spec = args.include
        items = Path(spec[1:]).read_text(encoding="utf-8").split() if spec.startswith("@") else spec.split(",")
        include = {x.strip() for x in items if x.strip()}
    if (args.limit is not None or include) and args.out_dir is None:
        raise SystemExit("a sample build (--limit / --include) needs its own --out-dir")
    p = _paths.load_paths(args.paths)
    local = bv.load_local_config(p)
    bv.apply_local_config(local)
    kw: Dict[str, Any] = {"analysis_source": args.analysis_source, "use_holdouts_module": not args.no_holdouts_module,
                          "drop_exposed_below_cer": None if args.drop_exposed_cer < 0 else args.drop_exposed_cer,
                          "max_row_tokens": args.max_row_tokens}
    if args.exposure is not None:
        kw["exposure_path"] = args.exposure
    if args.holdouts is not None:
        kw["holdouts_path"] = args.holdouts
    cfg = V2Config.from_paths(p, args.out_dir, local, **kw)
    result = build_v02(cfg, limit=args.limit, include=include)
    build_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    build_args["include_n"] = len(include) if include else 0
    build_args.pop("include", None)
    report = write_v02(result, cfg, build_args)
    viol = {k: v for k, v in report["checks"].items() if k.endswith("_violations") and v}
    print(json.dumps({"rows": report["counts"]["total"], "families": {f: v["rows"] for f, v in report["families"].items()},
                      "violations": viol, "seconds": report["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
