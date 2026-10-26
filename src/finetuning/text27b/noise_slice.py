# File name: noise_slice.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Noise slice (plan §9.1): real failing reads of train documents as inputs for human targets.

v0_1 kept a machine read only when it passed the read gate (CER ≤ 0.45, coverage 0.6–1.5). The
reads it dropped show the failure modes a serving model meets (missing lines, sides run together,
a neighbouring document's text) that the synthetic ``noised_edition`` copies never produce. This
module selects some of them as inputs for the SAME human targets the builder emits for the
document's edition row. The read is only ever the input; every target is human text.

Selection, per (train document, family):

* ``read_hard`` -- coverage 0.6–1.5 and 0.45 < CER ≤ 0.9; the lowest-CER failing read; for the
  translate families only when that CER is ≤ 0.75.
* ``read_partial`` -- only when no ``read_hard`` read exists, only for ``catalogue_json`` and
  ``describe_pgp``: coverage 0.4–0.6, CER ≤ 0.9, kept where :func:`textnorm.grounding` of the
  target's facts in the read is ≥ 0.5; the lowest-CER read among those that pass.
* never: Hebrew-script reads of Arabic editions (and the reverse), CER > 0.9, coverage outside
  0.4–2.5, reads by models outside :attr:`SliceConfig.read_models`, documents that are not in the
  train split, documents in the hold-out set H (by pgpid, canonical id or shelfmark key), documents
  whose ``lang_class`` is in :attr:`SliceConfig.exclude_lang_classes` (default ``AR``: no noised twin
  after B8; the Arabic read condition waits for the next reader).

CER and coverage are recomputed with :func:`textnorm.cer` / :func:`textnorm.coverage` on the
loop-cleaned read (:func:`textnorm.clean_read`) against the Latin-line-stripped edition
(:func:`textnorm.strip_latin_lines`); ``translate_he`` measures against ``edition_text_he`` when the
pairs file has one (plan B7), every other family against ``edition_text``.

Rows use the builder's prompts with the one v0_2 read note (:data:`rows.COND_NOTE`) and carry
``meta.condition`` ∈ {read_hard, read_partial}, ``noise_band``, ``noise_class``, ``read_model``,
``input_cer``, ``read_coverage``, ``read_agreed_share``, ``read_grounding``, ``reader_exposed``,
``exposure_level`` and ``swap_for = "noised_edition"`` (the mixture swaps a slice row in for the
noised copy of the same document and family; this module does not drop anything).

Targets come from the v0_2 builder's family files (``families/<family>.jsonl``, edition rows,
whole document) when they exist; otherwise from v0_1 (``text_ft_v0_1/{train,dev,test}.jsonl``)
with the v0_2 target strips applied here (B3 line markers on prose, B4 citations + project names on the
description and the catalogue ``summary``); the manifest records which source was used.

Hold-outs come from ``<out_dir>/holdouts.json`` through the holdouts module when both exist;
otherwise a fallback H is built from the benchmark lists (PGP-131 canonical ids → pgpids and
shelfmark keys, the benchmark registries), the v0_1 vision-benchmark flags, the probe samples and
every v0_1 dev / test document; the manifest says which.

Public API: :class:`SliceConfig`, :class:`TokenFns`, :class:`ReadEval`, :class:`Target`,
:class:`FallbackHoldouts`, :func:`noise_band`, :func:`classify_read`, :func:`read_script`,
:func:`evaluate_reads`, :func:`select_read`, :func:`grounding_text`, :func:`strip_target`,
:func:`truncate_lines`, :func:`make_slice_row`, :func:`build_slice`, :func:`load_targets`,
:func:`load_holdouts`, :func:`build_manifest`, :func:`main`.

CLI::

    .venv/bin/python -m src.finetuning.text27b.noise_slice            # full run
    .venv/bin/python -m src.finetuning.text27b.noise_slice --max-docs 500 --out-name noise_slice_sample
"""
from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import shelfmarks as SM
from src.finetuning.text27b import textnorm as TN

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------ constants
#: Families that get slice rows (analyse_document is excluded: its targets are book commentary).
SLICE_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations")
#: Families whose rows may not truncate their input (a translation must see the whole text).
TRANSLATE_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he")
#: Families that may take a partial read (their targets survive missing text; plan §9.1).
PARTIAL_FAMILIES: Tuple[str, ...] = ("catalogue_json", "describe_pgp")
#: E7 CER bands (``(lo, hi]``, the first band includes 0).
NOISE_BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9))
#: Read classes in the order :func:`classify_read` tests them.
READ_CLASSES: Tuple[str, ...] = ("empty", "script_mismatch", "cer_gt_limit", "coverage_out", "pass", "hard", "partial",
                                 "over")
#: Slice condition per read class.
CONDITION_OF_CLASS: Dict[str, str] = {"hard": "read_hard", "partial": "read_partial"}
#: Meta fields copied from the target's edition row.
COPY_META: Tuple[str, ...] = ("lang_class", "shelfmark", "shelfmark_group", "held_out_vision_benchmark", "licence",
                              "provenance", "source", "target_lang", "side", "n_relations", "target_raw", "strip_ops",
                              "pgp_description_dup")
#: Marker appended to a truncated input (the v0 builder's convention).
TRUNCATION_MARK = "\n[...]"


@dataclass(frozen=True)
class SliceConfig:
    """Selection thresholds and limits (defaults = plan §9.1 and the v0 builder)."""

    #: In-gate read (v0 builder ``READ_GATE``): such a read is already a ``read`` row, not a slice row.
    pass_cer_max: float = 0.45
    #: Coverage window of ``read_hard`` (and of the read gate).
    hard_coverage: Tuple[float, float] = (0.6, 1.5)
    #: Upper CER limit of every slice read.
    cer_max: float = 0.9
    #: ``read_hard`` CER cap for the translate families.
    translate_cer_max: float = 0.75
    #: Coverage window of ``read_partial`` (``[lo, hi)``).
    partial_coverage: Tuple[float, float] = (0.4, 0.6)
    #: Minimum grounding of the target's facts in a partial read.
    partial_grounding_min: float = 0.5
    #: Reads with coverage outside this window are never used.
    coverage_limits: Tuple[float, float] = (0.4, 2.5)
    #: Readers whose outputs may become training inputs (eval-only systems are not listed).
    read_models: Tuple[str, ...] = ("a50", "v21b")
    #: Families that get slice rows.
    families: Tuple[str, ...] = SLICE_FAMILIES
    #: Row cap in rendered tokens (the v0 builder's ``MAX_TOKENS``).
    max_tokens: int = 6000
    #: Smallest input budget worth a truncated row.
    min_input_budget: int = 300
    #: Tokens kept free when truncating (BPE merges at the cut differ from a standalone count).
    truncation_margin: int = 8
    #: Document language classes left out of the slice (decision D7: Arabic documents have no noised
    #: twin after B8 and the Arabic read condition waits for the next reader's reads).
    exclude_lang_classes: Tuple[str, ...] = ("AR",)


@dataclass
class TokenFns:
    """Token counters (injectable so tests need no tokenizer)."""

    #: Tokens of a plain text.
    count: Callable[[str], int]
    #: Tokens of a rendered chat (``enable_thinking=False``).
    render_count: Callable[[Sequence[Dict[str, Any]]], int]


def default_token_fns(tokenizer_path: Optional[Union[str, Path]] = None,
                      template_path: Optional[Union[str, Path]] = None) -> TokenFns:
    """Exact counters: :func:`rows.count_tokens` and the rendered length of :func:`rows.render_chat`.

    :param tokenizer_path: Explicit ``tokenizer.json`` (default: ``paths.tokenizer_json``).
    :param template_path: Explicit ``chat_template.jinja`` (default: ``paths.chat_template``).
    :returns: The counters.
    """
    def count(text: str) -> int:
        """Count plain-text tokens.

        :param text: Text.
        :returns: Token count.
        """
        return R.count_tokens(text, tokenizer_path)

    def render_count(messages: Sequence[Dict[str, Any]]) -> int:
        """Count tokens of the rendered chat (render contract checked).

        :param messages: Chat turns.
        :returns: Token count.
        """
        return R.count_tokens(R.render_chat(messages, template_path=template_path), tokenizer_path)

    return TokenFns(count=count, render_count=render_count)


# ------------------------------------------------------------------------------------ reads
@dataclass
class ReadEval:
    """One machine read of a document, measured against one edition text."""

    model: str
    text: str
    cer: Optional[float]
    coverage: Optional[float]
    agreed_share: Optional[float]
    loop_lines_removed: int
    script: str
    n_pages: int
    pages: List[str]
    cls: str


def noise_band(cer_value: Optional[float]) -> Optional[str]:
    """E7 CER band label of a read (``"0.45-0.6"``; ``">0.9"`` above the last band).

    :param cer_value: CER (None → None).
    :returns: Band label.
    """
    if cer_value is None:
        return None
    for lo, hi in NOISE_BANDS:
        if lo < cer_value <= hi or (lo == 0.0 and cer_value == 0.0):
            return f"{lo:g}-{hi:g}"
    return f">{NOISE_BANDS[-1][1]:g}"


def read_script(text: str) -> str:
    """Script a read is written in: the more frequent of Hebrew and Arabic letters.

    :param text: Read text.
    :returns: ``hebrew``, ``arabic`` or ``none`` (no letters of either, or a tie).
    """
    counts = TN.script_counts(text)
    if counts["hebrew"] > counts["arabic"]:
        return "hebrew"
    if counts["arabic"] > counts["hebrew"]:
        return "arabic"
    return "none"


def classify_read(cer_value: Optional[float], cov: Optional[float], script_ok: bool, cfg: SliceConfig) -> str:
    """Class of a read (first matching rule of :data:`READ_CLASSES`).

    :param cer_value: CER against the edition (None when either side is empty).
    :param cov: Coverage (read length / edition length after normalisation).
    :param script_ok: False for a Hebrew-script read of an Arabic edition (or the reverse).
    :param cfg: Thresholds.
    :returns: ``empty`` / ``script_mismatch`` / ``cer_gt_limit`` / ``coverage_out`` / ``pass`` / ``hard`` /
        ``partial`` / ``over``.
    """
    if cer_value is None or cov is None or cov == 0:
        return "empty"
    if not script_ok:
        return "script_mismatch"
    if cer_value > cfg.cer_max:
        return "cer_gt_limit"
    if not (cfg.coverage_limits[0] <= cov <= cfg.coverage_limits[1]):
        return "coverage_out"
    lo, hi = cfg.hard_coverage
    if lo <= cov <= hi:
        return "pass" if cer_value <= cfg.pass_cer_max else "hard"
    if cfg.partial_coverage[0] <= cov < cfg.partial_coverage[1]:
        return "partial"
    return "over"


def _agreed_share(read: Dict[str, Any]) -> Optional[float]:
    """Share of the read's lines that Kraken agreed with (the v0 builder's ``agreed_share``).

    :param read: A ``reads`` entry of the pairs file (``pages`` with ``n_lines`` / ``n_agreed``).
    :returns: Rounded share, or None without line counts.
    """
    n_lines = sum(pg.get("n_lines") or 0 for pg in read.get("pages") or [])
    n_agr = sum(pg.get("n_agreed") or 0 for pg in read.get("pages") or [])
    return round(n_agr / n_lines, 3) if n_lines else None


def evaluate_reads(prow: Dict[str, Any], edition: str, cfg: SliceConfig) -> List[ReadEval]:
    """Measure every read of a pairs row against ``edition`` (the v0 builder's ``read_variants`` without the gate).

    :param prow: ``pgp_parallel`` row (``reads``, ``edition_script``).
    :param edition: Latin-line-stripped edition text.
    :param cfg: Thresholds.
    :returns: One :class:`ReadEval` per read, in file order.
    """
    out: List[ReadEval] = []
    ed_script = prow.get("edition_script") or ""
    for rd in prow.get("reads") or []:
        txt, removed = TN.clean_read(rd.get("text") or "")
        norm = TN.normalize_for_cer(txt)
        cov = TN.coverage(txt, edition) if norm else None
        cer_value = TN.cer(txt, edition) if (norm and cov is not None) else None
        script = read_script(txt)
        script_ok = not ((ed_script == "arabic" and script == "hebrew") or (ed_script == "hebrew" and script == "arabic"))
        pages = [f"{pg.get('doc_id')}#{pg.get('image_index')}" for pg in rd.get("pages") or []]
        out.append(ReadEval(model=str(rd.get("model")), text=txt,
                            cer=None if cer_value is None else round(cer_value, 4),
                            coverage=None if cov is None else round(cov, 3), agreed_share=_agreed_share(rd),
                            loop_lines_removed=removed, script=script, n_pages=len(pages), pages=pages,
                            cls=classify_read(cer_value, cov, script_ok, cfg)))
    return out


def grounding_text(family: str, target: str) -> str:
    """Prose whose facts a partial read must support: the description, or the catalogue's summary + names.

    For ``catalogue_json`` the JSON is reduced to its summary followed by the people, places,
    origin and destination it names (as a comma list, so :func:`textnorm.entity_candidates` sees the
    names mid-sentence); keys, tags and language labels are not facts about the text.

    :param family: Row family.
    :param target: Assistant target.
    :returns: Prose for :func:`textnorm.grounding`.
    """
    if family != "catalogue_json":
        return target
    cat = json.loads(target)
    names: List[str] = [str(p.get("name")) for p in cat.get("people") or [] if isinstance(p, dict) and p.get("name")]
    names += [str(x) for x in cat.get("places_mentioned") or [] if x]
    names += [str(cat[k]) for k in ("origin", "destination") if cat.get(k)]
    summary = str(cat.get("summary") or "").strip()
    return summary + (f" Named: {', '.join(names)}." if names else "")


@dataclass
class Selection:
    """The read chosen for one (document, family)."""

    read: ReadEval
    condition: str
    grounding: Optional[float]


def select_read(family: str, reads: Sequence[ReadEval], target: str, scholars: Iterable[str],
                cfg: SliceConfig) -> Tuple[Optional[Selection], str]:
    """Choose the slice read of one (document, family).

    :param family: Row family.
    :param reads: Reads measured against the family's edition text.
    :param target: The family's human target (for the partial grounding test).
    :param scholars: Folded scholar surnames (not grounding facts).
    :param cfg: Thresholds.
    :returns: ``(selection or None, outcome)`` with outcome ``read_hard`` / ``read_partial`` /
        ``hard_over_translate_cap`` / ``partial_grounding_fail`` / ``partial_family_not_allowed`` /
        ``no_candidate``.
    """
    usable = [r for r in reads if r.model in cfg.read_models]
    hard = sorted((r for r in usable if r.cls == "hard"), key=lambda r: (r.cer, r.model))
    if hard:
        best = hard[0]
        if family in TRANSLATE_FAMILIES and best.cer > cfg.translate_cer_max:
            return None, "hard_over_translate_cap"
        g = None
        if family in PARTIAL_FAMILIES:
            g = round(TN.grounding(grounding_text(family, target), best.text, scholars)[0], 3)
        return Selection(best, "read_hard", g), "read_hard"
    partial = sorted((r for r in usable if r.cls == "partial"), key=lambda r: (r.cer, r.model))
    if not partial:
        return None, "no_candidate"
    if family not in PARTIAL_FAMILIES:
        return None, "partial_family_not_allowed"
    prose = grounding_text(family, target)
    for r in partial:
        g = TN.grounding(prose, r.text, scholars)[0]
        if g >= cfg.partial_grounding_min:
            return Selection(r, "read_partial", round(g, 3)), "read_partial"
    return None, "partial_grounding_fail"


# ------------------------------------------------------------------------------------ targets
@dataclass
class Target:
    """The human target of one (document, family), taken from its edition row."""

    pgpid: str
    family: str
    split: str
    text: str
    meta: Dict[str, Any]
    #: Rendered / approximate tokens of the noised copy this slice row would replace (None: no copy).
    noised_tokens: Optional[int] = None


def _strip_prose(text: str) -> Tuple[str, List[str]]:
    """B3 + B4 strips of a prose target: line markers, citations, project-name strings.

    :param text: Description or catalogue summary.
    :returns: ``(stripped text, ops)``.
    """
    t, ops = TN.strip_line_markers(text)
    t, cops = TN.strip_citations(t)
    t, n_scrub = TN.strip_source_names(t)
    out = list(ops) + [o for o in cops if o not in ops]
    if n_scrub:
        out.append(TN.SCRUB_OP)
    return t.strip(), out


def strip_target(family: str, target: str) -> Tuple[str, List[str]]:
    """Apply the v0_2 target strips to a v0_1 target (used only when the v0_2 family files are absent).

    Translations: line markers (B3). Description: markers, citations, project names (B3 + B4). Catalogue:
    the same on its ``summary`` field, JSON re-serialised the builder's way (``ensure_ascii=False``).
    Relations: unchanged.

    :param family: Row family.
    :param target: v0_1 target.
    :returns: ``(target, ops)``; the target is '' when stripping emptied it.
    """
    if family in TRANSLATE_FAMILIES:
        t, ops = TN.strip_line_markers(target)
        return t.strip(), list(ops)
    if family == "describe_pgp":
        return _strip_prose(target)
    if family == "catalogue_json":
        cat = json.loads(target)
        summary, ops = _strip_prose(str(cat.get("summary") or ""))
        if not summary:
            return "", ops
        cat["summary"] = summary
        return json.dumps(cat, ensure_ascii=False), ops
    return target, []


def _edition_rows(path: Path) -> Iterator[Dict[str, Any]]:
    """Whole-document edition and noised rows of a JSONL file.

    :param path: Family / split file.
    :returns: Iterator of rows with ``condition`` in {edition, noised_edition} and no side.
    """
    for row in R.read_jsonl(path):
        m = row["meta"]
        if m.get("condition") in ("edition", "noised_edition") and m.get("side") is None:
            yield row


def _collect(rows_iter: Iterable[Dict[str, Any]], families: Sequence[str], split_map: Callable[[str], str],
             strip: bool) -> Tuple[Dict[Tuple[str, str], Target], collections.Counter]:
    """Index edition-row targets and noised-row tokens by (pgpid, family).

    :param rows_iter: Edition / noised rows.
    :param families: Families to keep.
    :param split_map: Maps a row's split to the split recorded on the target.
    :param strip: Apply :func:`strip_target` (v0_1 source) and set ``target_raw`` / ``strip_ops`` / dup flag.
    :returns: ``(targets, counters)``.
    """
    targets: Dict[Tuple[str, str], Target] = {}
    noised: Dict[Tuple[str, str], int] = {}
    c: collections.Counter = collections.Counter()
    for row in rows_iter:
        m = row["meta"]
        fam = m.get("family") or m.get("task")
        if fam not in families:
            continue
        k = (str(m["pgpid"]), fam)
        if m["condition"] == "noised_edition":
            noised[k] = int(m.get("tokens") or 0)
            continue
        text = row["messages"][2]["content"]
        meta = {f: m.get(f) for f in COPY_META if f in m}
        if strip:
            new, ops = strip_target(fam, text)
            meta["strip_ops"] = ops
            meta["target_raw"] = text if new != text else None
            if not new or TN.has_source_names(new):
                c[f"{fam}:target_dropped_after_strip"] += 1
                continue
            text = new
        targets[k] = Target(pgpid=k[0], family=fam, split=split_map(m["split"]), text=text, meta=meta)
    for k, n in noised.items():
        if k in targets:
            targets[k].noised_tokens = n
    if strip:  # B5 flag: the description equals the catalogue summary
        for (p, fam), t in targets.items():
            if fam == "describe_pgp":
                cat = targets.get((p, "catalogue_json"))
                t.meta["pgp_description_dup"] = bool(cat and json.loads(cat.text).get("summary") == t.text)
    c["targets"] = len(targets)
    return targets, c


def load_targets(p: _paths.Paths, families: Sequence[str] = SLICE_FAMILIES, source: str = "auto"
                 ) -> Tuple[Dict[Tuple[str, str], Target], str, Dict[str, int]]:
    """Human targets per (pgpid, family): v0_2 family files when present, else v0_1 with the strips applied.

    :param p: Build paths.
    :param families: Families to load.
    :param source: ``auto`` (v0_2 if every family file exists), ``v0_2`` or ``v0_1``.
    :returns: ``(targets, source used, counters)``.
    :raises FileNotFoundError: When ``source='v0_2'`` and a family file is missing.
    """
    fam_files = {f: p.out("families", f"{f}.jsonl") for f in families}
    have_v02 = all(fp.exists() for fp in fam_files.values())
    if source == "v0_2" and not have_v02:
        raise FileNotFoundError(f"missing v0_2 family files: {[str(f) for f in fam_files.values() if not f.exists()]}")
    if source == "v0_2" or (source == "auto" and have_v02):
        def chain() -> Iterator[Dict[str, Any]]:
            """Edition rows of every family file.

            :returns: Iterator of rows.
            """
            for fp in fam_files.values():
                yield from _edition_rows(fp)
        targets, c = _collect(chain(), families, lambda s: s, strip=False)
        return targets, "v0_2_families", dict(c)

    def chain01() -> Iterator[Dict[str, Any]]:
        """Edition rows of the v0_1 split files.

        :returns: Iterator of rows.
        """
        for s in ("train", "dev", "test"):
            yield from _edition_rows(p.text_ft_v0_1 / f"{s}.jsonl")
    targets, c = _collect(chain01(), families, lambda s: s, strip=True)
    return targets, "v0_1_fallback_stripped", dict(c)


# ----------------------------------------------------------------------------------- hold-outs
def _walk_pgpids(obj: Any, out: Set[str]) -> None:
    """Collect every ``pgpid`` value of a JSON structure.

    :param obj: JSON value.
    :param out: Set to add to.
    :returns: None.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "pgpid" and v is not None:
                out.add(str(v))
            else:
                _walk_pgpids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_pgpids(v, out)


def _as_list(value: Union[None, str, Sequence[str]]) -> List[str]:
    """Normalise a None / one string / list argument to a list of strings.

    :param value: Value.
    :returns: List (empty for None).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


@dataclass
class FallbackHoldouts:
    """H built without ``holdouts.json`` (benchmark lists + v0_1 eval documents), same query API as the holdouts module."""

    benchmark_pgpids: Set[str] = field(default_factory=set)
    benchmark_cids: Set[str] = field(default_factory=set)
    benchmark_keys: Set[str] = field(default_factory=set)
    eval_pgpids: Set[str] = field(default_factory=set)
    eval_keys: Set[str] = field(default_factory=set)
    source: str = "fallback"

    @staticmethod
    def _doc_keys(shelfmark: Union[None, str, Sequence[str]], canonical: Union[None, str, Sequence[str]]) -> Set[str]:
        """Shelfmark keys of a document (every ``+`` part, plus every canonical id's key).

        :param shelfmark: Shelfmark string(s).
        :param canonical: Canonical id(s).
        :returns: Non-empty keys.
        """
        ks: Set[str] = set()
        for sm in _as_list(shelfmark):
            ks.update(SM.keys(sm))
        for cid in _as_list(canonical):
            k = SM.canonical_to_key(cid)
            if k:
                ks.add(k)
        return ks

    def is_benchmark(self, pgpid: Optional[str], shelfmark: Union[None, str, Sequence[str]] = None,
                     canonical: Union[None, str, Sequence[str]] = None) -> Optional[str]:
        """Reason a document is a benchmark / probe hold-out, or None.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark string(s).
        :param canonical: Merged canonical id(s).
        :returns: ``pgpid`` / ``canonical`` / ``shelfmark_key`` or None.
        """
        if pgpid is not None and str(pgpid) in self.benchmark_pgpids:
            return "pgpid"
        if any(c in self.benchmark_cids for c in _as_list(canonical)):
            return "canonical"
        if self._doc_keys(shelfmark, canonical) & self.benchmark_keys:
            return "shelfmark_key"
        return None

    def is_eval_doc(self, pgpid: Optional[str], shelfmark: Union[None, str, Sequence[str]] = None,
                    canonical: Union[None, str, Sequence[str]] = None) -> Optional[str]:
        """Reason a document is a v0_1 dev / test document (or shares a shelfmark key with one), or None.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark string(s).
        :param canonical: Merged canonical id(s).
        :returns: ``pgpid`` / ``shelfmark_key`` or None.
        """
        if pgpid is not None and str(pgpid) in self.eval_pgpids:
            return "pgpid"
        if self._doc_keys(shelfmark, canonical) & self.eval_keys:
            return "shelfmark_key"
        return None

    def counts(self) -> Dict[str, int]:
        """Sizes of the fallback sets.

        :returns: Counts per set.
        """
        return {k: len(getattr(self, k)) for k in ("benchmark_pgpids", "benchmark_cids", "benchmark_keys",
                                                     "eval_pgpids", "eval_keys")}


def build_fallback_holdouts(p: _paths.Paths) -> FallbackHoldouts:
    """Fallback H: PGP-131 + registries (pgpids, canonical ids, keys), v0_1 vision flags, probe docs, v0_1 dev/test.

    :param p: Build paths.
    :returns: The fallback hold-out sets.
    """
    from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents

    reg_cids, reg_pgpids = registered_benchmark_documents()
    with open(p.benchmark_ids, encoding="utf-8") as fh:
        b131 = {str(x) for x in json.load(fh)}
    cids = set(reg_cids) | b131
    with open(p.merged_pgpid_map, encoding="utf-8") as fh:
        merged = json.load(fh)
    bench = {str(x) for x in reg_pgpids}
    for cid in cids:
        bench.update(str(x) for x in (merged.get(cid) or {}).get("pgpids", []))
    for src in (p.samples, p.exclusions_v1):
        with open(src, encoding="utf-8") as fh:
            _walk_pgpids(json.load(fh), bench)
    eval_pgpids: Set[str] = set()
    eval_keys: Set[str] = set()
    for s in ("train", "dev", "test"):
        for row in R.read_jsonl(p.text_ft_v0_1 / f"{s}.jsonl"):
            m = row["meta"]
            if m.get("held_out_vision_benchmark"):
                bench.add(str(m["pgpid"]))
            if s != "train":
                eval_pgpids.add(str(m["pgpid"]))
                eval_keys.update(SM.keys(m.get("shelfmark")))
    keys = {k for k in (SM.canonical_to_key(c) for c in cids) if k}
    return FallbackHoldouts(benchmark_pgpids=bench, benchmark_cids=cids, benchmark_keys=keys,
                            eval_pgpids=eval_pgpids, eval_keys=eval_keys,
                            source="fallback:benchmark_ids+registries+v0_1_vision+probe+v0_1_dev_test")


def load_holdouts(p: _paths.Paths, mode: str = "auto") -> Tuple[Any, str]:
    """The hold-out checker: the holdouts module over ``<out_dir>/holdouts.json``, or the fallback.

    :param p: Build paths.
    :param mode: ``auto`` (file when present, else fallback), ``file`` or ``fallback``.
    :returns: ``(object with is_benchmark / is_eval_doc, source description)``.
    :raises FileNotFoundError: When ``mode='file'`` and the file or module is missing.
    """
    hfile = p.out("holdouts.json")
    have_mod = importlib.util.find_spec("src.finetuning.text27b.holdouts") is not None
    if mode == "file" and not (hfile.exists() and have_mod):
        raise FileNotFoundError(f"holdouts.json or the holdouts module missing ({hfile})")
    if mode != "fallback" and hfile.exists() and have_mod:
        mod = importlib.import_module("src.finetuning.text27b.holdouts")
        return mod.Holdouts.load(hfile, p), f"holdouts module over {hfile.name}"
    return build_fallback_holdouts(p), "fallback (no holdouts.json): benchmark ids + registries + v0_1 vision + probe + v0_1 dev/test"


def holdout_reason(h: Any, pgpid: str, shelfmark: Optional[str], cids: Sequence[str]) -> Optional[str]:
    """Why a document is in H (``benchmark:<how>`` / ``eval:<how>``), or None.

    :param h: Hold-out checker (``is_benchmark`` / ``is_eval_doc`` taking pgpid, shelfmark, canonical).
    :param pgpid: PGP id.
    :param shelfmark: Shelfmark.
    :param cids: Merged canonical ids of the document.
    :returns: Reason or None.
    """
    for cid in list(cids) or [None]:
        r = h.is_benchmark(pgpid, shelfmark, cid)
        if r:
            return f"benchmark:{r}"
    for cid in list(cids) or [None]:
        r = h.is_eval_doc(pgpid, shelfmark, cid)
        if r:
            return f"eval:{r}"
    return None


# ------------------------------------------------------------------------------------- rows
def truncate_lines(text: str, max_tokens: int, count: Callable[[str], int]) -> Tuple[str, bool]:
    """Keep the leading lines of ``text`` within ``max_tokens`` (port of the v0 builder; marks the cut with ``[...]``).

    :param text: Text.
    :param max_tokens: Budget for the kept text.
    :param count: Token counter.
    :returns: ``(text, truncated?)``.
    """
    if count(text) <= max_tokens:
        return text, False
    lines = text.split("\n")
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count("\n".join(lines[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return "\n".join(lines[:lo]).rstrip() + TRUNCATION_MARK, True


def make_slice_row(target: Target, sel: Selection, script: str, tokens: TokenFns, cfg: SliceConfig
                   ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Assemble one slice row (input truncated for catalogue-type families when over the cap).

    :param target: The family's human target.
    :param sel: Selected read.
    :param script: Edition script (names the script in the instruction, as the builder does).
    :param tokens: Token counters.
    :param cfg: Limits.
    :returns: ``(row or None, outcome)`` with outcome ``ok`` / ``over_cap`` / ``budget_too_small`` /
        ``over_cap_after_truncation`` / ``target_in_input``.
    """
    fam, rd = target.family, sel.read
    text = rd.text
    if TN.overlap60(target.text, text):
        return None, "target_in_input"
    msgs = R.make_messages(R.user_prompt(fam, sel.condition, script, text), target.text)
    total = tokens.render_count(msgs)
    truncated = False
    if total > cfg.max_tokens:
        if fam in TRANSLATE_FAMILIES:
            return None, "over_cap"
        budget = cfg.max_tokens - (total - tokens.count(text)) - cfg.truncation_margin
        if budget < cfg.min_input_budget:
            return None, "budget_too_small"
        text, truncated = truncate_lines(text, budget, tokens.count)
        msgs = R.make_messages(R.user_prompt(fam, sel.condition, script, text), target.text)
        total = tokens.render_count(msgs)
        if total > cfg.max_tokens:
            return None, "over_cap_after_truncation"
    tm = target.meta
    prov = dict(tm.get("provenance") or {}) if isinstance(tm.get("provenance"), dict) else {"source": tm.get("provenance")}
    prov["read_pages"] = rd.pages
    meta = R.new_meta(
        fam, sel.condition, target.split, target.pgpid, rd.model,
        lang_class=tm.get("lang_class"), pgpid=target.pgpid, shelfmark=tm.get("shelfmark"),
        shelfmark_group=tm.get("shelfmark_group"), held_out_vision_benchmark=bool(tm.get("held_out_vision_benchmark")),
        target_chars=len(target.text), target_tokens=tokens.count(target.text), tokens=total,
        input_cer=rd.cer, input_source=f"read:{rd.model}", input_truncated=truncated, licence=tm.get("licence"),
        provenance=prov, source=tm.get("source"), edition_available=True, read_coverage=rd.coverage,
        read_agreed_share=rd.agreed_share, noise_band=noise_band(rd.cer), noise_class=rd.cls,
        read_grounding=sel.grounding, loop_lines_removed=rd.loop_lines_removed, reader_exposed=None,
        exposure_level=None, target_raw=tm.get("target_raw"), strip_ops=list(tm.get("strip_ops") or []),
        swap_for="noised_edition", side=tm.get("side"))
    for k in ("target_lang", "n_relations", "pgp_description_dup"):
        if k in tm:
            meta[k] = tm[k]
    return {"messages": msgs, "meta": meta}, "ok"


@dataclass
class SliceStats:
    """Counters of one build."""

    docs: collections.Counter = field(default_factory=collections.Counter)
    holdout_reasons: collections.Counter = field(default_factory=collections.Counter)
    read_classes: collections.Counter = field(default_factory=collections.Counter)
    outcomes: collections.Counter = field(default_factory=collections.Counter)
    row_drops: collections.Counter = field(default_factory=collections.Counter)
    replaced_noised_tokens: int = 0
    rows_without_noised_twin: int = 0
    #: Merged canonical ids of every document that got a row (for the post-build hold-out re-check).
    doc_cids: Dict[str, List[str]] = field(default_factory=dict)


def build_slice(prows: Iterable[Dict[str, Any]], targets: Dict[Tuple[str, str], Target], holdouts: Any,
                tokens: TokenFns, scholars: Iterable[str] = (), cfg: SliceConfig = SliceConfig()
                ) -> Tuple[List[Dict[str, Any]], SliceStats]:
    """Select the slice reads and emit rows.

    :param prows: ``pgp_parallel`` rows.
    :param targets: Human targets per (pgpid, family) (:func:`load_targets`).
    :param holdouts: Hold-out checker (:func:`load_holdouts`).
    :param tokens: Token counters.
    :param scholars: Folded scholar surnames for the grounding test.
    :param cfg: Thresholds.
    :returns: ``(rows sorted by family then pgpid, stats)``.
    """
    scholars = set(scholars)
    st = SliceStats()
    out: List[Dict[str, Any]] = []
    for prow in prows:
        pid = str(prow["pgpid"])
        fams = [f for f in cfg.families if (pid, f) in targets]
        st.docs["pairs_rows"] += 1
        if not fams:
            st.docs["no_target"] += 1
            continue
        if not prow.get("reads"):
            st.docs["no_reads"] += 1
            continue
        split = targets[(pid, fams[0])].split
        if split != "train":
            st.docs[f"split_{split}"] += 1
            continue
        reason = holdout_reason(holdouts, pid, prow.get("shelfmark"), prow.get("merged_doc_ids") or [])
        if reason:
            st.docs["train_doc_in_H"] += 1
            st.holdout_reasons[reason] += 1
            continue
        lang = targets[(pid, fams[0])].meta.get("lang_class")
        if lang in cfg.exclude_lang_classes:
            st.docs[f"excluded_lang_{lang}"] += 1
            continue
        st.docs["train_docs_with_reads"] += 1
        cache: Dict[str, List[ReadEval]] = {}
        for fam in fams:
            ed_field = "edition_text_he" if fam == "translate_he" and prow.get("edition_text_he") else "edition_text"
            if ed_field not in cache:
                cache[ed_field] = evaluate_reads(prow, TN.strip_latin_lines(prow.get(ed_field) or "")[0], cfg)
            reads = cache[ed_field]
            for r in reads:
                st.read_classes[(fam, r.model, r.cls)] += 1
            tgt = targets[(pid, fam)]
            sel, outcome = select_read(fam, reads, tgt.text, scholars, cfg)
            st.outcomes[(fam, outcome)] += 1
            if sel is None:
                continue
            row, how = make_slice_row(tgt, sel, prow.get("edition_script") or "hebrew", tokens, cfg)
            if row is None:
                st.row_drops[(fam, sel.condition, how)] += 1
                continue
            row["meta"]["cer_reference"] = ed_field
            st.doc_cids[pid] = list(prow.get("merged_doc_ids") or [])
            if tgt.noised_tokens is None:
                st.rows_without_noised_twin += 1
            else:
                st.replaced_noised_tokens += tgt.noised_tokens
            out.append(row)
    order = {f: i for i, f in enumerate(cfg.families)}
    out.sort(key=lambda r: (order[r["meta"]["family"]], int(r["meta"]["pgpid"]) if str(r["meta"]["pgpid"]).isdigit()
                            else 0, str(r["meta"]["pgpid"])))
    return out, st


# ---------------------------------------------------------------------------------- exposure
def apply_exposure(rows_: List[Dict[str, Any]], p: _paths.Paths, mode: str = "auto") -> Tuple[str, Dict[str, int]]:
    """Tag ``reader_exposed`` / ``exposure_level`` with the exposure module's ``tag_rows`` when ``exposure.jsonl`` exists.

    Exposed reads stay in the slice (plan B2: train keeps them, tagged); the mixture and the evals
    decide what to do with the tag.

    :param rows_: Slice rows (tagged in place).
    :param p: Build paths.
    :param mode: ``auto`` (tag when the file and module exist) or ``off``.
    :returns: ``(status for the manifest, level counts)``.
    """
    efile = p.out("exposure.jsonl")
    if mode == "off":
        return "not applied (--exposure off): reader_exposed=None", {}
    if not efile.exists() or importlib.util.find_spec("src.finetuning.text27b.exposure") is None:
        return "not applied (no exposure.jsonl / exposure module): reader_exposed=None", {}
    mod = importlib.import_module("src.finetuning.text27b.exposure")
    stats: collections.Counter = collections.Counter()
    mod.tag_rows(rows_, efile, stats)
    return f"applied from {efile.name}", dict(stats)


# ---------------------------------------------------------------------------------- manifest
def _counter_table(rows_: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Rows / tokens / target tokens grouped by meta fields (``"a|b|c"`` keys).

    :param rows_: Rows.
    :param keys: Meta fields to group by.
    :returns: ``{group: {"rows", "tokens", "target_tokens"}}`` sorted by group.
    """
    agg: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"rows": 0, "tokens": 0, "target_tokens": 0})
    for r in rows_:
        m = r["meta"]
        g = "|".join(str(m.get(k)) for k in keys)
        agg[g]["rows"] += 1
        agg[g]["tokens"] += int(m.get("tokens") or 0)
        agg[g]["target_tokens"] += int(m.get("target_tokens") or 0)
    return {k: agg[k] for k in sorted(agg)}


def _nested(counter: collections.Counter) -> Dict[str, Dict[str, int]]:
    """Turn a counter keyed by tuples into ``{"a|b": {last: n}}``.

    :param counter: Counter with tuple keys.
    :returns: Nested dict.
    """
    out: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    for k, v in sorted(counter.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out["|".join(str(x) for x in k[:-1])][str(k[-1])] = v
    return dict(out)


def run_checks(rows_: Sequence[Dict[str, Any]], holdouts: Any, cfg: SliceConfig,
               doc_cids: Optional[Dict[str, List[str]]] = None) -> Dict[str, int]:
    """Post-build checks (all must be 0).

    :param rows_: Slice rows.
    :param holdouts: Hold-out checker.
    :param cfg: Thresholds.
    :param doc_cids: pgpid → merged canonical ids (:attr:`SliceStats.doc_cids`), so the H re-check sees them too.
    :returns: Violation counts.
    """
    doc_cids = doc_cids or {}
    ids = collections.Counter(r["meta"]["row_id"] for r in rows_)
    keys = collections.Counter((r["meta"]["pgpid"], r["meta"]["family"]) for r in rows_)
    c = {"schema_problems": 0, "not_train": 0, "in_H": 0, "scrub_in_target": 0, "duplicate_row_id": 0,
         "duplicate_doc_family": 0, "translate_over_cer_cap": 0, "partial_in_wrong_family": 0, "cer_over_limit": 0,
         "over_max_tokens": 0, "read_model_not_allowed": 0, "excluded_lang_class": 0}
    c["duplicate_row_id"] = sum(n - 1 for n in ids.values() if n > 1)
    c["duplicate_doc_family"] = sum(n - 1 for n in keys.values() if n > 1)
    for r in rows_:
        m = r["meta"]
        c["schema_problems"] += bool(R.validate_row(r))
        c["not_train"] += m["split"] != "train"
        c["in_H"] += bool(holdout_reason(holdouts, m["pgpid"], m.get("shelfmark"), doc_cids.get(m["pgpid"], [])))
        c["scrub_in_target"] += TN.has_source_names(r["messages"][2]["content"])
        c["translate_over_cer_cap"] += m["family"] in TRANSLATE_FAMILIES and m["input_cer"] > cfg.translate_cer_max
        c["partial_in_wrong_family"] += m["condition"] == "read_partial" and m["family"] not in PARTIAL_FAMILIES
        c["cer_over_limit"] += m["input_cer"] > cfg.cer_max
        c["over_max_tokens"] += m["tokens"] > cfg.max_tokens
        c["read_model_not_allowed"] += m["read_model"] not in cfg.read_models
        c["excluded_lang_class"] += m.get("lang_class") in cfg.exclude_lang_classes
    return c


def build_manifest(rows_: Sequence[Dict[str, Any]], st: SliceStats, checks: Dict[str, int],
                   build_args: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest of the slice (brief rule 6: ``counts`` + ``build_args``; plus bands, classes, drops).

    :param rows_: Slice rows.
    :param st: Build counters.
    :param checks: :func:`run_checks` output.
    :param build_args: CLI arguments and config.
    :param info: Sources used (targets, hold-outs, exposure) and timing.
    :returns: Manifest dict.
    """
    total_tokens = sum(int(r["meta"]["tokens"]) for r in rows_)
    return {
        "build": R.BUILD, "family_file": "families/noise_slice.jsonl", "n_rows": len(rows_), "total_tokens": total_tokens,
        "total_target_tokens": sum(int(r["meta"]["target_tokens"]) for r in rows_),
        "fingerprint": R.fingerprint(rows_),
        "counts": _counter_table(rows_, ("family", "condition", "lang_class", "split")),
        "by_family": _counter_table(rows_, ("family",)),
        "by_family_condition": _counter_table(rows_, ("family", "condition")),
        "by_family_lang": _counter_table(rows_, ("family", "lang_class")),
        "by_noise_band": _counter_table(rows_, ("condition", "noise_band")),
        "by_family_noise_band": _counter_table(rows_, ("family", "noise_band")),
        "by_noise_class": _counter_table(rows_, ("noise_class",)),
        "by_read_model": _counter_table(rows_, ("family", "read_model")),
        "by_exposure": _counter_table(rows_, ("condition", "exposure_level")),
        "truncated_inputs": sum(bool(r["meta"]["input_truncated"]) for r in rows_),
        "swap": {"replaced_noised_tokens": st.replaced_noised_tokens,
                 "rows_without_noised_twin": st.rows_without_noised_twin,
                 "net_tokens_if_swapped": total_tokens - st.replaced_noised_tokens},
        "docs": dict(st.docs), "holdout_reasons": dict(st.holdout_reasons),
        "read_classes_train_docs": _nested(st.read_classes), "selection_outcomes": _nested(st.outcomes),
        "row_drops": _nested(st.row_drops), "checks": checks, "build_args": build_args, **info,
    }


# --------------------------------------------------------------------------------------- CLI
def _iter_pairs(path: Path, max_docs: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Stream the pairs file, optionally only its first ``max_docs`` rows.

    :param path: ``pgp_parallel`` JSONL.
    :param max_docs: Row limit (None = all).
    :returns: Iterator of rows.
    """
    for i, row in enumerate(R.read_jsonl(path)):
        if max_docs is not None and i >= max_docs:
            return
        yield row


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: build ``families/<out-name>.jsonl`` and ``families/<out-name>_manifest.json``.

    :param argv: Arguments (default: ``sys.argv``).
    :returns: Exit code (1 when a check fails).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paths", default=None, help="paths.json (default: $TEXT27B_PATHS or logs/text_finetune_27b/paths.json)")
    ap.add_argument("--targets", choices=("auto", "v0_2", "v0_1"), default="auto")
    ap.add_argument("--holdouts", choices=("auto", "file", "fallback"), default="auto")
    ap.add_argument("--exposure", choices=("auto", "off"), default="auto")
    ap.add_argument("--max-docs", type=int, default=None, help="only the first N pairs rows (development)")
    ap.add_argument("--out-name", default="noise_slice")
    ap.add_argument("--max-tokens", type=int, default=SliceConfig.max_tokens)
    ap.add_argument("--read-models", default=",".join(SliceConfig.read_models))
    ap.add_argument("--exclude-lang", default=",".join(SliceConfig.exclude_lang_classes),
                    help="document lang_class values left out of the slice (comma list; '' keeps all; D7: AR)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    p = _paths.load_paths(a.paths)
    cfg = SliceConfig(max_tokens=a.max_tokens, read_models=tuple(x for x in a.read_models.split(",") if x),
                      exclude_lang_classes=tuple(x.strip() for x in a.exclude_lang.split(",") if x.strip()))
    targets, tsrc, tcounts = load_targets(p, cfg.families, a.targets)
    LOG.info("targets: %s (%s) %s", len(targets), tsrc, tcounts)
    holdouts, hsrc = load_holdouts(p, a.holdouts)
    LOG.info("hold-outs: %s %s", hsrc, holdouts.counts() if hasattr(holdouts, "counts") else "")
    scholars = TN.scholar_surnames(p.pgp_export / "sources.csv")
    rows_, st = build_slice(_iter_pairs(p.pgp_parallel, a.max_docs), targets, holdouts, default_token_fns(),
                            scholars, cfg)
    exposure, exposure_counts = apply_exposure(rows_, p, a.exposure)
    checks = run_checks(rows_, holdouts, cfg, st.doc_cids)
    out = p.out("families", f"{a.out_name}.jsonl", mkdir=True)
    R.write_jsonl(out, rows_)
    info = {"target_source": tsrc, "target_counters": tcounts, "holdout_source": hsrc,
            "holdout_counts": holdouts.counts() if hasattr(holdouts, "counts") else None, "exposure": exposure,
            "exposure_counts": exposure_counts,
            "pairs_file": p.pgp_parallel.name, "seconds": round(time.time() - t0, 1)}
    build_args = {**{k: v for k, v in vars(a).items() if k != "paths"}, "config": asdict(cfg)}
    man = build_manifest(rows_, st, checks, build_args, info)
    with open(p.out("families", f"{a.out_name}_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    LOG.info("wrote %s rows (%s tokens) to %s", len(rows_), man["total_tokens"], out)
    for g, v in man["by_family_condition"].items():
        LOG.info("  %-32s rows %5d tokens %10d", g, v["rows"], v["tokens"])
    LOG.info("checks %s", checks)
    return 1 if any(checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
