# File name: mixture.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Materialise one training epoch of the text27b mixtures (``t27a`` / ``t27b``) from the v0_2 family files.

Plan §4 / §9.3 and build brief section I. The family builders write ``families/<family>.jsonl`` under
``paths.out_dir`` (every split in one file); this module reads them, keeps ``split == "train"`` only and
writes one seeded, fully materialised epoch per mixture:

``t27a`` (translate pilot, the "no slice, no literature" baseline)
    every train row of ``translate_en`` + ``translate_he`` (conditions ``edition``, ``noised_edition``,
    ``read``, ``edition_rescript``) twice, plus the L5 translate pairs (``families/book_pairs.jsonl``,
    optional) twice -- the same translate data as t27b, so the two runs differ only in the other
    components. Noise-slice rows and the L5 analysis rows are not part of t27a.
``t27b`` (joint run, plan §9.3)
    * translate_en / translate_he ×2, all conditions, noise slice swapped in (below);
    * L5 pairs (``families/book_pairs.jsonl``; skipped and logged when absent): translate rows ×2,
      catalogue / describe rows join the one-row-per-document pool, relations rows the relations pool;
    * catalogue_json ∪ describe_pgp: ONE row per document per epoch (the one-condition rule, below);
    * extract_relations: ``round(0.4 × rows)`` documents (seeded), one row each, same rule;
    * analyse_document ×2 (``families/analyse_document_v2.jsonl``, the widened L2 file, replaces the
      v0_1-carried ``analyse_document.jsonl`` when present); rag_context all; domain_reading train rows
      (``probe`` excluded) within a 2.70 M token budget per epoch (seeded row subsample, fix pass F5);
    * the public-domain edition pairs (``families/pd_editions.jsonl``, translate_en rows of :mod:`canon_diff`'s PD
      step) ×2 as component ``pd_editions``, counted as L5 literature (design owner, 2026-10-26: book-sourced human
      translation pairs);
    * ``canon_diff`` (canon fold-in, 2026-10-26) within a 2.50 M token budget per epoch, drawn by a WORK-WEIGHTED
      seeded subsample (:func:`apply_token_caps`: Talmud Bavli rows weight 3, Mishnah 2, Mishneh Torah 1, as
      sampling-probability multipliers; every (work, title) -- tractate / book of hilkhot -- of the pool keeps at
      least one row). ``canon_identify`` stays out of the mixtures (its family file and eval set E8 serve probes).

Canon block and the literature floor (design-owner decision, 2026-10-26)
    The canon components (:data:`ADDITIVE_COMPONENTS`) are a user-approved additive block with their own token caps.
    The F13 literature floor is applied to the CORE epoch (the epoch minus that block), so the approved documentary +
    literature core keeps exactly its verified composition; the gate's ``share`` is the core share. The manifest
    reports three shares side by side (``literature_shares``): literature share of the core (gated, ≥ 17.5 %), of
    the whole epoch, and literature-plus-canon share of the whole epoch.

Exclusions and caps (decisions D5 / D11)
    Rows listed in ``excluded_rows.json`` (contamination module: train rows whose target leaks an eval
    reference) never enter a mixture (``--exclude-rows``, default ``<out_dir>/excluded_rows.json`` when it
    exists). Identical targets of the capped families (default ``extract_relations``) appear at most
    ``max_copies_per_target`` times per epoch (t27b: 5): the relations subsample walks the seeded document
    ranking and skips a document whose target already has that many copies, so the 40 % row budget is
    still met. The manifest reports the RAG twin coverage (rag rows whose no-literature twin row is in
    the same epoch).

Swap rule (plan §9.1 "swap, don't add")
    A noise-slice row (``condition`` ∈ ``read_hard`` / ``read_partial``) replaces the ``noised_edition``
    row(s) of the same (document, family): those noised rows leave the pool and the slice row takes their
    place. A slice row whose (document, family) has no noised copy is kept and counted
    (``slice_without_noised``).
    Prep pass 2026-10-26: t27b never draws ``read_partial`` rows (:data:`T27B_EXCLUDE_CONDITIONS`; they stay in the
    family file tagged ``meta.mixture_excluded``), so the noised copy of their (document, family) stays in the pool.

One-condition rule (catalogue ∪ describe, relations)
    Per document, the candidate rows are grouped into condition classes ``edition`` (edition,
    edition_rescript), ``read`` (read) and ``noised`` (noised_edition and, after the swap, slice rows).
    With ``slice_policy="single"`` (default, plan §9.1: "for catalogue∪describe it becomes that
    document's single row") a document that has a slice row always gets a slice row. Otherwise a class
    is drawn with weights edition 0.4 / read 0.4 / noised 0.2, renormalised over the classes the
    document has (``slice_policy="noised_class"`` = the slice row only competes inside the noised class).
    Inside the class a family is drawn uniformly, then a row uniformly. Every draw is a hash of
    (seed, epoch, purpose, document, step), so a document's choice does not depend on file order or on
    other documents.

Order
    Copy ``k`` of a row repeated ``r`` times gets the sort key ``(k + u) / r`` with ``u`` a seeded hash
    uniform: single rows spread over the whole epoch, the two copies of a ×2 row fall in different halves
    (so the 50 % milestone has seen every translate row once), families stay interleaved throughout.

Outputs (under ``out_dir/mixtures/``)
    ``<name>_epoch<e>.jsonl`` (rows with ``meta.mixture``, ``mix_component``, ``mix_rule``, ``mix_copy``,
    ``mix_repeats`` added), ``mixture_manifest.json`` (one section per mixture: ``counts`` per family ×
    condition × lang_class × split, tokens per family × condition × lang, components and shares, noise-
    slice share, repeats per target / row / document histograms, total tokens, projected steps at 16 rows
    and at 16.6 k tokens per step, text fingerprint, selection and drop statistics, gates, ``build_args``)
    and ``render_check.json`` (G3: three rows per family rendered with ``enable_thinking=False`` -- the
    longest plus two seeded picks -- exactly one empty think block, no effort sentence, prompt/completion
    split intact, ≤ ``max_seq`` tokens).

CLI::

    .venv/bin/python -m src.finetuning.text27b.mixture --mixture t27a t27b --seed 3407
        [--slice-policy single] [--exclude-rows auto|none|FILE] [--max-copies-per-target 5]
        [--without book_pairs analyse_document_v2]   # leave optional files out (e.g. while their agents run)
        [--token-caps domain_reading=2700000 canon_diff=2500000]   # per-epoch family token budgets (t27b default)
        [--literature-floor min_share=0.175 l1_whole_pool=1 | none]   # literature floor gate (t27b default; F13 restated)

Public API: :data:`MIXTURES`, :class:`MixConfig`, :func:`default_config`, :class:`Item`,
:class:`Instance`, :func:`unit_hash`, :func:`pick_weighted`, :func:`doc_key`, :func:`cond_class`,
:func:`load_pool`, :func:`load_exclusions`, :func:`apply_swap`, :func:`choose_one`, :func:`select_one_per_doc`,
:func:`select_subsample`, :func:`apply_token_caps`, :func:`literature_floor_check`, :func:`select`, :func:`order_instances`, :func:`instance_row`,
:func:`count_item_tokens`, :func:`summarise`, :func:`rag_twin_coverage`, :func:`render_check`, :func:`check_gates`,
:func:`build_mixture`, :func:`main`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import textnorm as _tn

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------ constants
MIXTURES: Tuple[str, ...] = ("t27a", "t27b")
DEFAULT_SEED = 3407
TRANSLATE_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he")
#: Noise-slice conditions (module C); they replace the noised copy of the same (doc, family).
SLICE_CONDITIONS: Tuple[str, ...] = ("read_hard", "read_partial")
#: Condition → class used by the one-condition rule (conditions not listed are never chosen there).
COND_CLASS: Dict[str, str] = {"edition": "edition", "edition_rescript": "edition", "read": "read",
                              "noised_edition": "noised", "read_hard": "noised", "read_partial": "noised"}
CLASS_ORDER: Tuple[str, ...] = ("edition", "read", "noised")
DEFAULT_CLASS_WEIGHTS: Dict[str, float] = {"edition": 0.4, "read": 0.4, "noised": 0.2}
SLICE_POLICIES: Tuple[str, ...] = ("single", "noised_class")
#: Families that share ONE row-per-document budget (B5: catalogue summary == describe target).
ONE_PER_DOC_FAMILIES: Tuple[str, ...] = ("catalogue_json", "describe_pgp")
#: Family files under ``families/`` (stems). ``noise_slice`` and ``book_pairs`` hold rows of PGP families.
NOISE_SLICE_FILE = "noise_slice"
BOOK_PAIRS_FILE = "book_pairs"
#: The web translation pairs (:mod:`web_pairs`, 2026-10-26): translate rows ×2 in t27b, component ``web_pairs``
#: (counted with the L5 literature pairs); not part of t27a (t27a stays as pinned).
WEB_PAIRS_FILE = "web_pairs"
#: The widened L2 analysis file; when present it replaces the v0_1-carried ``analyse_document`` file.
ANALYSE_V2_FILE = "analyse_document_v2"
#: Family file stems that a present replacement file supersedes (replacement → replaced).
REPLACES: Dict[str, str] = {ANALYSE_V2_FILE: "analyse_document"}
#: Row exclusions written by the contamination module (D5).
EXCLUDED_ROWS_FILE = "excluded_rows.json"
#: Families whose identical targets are capped per epoch (D11).
CAP_FAMILIES: Tuple[str, ...] = ("extract_relations",)
#: The public-domain edition pairs (:mod:`canon_diff` PD step): translate_en rows, component ``pd_editions`` (L5).
PD_EDITIONS_FILE = "pd_editions"
#: The canon-comparison apparatus family (fold-in 2026-10-26; ``canon_identify`` is not mixed).
CANON_DIFF_FILE = "canon_diff"
#: Per-epoch token budgets of t27b (fix pass F5: the L1 budget, ≈ 10 % of the epoch as approved; canon fold-in
#: 2026-10-26: canon_diff 2.50 M, user-approved).
T27B_TOKEN_CAPS: Dict[str, int] = {"domain_reading": 2_700_000, CANON_DIFF_FILE: 2_500_000}
#: Work weights of the canon subsample (sampling-probability multipliers by ``meta.work``; works not listed weigh 1).
T27B_CAP_WEIGHTS: Dict[str, Dict[str, float]] = {CANON_DIFF_FILE: {"Talmud Bavli": 3.0, "Mishnah": 2.0,
                                                                   "Mishneh Torah": 1.0}}
#: Families whose capped subsample keeps at least one row of every (work, title) stratum of the pool.
T27B_CAP_COVER: Tuple[str, ...] = (CANON_DIFF_FILE,)
#: Components outside the core epoch on which the literature floor is computed (the canon block; design owner
#: 2026-10-26).
ADDITIVE_COMPONENTS: Tuple[str, ...] = tuple(R.CANON_FAMILIES)
#: Components that carry literature text: L1 ``domain_reading``, L3 ``rag_context``, L2 / L5 ``analyse_document``,
#: the L5 translate pairs ``book_pairs``, the public-domain edition pairs ``pd_editions`` (L5, 2026-10-26) and the
#: web translation pairs ``web_pairs`` (human translations printed in articles / posts, 2026-10-26).
LITERATURE_COMPONENTS: Tuple[str, ...] = ("domain_reading", "rag_context", "analyse_document", "book_pairs",
                                          "pd_editions", "web_pairs")
#: The t27b literature floor (fix pass 2, F13, restated 2026-10-26): the literature components ≥ 17.5 % of the core
#: epoch's tokens (``min_share``) and the L1 rule. F13 held L1 at its 2.70 M budget ± 1 % (``l1_tolerance``); the
#: design owner reverted the per-work L1 cap to 0.30 M (concentration of one monograph) and restated the L1 gate as
#: "L1 = the whole filtered pool after the per-work caps, ≤ 2.70 M, reported" (``l1_whole_pool``: every offered L1
#: row is in the epoch when the pool fits the budget; no floor). ``l1_tolerance`` is still honoured when given.
T27B_LITERATURE_FLOOR: Dict[str, float] = {"min_share": 0.175, "l1_whole_pool": 1.0}
T27A_FILES: Tuple[str, ...] = TRANSLATE_FAMILIES + (BOOK_PAIRS_FILE,)
T27B_FILES: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations",
                               "analyse_document", ANALYSE_V2_FILE, NOISE_SLICE_FILE, BOOK_PAIRS_FILE, WEB_PAIRS_FILE,
                               "rag_context", "domain_reading", PD_EDITIONS_FILE, CANON_DIFF_FILE)
T27B_REQUIRED: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations")
ROWS_PER_STEP = 16
TOKENS_PER_STEP = 16_600
MAX_SEQ = 6144
#: B2: reader-exposed reads this close to the edition are near-duplicates of the edition row.
EXPOSED_NEAR_DUP_CER = 0.05
#: Slice conditions left out of the t27b epoch (prep pass 2026-10-26: partial reads cannot support a whole-document
#: target; their rows stay in ``families/noise_slice.jsonl`` with ``meta.mixture_excluded``). Rows whose
#: ``meta.mixture_excluded`` is set are dropped too.
T27B_EXCLUDE_CONDITIONS: Tuple[str, ...] = ("read_partial",)
#: Plan §9.1 gate for the noise-slice token share of t27b (informational warning).
SLICE_SHARE_BAND: Tuple[float, float] = (0.05, 0.08)
MANIFEST_NAME = "mixture_manifest.json"
RENDER_CHECK_NAME = "render_check.json"
MIX_DIR = "mixtures"


# ---------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class MixConfig:
    """Selection parameters of one mixture.

    ``repeats`` is keyed by component (``translate_en``, ``translate_he``, ``book_pairs``,
    ``analyse_document``, ``rag_context``, ``domain_reading``); ``subsample`` maps a family to the
    fraction of its rows whose count of documents is drawn (one row each); ``token_caps`` maps a family to
    its token budget per epoch (a seeded subsample of its rows, :func:`apply_token_caps`; ``cap_weights`` maps a
    capped family to per-work sampling weights, ``cap_cover`` lists capped families that keep every (work, title)
    stratum); ``exclude_conditions`` lists conditions never drawn (t27b: ``read_partial``, prep pass 2026-10-26; rows
    tagged ``meta.mixture_excluded`` are never drawn either); ``literature_floor`` (``min_share``, ``l1_tolerance``) is the gate on the literature components
    (:data:`LITERATURE_COMPONENTS`) and the L1 budget (F13; empty = no gate), computed on the epoch minus the
    ``additive_components`` (the canon block, design owner 2026-10-26).
    """

    name: str
    seed: int = DEFAULT_SEED
    epoch: int = 1
    files: Tuple[str, ...] = T27B_FILES
    required: Tuple[str, ...] = T27B_REQUIRED
    families: Tuple[str, ...] = R.FAMILIES
    include_slice: bool = True
    repeats: Dict[str, int] = field(default_factory=dict)
    one_per_doc: Tuple[str, ...] = ONE_PER_DOC_FAMILIES
    subsample: Dict[str, float] = field(default_factory=dict)
    class_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    slice_policy: str = "single"
    exposed_near_dup_cer: Optional[float] = EXPOSED_NEAR_DUP_CER
    max_seq: int = MAX_SEQ
    drop_over_max_seq: bool = False
    rows_per_step: int = ROWS_PER_STEP
    tokens_per_step: int = TOKENS_PER_STEP
    render_check_n: int = 3
    recount_tokens: bool = False
    max_copies_per_target: Optional[int] = None
    cap_families: Tuple[str, ...] = CAP_FAMILIES
    exclude_rows_file: Optional[str] = None
    token_caps: Dict[str, int] = field(default_factory=dict)
    literature_floor: Dict[str, float] = field(default_factory=dict)
    cap_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cap_cover: Tuple[str, ...] = ()
    additive_components: Tuple[str, ...] = ADDITIVE_COMPONENTS
    exclude_conditions: Tuple[str, ...] = ()


def default_config(name: str, **overrides: Any) -> MixConfig:
    """Return the plan's configuration for ``t27a`` or ``t27b`` with optional overrides.

    :param name: ``t27a`` or ``t27b``.
    :param overrides: :class:`MixConfig` fields to replace (``repeats`` / ``subsample`` /
        ``class_weights`` / ``token_caps`` dicts are merged into the defaults, not replaced; a token cap ≤ 0
        removes that family's cap).
    :returns: The configuration.
    :raises ValueError: On an unknown mixture name, slice policy or class weight.
    """
    if name == "t27a":
        base = MixConfig(name=name, files=T27A_FILES, required=TRANSLATE_FAMILIES, families=TRANSLATE_FAMILIES,
                         include_slice=False, repeats={"translate_en": 2, "translate_he": 2, "book_pairs": 2},
                         one_per_doc=(),
                         subsample={})
    elif name == "t27b":
        base = MixConfig(name=name, repeats={"translate_en": 2, "translate_he": 2, "book_pairs": 2, "web_pairs": 2,
                                             "analyse_document": 2, "rag_context": 1, "domain_reading": 1,
                                             "pd_editions": 2, CANON_DIFF_FILE: 1},
                         subsample={"extract_relations": 0.4}, max_copies_per_target=5,
                         token_caps=dict(T27B_TOKEN_CAPS), literature_floor=dict(T27B_LITERATURE_FLOOR),
                         cap_weights={k: dict(v) for k, v in T27B_CAP_WEIGHTS.items()}, cap_cover=T27B_CAP_COVER,
                         exclude_conditions=T27B_EXCLUDE_CONDITIONS)
    else:
        raise ValueError(f"unknown mixture {name!r}; expected one of {MIXTURES}")
    for key in ("repeats", "subsample", "class_weights", "token_caps"):
        if key in overrides:
            overrides[key] = {**getattr(base, key), **(overrides[key] or {})}
    cfg = replace(base, **overrides)
    if cfg.max_copies_per_target is not None and cfg.max_copies_per_target <= 0:
        cfg = replace(cfg, max_copies_per_target=None)
    if any(v <= 0 for v in cfg.token_caps.values()):
        cfg = replace(cfg, token_caps={k: v for k, v in cfg.token_caps.items() if v > 0})
    if cfg.slice_policy not in SLICE_POLICIES:
        raise ValueError(f"slice_policy {cfg.slice_policy!r} not in {SLICE_POLICIES}")
    unknown = set(cfg.class_weights) - set(CLASS_ORDER)
    if unknown or any(w < 0 for w in cfg.class_weights.values()):
        raise ValueError(f"class weights must be non-negative over {CLASS_ORDER}: {cfg.class_weights}")
    return cfg


# ------------------------------------------------------------------------------------- row items
@dataclass
class Item:
    """One distinct train row in the selection pool."""

    row: Dict[str, Any]
    src: str
    uid: str
    tokens: Optional[int] = None

    @property
    def meta(self) -> Dict[str, Any]:
        """The row's meta block.

        :returns: ``row["meta"]``.
        """
        return self.row["meta"]

    @property
    def family(self) -> str:
        """Row family (``meta.family``, falling back to ``meta.task``).

        :returns: Family name.
        """
        return family_of(self.meta)

    @property
    def condition(self) -> str:
        """Input condition.

        :returns: ``meta.condition``.
        """
        return str(self.meta.get("condition"))

    @property
    def row_id(self) -> str:
        """Row id written by the family builder.

        :returns: ``meta.row_id`` ('' when absent).
        """
        return str(self.meta.get("row_id") or "")

    @property
    def doc(self) -> str:
        """Document key used by the swap and one-condition rules.

        :returns: See :func:`doc_key`.
        """
        return doc_key(self.meta)

    @property
    def component(self) -> str:
        """Mixture component (§9.3 row): L5 translate rows are ``book_pairs``, the public-domain edition pairs
        ``pd_editions``, the web translation pairs ``web_pairs``, catalogue / describe share ``catalogue_describe``, everything else is its family.

        :returns: Component name.
        """
        if self.src == BOOK_PAIRS_FILE and self.family in TRANSLATE_FAMILIES:
            return "book_pairs"
        if self.src == PD_EDITIONS_FILE and self.family in TRANSLATE_FAMILIES:
            return "pd_editions"
        if self.src == WEB_PAIRS_FILE and self.family in TRANSLATE_FAMILIES:
            return "web_pairs"
        if self.family in ONE_PER_DOC_FAMILIES:
            return "catalogue_describe"
        return self.family


@dataclass
class Instance:
    """One occurrence of an item in the epoch."""

    item: Item
    copy: int
    repeats: int
    rule: str
    order: float = 0.0


def family_of(meta: Mapping[str, Any]) -> str:
    """Family of a row's meta block.

    :param meta: Row meta.
    :returns: ``meta.family`` or, for rows without it, ``meta.task`` ('' when neither is set).
    """
    return str(meta.get("family") or meta.get("task") or "")


def doc_key(meta: Mapping[str, Any]) -> str:
    """Document identity for the swap and one-condition rules.

    :param meta: Row meta.
    :returns: ``pgpid:<id>``, else ``doc_id:<id>``, else ``shelfmark:<mark>``, else ``row:<row_id>``.
    """
    for k in ("pgpid", "doc_id", "shelfmark"):
        v = meta.get(k)
        if v not in (None, ""):
            return f"{k}:{v}"
    return f"row:{meta.get('row_id')}"


def cond_class(condition: str) -> Optional[str]:
    """Condition class of the one-condition rule.

    :param condition: ``meta.condition``.
    :returns: ``edition`` / ``read`` / ``noised``, or None for conditions the rule never picks.
    """
    return COND_CLASS.get(condition)


def _digest(*texts: str) -> str:
    """sha1 hex digest of joined texts.

    :param texts: Texts.
    :returns: 40-character hex digest.
    """
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _content(row: Mapping[str, Any], role: str) -> str:
    """Content of the first message with ``role``.

    :param row: Row.
    :param role: ``system`` / ``user`` / ``assistant``.
    :returns: The content ('' when absent).
    """
    return next((str(m.get("content") or "") for m in row.get("messages") or [] if m.get("role") == role), "")


def target_hash(row: Mapping[str, Any]) -> str:
    """Identity of a row's supervised target (the assistant text).

    :param row: Row.
    :returns: sha1 hex of the stripped assistant content.
    """
    return _digest(_content(row, "assistant").strip())


def unit_hash(seed: int, *parts: Any) -> float:
    """Deterministic uniform number in [0, 1) from a seed and a key.

    :param seed: Mixture seed.
    :param parts: Key parts (stringified).
    :returns: A float in [0, 1).
    """
    h = hashlib.sha256("|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def pick_weighted(u: float, options: Sequence[Tuple[str, float]]) -> str:
    """Pick one option by cumulative weight (weights are renormalised over the options given).

    :param u: Uniform number in [0, 1).
    :param options: ``(name, weight)`` pairs with positive total weight.
    :returns: The chosen name.
    :raises ValueError: When there is no option with positive weight.
    """
    total = sum(w for _, w in options if w > 0)
    if total <= 0:
        raise ValueError("no option with positive weight")
    x = u * total
    acc = 0.0
    chosen = None
    for name, w in options:
        if w <= 0:
            continue
        chosen = name
        acc += w
        if x < acc:
            return name
    return chosen  # type: ignore[return-value]


def _pick_index(u: float, n: int) -> int:
    """Index in ``range(n)`` from a uniform number.

    :param u: Uniform number in [0, 1).
    :param n: Number of choices (≥1).
    :returns: The index.
    """
    return min(int(u * n), n - 1)


# ------------------------------------------------------------------------------------------ load
def file_sha256(path: Union[str, Path]) -> str:
    """sha256 of a file's bytes.

    :param path: File.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _structural_problems(row: Mapping[str, Any], family: str) -> Tuple[List[str], List[str]]:
    """Split :func:`rows.validate_row` problems into fatal (unrenderable) and informational ones.

    :param row: Row.
    :param family: Its family (v0_1 meta fields are required for PGP families only).
    :returns: ``(fatal, other)`` problem lists.
    """
    problems = R.validate_row(dict(row), require_v01=family in R.PGP_FAMILIES)
    fatal = [p for p in problems if p.startswith("messages must be") or p.startswith("empty ")
             or p == "meta missing"]
    return fatal, [p for p in problems if p not in fatal]


def _problem_key(problem: str) -> str:
    """Counter key of a schema problem (field name only, so lists and values do not fragment the count).

    :param problem: A :func:`rows.validate_row` message.
    :returns: Short key such as ``meta.build`` or ``missing v0_1 meta fields``.
    """
    if problem.startswith("missing v0_1 meta fields"):
        return "missing v0_1 meta fields"
    return problem.split("=")[0].split(" not in")[0].split(" must")[0][:60]


def load_exclusions(path: Optional[Union[str, Path]]) -> Dict[str, str]:
    """Row exclusions (``excluded_rows.json``: ``{row_id: reason}``; a list of row ids is also accepted).

    :param path: File (None or missing → no exclusions).
    :returns: row_id → reason.
    """
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {str(k): "listed" for k in data}
    return {str(k): str(v) for k, v in data.items()}


def load_pool(families_dir: Union[str, Path], cfg: MixConfig,
              exclude_rows: Optional[Mapping[str, str]] = None) -> Tuple[List[Item], Dict[str, Any]]:
    """Read the family files of a mixture and keep the distinct, valid train rows.

    Dropped (and counted): rows whose split is not ``train`` (dev, test_indist, heldout,
    heldout_vision, probe, missing), rows flagged ``held_out_vision_benchmark``, families outside the
    mixture, slice rows when ``include_slice`` is off, rows of ``cfg.exclude_conditions`` or tagged
    ``meta.mixture_excluded`` (``condition_excluded:<reason>``), unrenderable rows, reader-exposed ``read`` rows
    with ``input_cer`` below ``exposed_near_dup_cer`` (B2 near-duplicates), rows listed in
    ``exclude_rows`` (D5) and exact duplicates (same row id and same messages). Rows that share a row
    id but differ in content are kept and reported (``row_id_collisions``). A file superseded by a
    present replacement (:data:`REPLACES`) is not read.

    :param families_dir: Directory holding ``<stem>.jsonl`` family files.
    :param cfg: Mixture configuration.
    :param exclude_rows: row_id → reason of rows that never enter the mixture.
    :returns: ``(items, stats)``.
    :raises FileNotFoundError: When a required family file is missing.
    """
    fdir = Path(families_dir)
    stats: Dict[str, Any] = {"inputs": {}, "missing_inputs": [], "ignored_files": [],
                             "dropped": Counter(), "dropped_split": Counter(), "schema_problems": Counter(),
                             "row_id_collisions": 0, "replaced_inputs": {}, "excluded_rows_by_family": Counter()}
    exclude_rows = exclude_rows or {}
    known = set(cfg.files)
    superseded = {old: new for new, old in REPLACES.items() if new in known and (fdir / f"{new}.jsonl").exists()}
    if fdir.is_dir():
        stats["ignored_files"] = sorted(p.name for p in fdir.glob("*.jsonl") if p.stem not in known)
    items: List[Item] = []
    seen_uid: set = set()
    seen_rid: Dict[str, str] = {}
    for stem in cfg.files:
        path = fdir / f"{stem}.jsonl"
        if stem in superseded:
            stats["replaced_inputs"][f"families/{stem}.jsonl"] = f"families/{superseded[stem]}.jsonl"
            LOG.info("%s: families/%s.jsonl replaced by families/%s.jsonl", cfg.name, stem, superseded[stem])
            continue
        if not path.exists():
            if stem in cfg.required:
                raise FileNotFoundError(f"required family file missing: families/{stem}.jsonl")
            LOG.warning("%s: optional family file families/%s.jsonl not found, skipped", cfg.name, stem)
            stats["missing_inputs"].append(f"families/{stem}.jsonl")
            continue
        n_read = n_kept = 0
        for row in R.read_jsonl(path):
            n_read += 1
            meta = row.get("meta") or {}
            fam = family_of(meta)
            split = meta.get("split")
            if split != "train":
                stats["dropped_split"][str(split)] += 1
                continue
            if meta.get("held_out_vision_benchmark") is True:
                stats["dropped"]["held_out_vision_flag_in_train"] += 1
                continue
            if fam not in cfg.families:
                stats["dropped"]["family_not_in_mixture"] += 1
                continue
            if meta.get("condition") in SLICE_CONDITIONS and not cfg.include_slice:
                stats["dropped"]["slice_not_in_mixture"] += 1
                continue
            if meta.get("condition") in cfg.exclude_conditions or meta.get("mixture_excluded"):
                stats["dropped"][f"condition_excluded:{meta.get('mixture_excluded') or meta.get('condition')}"] += 1
                continue
            fatal, other = _structural_problems(row, fam)
            for p in other:
                stats["schema_problems"][_problem_key(p)] += 1
            if fatal:
                stats["dropped"]["unrenderable"] += 1
                continue
            cer = meta.get("input_cer")
            if (cfg.exposed_near_dup_cer is not None and meta.get("condition") == "read"
                    and meta.get("reader_exposed") is True and isinstance(cer, (int, float))
                    and cer < cfg.exposed_near_dup_cer):
                stats["dropped"]["exposed_read_near_duplicate"] += 1
                continue
            rid = str(meta.get("row_id") or "")
            if rid and rid in exclude_rows:
                stats["dropped"]["excluded_rows"] += 1
                stats["excluded_rows_by_family"][f"{fam}|{meta.get('condition')}"] += 1
                continue
            content = _digest(_content(row, "system"), _content(row, "user"), _content(row, "assistant"))
            uid = f"{rid}:{content[:10]}"
            if uid in seen_uid:
                stats["dropped"]["duplicate_row"] += 1
                continue
            if rid in seen_rid and seen_rid[rid] != uid:
                stats["row_id_collisions"] += 1
            seen_uid.add(uid)
            seen_rid.setdefault(rid, uid)
            tok = meta.get("tokens")
            items.append(Item(row=row, src=stem, uid=uid,
                              tokens=tok if isinstance(tok, int) and not isinstance(tok, bool) else None))
            n_kept += 1
        stats["inputs"][f"families/{stem}.jsonl"] = {"sha256": file_sha256(path), "rows": n_read,
                                                     "train_rows_kept": n_kept}
        LOG.info("%s: families/%s.jsonl %d rows, %d kept", cfg.name, stem, n_read, n_kept)
    if stats["row_id_collisions"]:
        LOG.warning("%s: %d rows share a row id with different content (kept)", cfg.name,
                    stats["row_id_collisions"])
    return items, stats


# ---------------------------------------------------------------------------------------- select
def apply_swap(items: Sequence[Item]) -> Tuple[List[Item], Dict[str, Any]]:
    """The swap rule: a slice row replaces the ``noised_edition`` row(s) of the same (document, family).

    :param items: Pool items.
    :returns: ``(items without the replaced noised rows, stats)``; stats count slice rows, removed
        noised rows, swapped (doc, family) pairs and slice pairs that had no noised copy, per family.
    """
    slice_pairs = {(it.doc, it.family) for it in items if it.condition in SLICE_CONDITIONS}
    noised_pairs = {(it.doc, it.family) for it in items if it.condition == "noised_edition"}
    kept: List[Item] = []
    by_family: Dict[str, Counter] = defaultdict(Counter)
    for it in items:
        if it.condition == "noised_edition" and (it.doc, it.family) in slice_pairs:
            by_family[it.family]["noised_removed"] += 1
            continue
        if it.condition in SLICE_CONDITIONS:
            by_family[it.family]["slice_rows"] += 1
        kept.append(it)
    for doc, fam in slice_pairs:
        by_family[fam]["pairs_swapped" if (doc, fam) in noised_pairs else "slice_without_noised"] += 1
    totals: Counter = Counter()
    for c in by_family.values():
        totals.update(c)
    for k in ("slice_rows", "noised_removed", "pairs_swapped", "slice_without_noised"):
        totals.setdefault(k, 0)
    return kept, {**dict(totals), "by_family": {f: dict(c) for f, c in sorted(by_family.items())}}


def choose_one(doc: str, items: Sequence[Item], seed: int, purpose: str, weights: Mapping[str, float],
               slice_policy: str = "single", epoch: int = 1) -> Tuple[Optional[Item], Optional[str]]:
    """The one-condition rule for one document.

    :param doc: Document key.
    :param items: The document's candidate rows (one or more families).
    :param seed: Mixture seed.
    :param purpose: Budget name (``catalogue_describe``, ``extract_relations``); decorrelates draws.
    :param weights: Class weights (``edition`` / ``read`` / ``noised``), renormalised over the classes present.
    :param slice_policy: ``single`` (a slice row, when present, is the document's row) or ``noised_class``.
    :param epoch: Epoch number (part of the hash key).
    :returns: ``(chosen item, class)``; ``(None, None)`` when no candidate has a selectable class.
    """
    usable = [it for it in items if cond_class(it.condition) is not None]
    if not usable:
        return None, None
    slice_items = [it for it in usable if it.condition in SLICE_CONDITIONS]
    if slice_items and slice_policy == "single":
        cls, pool = "slice", slice_items
    else:
        by_cls: Dict[str, List[Item]] = defaultdict(list)
        for it in usable:
            by_cls[cond_class(it.condition)].append(it)  # type: ignore[index]
        options = [(c, float(weights.get(c, 0.0))) for c in CLASS_ORDER if by_cls.get(c)]
        if not any(w > 0 for _, w in options):
            return None, None
        cls = pick_weighted(unit_hash(seed, epoch, purpose, doc, "class"), options)
        pool = by_cls[cls]
    fams = sorted({it.family for it in pool})
    fam = fams[_pick_index(unit_hash(seed, epoch, purpose, doc, "family"), len(fams))]
    cands = sorted((it for it in pool if it.family == fam), key=lambda it: it.uid)
    chosen = cands[_pick_index(unit_hash(seed, epoch, purpose, doc, "row"), len(cands))]
    if cls != "slice" and chosen.condition in SLICE_CONDITIONS:
        cls = "slice"
    return chosen, cls


def _group_by_doc(items: Iterable[Item]) -> Dict[str, List[Item]]:
    """Group items by document key.

    :param items: Items.
    :returns: ``{doc: [items]}``.
    """
    out: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        out[it.doc].append(it)
    return out


def select_one_per_doc(items: Sequence[Item], cfg: MixConfig, purpose: str,
                       docs: Optional[Iterable[str]] = None) -> Tuple[List[Item], Dict[str, Any]]:
    """Apply the one-condition rule to every document (or to ``docs``) of a pool.

    :param items: Candidate rows (all families sharing the budget).
    :param cfg: Mixture configuration (seed, epoch, class weights, slice policy).
    :param purpose: Budget name.
    :param docs: Restrict to these documents (default: every document in ``items``).
    :returns: ``(chosen items, stats)`` with documents, rows offered, realised classes and families.
    """
    by_doc = _group_by_doc(items)
    wanted = sorted(by_doc) if docs is None else sorted(set(docs) & set(by_doc))
    chosen: List[Item] = []
    classes: Counter = Counter()
    fams: Counter = Counter()
    no_choice = 0
    for d in wanted:
        it, cls = choose_one(d, by_doc[d], cfg.seed, purpose, cfg.class_weights, cfg.slice_policy, cfg.epoch)
        if it is None:
            no_choice += 1
            continue
        chosen.append(it)
        classes[cls] += 1
        fams[it.family] += 1
    stats = {"docs_in_pool": len(by_doc), "docs_selected": len(wanted), "rows_offered": len(items),
             "rows_chosen": len(chosen), "docs_without_selectable_row": no_choice,
             "classes": dict(sorted(classes.items())), "families": dict(sorted(fams.items()))}
    return chosen, stats


def select_subsample(items: Sequence[Item], cfg: MixConfig, family: str, frac: float,
                     max_copies: Optional[int] = None) -> Tuple[List[Item], Dict[str, Any]]:
    """Subsampled family (relations 40 %): ``round(frac × rows)`` documents drawn by seeded hash
    (capped at the number of documents), one row each by the one-condition rule.

    With ``max_copies`` the documents are taken in ranking order and a document whose chosen row's
    target already appears ``max_copies`` times is skipped (D11: identical relations targets at most
    five times per epoch); the walk continues down the ranking until the budget is met or the ranking
    is exhausted.

    :param items: The family's rows.
    :param cfg: Mixture configuration.
    :param family: Family name (the budget's purpose key).
    :param frac: Fraction of the family's rows to keep.
    :param max_copies: Cap on copies of one target (None = no cap).
    :returns: ``(chosen items, stats)``.
    """
    by_doc = _group_by_doc(items)
    k = min(len(by_doc), int(round(frac * len(items))))
    ranked = sorted(by_doc, key=lambda d: (unit_hash(cfg.seed, cfg.epoch, family, "doc", d), d))
    if not max_copies:
        chosen, stats = select_one_per_doc(items, cfg, family, docs=ranked[:k])
    else:
        chosen = []
        classes: Counter = Counter()
        fams: Counter = Counter()
        per_target: Counter = Counter()
        skipped = no_choice = walked = 0
        for d in ranked:
            if len(chosen) >= k:
                break
            walked += 1
            it, cls = choose_one(d, by_doc[d], cfg.seed, family, cfg.class_weights, cfg.slice_policy, cfg.epoch)
            if it is None:
                no_choice += 1
                continue
            th = target_hash(it.row)
            if per_target[th] >= max_copies:
                skipped += 1
                continue
            per_target[th] += 1
            chosen.append(it)
            classes[cls] += 1
            fams[it.family] += 1
        stats = {"docs_in_pool": len(by_doc), "docs_selected": len(chosen), "docs_walked": walked,
                 "rows_offered": len(items), "rows_chosen": len(chosen), "docs_without_selectable_row": no_choice,
                 "docs_skipped_target_cap": skipped, "max_copies_per_target": max_copies,
                 "max_copies_seen": max(per_target.values(), default=0),
                 "classes": dict(sorted(classes.items())), "families": dict(sorted(fams.items()))}
    stats.update({"frac": frac, "target_rows": int(round(frac * len(items))), "capped_at_docs": k < round(frac * len(items))})
    return chosen, stats


def stratum_of(meta: Mapping[str, Any]) -> Tuple[str, str]:
    """Coverage stratum of a row: its work and title (for canon rows the Sefaria title -- tractate / book of hilkhot).

    :param meta: Row meta.
    :returns: ``(work, title)`` (``''`` parts when absent).
    """
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    title = prov.get("sefaria_title") or meta.get("catalogue_title") or meta.get("page_range") or ""
    return str(meta.get("work") or ""), str(title)


def weighted_rank_key(u: float, weight: float) -> float:
    """Sort key of weighted sampling without replacement (Efraimidis–Spirakis, exponential form): ascending
    ``-ln(u) / w`` orders the rows as successive draws with probability proportional to ``w``.

    :param u: Seeded uniform number in [0, 1).
    :param weight: The row's weight (> 0).
    :returns: The key (smaller = drawn earlier).
    :raises ValueError: On a non-positive weight.
    """
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")
    return -math.log(max(u, 1e-300)) / weight


def apply_token_caps(items: Sequence[Item], cfg: MixConfig) -> Tuple[List[Item], Dict[str, Any]]:
    """Seeded subsample of the families with a token budget (``cfg.token_caps``; fix pass F5, canon fold-in).

    The family's rows are ranked by a seeded hash of their uid -- uniformly, or, when ``cfg.cap_weights`` names
    the family, by weighted sampling without replacement over ``meta.work`` (:func:`weighted_rank_key`; works not
    listed weigh 1). For a family in ``cfg.cap_cover`` the best-ranked row of every (work, title) stratum
    (:func:`stratum_of`) moves to the front, so each tractate / book keeps a row. Rows are then taken in that order
    while the epoch tokens (row tokens × the component's repeat count) stay within the cap; a row that would
    overshoot is skipped and the walk goes on, so smaller rows still fill the budget. Rows of other families pass
    unchanged.

    :param items: Pool items (after the swap).
    :param cfg: Mixture configuration.
    :returns: ``(kept items, {family: {"cap", "rows_offered", "rows_kept", "tokens_offered", "tokens_kept", ...}})``;
        weighted families add ``weights`` and ``by_work`` (rows / tokens offered and kept per work), covered ones
        ``strata_offered``, ``strata_kept`` and ``strata_missing``.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    drop: set = set()
    for fam, cap in sorted(cfg.token_caps.items()):
        fam_items = [it for it in items if it.family == fam]
        if not fam_items:
            continue
        weights = cfg.cap_weights.get(fam)

        def key(it: Item) -> Tuple[float, str]:
            """Rank key of one row.

            :param it: Item.
            :returns: ``(key, uid)``.
            """
            u = unit_hash(cfg.seed, cfg.epoch, cfg.name, "token_cap", fam, it.uid)
            if weights is None:
                return u, it.uid
            return weighted_rank_key(u, float(weights.get(str(it.meta.get("work")), 1.0))), it.uid

        ranked = sorted(fam_items, key=key)
        if fam in cfg.cap_cover:
            firsts: Dict[Tuple[str, str], Item] = {}
            for it in ranked:
                firsts.setdefault(stratum_of(it.meta), it)
            lead = {it.uid for it in firsts.values()}
            ranked = [it for it in ranked if it.uid in lead] + [it for it in ranked if it.uid not in lead]
        used = kept = offered = 0
        work_c: Dict[str, Counter] = defaultdict(Counter)
        kept_strata: set = set()
        for it in ranked:
            t = int(it.tokens or 0) * max(1, int(cfg.repeats.get(it.component, 1) or 1))
            offered += t
            w = str(it.meta.get("work"))
            work_c[w]["rows_offered"] += 1
            work_c[w]["tokens_offered"] += t
            if used + t <= cap:
                used += t
                kept += 1
                work_c[w]["rows_kept"] += 1
                work_c[w]["tokens_kept"] += t
                kept_strata.add(stratum_of(it.meta))
            else:
                drop.add(it.uid)
        st: Dict[str, Any] = {"cap": int(cap), "rows_offered": len(fam_items), "rows_kept": kept,
                              "tokens_offered": offered, "tokens_kept": used}
        if weights is not None:
            st["weights"] = dict(weights)
            st["by_work"] = {w: {k: int(c[k]) for k in ("rows_offered", "rows_kept", "tokens_offered", "tokens_kept")}
                             for w, c in sorted(work_c.items())}
        if fam in cfg.cap_cover:
            strata = {stratum_of(it.meta) for it in fam_items}
            st["strata_offered"] = len(strata)
            st["strata_kept"] = len(kept_strata)
            st["strata_missing"] = sorted("|".join(x) for x in strata - kept_strata)
        stats[fam] = st
    return [it for it in items if it.uid not in drop], stats


def select(items: Sequence[Item], cfg: MixConfig) -> Tuple[List[Instance], Dict[str, Any]]:
    """Turn the pool into the epoch's instances (before ordering).

    Order of rules per row: the swap (always, when slice rows are present); the token budgets
    (:func:`apply_token_caps`); family in ``cfg.one_per_doc`` → one shared row per document; family in
    ``cfg.subsample`` → subsampled documents; otherwise the component's ``repeats`` (rows of a component
    without a repeat count, or with 0, are dropped and counted).

    :param items: Pool items from :func:`load_pool`.
    :param cfg: Mixture configuration.
    :returns: ``(instances, selection stats)``.
    """
    pool, swap_stats = apply_swap(items)
    stats: Dict[str, Any] = {"swap": swap_stats, "rules": {}, "dropped_no_rule": Counter()}
    if cfg.token_caps:
        pool, stats["token_caps"] = apply_token_caps(pool, cfg)
    one_doc = [it for it in pool if it.family in cfg.one_per_doc]
    instances: List[Instance] = []
    if one_doc:
        chosen, st = select_one_per_doc(one_doc, cfg, "catalogue_describe")
        stats["rules"]["one_per_doc"] = st
        instances += [Instance(it, 0, 1, "one_per_doc") for it in chosen]
    for fam, frac in sorted(cfg.subsample.items()):
        fam_items = [it for it in pool if it.family == fam and it.family not in cfg.one_per_doc]
        if not fam_items:
            continue
        cap = cfg.max_copies_per_target if fam in cfg.cap_families else None
        chosen, st = select_subsample(fam_items, cfg, fam, frac, cap)
        stats["rules"][f"subsample:{fam}"] = st
        instances += [Instance(it, 0, 1, f"subsample:{fam}") for it in chosen]
    rep: Counter = Counter()
    for it in pool:
        if it.family in cfg.one_per_doc or it.family in cfg.subsample:
            continue
        r = int(cfg.repeats.get(it.component, 0))
        if r <= 0:
            stats["dropped_no_rule"][it.component] += 1
            continue
        rep[f"{it.component}×{r}"] += 1
        instances += [Instance(it, k, r, f"repeat×{r}") for k in range(r)]
    stats["rules"]["repeat"] = dict(sorted(rep.items()))
    stats["dropped_no_rule"] = dict(stats["dropped_no_rule"])
    if stats["dropped_no_rule"]:
        LOG.warning("%s: rows without a mixture rule dropped: %s", cfg.name, stats["dropped_no_rule"])
    return instances, stats


def order_instances(instances: Sequence[Instance], cfg: MixConfig) -> List[Instance]:
    """Seeded stratified order: copy ``k`` of an ``r``-times row sorts at ``(k + u) / r``.

    :param instances: Instances from :func:`select`.
    :param cfg: Mixture configuration (seed, epoch, name).
    :returns: The instances sorted into the epoch order (``order`` filled in).
    """
    for inst in instances:
        u = unit_hash(cfg.seed, cfg.epoch, cfg.name, "order", inst.item.uid, inst.copy)
        inst.order = (inst.copy + u) / inst.repeats
    return sorted(instances, key=lambda i: (i.order, i.item.uid, i.copy))


def instance_row(inst: Instance, cfg: MixConfig) -> Dict[str, Any]:
    """The output row of one instance (messages shared, meta copied with mixture fields).

    :param inst: Instance.
    :param cfg: Mixture configuration.
    :returns: Row with ``meta.mixture``, ``mix_component``, ``mix_rule``, ``mix_copy``, ``mix_repeats``
        (and ``tokens`` when recounted).
    """
    meta = dict(inst.item.meta)
    if inst.item.tokens is not None:
        meta["tokens"] = inst.item.tokens
    meta.update({"mixture": cfg.name, "mix_component": inst.item.component, "mix_rule": inst.rule,
                 "mix_copy": inst.copy, "mix_repeats": inst.repeats})
    return {"messages": inst.item.row["messages"], "meta": meta}


# ---------------------------------------------------------------------------------------- tokens
def count_item_tokens(items: Sequence[Item], recount: bool = False,
                      template_path: Optional[Union[str, Path]] = None,
                      tokenizer_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Fill exact rendered token counts (``enable_thinking=False``) where missing, or for all items.

    :param items: Items (``tokens`` updated in place).
    :param recount: Recount every item, not only those without ``meta.tokens``.
    :param template_path: Explicit chat template (default: ``paths.chat_template``).
    :param tokenizer_path: Explicit tokenizer (default: ``paths.tokenizer_json``).
    :returns: ``{"counted", "mismatch", "max_abs_diff"}`` (mismatch against the builders' ``meta.tokens``).
    """
    todo = [it for it in items if recount or it.tokens is None]
    if not todo:
        return {"counted": 0, "mismatch": 0, "max_abs_diff": 0}
    rendered = [R.render_chat(it.row["messages"], template_path=template_path, check=False) for it in todo]
    counts = R.count_tokens_batch(rendered, tokenizer_path)
    mismatch, max_diff = 0, 0
    for it, n in zip(todo, counts):
        if it.tokens is not None and it.tokens != n:
            mismatch += 1
            max_diff = max(max_diff, abs(it.tokens - n))
        it.tokens = n
    return {"counted": len(todo), "mismatch": mismatch, "max_abs_diff": max_diff}


# -------------------------------------------------------------------------------------- manifest
def histogram(values: Iterable[int]) -> Dict[str, int]:
    """Histogram of integer values with string keys in numeric order.

    :param values: Values.
    :returns: ``{"value": count}``.
    """
    c = Counter(values)
    return {str(k): c[k] for k in sorted(c)}


def _tok(inst: Instance) -> int:
    """Token count of an instance (0 when unknown).

    :param inst: Instance.
    :returns: Tokens.
    """
    return int(inst.item.tokens or 0)


def summarise(ordered: Sequence[Instance], cfg: MixConfig) -> Dict[str, Any]:
    """Manifest statistics of a materialised epoch.

    :param ordered: Instances in epoch order.
    :param cfg: Mixture configuration.
    :returns: Dict with ``counts`` (rows / tokens per family|condition|lang_class|split), ``tokens``
        nested per family → condition → lang, ``components`` (rows, tokens, share), ``noise_slice``,
        ``totals``, ``repeats_per_target`` / ``repeats_per_row`` / ``repeats_per_document`` histograms,
        ``projected_steps`` and ``fingerprint``.
    """
    rows_c: Counter = Counter()
    tok_c: Counter = Counter()
    nested: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    comp_rows: Counter = Counter()
    comp_tok: Counter = Counter()
    targets: Counter = Counter()
    target_fams: Dict[str, set] = defaultdict(set)
    docs: Counter = Counter()
    total_tok = 0
    sup_tok = 0
    slice_rows = slice_tok = 0
    for inst in ordered:
        m = inst.item.meta
        fam, cond, lang = inst.item.family, inst.item.condition, str(m.get("lang_class") or "unknown")
        key = f"{fam}|{cond}|{lang}|{m.get('split')}"
        t = _tok(inst)
        rows_c[key] += 1
        tok_c[key] += t
        nested[fam][cond][lang] += t
        comp_rows[inst.item.component] += 1
        comp_tok[inst.item.component] += t
        th = target_hash(inst.item.row)
        targets[th] += 1
        target_fams[th].add(fam)
        if fam in R.PGP_FAMILIES or m.get("pgpid") not in (None, ""):
            docs[inst.item.doc] += 1
        total_tok += t
        tt = m.get("target_tokens")
        sup_tok += int(tt) if isinstance(tt, int) and not isinstance(tt, bool) else 0
        if cond in SLICE_CONDITIONS:
            slice_rows += 1
            slice_tok += t
    by_family_rep: Dict[str, List[int]] = defaultdict(list)
    for th, n in targets.items():
        for fam in target_fams[th]:
            by_family_rep[fam].append(n)
    n_rows = len(ordered)
    share = (lambda x: round(x / total_tok, 4) if total_tok else 0.0)
    return {
        "counts": {"rows": dict(sorted(rows_c.items())), "tokens": dict(sorted(tok_c.items()))},
        "tokens": {f: {c: dict(sorted(l.items())) for c, l in sorted(cs.items())} for f, cs in sorted(nested.items())},
        "components": {c: {"rows": comp_rows[c], "tokens": comp_tok[c], "share": share(comp_tok[c])}
                       for c in sorted(comp_rows)},
        "noise_slice": {"rows": slice_rows, "tokens": slice_tok, "share": share(slice_tok)},
        "totals": {"rows": n_rows, "distinct_rows": len({i.item.uid for i in ordered}), "tokens": total_tok,
                   "supervised_tokens_meta": sup_tok, "distinct_targets": len(targets)},
        "repeats_per_target": {"histogram": histogram(targets.values()),
                               "by_family": {f: histogram(v) for f, v in sorted(by_family_rep.items())},
                               "max": max(targets.values(), default=0),
                               "mean": round(sum(targets.values()) / len(targets), 3) if targets else 0.0},
        "repeats_per_row": histogram(Counter(i.item.uid for i in ordered).values()),
        "repeats_per_document": {"histogram": histogram(docs.values()), "documents": len(docs)},
        "projected_steps": {"rows_per_step": cfg.rows_per_step,
                            "unpacked_steps": math.ceil(n_rows / cfg.rows_per_step) if n_rows else 0,
                            "tokens_per_step": cfg.tokens_per_step,
                            "packed_steps": math.ceil(total_tok / cfg.tokens_per_step) if total_tok else 0},
    }


def rag_twin_coverage(ordered: Sequence[Instance], items: Sequence[Item]) -> Dict[str, Any]:
    """How many RAG rows have their no-literature twin (``meta.twin_row_id``) in the epoch / in the pool.

    :param ordered: Instances of the epoch.
    :param items: The pool the epoch was drawn from (after exclusions).
    :returns: ``{"rag_rows", "twin_in_epoch", "twin_in_pool_only", "twin_missing", "share_in_epoch", "by_task"}``.
    """
    in_epoch = {i.item.row_id for i in ordered}
    in_pool = {it.row_id for it in items}
    rag = {i.item.uid: i.item for i in ordered if i.item.family == "rag_context"}
    out: Counter = Counter()
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for it in rag.values():
        tw = str(it.meta.get("twin_row_id") or "")
        where = "twin_in_epoch" if tw in in_epoch else "twin_in_pool_only" if tw in in_pool else "twin_missing"
        out[where] += 1
        by_task[str(it.meta.get("task"))][where] += 1
    n = len(rag)
    return {"rag_rows": n, "twin_in_epoch": out["twin_in_epoch"], "twin_in_pool_only": out["twin_in_pool_only"],
            "twin_missing": out["twin_missing"], "share_in_epoch": round(out["twin_in_epoch"] / n, 4) if n else None,
            "by_task": {t: dict(c) for t, c in sorted(by_task.items())}}


# ---------------------------------------------------------------------------------------- render
def _render_sample(ordered: Sequence[Instance], n: int, cfg: MixConfig) -> List[Item]:
    """Rows to render-check: per family the longest plus ``n - 1`` seeded picks (distinct rows).

    :param ordered: Instances.
    :param n: Rows per family.
    :param cfg: Mixture configuration (seed).
    :returns: Items to check.
    """
    by_fam: Dict[str, Dict[str, Item]] = defaultdict(dict)
    for inst in ordered:
        by_fam[inst.item.family][inst.item.uid] = inst.item
    out: List[Item] = []
    for fam in sorted(by_fam):
        its = sorted(by_fam[fam].values(), key=lambda it: it.uid)
        if not its or n <= 0:
            continue
        longest = max(its, key=lambda it: (it.tokens or 0, it.uid))
        rest = sorted((it for it in its if it.uid != longest.uid),
                      key=lambda it: unit_hash(cfg.seed, cfg.name, "render", it.uid))
        out += [longest] + rest[: n - 1]
    return out


def render_check(ordered: Sequence[Instance], cfg: MixConfig, template_path: Optional[Union[str, Path]] = None,
                 tokenizer_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """G3 render check: render sampled rows with ``enable_thinking=False`` and test the contract.

    Per row: exactly one empty think block, no reasoning-effort sentence, the prompt/completion split
    reproduces the render (the scaffold is maskable), total tokens ≤ ``cfg.max_seq``.

    :param ordered: Instances of the epoch.
    :param cfg: Mixture configuration.
    :param template_path: Explicit chat template.
    :param tokenizer_path: Explicit tokenizer.
    :returns: ``{"ok", "n_checked", "n_failed", "max_seq", "rows": [...]}``.
    """
    results: List[Dict[str, Any]] = []
    for it in _render_sample(ordered, cfg.render_check_n, cfg):
        msgs = it.row["messages"]
        rendered = R.render_chat(msgs, template_path=template_path, check=False)
        res: Dict[str, Any] = {"row_id": it.row_id, "family": it.family, "condition": it.condition,
                               "lang_class": it.meta.get("lang_class"),
                               "n_empty_think": rendered.count(R.EMPTY_THINK),
                               "n_think_tags": rendered.count("<think>"),
                               "effort_sentence": R.EFFORT_MARKER in rendered,
                               "tokens": R.count_tokens(rendered, tokenizer_path), "error": None}
        try:
            R.check_render(rendered, msgs)
            prompt, completion = R.split_prompt_completion(msgs, template_path=template_path)
            res["prompt_tokens"] = R.count_tokens(prompt, tokenizer_path)
            res["completion_tokens"] = R.count_tokens(completion, tokenizer_path)
            res["mask_ok"] = prompt + completion == rendered
        except R.RenderError as exc:  # an expected failure mode: recorded, not raised
            res["error"] = str(exc)
            res["mask_ok"] = False
        res["within_max_seq"] = res["tokens"] <= cfg.max_seq
        res["ok"] = (res["error"] is None and res["mask_ok"] and res["n_empty_think"] == 1
                     and res["n_think_tags"] == 1 and not res["effort_sentence"] and res["within_max_seq"])
        results.append(res)
    failed = [r for r in results if not r["ok"]]
    return {"ok": bool(results) and not failed, "n_checked": len(results), "n_failed": len(failed),
            "max_seq": cfg.max_seq, "families": sorted({r["family"] for r in results}), "rows": results}


# ----------------------------------------------------------------------------------------- gates
def literature_floor_check(ordered: Sequence[Instance], cfg: MixConfig,
                           cap_stats: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The literature floor of an epoch (F13, restated 2026-10-26): literature components' share of the CORE epoch's
    tokens ≥ ``min_share`` and the L1 (``domain_reading``) rule: with ``l1_whole_pool``, L1 ≤ its token budget and,
    when the offered pool fits the budget, every offered token is in the epoch (no floor); with ``l1_tolerance``
    (F13 as first written), L1 within that fraction of its budget.

    The core epoch is the epoch minus ``cfg.additive_components`` (the canon block; design-owner decision 2026-10-26:
    the canon families are a user-approved additive component with their own caps, so the documentary + literature
    core keeps its verified composition). Three shares are reported side by side: ``share_core`` (gated; also
    ``share``), ``share_whole_epoch`` and ``share_literature_plus_additive_whole_epoch``.

    :param ordered: Instances in epoch order.
    :param cfg: Mixture configuration (``literature_floor``, ``token_caps``, ``additive_components``).
    :param cap_stats: :func:`apply_token_caps` statistics (``tokens_offered`` of the L1 pool; None = unknown).
    :returns: ``{"pass", "violations", "share", "share_core", "share_whole_epoch",
        "share_literature_plus_additive_whole_epoch", "min_share", "l1_tokens", "l1_budget", "l1_tolerance",
        "l1_rule", "l1_pool_tokens", "core_tokens", "additive_tokens", "tokens_by_component", "share_by_component"}``
        (component shares of the core).
    """
    total = sum(_tok(i) for i in ordered)
    additive = set(cfg.additive_components)
    add_tok = sum(_tok(i) for i in ordered if i.item.component in additive)
    core = total - add_tok
    by: Counter = Counter()
    for i in ordered:
        if i.item.component in LITERATURE_COMPONENTS:
            by[i.item.component] += _tok(i)
    lit = sum(by.values())
    share = lit / core if core else 0.0
    min_share = float(cfg.literature_floor.get("min_share", 0.0))
    tol = float(cfg.literature_floor.get("l1_tolerance", 0.0))
    whole = bool(cfg.literature_floor.get("l1_whole_pool", 0.0))
    budget = cfg.token_caps.get("domain_reading")
    pool = (cap_stats or {}).get("domain_reading", {}).get("tokens_offered") if cap_stats else None
    l1 = by["domain_reading"]
    if whole:
        rule = "whole_pool_le_budget"
        l1_ok = (budget is None or l1 <= budget) and (pool is None or (budget is not None and pool > budget)
                                                       or l1 == pool)
    elif tol:
        rule = "budget_tolerance"
        l1_ok = budget is None or abs(l1 - budget) <= tol * budget
    else:
        rule, l1_ok = "none", True
    ok = share >= min_share and l1_ok
    return {"pass": ok, "violations": int(share < min_share) + int(not l1_ok), "share": round(share, 5),
            "gated_on": "core epoch (epoch minus additive components)", "share_core": round(share, 5),
            "share_whole_epoch": round(lit / total, 5) if total else 0.0,
            "share_literature_plus_additive_whole_epoch": round((lit + add_tok) / total, 5) if total else 0.0,
            "min_share": min_share, "literature_tokens": lit, "total_tokens": total, "core_tokens": core,
            "additive_components": sorted(additive), "additive_tokens": add_tok, "l1_tokens": by["domain_reading"],
            "l1_budget": budget, "l1_tolerance": tol, "l1_rule": rule, "l1_pool_tokens": pool,
            "tokens_by_component": {c: by[c] for c in LITERATURE_COMPONENTS},
            "share_by_component": {c: round(by[c] / core, 5) if core else 0.0 for c in LITERATURE_COMPONENTS}}


def check_gates(ordered: Sequence[Instance], cfg: MixConfig, summary: Mapping[str, Any],
                render: Optional[Mapping[str, Any]], cap_stats: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Hard gates of the materialised epoch (plus informational warnings).

    :param ordered: Instances in epoch order.
    :param cfg: Mixture configuration.
    :param summary: Output of :func:`summarise`.
    :param render: Output of :func:`render_check`, or None when skipped.
    :param cap_stats: :func:`apply_token_caps` statistics (the L1 pool for the literature floor's L1 rule).
    :returns: ``{"ok", "checks": {name: {"pass", ...}}, "warnings": [...]}``.
    """
    checks: Dict[str, Dict[str, Any]] = {}
    bad_split = sum(1 for i in ordered if i.item.meta.get("split") != "train")
    checks["train_split_only"] = {"pass": bad_split == 0, "violations": bad_split}
    vis = sum(1 for i in ordered if i.item.meta.get("held_out_vision_benchmark") is True)
    checks["no_vision_benchmark_flag"] = {"pass": vis == 0, "violations": vis}
    per_doc: Counter = Counter(i.item.doc for i in ordered if i.item.family in cfg.one_per_doc)
    multi = sum(1 for n in per_doc.values() if n > 1)
    checks["one_row_per_doc_catalogue_describe"] = {"pass": multi == 0, "violations": multi}
    for fam in sorted(cfg.subsample):
        c = Counter(i.item.doc for i in ordered if i.item.family == fam)
        m = sum(1 for n in c.values() if n > 1)
        checks[f"one_row_per_doc_{fam}"] = {"pass": m == 0, "violations": m}
    slice_pairs = {(i.item.doc, i.item.family) for i in ordered if i.item.condition in SLICE_CONDITIONS}
    both = len({(i.item.doc, i.item.family) for i in ordered if i.item.condition == "noised_edition"} & slice_pairs)
    checks["swap_no_noised_beside_slice"] = {"pass": both == 0, "violations": both}
    ar = sum(1 for i in ordered if i.item.condition == "noised_edition" and i.item.meta.get("lang_class") == "AR")
    checks["no_arabic_noised_rows"] = {"pass": ar == 0, "violations": ar}
    over = [i.item.row_id for i in ordered if _tok(i) > cfg.max_seq]
    checks["max_seq"] = {"pass": not over, "violations": len(over), "max_seq": cfg.max_seq,
                         "examples": sorted(set(over))[:20]}
    if cfg.max_copies_per_target:
        for fam in cfg.cap_families:
            per_t = Counter(target_hash(i.item.row) for i in ordered if i.item.family == fam)
            over_t = sum(1 for n in per_t.values() if n > cfg.max_copies_per_target)
            checks[f"max_copies_per_target_{fam}"] = {"pass": over_t == 0, "violations": over_t,
                                                       "max_seen": max(per_t.values(), default=0)}
    unknown_tok = sum(1 for i in ordered if i.item.tokens is None)
    checks["token_counts_known"] = {"pass": unknown_tok == 0, "violations": unknown_tok}
    for fam, cap in sorted(cfg.token_caps.items()):
        used = sum(_tok(i) for i in ordered if i.item.family == fam)
        checks[f"token_cap_{fam}"] = {"pass": used <= cap, "violations": int(used > cap), "tokens": used, "cap": cap}
    if cfg.literature_floor:
        checks["literature_floor"] = literature_floor_check(ordered, cfg, cap_stats)
    distinct = list({i.item.uid: i.item for i in ordered}.values())
    non_nfc = sorted({it.row_id for it in distinct if R.non_nfc_roles(it.row)})
    checks["nfc_messages"] = {"pass": not non_nfc, "violations": len(non_nfc), "examples": non_nfc[:20]}
    named = sorted({it.row_id for it in distinct if _tn.has_source_names(json.dumps(it.row, ensure_ascii=False))})
    checks["no_source_names_anywhere"] = {"pass": not named, "violations": len(named), "examples": named[:20]}
    if render is not None:
        checks["render_check"] = {"pass": bool(render.get("ok")), "violations": render.get("n_failed", 0)}
    warnings: List[str] = []
    if cfg.include_slice and summary["totals"]["tokens"]:
        s = summary["noise_slice"]["share"]
        lo, hi = SLICE_SHARE_BAND
        if not lo <= s <= hi:
            warnings.append(f"noise-slice token share {s:.3f} outside the plan band {lo}-{hi}")
    if render is None:
        warnings.append("render check skipped")
    return {"ok": all(c["pass"] for c in checks.values()), "checks": checks, "warnings": warnings}


# ------------------------------------------------------------------------------------------ build
def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON atomically (temporary file in the same directory, then rename).

    :param path: Destination.
    :param obj: JSON-serialisable object.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _merge_section(path: Path, name: str, section: Mapping[str, Any]) -> None:
    """Set ``obj[name] = section`` in a JSON file keyed by mixture name (other sections kept).

    :param path: JSON file.
    :param name: Mixture name.
    :param section: Section content.
    :returns: None.
    """
    obj: Dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    obj[name] = section
    _write_json_atomic(path, obj)


def _plain(obj: Any) -> Any:
    """Convert Counters / tuples / Paths inside stats into JSON-friendly values.

    :param obj: Any nested structure.
    :returns: The converted structure.
    """
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def l1_build_info(families_dir: Path) -> Optional[Dict[str, Any]]:
    """The L1 build parameters recorded in the mixture manifest (per-work token cap, works, offered train tokens).

    :param families_dir: Directory of the family files (``domain_reading_manifest.json``).
    :returns: ``{"per_work_token_cap", "works", "train_rows", "train_tokens", "capped_rows", "fingerprint"}`` or None.
    """
    path = Path(families_dir) / "domain_reading_manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    args = m.get("build_args") or {}
    tot = m.get("totals") or {}
    return {"per_work_token_cap": args.get("token_cap"), "works": len(args.get("works") or m.get("works") or []),
            "train_rows": tot.get("train_rows"), "train_tokens": tot.get("train_row_tokens"),
            "capped_rows": m.get("capped_rows"), "fingerprint": m.get("fingerprint")}


def build_mixture(cfg: MixConfig, families_dir: Union[str, Path], out_dir: Union[str, Path],
                  template_path: Optional[Union[str, Path]] = None,
                  tokenizer_path: Optional[Union[str, Path]] = None, run_render_check: bool = True,
                  exclude_rows: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Build one mixture epoch end to end and write its rows, manifest section and render check.

    :param cfg: Mixture configuration.
    :param families_dir: Directory of the family files.
    :param out_dir: Build output root (``paths.out_dir``); results go to ``out_dir/mixtures/``.
    :param template_path: Explicit chat template (default: ``paths.chat_template``).
    :param tokenizer_path: Explicit tokenizer (default: ``paths.tokenizer_json``).
    :param run_render_check: Run the G3 render check.
    :param exclude_rows: row_id → reason of rows kept out (D5; default: read ``cfg.exclude_rows_file`` when set).
    :returns: The manifest section (also written to ``mixtures/mixture_manifest.json``).
    """
    out = Path(out_dir)
    mix_dir = out / MIX_DIR
    if exclude_rows is None and cfg.exclude_rows_file:
        exclude_rows = load_exclusions(cfg.exclude_rows_file)
    items, load_stats = load_pool(families_dir, cfg, exclude_rows)
    instances, sel_stats = select(items, cfg)
    used = list({i.item.uid: i.item for i in instances}.values())
    tok_stats = count_item_tokens(used, cfg.recount_tokens, template_path, tokenizer_path)
    dropped_over: List[str] = []
    if cfg.drop_over_max_seq:
        dropped_over = sorted({i.item.row_id for i in instances if _tok(i) > cfg.max_seq})
        instances = [i for i in instances if _tok(i) <= cfg.max_seq]
    ordered = order_instances(instances, cfg)
    out_rows = [instance_row(i, cfg) for i in ordered]
    mix_path = mix_dir / f"{cfg.name}_epoch{cfg.epoch}.jsonl"
    R.write_jsonl(mix_path, out_rows)
    summary = summarise(ordered, cfg)
    summary["rag_twins"] = rag_twin_coverage(ordered, items)
    render = render_check(ordered, cfg, template_path, tokenizer_path) if run_render_check else None
    if render is not None:
        _merge_section(mix_dir / RENDER_CHECK_NAME, cfg.name, render)
    gates = check_gates(ordered, cfg, summary, render, sel_stats.get("token_caps"))
    lf = literature_floor_check(ordered, cfg, sel_stats.get("token_caps"))
    summary["literature_shares"] = {k: lf[k] for k in ("share_core", "share_whole_epoch",
                                                       "share_literature_plus_additive_whole_epoch", "literature_tokens",
                                                       "core_tokens", "additive_tokens", "total_tokens",
                                                       "additive_components", "l1_tokens", "l1_pool_tokens", "l1_rule",
                                                       "l1_budget")}
    summary["l1_build"] = l1_build_info(Path(families_dir))
    try:
        families_rel = str(Path(families_dir).resolve().relative_to(out.resolve()))
    except ValueError:
        families_rel = Path(families_dir).name
    section: Dict[str, Any] = {
        "mixture": cfg.name, "build": R.BUILD, "epoch": cfg.epoch, "seed": cfg.seed,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "output": f"{MIX_DIR}/{mix_path.name}", "output_sha256": file_sha256(mix_path),
        "fingerprint": R.fingerprint(out_rows), **summary,
        "selection": {**sel_stats, "dropped_over_max_seq": dropped_over},
        "load": {"inputs": load_stats["inputs"], "missing_inputs": load_stats["missing_inputs"],
                 "ignored_files": load_stats["ignored_files"], "dropped": dict(load_stats["dropped"]),
                 "dropped_split": dict(load_stats["dropped_split"]),
                 "schema_problems": dict(load_stats["schema_problems"]),
                 "row_id_collisions": load_stats["row_id_collisions"], "pool_rows": len(items),
                 "replaced_inputs": load_stats["replaced_inputs"],
                 "excluded_rows": {"file": Path(cfg.exclude_rows_file).name if cfg.exclude_rows_file else None,
                                   "sha256": file_sha256(cfg.exclude_rows_file)
                                   if cfg.exclude_rows_file and Path(cfg.exclude_rows_file).exists() else None,
                                   "listed": len(exclude_rows or {}),
                                   "dropped_by_family_condition": dict(sorted(load_stats["excluded_rows_by_family"].items()))}},
        "token_count": tok_stats, "gates": gates,
        "render_check": None if render is None else {k: render[k] for k in ("ok", "n_checked", "n_failed")},
        "build_args": {**asdict(cfg), "exclude_rows_file": Path(cfg.exclude_rows_file).name if cfg.exclude_rows_file else None,
                       "families_dir": families_rel, "run_render_check": run_render_check},
    }
    section = _plain(section)
    _merge_section(mix_dir / MANIFEST_NAME, cfg.name, section)
    LOG.info("%s: %d rows, %d tokens → %s (gates ok=%s)", cfg.name, summary["totals"]["rows"],
             summary["totals"]["tokens"], mix_path.name, gates["ok"])
    return section


# -------------------------------------------------------------------------------------------- CLI
def _parse_kv(pairs: Optional[Sequence[str]], cast: Any) -> Optional[Dict[str, Any]]:
    """Parse ``key=value`` CLI items (comma-separated items allowed).

    :param pairs: Items such as ``["edition=0.4,read=0.4", "noised=0.2"]``.
    :param cast: Value type (``int`` / ``float``).
    :returns: The mapping, or None when no items were given.
    :raises argparse.ArgumentTypeError: On an item without ``=``.
    """
    if not pairs:
        return None
    out: Dict[str, Any] = {}
    for chunk in pairs:
        for item in filter(None, (s.strip() for s in chunk.split(","))):
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
            k, v = item.split("=", 1)
            out[k.strip()] = cast(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser.

    :returns: The parser.
    """
    ap = argparse.ArgumentParser(prog="python -m src.finetuning.text27b.mixture",
                                 description="Materialise the t27a / t27b training epochs from the v0_2 family files.")
    ap.add_argument("--mixture", nargs="+", choices=MIXTURES, default=list(MIXTURES))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--paths", default=None, help="paths.json (default: $TEXT27B_PATHS or the repo default)")
    ap.add_argument("--families-dir", default=None, help="default: <out_dir>/families")
    ap.add_argument("--out-dir", default=None, help="default: paths.out_dir")
    ap.add_argument("--relations-frac", type=float, default=None)
    ap.add_argument("--class-weights", nargs="*", default=None, help="edition=0.4 read=0.4 noised=0.2")
    ap.add_argument("--repeat", nargs="*", default=None, help="component=n, e.g. translate_en=2 analyse_document=2")
    ap.add_argument("--slice-policy", choices=SLICE_POLICIES, default=None)
    ap.add_argument("--exposed-near-dup-cer", type=float, default=None,
                    help=f"drop reader-exposed reads below this CER (default {EXPOSED_NEAR_DUP_CER}; <0 disables)")
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--drop-over-max-seq", action="store_true")
    ap.add_argument("--recount-tokens", action="store_true", help="render + count every row instead of meta.tokens")
    ap.add_argument("--render-check-n", type=int, default=None)
    ap.add_argument("--no-render-check", action="store_true")
    ap.add_argument("--exclude-rows", default="auto",
                    help="excluded_rows.json (D5); 'auto' = <out_dir>/excluded_rows.json when present, 'none' = off")
    ap.add_argument("--max-copies-per-target", type=int, default=None,
                    help="cap identical targets of the capped families per epoch (t27b default 5; 0 disables)")
    ap.add_argument("--cap-families", nargs="*", default=None, help=f"families the cap applies to (default {list(CAP_FAMILIES)})")
    ap.add_argument("--token-caps", nargs="*", default=None,
                    help=f"family=tokens budgets per epoch, seeded subsample (t27b default {T27B_TOKEN_CAPS}, "
                         f"work weights {T27B_CAP_WEIGHTS}; 0 removes a cap)")
    ap.add_argument("--literature-floor", nargs="*", default=None,
                    help=f"min_share=x [l1_whole_pool=1 | l1_tolerance=y] gate on the literature components and the L1 rule "
                         f"(t27b default {T27B_LITERATURE_FLOOR}; F13 restated 2026-10-26); 'none' switches it off "
                         f"(synthetic tests)")
    ap.add_argument("--exclude-conditions", nargs="*", default=None,
                    help=f"conditions never drawn (t27b default {list(T27B_EXCLUDE_CONDITIONS)}; 'none' = keep all)")
    ap.add_argument("--without", nargs="*", default=None,
                    help="optional family file stems left out (e.g. book_pairs analyse_document_v2 while their agents run)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :returns: 0 when every mixture passes its gates, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = _paths.load_paths(args.paths)
    out_dir = Path(args.out_dir) if args.out_dir else p.out_dir
    fam_dir = Path(args.families_dir) if args.families_dir else out_dir / "families"
    near = args.exposed_near_dup_cer
    if args.exclude_rows == "none":
        ex_file: Optional[str] = None
    elif args.exclude_rows == "auto":
        ex_file = str(out_dir / EXCLUDED_ROWS_FILE) if (out_dir / EXCLUDED_ROWS_FILE).exists() else None
    else:
        ex_file = args.exclude_rows
        if not Path(ex_file).exists():
            raise FileNotFoundError(f"--exclude-rows file not found: {ex_file}")
    if ex_file is None:
        LOG.warning("no row exclusions applied (excluded_rows.json absent or --exclude-rows none)")
    status = 0
    for name in args.mixture:
        overrides: Dict[str, Any] = {"seed": args.seed, "epoch": args.epoch,
                                     "class_weights": _parse_kv(args.class_weights, float),
                                     "repeats": _parse_kv(args.repeat, int), "slice_policy": args.slice_policy,
                                     "max_seq": args.max_seq, "render_check_n": args.render_check_n,
                                     "drop_over_max_seq": args.drop_over_max_seq or None,
                                     "recount_tokens": args.recount_tokens or None,
                                     "exclude_rows_file": ex_file,
                                     "cap_families": tuple(args.cap_families) if args.cap_families else None,
                                     "token_caps": _parse_kv(args.token_caps, int)}
        if args.max_copies_per_target is not None:
            overrides["max_copies_per_target"] = args.max_copies_per_target
        if args.exclude_conditions is not None:
            overrides["exclude_conditions"] = () if args.exclude_conditions in ([], ["none"]) else \
                tuple(args.exclude_conditions)
        if args.literature_floor is not None:
            overrides["literature_floor"] = {} if args.literature_floor in ([], ["none"]) else \
                _parse_kv(args.literature_floor, float)
        if args.without:
            base_files = default_config(name).files
            required = default_config(name).required
            bad = sorted(set(args.without) & set(required))
            if bad:
                raise ValueError(f"--without cannot drop required family files: {bad}")
            overrides["files"] = tuple(f for f in base_files if f not in set(args.without))
        if args.relations_frac is not None and name == "t27b":
            overrides["subsample"] = {"extract_relations": args.relations_frac}
        if near is not None:
            overrides["exposed_near_dup_cer"] = None if near < 0 else near
        cfg = default_config(name, **{k: v for k, v in overrides.items()
                                      if v is not None or k in ("exposed_near_dup_cer", "exclude_rows_file")})
        sec = build_mixture(cfg, fam_dir, out_dir, p.chat_template, p.tokenizer_json,
                            run_render_check=not args.no_render_check)
        tot, steps = sec["totals"], sec["projected_steps"]
        print(f"\n== {name} (seed {cfg.seed}, epoch {cfg.epoch}) ==")
        print(f"rows {tot['rows']:,} (distinct {tot['distinct_rows']:,}), tokens {tot['tokens']:,}, "
              f"steps {steps['unpacked_steps']:,} unpacked @{steps['rows_per_step']} rows / "
              f"{steps['packed_steps']:,} packed @{steps['tokens_per_step']:,} tok")
        for comp, c in sec["components"].items():
            print(f"  {comp:20s} rows {c['rows']:7,}  tokens {c['tokens']:12,}  share {c['share']:.3f}")
        ns = sec["noise_slice"]
        print(f"  {'(noise slice)':20s} rows {ns['rows']:7,}  tokens {ns['tokens']:12,}  share {ns['share']:.3f}")
        print(f"  repeats per target: {sec['repeats_per_target']['histogram']}")
        print(f"  excluded rows dropped: {sec['load']['dropped'].get('excluded_rows', 0)}; rag twins: {sec['rag_twins']}")
        if sec["selection"].get("token_caps"):
            print(f"  token caps: {sec['selection']['token_caps']}")
        print(f"  gates ok={sec['gates']['ok']} warnings={sec['gates']['warnings']}")
        for k, c in sec["gates"]["checks"].items():
            if not c["pass"]:
                print(f"  FAILED {k}: {c}")
        status = status or (0 if sec["gates"]["ok"] else 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
