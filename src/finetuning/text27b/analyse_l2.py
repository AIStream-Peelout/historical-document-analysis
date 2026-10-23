# File name: analyse_l2.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""L2: the widened ``analyse_document`` family of the text27b build (plan §9.2 L2; phase-2 brief L2, D3/D14/D15).

The target of every row is an editor's human commentary on one document (introduction plus notes);
the input is the document text. Reads are only ever inputs. Three sources:

``span_pgp_edition``
    Document-treatment spans of the literature (``rag_spans.jsonl``, re-detected here with the
    section-sample cap lifted so the full intro / notes lines are available) joined to exactly one
    PGP document with a digital edition in the v0_2 builder output. Input = that document's
    edition (and the read / noised variants its ``catalogue_json`` rows carry).
``span_book_edition``
    Spans of documents without a PGP edition whose pages print the edition: input = the book's own
    edition text (the span's ``edition`` lines, line numbers removed).
``nas_commentary``
    Book-edition entries (``paths.book_editions_nas``: every sub-directory holding
    ``entries.jsonl``): introduction / discussion plus commentary / textual notes as the target,
    the entry's transcription as the input. ``--skip-pgp-overlap SLUG`` drops entries of a book
    whose document is already in PGP.

Gates (in order): commentary length; lines / sentences copying the input dropped (12-gram share
≥ 0.3, the v0 rule); provenance (span targets: share of their 5-word shingles found in the OCR
text of the span pages ≥ 0.8); held-out mentions (a train target naming a benchmark / eval document
by shelfmark is dropped); the v0_2 target strips (:func:`builder.clean_target`: citations, project names,
dimensions, line markers; D3: a target emptied by a strip is dropped and counted); the v0
target-vs-input overlap gate (no 60-character window of the target in the input); grounding tiers:

* ``A'`` -- :func:`textnorm.grounding` share ≥ 0.5 against input + catalogue context;
* ``B`` (opt-in, ``--tiers``) -- ≥ 0.6 with the book's own translation added (the v0 tier B);
* ``C`` -- ≥ 0.6 against input + catalogue + a literature block: up to ``k_max`` non-span pages of
  the same work ranked by BM25 against the INPUT (document + catalogue context, never the target),
  the target span ±1 page excluded, pages of every document span of the work excluded, pages in
  ``excluded_pages.json`` / ``extra_excluded_pages.json`` or mentioning a non-train document (±1)
  excluded, pages failing the L1 quality filter excluded, and the block holding < 5 % of the
  target's word 8-grams (no page with a 60-character target window);

then the eval-reference n-gram gate (:mod:`contamination`) on train targets, blocks and book inputs.
Train targets are gated at text level (their own slice: held mentions, n-gram leaks) and, by default (fix pass
F2; ``--no-page-level-exclusion`` switches it off), at page level: a train target whose span pages (the target
source, and for book-edition spans the input) appear in ``excluded_pages.json`` / the extra list with any reason
other than ``train_target_dup`` (the L1-only dedupe) is dropped (the strict §9.4.2 reading); literature blocks
never use an excluded page. Every row carries its span and block pages in ``meta.page_keys``; a book-edition
entry row carries the pages of its entry's PDF range (:class:`book_pairs.EntryPages`, fix pass F11), so the
same page rule drops a train entry on an excluded page; an entry that cannot be mapped is flagged
``meta.page_keys_unmapped`` and excluded from training by :mod:`contamination` (``unmapped_entry_pages``).

Splits (D14). A document's identity is its shelfmark key. A document matching the benchmark tier of
H (:mod:`holdouts`, by pgpid, canonical id or shelfmark key) → ``heldout``; a document in the v0_2
builder output keeps the builder's split; a document matching an eval document (v0_1 dev / test) →
``heldout``; a document sharing a shelfmark key with builder documents takes ``train`` when they
are all train, else ``heldout``; any other document gets ``sha1(doc key)`` buckets 80 / 10 / 10 →
``train`` / ``dev`` / ``test_indist``.

Outputs (``paths.out_dir``): ``families/analyse_document_v2.jsonl`` (all splits),
``families/analyse_v2_manifest.json`` (counts, tiers, grounding distributions, drops by reason,
tokens, build args) and ``audit/analyse_v2_sample_<n>.jsonl`` (stratified hand-audit sample).
No holdings are named here: works, books and their licences come from the local manifests.

Public API: :data:`FAMILY`, :class:`L2Config`, :class:`TokenFns`, :func:`default_token_fns`,
:class:`Variant`, :class:`DocInfo`, :class:`DocIndex`, :func:`load_doc_index`, :func:`detect_full_spans`,
:func:`attach_sections`, :func:`clean_book_edition`, :func:`render_transcription`, :class:`Candidate`,
:func:`span_candidates`, :func:`load_nas_books`, :func:`nas_commentary`, :func:`pgp_matches`,
:func:`nas_candidates`, :func:`hash_split`, :func:`assign_split`, :func:`drop_copied_units`,
:func:`span_target`, :func:`nas_target`, :func:`ocr_containment`, :func:`context_for`,
:func:`bm25_tokens`, :class:`BM25`, :class:`LitPool`, :func:`build_pools`, :func:`select_block`,
:func:`decide_tier`, :func:`fit_row`, :class:`L2Inputs`, :func:`build_rows`, :func:`audit_sample`,
:func:`load_inputs`, :func:`run`, :func:`main`.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.analyse_l2 build [--skip-pgp-overlap SLUG ...]
        [--tiers "A',C"] [--k-max 3] [--no-ngram] [--no-nas] [--no-spans] [--audit-n 30] [--out-dir DIR]
        [--no-page-level-exclusion]
"""
from __future__ import annotations

import __future__
import argparse
import collections
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from src.finetuning.text27b import builder as _bld
from src.finetuning.text27b import builder_v0 as _bv
from src.finetuning.text27b import pages as _pg
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rag_l3 as _rag
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as _tn

LOG = logging.getLogger(__name__)

FAMILY = "analyse_document"
OUT_FILE = "analyse_document_v2.jsonl"
MANIFEST_FILE = "analyse_v2_manifest.json"
#: Source kinds.
SPAN_PGP = "span_pgp_edition"
SPAN_BOOK = "span_book_edition"
NAS = "nas_commentary"
#: ``meta.page_keys_source`` of rows whose page keys come from the entry → PDF page-range mapping (F11; the value of
#: :data:`book_pairs.NAS_PAGE_SOURCE`).
NAS_PAGE_SOURCE = "nas_entry"
KIND_ORDER: Tuple[str, ...] = (SPAN_PGP, SPAN_BOOK, NAS)
#: Grounding tiers.
TIER_A = "A_prime"
TIER_B = "B"
TIER_C = "C"
TIER_ALIASES: Dict[str, str] = {"A": TIER_A, "A'": TIER_A, "Ap": TIER_A, "A_prime": TIER_A, "B": TIER_B, "C": TIER_C}
#: Families whose rows describe a builder document (for the document index), in preference order.
DOC_FAMILIES: Tuple[str, ...] = ("catalogue_json", "describe_pgp", "extract_relations", "translate_en", "translate_he")
#: Input conditions an L2 row may take from the document's catalogue rows.
DEFAULT_CONDITIONS: Tuple[str, ...] = ("edition", "noised_edition", "read")
#: Fields of a book-edition entry used as the introduction / the notes of the target.
NAS_INTRO_FIELDS: Tuple[str, ...] = ("introduction", "discussion")
NAS_NOTES_FIELDS: Tuple[str, ...] = ("commentary", "textual_notes", "notes")
SUPPORTED_SCRIPTS: Tuple[str, ...] = ("hebrew", "arabic")
#: Input meta fields copied from the source row of a variant.
VARIANT_META: Tuple[str, ...] = ("input_source", "input_cer", "input_truncated", "read_model", "read_coverage",
                                 "read_agreed_share", "edition_available", "reader_exposed", "exposure_level",
                                 "noise_band", "noise_class", "noise_target_cer")
NOTES_HEADER = "\n\nNotes\n\n"
CONTEXT_HEADER = "Known catalogue context:\n"
TRANSLATION_HEADER = "Translation (from the edition):\n"
_SCRIPT_RE = re.compile(r"\((Hebrew script|Arabic script|mixed script)\)")
_SCRIPT_FROM_NAME = {v: k for k, v in _rows.SCRIPT_NAME.items()}
#: The sample cap ``rag_l3.classify_lines`` applies to span sections (lifted here).
_SAMPLE_CAP = "< 1500"


# ============================================================================== configuration
@dataclass
class L2Config:
    """Thresholds and switches of one L2 build (defaults = plan §9.2 L2)."""

    tiers: Tuple[str, ...] = (TIER_A, TIER_C)
    tier_a_pass: float = 0.5
    tier_b_pass: float = 0.6
    tier_c_pass: float = 0.6
    block_max_share: float = 0.05
    k_max: int = 3
    min_commentary_chars: int = 300
    min_target_chars: int = 300
    min_target_after_strip: int = _bld.ANALYSIS_MIN_CHARS
    min_book_edition_chars: int = 150
    min_book_translation_chars: int = 150
    min_page_chars: int = 200
    min_ocr_containment: float = 0.8
    allow_missing_ocr: bool = False
    copy_ngram: int = 12
    copy_max_share: float = 0.3
    max_row_tokens: int = _bld.MAX_ROW_TOKENS
    conditions: Tuple[str, ...] = DEFAULT_CONDITIONS
    strict_non_span: bool = True
    window: int = 1
    skip_pgp_overlap: Tuple[str, ...] = ()
    excluded_books: Tuple[str, ...] = ()
    page_level_exclusion: bool = True
    #: Page-exclusion reason classes that do NOT make a span page unusable (contamination.L1_ONLY_REASONS: the D5
    #: dedupe binds L1 rows only, and every L2 train target's own span pages carry it).
    page_exclusion_ignore: Tuple[str, ...] = ("train_target_dup",)
    split_buckets: Tuple[Tuple[int, str], ...] = ((80, "train"), (90, "dev"), (100, "test_indist"))
    seed: int = 3407


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

    :param tokenizer_path: Explicit ``tokenizer.json`` (default ``paths.tokenizer_json``).
    :param template_path: Explicit ``chat_template.jinja`` (default ``paths.chat_template``).
    :returns: The counters.
    """
    def count(text: str) -> int:
        """Plain-text tokens.

        :param text: Text.
        :returns: Token count.
        """
        return _rows.count_tokens(text, tokenizer_path)

    def render_count(messages: Sequence[Dict[str, Any]]) -> int:
        """Tokens of the rendered chat (render contract checked).

        :param messages: Chat turns.
        :returns: Token count.
        """
        return _rows.count_tokens(_rows.render_chat(messages, template_path=template_path), tokenizer_path)

    return TokenFns(count=count, render_count=render_count)


# ============================================================================== builder documents
@dataclass
class Variant:
    """One input of a document (condition, text, input meta copied from its source row)."""

    condition: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocInfo:
    """A document of the v0_2 builder output (split and identity from its rows; texts when wanted)."""

    pgpid: str
    split: str
    group: Optional[str]
    lang_class: str
    shelfmark: Optional[str]
    held_out_vision_benchmark: bool
    holdout_reasons: List[str]
    script: Optional[str] = None
    edition: Optional[str] = None
    edition_truncated: bool = False
    variants: List[Variant] = field(default_factory=list)
    catalogue: Optional[Dict[str, Any]] = None
    families: Set[str] = field(default_factory=set)
    edition_row: bool = False

    @property
    def has_edition(self) -> bool:
        """Whether the builder had a PGP edition for this document (an edition-condition row without side split).

        :returns: True / False (set for every document, also when its texts were not kept).
        """
        return self.edition_row or self.edition is not None


@dataclass
class DocIndex:
    """Builder documents by pgpid and by shelfmark key."""

    by_pgpid: Dict[str, DocInfo] = field(default_factory=dict)
    by_key: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    def get(self, pgpid: Any) -> Optional[DocInfo]:
        """Document by pgpid.

        :param pgpid: PGP id.
        :returns: The document or None.
        """
        return self.by_pgpid.get(str(pgpid)) if pgpid not in (None, "") else None

    def has_edition(self, pgpid: Any) -> bool:
        """Whether a pgpid has a PGP edition in the builder output.

        :param pgpid: PGP id.
        :returns: True / False.
        """
        d = self.get(pgpid)
        return bool(d and d.has_edition)

    def by_shelfmarks(self, shelfmarks: Iterable[str]) -> Set[str]:
        """pgpids of builder documents sharing a shelfmark key with any of ``shelfmarks``.

        :param shelfmarks: Shelfmark strings (joins split).
        :returns: pgpids.
        """
        out: Set[str] = set()
        for s in shelfmarks:
            for k in _sm.keys(s):
                out |= self.by_key.get(k, set())
        return out


def document_text(user: str) -> str:
    """The text between ``<document>`` tags of a user turn.

    :param user: User message.
    :returns: Document text ('' when absent).
    """
    i, j = user.find("<document>\n"), user.find("\n</document>")
    if i < 0 or j < 0:
        return ""
    return user[i + len("<document>\n"):j]


def script_of_prompt(user: str) -> Optional[str]:
    """Script named by the instruction of a PGP-family prompt (``(Hebrew script)`` → ``hebrew``).

    :param user: User message.
    :returns: ``hebrew`` / ``arabic`` / ``mixed`` or None.
    """
    m = _SCRIPT_RE.search(user[:400])
    return _SCRIPT_FROM_NAME.get(m.group(1)) if m else None


def load_doc_index(families_dir: Path, want: Optional[Set[str]] = None,
                   conditions: Sequence[str] = DEFAULT_CONDITIONS,
                   families: Sequence[str] = DOC_FAMILIES) -> DocIndex:
    """Index the documents of the v0_2 builder output.

    Every document gets its split and identity fields; documents in ``want`` (all when None) also
    get their full edition text (an edition-condition row without truncation or side split, by
    family preference), their input variants (from the first family in ``families`` that has rows
    for them, ``conditions`` only) and their catalogue record (the ``catalogue_json`` target).

    :param families_dir: Directory with ``<family>.jsonl``.
    :param want: pgpids whose texts are kept (None = all).
    :param conditions: Conditions kept as variants.
    :param families: Family files read, in preference order.
    :returns: The index.
    """
    idx = DocIndex()
    rank = {f: i for i, f in enumerate(families)}
    var_family: Dict[str, str] = {}
    ed_rank: Dict[str, int] = {}
    for fam in families:
        path = Path(families_dir) / f"{fam}.jsonl"
        if not path.exists():
            LOG.warning("doc index: %s missing", path.name)
            continue
        for r in _rows.read_jsonl(path):
            m = r["meta"]
            pid = str(m.get("pgpid") or "")
            if not pid:
                continue
            d = idx.by_pgpid.get(pid)
            if d is None:
                d = DocInfo(pgpid=pid, split=m["split"], group=m.get("shelfmark_group"), lang_class=m.get("lang_class") or "OTHER",
                            shelfmark=m.get("shelfmark"), held_out_vision_benchmark=bool(m.get("held_out_vision_benchmark")),
                            holdout_reasons=list(m.get("holdout_reasons") or []))
                idx.by_pgpid[pid] = d
                for k in _sm.keys(d.shelfmark):
                    idx.by_key[k].add(pid)
            d.families.add(fam)
            cond = m.get("condition")
            if cond == "edition" and not m.get("side"):
                d.edition_row = True
            if want is not None and pid not in want:
                continue
            user, target = r["messages"][1]["content"], r["messages"][2]["content"]
            if d.script is None:
                d.script = script_of_prompt(user)
            if cond == "edition" and not m.get("side"):
                trunc = bool(m.get("input_truncated"))
                score = rank[fam] + (100 if trunc else 0)
                if pid not in ed_rank or score < ed_rank[pid]:
                    d.edition, d.edition_truncated, ed_rank[pid] = document_text(user), trunc, score
            if fam == "catalogue_json" and d.catalogue is None:
                try:
                    d.catalogue = json.loads(target)
                except json.JSONDecodeError:
                    d.catalogue = None
            if cond in conditions and not m.get("side") and var_family.setdefault(pid, fam) == fam:
                d.variants.append(Variant(cond, document_text(user), {k: m.get(k) for k in VARIANT_META if k in m}))
    return idx


# ============================================================================== span sections
def _uncapped_detector() -> Callable[..., List[Any]]:
    """``rag_l3.detect_spans`` with the 1,500-character section-sample cap lifted (private namespace).

    :returns: A ``detect_spans(pages, corpus, config)`` that keeps every section line.
    :raises RuntimeError: When ``rag_l3.classify_lines`` no longer has the two cap checks.
    """
    src = inspect.getsource(_rag.classify_lines)
    if src.count(_SAMPLE_CAP) != 2:
        raise RuntimeError("rag_l3.classify_lines changed: update the section-cap patch in analyse_l2")
    ns: Dict[str, Any] = dict(_rag.__dict__)
    flags = __future__.annotations.compiler_flag
    exec(compile(src.replace(_SAMPLE_CAP, "< 10**12"), "classify_lines_uncapped", "exec", flags=flags,  # noqa: S102
                 dont_inherit=True), ns)
    exec(compile(inspect.getsource(_rag.detect_spans), "detect_spans_uncapped", "exec", flags=flags,  # noqa: S102
                 dont_inherit=True), ns)
    return ns["detect_spans"]


def detect_full_spans(pages: Sequence[Any], corpus: Any, config: Any) -> List[Any]:
    """Re-run the span detector of :mod:`rag_l3` with full section lines.

    :param pages: Literature pages of the span works (:func:`pages.iter_pages`).
    :param corpus: :class:`shelfmarks.Corpus`.
    :param config: :class:`rag_l3.SpanConfig`.
    :returns: Spans (``rag_l3.Span``) with ``samples`` holding every line per section.
    """
    return _uncapped_detector()(pages, corpus, config)


def attach_sections(span_rows: Sequence[Dict[str, Any]], spans: Sequence[Any]) -> List[Dict[str, Any]]:
    """Join the full section lines to the ``rag_spans.jsonl`` rows (same detection order).

    :param span_rows: Rows of ``rag_spans.jsonl``.
    :param spans: Output of :func:`detect_full_spans`.
    :returns: Copies of the rows with ``sections`` = ``{intro|edition|translation|notes: [lines]}``.
    :raises ValueError: When the two disagree (rebuild ``rag_spans.jsonl`` with ``rag_l3 build_index``).
    """
    if len(span_rows) != len(spans):
        raise ValueError(f"rag_spans.jsonl has {len(span_rows)} spans, the detector {len(spans)}: rebuild the index")
    out = []
    for r, sp in zip(span_rows, spans):
        if (sp.book, sp.part, sp.label) != (r["book"], r["part"], r["doc_label"]):
            raise ValueError(f"span mismatch at {r['book']}/{r['doc_label']}: rebuild the index")
        full = {k: sum(len(x) for x in v) for k, v in sp.samples.items()}
        for k in ("intro", "notes", "translation", "edition"):
            if full.get(k, 0) != r.get(f"{k}_chars", 0):
                raise ValueError(f"section {k} of {r['book']}/{r['doc_label']} differs: rebuild the index")
        out.append({**r, "sections": {k: list(v) for k, v in sp.samples.items()}})
    return out


# ============================================================================== candidates
@dataclass
class Candidate:
    """One commentary target with its document, before the gates."""

    kind: str
    source: str
    work: str
    part: str
    label: str
    title: str
    pgpids: List[str]
    merged_ids: List[str]
    shelfmarks: List[str]
    intro: Union[List[str], str]
    notes: Union[List[str], str]
    edition_text: Optional[str] = None
    translation_text: str = ""
    script: Optional[str] = None
    lang_hint: Optional[str] = None
    page_keys: List[str] = field(default_factory=list)
    printed_pages: List[Any] = field(default_factory=list)
    hints: Dict[str, str] = field(default_factory=dict)
    licence: Optional[str] = None
    licence_class: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    edition_pgpid: Optional[str] = None


_LINE_NUM_LEAD = re.compile(r"^\s*[:.]?\s*\(?\d{1,3}[a-z]?\)?[.:]?\s+")
_LINE_NUM_TRAIL = re.compile(r"\s+\(?\d{1,3}\)?\s*$")
_RTL_LETTER = re.compile(r"[֐-׿؀-ۿ]")


def clean_book_edition(lines: Sequence[str]) -> str:
    """A book's printed edition lines as a plain input text (line numbers and Latin lines removed).

    :param lines: The span's ``edition`` lines.
    :returns: The text ('' when no Hebrew / Arabic line is left).
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or not _RTL_LETTER.search(s):
            continue
        s = _LINE_NUM_TRAIL.sub("", _LINE_NUM_LEAD.sub("", s))
        s = re.sub(r"^[:\s]+", "", s)
        s = re.sub(r"[ \t]{2,}", " ", s).strip()
        if s:
            out.append(s)
    text, _ = _tn.strip_latin_lines("\n".join(out))
    return text.strip()


def _lang_of_text(text: str, script: Optional[str]) -> str:
    """Coarse language class of an edition text without PGP metadata.

    :param text: Edition text.
    :param script: Its script.
    :returns: ``AR`` / ``JA`` / ``HE`` / ``OTHER``.
    """
    if script == "arabic":
        return "AR"
    if script == "hebrew":
        return "HE" if _pg.ja_or_he(text) == "he" else "JA"
    return "OTHER"


def _work_licence(work: str, local_licences: Mapping[str, str], manifest: Mapping[str, Dict[str, Any]]
                  ) -> Tuple[Optional[str], Optional[str]]:
    """Licence string and class of a literature work (local licence map, then the literature manifest).

    :param work: Work directory.
    :param local_licences: ``licence_by_book`` of the local builder config.
    :param manifest: Literature manifest ``{work_dir: entry}``.
    :returns: ``(licence, licence_class)``.
    """
    e = manifest.get(work) or {}
    lic = local_licences.get(work) or e.get("licence") or "unknown"
    cls = e.get("licence_class") if e.get("licence_class") in _rows.LICENCE_CLASSES else None
    return lic, cls or _licence_class(lic)


def _licence_class(licence: Optional[str]) -> str:
    """Licence class of a licence statement (``literature_l1.licence_class``).

    :param licence: Licence text.
    :returns: A class of :data:`rows.LICENCE_CLASSES`.
    """
    from src.finetuning.text27b import literature_l1 as _l1  # sibling module (heavy imports kept lazy)

    return _l1.licence_class(licence)


def span_candidates(span_rows: Sequence[Dict[str, Any]], docs: DocIndex, cfg: L2Config,
                    local_licences: Optional[Mapping[str, str]] = None,
                    manifest: Optional[Mapping[str, Dict[str, Any]]] = None) -> Tuple[List[Candidate], collections.Counter]:
    """Commentary candidates from the document-treatment spans (with full ``sections``).

    :param span_rows: Output of :func:`attach_sections`.
    :param docs: Builder documents.
    :param cfg: Build configuration (``min_commentary_chars``, ``excluded_books``, ``min_book_edition_chars``).
    :param local_licences: Licence strings per work.
    :param manifest: Literature manifest.
    :returns: ``(candidates, drop counter)``.
    """
    drops: collections.Counter = collections.Counter()
    out: List[Candidate] = []
    local_licences, manifest = local_licences or {}, manifest or {}
    for sp in span_rows:
        book = sp["book"]
        if not sp.get("boundary_clean"):
            drops["span_boundary_unclean"] += 1
            continue
        if sp.get("intro_chars", 0) + sp.get("notes_chars", 0) < cfg.min_commentary_chars:
            drops["span_commentary_short"] += 1
            continue
        if book in cfg.excluded_books:
            drops["span_excluded_book"] += 1
            continue
        secs = sp.get("sections") or {}
        pgpids = [str(x) for x in sp.get("pgpids") or []]
        with_ed = [x for x in pgpids if docs.has_edition(x)]
        if len(with_ed) > 1:
            drops["span_several_pgpids_with_edition"] += 1
            continue
        lic, lic_cls = _work_licence(book, local_licences, manifest)
        base = dict(source=f"book:{book}", work=book, part=sp["part"], label=str(sp["doc_label"]),
                    title=str(sp.get("title") or ""), pgpids=pgpids, merged_ids=list(sp.get("merged_doc_ids") or []),
                    shelfmarks=list(sp.get("shelfmarks") or []), intro=list(secs.get("intro", [])),
                    notes=list(secs.get("notes", [])), page_keys=list(sp.get("page_keys") or []),
                    printed_pages=list(sp.get("printed_pages") or []), licence=lic, licence_class=lic_cls,
                    translation_text=_bv.clean_commentary(list(secs.get("translation", [])))[0],
                    provenance={"book": book, "part": sp["part"], "doc_label": sp["doc_label"], "pages": sp.get("pages"),
                                "printed_pages": sp.get("printed_pages"), "join_basis": sp.get("join_basis")})
        if with_ed:
            d = docs.get(with_ed[0])
            out.append(Candidate(kind=SPAN_PGP, script=d.script if d else None, edition_pgpid=with_ed[0], **base))
            continue
        ed = clean_book_edition(secs.get("edition", []))
        script = _tn.dominant_script(ed) if ed else "none"
        if len(ed) < cfg.min_book_edition_chars:
            drops["span_no_input"] += 1
            continue
        if script not in SUPPORTED_SCRIPTS:
            drops["span_book_edition_script"] += 1
            continue
        out.append(Candidate(kind=SPAN_BOOK, edition_text=ed, script=script, lang_hint=_lang_of_text(ed, script), **base))
    return out, drops


# ------------------------------------------------------------------------------ book-edition entries
def load_nas_books(root: Path, books: Optional[Sequence[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Read every ``<root>/<slug>/entries.jsonl`` (read-only).

    :param root: Book-editions root (``paths.book_editions_nas``).
    :param books: Restrict to these slugs.
    :returns: ``{slug: entries}`` (slugs without ``entries.jsonl`` are skipped).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    root = Path(root)
    if not root.exists():
        LOG.warning("book editions root %s not found", root)
        return out
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if books is not None and d.name not in books:
            continue
        f = d / "entries.jsonl"
        if f.exists():
            out[d.name] = _rows.load_jsonl(f)
    return out


def _join_fields(entry: Mapping[str, Any], names: Sequence[str]) -> str:
    """Non-empty string fields of an entry joined by blank lines.

    :param entry: Entry.
    :param names: Field names in order.
    :returns: Text.
    """
    return "\n\n".join(str(entry[n]).strip() for n in names if isinstance(entry.get(n), str) and str(entry[n]).strip())


def nas_commentary(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Introduction and notes of a book-edition entry.

    :param entry: Entry of ``entries.jsonl``.
    :returns: ``(intro, notes)``.
    """
    return _join_fields(entry, NAS_INTRO_FIELDS), _join_fields(entry, NAS_NOTES_FIELDS)


def render_transcription(lines: Sequence[Mapping[str, Any]]) -> str:
    """An entry's transcription lines as an edition text with ``§`` side / part labels (PGP edition layout).

    A label line is written whenever the (side, part, block) of the lines changes and the entry has
    more than one such group or a named part.

    :param lines: ``transcription_lines`` (``{"n", "side", "part", "block", "text"}``).
    :returns: The text.
    """
    groups: List[Tuple[Tuple[str, str, str], List[str]]] = []
    for ln in lines:
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        g = tuple(str(ln.get(k) or "").strip() for k in ("side", "part", "block"))
        if not groups or groups[-1][0] != g:
            groups.append((g, []))  # type: ignore[arg-type]
        groups[-1][1].append(text)
    labelled = len(groups) > 1 or any(g[1] or g[2] for g, _ in groups)
    out: List[str] = []
    for (side, part, block), texts in groups:
        label = ", ".join(x for x in (side.capitalize() if side and side not in ("single", "both") else "", part, block) if x)
        if labelled and label:
            out.append(f"§ {label}")
        out.extend(texts)
    return "\n".join(out)


def pgp_suffix_index(corpus: Any) -> Dict[str, Set[str]]:
    """PGP shelfmarks by their last whitespace token (for entries that print only an inventory number).

    :param corpus: :class:`shelfmarks.Corpus`.
    :returns: ``{last token: pgpids}``.
    """
    idx: Dict[str, Set[str]] = collections.defaultdict(set)
    for pid, d in (getattr(corpus, "docs", None) or {}).items():
        sm = str(d.get("shelfmark") or "").strip()
        if " " in sm:
            idx[sm.rsplit(" ", 1)[1]].add(str(pid))
    return idx


def pgp_matches(entry: Mapping[str, Any], corpus: Any, suffix: Mapping[str, Set[str]]) -> Tuple[List[str], List[str]]:
    """PGP documents and merged ids of a book-edition entry.

    Sources: the entry's own ``pgp.pgpids`` cross-check, :func:`shelfmarks.match_shelfmark` on each
    shelfmark, and an inventory-number suffix match (a space-free number with a dot, ≥ 5 characters,
    equal to the last token of a PGP shelfmark).

    :param entry: Entry.
    :param corpus: Corpus (None → only the entry's own cross-check).
    :param suffix: :func:`pgp_suffix_index`.
    :returns: ``(pgpids, merged ids)``.
    """
    pg: Set[str] = set()
    cids: Set[str] = set()
    own = entry.get("pgp")
    if isinstance(own, Mapping):
        pg |= {str(x) for x in own.get("pgpids") or []}
    sms = [str(s) for s in (entry.get("shelfmarks_pgp") or entry.get("shelfmarks") or [entry.get("shelfmark")]) if s]
    if corpus is not None:
        for s in sms:
            m = _sm.match_shelfmark(s, corpus)
            pg |= {str(x) for x in m["pgpids"]}
            cids |= set(m["cids"])
    for s in sms:
        t = s.strip()
        if " " not in t and "." in t and len(t) >= 5 and re.search(r"\d", t):
            pg |= set(suffix.get(t, set()))
    if corpus is not None:
        for p in pg:
            cids |= set(getattr(corpus, "pgp_to_cids", {}).get(p, set()))
    return sorted(pg, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)), sorted(cids)


def nas_candidates(books: Mapping[str, List[Dict[str, Any]]], corpus: Any, cfg: L2Config,
                   entry_pages: Any = None) -> Tuple[List[Candidate], collections.Counter]:
    """Commentary candidates from the book-edition entries.

    With ``entry_pages`` (:class:`book_pairs.EntryPages`, fix pass F11) every candidate carries the literature page
    keys of its entry's PDF range (``page_keys``; provenance ``page_keys_source`` / ``entry_pages`` and, when the
    entry cannot be mapped, ``page_keys_unmapped``), so the page rule applies to it like any other row.

    :param books: :func:`load_nas_books`.
    :param corpus: Corpus for the PGP match (None allowed in tests).
    :param cfg: Build configuration (``skip_pgp_overlap``, ``min_commentary_chars``, ``min_book_edition_chars``).
    :param entry_pages: Entry → page-key mapper (None: candidates carry no page keys).
    :returns: ``(candidates, drop counter)``.
    """
    drops: collections.Counter = collections.Counter()
    out: List[Candidate] = []
    suffix = pgp_suffix_index(corpus) if corpus is not None else {}
    for slug, entries in sorted(books.items()):
        if entry_pages is not None:
            entry_pages.add_book(slug, entries)
        for e in entries:
            intro, notes = nas_commentary(e)
            if len(intro) + len(notes) < cfg.min_commentary_chars:
                drops["nas_commentary_short"] += 1
                continue
            ed = render_transcription(e.get("transcription_lines") or [])
            if len(ed) < cfg.min_book_edition_chars:
                drops["nas_no_transcription"] += 1
                continue
            script = str(e.get("script") or "")
            if script not in SUPPORTED_SCRIPTS:
                script = _tn.dominant_script(ed)
            if script not in SUPPORTED_SCRIPTS:
                drops["nas_script_unsupported"] += 1
                continue
            pgpids, cids = pgp_matches(e, corpus, suffix)
            if slug in cfg.skip_pgp_overlap and pgpids:
                drops["nas_pgp_overlap"] += 1
                continue
            langs = e.get("languages")
            lang = _bv.lang_from_pgp(" ".join(langs)) if isinstance(langs, list) and langs else _lang_of_text(ed, script)
            if lang in ("OTHER", "UNKNOWN"):
                lang = _lang_of_text(ed, script)
            lic = e.get("licence") if isinstance(e.get("licence"), str) else None
            hints = {"title": str(e.get("title") or ""), "date": str(e.get("date") or ""),
                     "description": str(e.get("description") or "")}
            sms = [str(s) for s in (e.get("shelfmarks") or [e.get("shelfmark")]) if s]
            pkeys: List[str] = []
            pprov: Dict[str, Any] = {}
            if entry_pages is not None:
                pkeys, why = entry_pages.keys(slug, e.get("entry_no"))
                pprov = {"page_keys_source": NAS_PAGE_SOURCE, "entry_pages": entry_pages.describe(slug, e.get("entry_no"))}
                if why:
                    pprov["page_keys_unmapped"] = why
            out.append(Candidate(
                kind=NAS, source=f"book:{slug}", work=slug, part="", label=str(e.get("entry_no")),
                title=hints["title"], pgpids=pgpids, merged_ids=cids, shelfmarks=sms, intro=intro, notes=notes,
                edition_text=ed, script=script, lang_hint=lang, hints=hints, licence=lic or "unknown",
                licence_class=_licence_class(lic), translation_text=_nas_translation(e),
                printed_pages=list(e.get("pages") or []), page_keys=pkeys,
                provenance={"book": slug, "entry_no": e.get("entry_no"), "pages": e.get("pages"),
                            "pdf_page": e.get("pdf_page"), **pprov}))
    return out, drops


def _nas_translation(entry: Mapping[str, Any]) -> str:
    """The entry's English translation as prose (lines or paragraphs).

    :param entry: Entry.
    :returns: Translation text ('' when absent).
    """
    parts = [str(x.get("text") or "").strip() for x in entry.get("translation_lines") or [] if isinstance(x, Mapping)]
    parts += [str(x.get("text") or "").strip() for x in entry.get("translation_paragraphs") or [] if isinstance(x, Mapping)]
    return "\n".join(p for p in parts if p)


# ============================================================================== splits (D14)
def doc_key_of(cand: Candidate, doc: Optional[DocInfo]) -> str:
    """Document identity used for the hash split and the per-document dedupe (shelfmark-keyed).

    :param cand: Candidate.
    :param doc: Matched builder document (its shelfmark wins), or None.
    :returns: A shelfmark key, else ``<source>#<label>``.
    """
    for sm in ([doc.shelfmark] if doc and doc.shelfmark else []) + list(cand.shelfmarks):
        k = _sm.key(sm)
        if k:
            return k
    return f"{cand.source}#{cand.label}"


def hash_split(doc_id: str, buckets: Sequence[Tuple[int, str]] = ((80, "train"), (90, "dev"), (100, "test_indist"))) -> str:
    """D14 split of a new document: ``sha1(doc_id)`` mod 100 into the buckets.

    :param doc_id: Document key.
    :param buckets: ``(upper bound, split)`` in ascending order.
    :returns: Split name.
    """
    v = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 100
    for bound, name in buckets:
        if v < bound:
            return name
    return buckets[-1][1]


@dataclass
class SplitDecision:
    """Where a candidate's rows go and why."""

    split: str
    reason: str
    doc: Optional[DocInfo]
    held_out_vision_benchmark: bool
    holdout_reasons: List[str]


def assign_split(cand: Candidate, docs: DocIndex, holdouts: Any, doc_key: str,
                 buckets: Sequence[Tuple[int, str]] = ((80, "train"), (90, "dev"), (100, "test_indist"))) -> SplitDecision:
    """D14 split of a candidate's document.

    Order: benchmark tier of H (any pgpid / merged id / shelfmark) → ``heldout``; a builder document
    (edition pgpid first, then any pgpid) keeps its split; an eval-tier match → ``heldout``; builder
    documents sharing a shelfmark key → ``train`` when all are train, else ``heldout``; otherwise
    :func:`hash_split` of ``doc_key``.

    :param cand: Candidate.
    :param docs: Builder documents.
    :param holdouts: :class:`holdouts.Holdouts` (or any object with ``match_any``).
    :param doc_key: Document key.
    :param buckets: Hash buckets.
    :returns: The decision.
    """
    bench = holdouts.match_any(pgpids=cand.pgpids, canonicals=cand.merged_ids, shelfmarks=cand.shelfmarks,
                               include_eval=False) if holdouts is not None else None
    order = ([cand.edition_pgpid] if cand.edition_pgpid else []) + [p for p in cand.pgpids if p != cand.edition_pgpid]
    doc = next((docs.get(p) for p in order if docs.get(p) is not None), None)
    if bench:
        vision = _holdout_is_vision(bench) or bool(doc is not None and doc.held_out_vision_benchmark)
        reasons = sorted(set(doc.holdout_reasons if doc is not None else []) | {bench})
        return SplitDecision("heldout", f"benchmark:{bench}", doc, vision, reasons)
    if doc is not None:
        return SplitDecision(doc.split, "builder", doc, doc.held_out_vision_benchmark, list(doc.holdout_reasons))
    ev = holdouts.match_any(pgpids=cand.pgpids, canonicals=cand.merged_ids, shelfmarks=cand.shelfmarks,
                            include_eval=True) if holdouts is not None else None
    if ev:
        return SplitDecision("heldout", f"eval:{ev}", None, False, [ev])
    same = docs.by_shelfmarks(cand.shelfmarks)
    if same:
        splits = {docs.by_pgpid[p].split for p in same}
        first = docs.by_pgpid[sorted(same)[0]]
        if splits == {"train"}:
            return SplitDecision("train", "shelfmark_key:train", first, False, [])
        return SplitDecision("heldout", "shelfmark_key:" + ",".join(sorted(splits)), first, False, [])
    return SplitDecision(hash_split(doc_key, buckets), "hash", None, False, [])


def _holdout_is_vision(reason: str) -> bool:
    """Whether a hold-out reason marks a vision-benchmark document.

    :param reason: Reason (``pgp131``, ``key:registry`` ...).
    :returns: True for PGP-131 / registry / vision-flag reasons.
    """
    return reason.split(":")[-1] in _bld.BENCH_REASONS


# ============================================================================== targets
def _char_grams(text: str, n: int) -> Set[str]:
    """Character n-grams of a text (whitespace collapsed).

    :param text: Text.
    :param n: n.
    :returns: Set of n-grams.
    """
    t = re.sub(r"\s+", " ", text)
    return {t[i:i + n] for i in range(0, max(len(t) - n + 1, 0))}


def drop_copied_units(units: Sequence[str], source: str, n: int = 12, max_share: float = 0.3) -> Tuple[List[str], int]:
    """Drop units (lines / sentences) whose character n-grams mostly occur in the source (the v0 rule).

    :param units: Target units.
    :param source: Input text.
    :param n: n-gram length (12 in v0).
    :param max_share: A unit with at least this share of its n-grams in the source is dropped.
    :returns: ``(kept units, number dropped)``.
    """
    src = _char_grams(source, n)
    kept, dropped = [], 0
    for u in units:
        t = re.sub(r"\s+", " ", u)
        grams = [t[i:i + n] for i in range(0, max(len(t) - n + 1, 0))]
        if grams and sum(g in src for g in grams) / len(grams) >= max_share:
            dropped += 1
            continue
        kept.append(u)
    return kept, dropped


_COPYRIGHT = re.compile(r"(?i)\u00a9|\(c\)\s*(?:1[89]|20)\d{2}|\bCC[\s-]*BY\b|creative\s+commons|all rights reserved|"
                        r"https?://|www\.")
_GARBLE = re.compile(r"(?:^|\s)[:.;,!?][\u0590-\u05FF\u0600-\u06FF]|(?:^|\s):[A-Za-z]|(?:^|\s)[.:]\d{1,3}(?=\s|$)")
_LATIN_WORD3 = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]{3,}")
_RTL_WORD2 = re.compile(r"[\u05d0-\u05ea\u0621-\u064a]{2,}")
_LABEL_WORDS = frozenset({"recto", "verso", "line", "lines", "margin", "notes", "note", "side", "fol", "col", "column",
                          "right", "left", "top", "bottom", "upper", "lower"})
_GAP_MARK = re.compile(r"\[\s*(?:\.{2,}|…)?\s*\]|\]\s*[^\[\]\n]{0,40}\[")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
_DIM_PAIR = re.compile(r"(?<![\w.])(?:\d{1,3}[.,]\d{1,2}\s*[×xX]\s*\d{1,3}(?:[.,]\d{1,2})?|"
                       r"\d{1,3}(?:[.,]\d{1,2})?\s*[×xX]\s*\d{1,3}[.,]\d{1,2}|"
                       r"\d{1,3}\s*[×xX]\s*\d{1,3}\s*(?:cm|mm|;))(?![\w.])")
_PHYS_TERMS = re.compile(r"(?i)\b(?:paper|parchment|vellum|leaf|leaves|folios?|bifolium|holes?|stains?|stained|torn|rubbed|"
                         r"faded|wormholes?|mutilated|damaged|ink|\d+\s+lines?\s*\((?:recto|verso)\))\b|"
                         r"נייר|קלף|גיליון|דהוי|קרעים|קרוע|כתמים|חורים|מטושטש")


def furniture_line(line: str) -> bool:
    """Whether a commentary line is page furniture (page number, download stamp, copyright / licence / URL line).

    :param line: Line.
    :returns: True for furniture.
    """
    from src.finetuning.text27b import literature_l1 as _l1  # sibling module (heavy imports kept lazy)

    s = line.strip()
    return bool(s) and (_l1.is_page_number_line(s) or _l1.is_boilerplate_line(s) or
                        (len(s) < 200 and bool(_COPYRIGHT.search(s))))


def garbled(unit: str) -> bool:
    """Whether a unit shows visual-order (bidi) extraction: punctuation glued to the start of a word.

    :param unit: Line or sentence.
    :returns: True when garbled.
    """
    return bool(_GARBLE.search(unit))


def commentary_mode(lines: Sequence[str]) -> str:
    """Script of a commentary: ``latin`` when Latin letters outnumber Hebrew / Arabic ones, else ``rtl``.

    :param lines: Commentary lines.
    :returns: ``latin`` / ``rtl``.
    """
    h, a, lat = _pg.letter_counts("\n".join(lines))
    return "latin" if lat >= h + a else "rtl"


def edition_line(line: str, mode: str) -> bool:
    """Whether a commentary line is a piece of the edition (not commentary).

    Latin commentary: a line without Latin letters but with Hebrew / Arabic ones. Hebrew commentary: a
    line with two lacuna / bracket marks, an Arabic-script line, or a Judaeo-Arabic line.

    :param line: Line.
    :param mode: :func:`commentary_mode`.
    :returns: True for edition text.
    """
    s = line.strip()
    if not s:
        return False
    if len(_GAP_MARK.findall(s)) >= 2:
        return True
    if mode == "latin":
        return not _LATIN.search(s) and bool(_RTL_LETTER.search(s))
    h, a, _ = _pg.letter_counts(s)
    return a > h or (len(s) >= 15 and _pg.ja_or_he(s) == "ja")


def thin(paragraph: str, mode: str, min_words: int = 3) -> bool:
    """Whether a paragraph has too few content words to be commentary (bare references, orphan line labels).

    :param paragraph: Paragraph.
    :param mode: :func:`commentary_mode` (``latin`` counts Latin words of ≥ 3 letters other than side /
        line labels; ``rtl`` counts Hebrew / Arabic words of ≥ 2 letters).
    :param min_words: Fewest content words of a kept paragraph.
    :returns: True when thin.
    """
    if mode == "latin":
        n = sum(1 for w in _LATIN_WORD3.findall(paragraph) if w.lower() not in _LABEL_WORDS)
    else:
        n = len(_RTL_WORD2.findall(paragraph))
    return n < min_words


def drop_thin(text: str, mode: str, acts: collections.Counter) -> str:
    """Remove thin paragraphs (:func:`thin`) from a cleaned section.

    :param text: Section prose (paragraphs separated by blank lines).
    :param mode: Commentary mode.
    :param acts: Action counter (updated).
    :returns: The section without thin paragraphs.
    """
    keep = []
    for para in text.split("\n\n"):
        if para.strip() and thin(para, mode):
            acts["thin_paragraph"] += 1
            continue
        keep.append(para)
    return "\n\n".join(p for p in keep if p.strip())


def strip_physical(text: str) -> Tuple[str, List[str]]:
    """Drop physical-description sentences (measurements without units, material / condition lists).

    A sentence goes when it holds a measurement pair (``21.5 × 14.6``, ``7 × 13.6``, ``20 × 14 cm``) or
    three or more material / condition terms
    (paper, leaf, ``23 lines (recto)``, holes, stains, torn ... and their Hebrew equivalents).

    :param text: Target prose.
    :returns: ``(text, ops)`` with op ``L2:physical_description`` when anything was removed.
    """
    out_paras, hit = [], False
    for para in re.split(r"\n[ \t]*\n", text):
        kept = []
        for sent in _sentences(para):
            if _DIM_PAIR.search(sent) or len(_PHYS_TERMS.findall(sent)) >= 3:
                hit = True
                continue
            kept.append(sent)
        if kept:
            out_paras.append(" ".join(kept))
    if not hit:
        return text, []
    return "\n\n".join(out_paras).strip(), ["L2:physical_description"]


def _filter_lines(lines: Sequence[str], mode: str, acts: collections.Counter) -> List[str]:
    """Drop furniture, edition and garbled lines of a span section.

    :param lines: Section lines.
    :param mode: :func:`commentary_mode`.
    :param acts: Action counter (updated).
    :returns: Kept lines.
    """
    out = []
    for ln in lines:
        if furniture_line(ln):
            acts["furniture_line"] += 1
        elif edition_line(ln, mode):
            acts["edition_line_in_commentary"] += 1
        elif garbled(ln):
            acts["bidi_garbled_line"] += 1
        else:
            out.append(ln)
    return out


def span_target(intro_lines: Sequence[str], notes_lines: Sequence[str], source: str, cfg: L2Config
                ) -> Tuple[str, Dict[str, int], List[str]]:
    """Target prose of a span: furniture / edition / garbled / copied lines dropped, then the v0 commentary cleaning.

    :param intro_lines: ``intro`` section lines.
    :param notes_lines: ``notes`` section lines.
    :param source: Input text (copy test).
    :param cfg: Build configuration.
    :returns: ``(target before the v0_2 strips, action counts, kept source lines)`` -- the kept lines
        are what :func:`ocr_containment` checks.
    """
    acts: collections.Counter = collections.Counter()
    mode = commentary_mode(list(intro_lines) + list(notes_lines))
    i_kept, n1 = drop_copied_units(_filter_lines(intro_lines, mode, acts), source, cfg.copy_ngram, cfg.copy_max_share)
    n_kept, n2 = drop_copied_units(_filter_lines(notes_lines, mode, acts), source, cfg.copy_ngram, cfg.copy_max_share)
    acts["lines_copying_input"] += n1 + n2
    intro, c1 = _bv.clean_commentary(i_kept)
    notes, c2 = _bv.clean_commentary(n_kept)
    acts.update(c1 + c2)
    intro, notes = drop_thin(intro, mode, acts), drop_thin(notes, mode, acts)
    return (intro + (NOTES_HEADER + notes if notes else "")).strip(), dict(acts), list(i_kept) + list(n_kept)


def _sentences(text: str) -> List[str]:
    """Sentences of one paragraph (:func:`textnorm.sentence_spans`).

    :param text: Paragraph.
    :returns: Sentence strings.
    """
    spans = _tn.sentence_spans(text)
    return [text[a:b].strip() for a, b in spans if text[a:b].strip()] or ([text.strip()] if text.strip() else [])


def nas_target(intro: str, notes: str, source: str, cfg: L2Config) -> Tuple[str, Dict[str, int]]:
    """Target prose of a book-edition entry: garbled, furniture and input-copying sentences dropped, paragraphs kept.

    :param intro: Introduction / discussion text.
    :param notes: Commentary / textual notes text.
    :param source: Input text (copy test).
    :param cfg: Build configuration.
    :returns: ``(target before the v0_2 strips, action counts)``.
    """
    dropped, n_garbled = 0, 0

    def clean(text: str) -> str:
        """Paragraph-wise copy filter.

        :param text: Text.
        :returns: Filtered text.
        """
        nonlocal dropped
        nonlocal n_garbled
        paras = []
        for para in re.split(r"\n\s*\n|\n", text):
            sents = _sentences(para)
            clean_s = [x for x in sents if not garbled(x) and not furniture_line(x)]
            n_garbled += len(sents) - len(clean_s)
            kept, n = drop_copied_units(clean_s, source, cfg.copy_ngram, cfg.copy_max_share)
            dropped += n
            if kept:
                paras.append(" ".join(kept))
        return "\n\n".join(paras)

    acts: collections.Counter = collections.Counter()
    mode = commentary_mode([intro, notes])
    a, b = drop_thin(clean(intro), mode, acts), drop_thin(clean(notes), mode, acts)
    acts.update({"sentences_copying_input": dropped, "bidi_garbled_sentence": n_garbled})
    return (a + (NOTES_HEADER + b if b else "")).strip(), dict(acts)


def ocr_containment(units: Sequence[str], ocr_texts: Sequence[Optional[str]], n: int = 5) -> Optional[float]:
    """Share of the target lines' ``n``-word shingles found in the OCR text of their pages.

    Shingles are taken within each unit (a structured-text line), so dropping a line never creates
    shingles across the gap; the reference is the OCR text as is plus its hyphen-joined form.

    :param units: Kept source lines of the target (before the commentary cleaning joins them).
    :param ocr_texts: OCR texts of the span pages (None for a page without OCR).
    :param n: Shingle length.
    :returns: Share, or None when no page has OCR text or no unit has ``n`` words.
    """
    texts = [t for t in ocr_texts if t]
    if not texts:
        return None
    raw = "\n".join(texts)
    ref = _pg.shingles(raw, n) | _pg.shingles(re.sub(r"-[ \t]*\n[ \t]*", "", raw), n)
    mine: Set[str] = set()
    for u in units:
        mine |= _pg.shingles(u, n)
    if not mine:
        return None
    return len(mine & ref) / len(mine)


def held_mentions(text: str, holdouts: Any, own: Iterable[str]) -> List[Tuple[str, str]]:
    """Shelfmarks in a text that resolve to a held document (benchmark or eval tier) other than its own.

    :param text: Text (a target before the strips).
    :param holdouts: Hold-out set (``match_any``).
    :param own: The document's own shelfmarks (their keys are ignored).
    :returns: ``[(shelfmark, reason)]``.
    """
    if holdouts is None:
        return []
    own_keys = {k for s in own for k in _sm.keys(s)}
    out = []
    for a, b in _tn.find_shelfmarks(text):
        s = text[a:b]
        if own_keys and set(_sm.keys(s)) & own_keys:
            continue
        r = holdouts.match_any(shelfmarks=[s], include_eval=True)
        if r:
            out.append((s, r))
    return out


# ============================================================================== context
def catalogue_from_corpus(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A context record from a PGP ``documents.csv`` row (type, date, places, stripped description).

    :param row: ``Corpus.docs`` entry.
    :returns: A record in the :func:`builder_v0.context_block` shape, or None without a usable description.
    """
    raw, _ = _bv.clean_description(str(row.get("description") or ""))
    summary, _ = _bld.clean_target(raw, "describe_pgp")
    if len(summary) < 40:
        return None
    std = str(row.get("doc_date_standard") or "").strip()
    return {"document_type": row.get("type") or None, "date": {"original": None, "standard": std} if std else None,
            "origin": row.get("origin") or None, "destination": row.get("destination") or None,
            "places_mentioned": _bv.split_multi(str(row.get("mentioned") or "")), "summary": summary}


def _strip_shelfmarks(text: str) -> str:
    """Delete shelfmark mentions from a short text (titles in context blocks).

    :param text: Text.
    :returns: Text without shelfmarks.
    """
    spans = _tn.find_shelfmarks(text)
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + text[b:]
    return re.sub(r"^[\s:;,.\-–]+", "", re.sub(r"\s{2,}", " ", text)).strip()


def hint_context(hints: Mapping[str, str]) -> Dict[str, Any]:
    """A context record from an edition's own heading (title, date, catalogue description), strips applied.

    :param hints: ``{"title", "date", "description"}``.
    :returns: Record with ``title``, ``date``, ``summary`` (None when empty).
    """
    title = _strip_shelfmarks(str(hints.get("title") or ""))
    desc, _ = _bld.clean_target(str(hints.get("description") or ""), "describe_pgp")
    desc, _ = _bld.strip_analysis_extras(desc)
    desc, _ = strip_physical(desc)
    return {"title": title or None, "date": str(hints.get("date") or "").strip() or None, "summary": desc.strip() or None}


def render_hint_context(rec: Mapping[str, Any]) -> str:
    """Context block text of :func:`hint_context`.

    :param rec: Record.
    :returns: Block ('' when empty).
    """
    lines = [f"Title: {rec['title']}" if rec.get("title") else "", f"Date: {rec['date']}" if rec.get("date") else "",
             f"Catalogue description: {rec['summary']}" if rec.get("summary") else ""]
    body = "\n".join(x for x in lines if x)
    return (CONTEXT_HEADER + body) if body else ""


def context_for(target: str, catalogue: Optional[Dict[str, Any]], hints: Optional[Mapping[str, str]] = None,
                leak_fn: Optional[Callable[[str], bool]] = None) -> Tuple[str, Optional[str]]:
    """The 'Known catalogue context' block of a row.

    The description is withheld when the target copies it (v0) or when it leaks an eval reference
    (``leak_fn``); a context that still leaks without its description is dropped.

    :param target: Stripped target.
    :param catalogue: PGP catalogue record (builder) or None.
    :param hints: Edition heading fields used when there is no catalogue record.
    :param leak_fn: Eval-reference n-gram test (None: skipped).
    :returns: ``(block text, note)`` where note ∈ {None, ``hint``, ``none``, ``description_withheld_overlap``,
        ``description_withheld_ngram``, ``context_dropped_ngram``}.
    """
    if catalogue:
        full, bare = CONTEXT_HEADER + _bv.context_block(catalogue), CONTEXT_HEADER + _bv.context_block({**catalogue, "summary": None})
        note: Optional[str] = None
    elif hints:
        rec = hint_context(hints)
        full, bare = render_hint_context(rec), render_hint_context({**rec, "summary": None})
        note = "hint" if full else "none"
    else:
        return "", "none"
    ctx = full
    if ctx and _tn.overlap60(target, ctx):
        ctx, note = bare, "description_withheld_overlap"
    if ctx and leak_fn is not None and leak_fn(ctx):
        ctx, note = bare, "description_withheld_ngram"
        if ctx and leak_fn(ctx):
            ctx, note = "", "context_dropped_ngram"
    if ctx.strip() == CONTEXT_HEADER.strip():
        ctx = ""
    return ctx, note


# ============================================================================== literature block (tier C)
_BM25_WORD = re.compile(r"[a-z֐-׿؀-ۿ]{3,}")
_BM25_STOP: Set[str] = set(_pg.EN_SW) | {
    "are", "were", "has", "have", "had", "not", "but", "his", "her", "him", "they", "them", "their", "there", "who", "whom",
    "what", "when", "where", "also", "been", "into", "one", "two", "its", "may", "would", "could", "should", "than", "then",
    "this", "these", "those", "such", "some", "any", "all", "other", "more", "most", "only", "very", "upon", "after",
    "before", "about", "document", "documents", "letter", "text", "line", "lines", "page", "known", "context", "catalogue",
    "description", "type", "date", "origin", "destination", "places", "mentioned", "recto", "verso", "margin"}


def bm25_tokens(text: str) -> List[str]:
    """BM25 terms: folded words of ≥ 3 letters (Latin, Hebrew, Arabic) without common English words.

    :param text: Text.
    :returns: Terms in order.
    """
    return [w for w in _BM25_WORD.findall(_tn.strip_marks(text or "")) if w not in _BM25_STOP]


class BM25:
    """A small Okapi BM25 index (``k1`` = 1.5, ``b`` = 0.75)."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Index tokenised documents.

        :param docs: Term lists.
        :param k1: Term-frequency saturation.
        :param b: Length normalisation.
        """
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl and sum(self.dl) else 1.0
        self.df: collections.Counter = collections.Counter()
        for t in self.tf:
            self.df.update(t.keys())
        self.n = len(self.tf)

    def idf(self, term: str) -> float:
        """Inverse document frequency (BM25+ style, always positive).

        :param term: Term.
        :returns: idf.
        """
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: Sequence[str]) -> List[float]:
        """Scores of every document for a query (each distinct query term counted once).

        :param query: Query terms.
        :returns: Scores in document order.
        """
        terms = [t for t in set(query) if t in self.df]
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


@dataclass
class LitPool:
    """The pages of one work a literature block may draw on (eligible pages only)."""

    work: str
    keys: List[str]
    texts: List[str]
    cites: List[str]
    loc: Dict[str, Tuple[str, int]]
    index: BM25


def _neighbour_keys(keys: Iterable[str], loc_all: Mapping[str, Tuple[str, int]],
                    by_loc: Mapping[Tuple[str, int], str], window: int) -> Set[str]:
    """Page keys within ``window`` pages (same work and part) of the given keys.

    :param keys: Page keys.
    :param loc_all: page key → (work/part, page number).
    :param by_loc: (work/part, page number) → page key.
    :param window: Window size.
    :returns: Keys (the given ones included).
    """
    out: Set[str] = set()
    for k in keys:
        if k not in loc_all:
            out.add(k)
            continue
        part, n = loc_all[k]
        for d in range(-window, window + 1):
            kk = by_loc.get((part, n + d))
            if kk:
                out.add(kk)
    return out


def build_pools(pages: Sequence[Any], works: Iterable[str], span_keys: Mapping[str, Set[str]],
                drop: Mapping[str, Sequence[str]], cfg: L2Config, manifest: Optional[Mapping[str, Dict[str, Any]]] = None,
                quality_fn: Optional[Callable[[Any], bool]] = None) -> Tuple[Dict[str, LitPool], collections.Counter]:
    """Eligible literature pages per work, indexed for BM25.

    A page is eligible when it is not a span page of the work (``strict_non_span``), carries no drop
    reason, passes the quality filter and has at least ``min_page_chars`` characters of display text.

    :param pages: Literature pages (:class:`pages.PageRec`).
    :param works: Works to pool.
    :param span_keys: work → page keys inside any document span of the work.
    :param drop: page key → reasons (excluded pages, non-train mentions ±1).
    :param cfg: Build configuration.
    :param manifest: Literature manifest (citation stubs).
    :param quality_fn: Page → bool (None: no quality filter).
    :returns: ``({work: pool}, page counts by reason)``.
    """
    works = set(works)
    stats: collections.Counter = collections.Counter()
    acc: Dict[str, Dict[str, list]] = {}
    for p in pages:
        if p.work not in works:
            continue
        k = p.page_key
        if cfg.strict_non_span and k in span_keys.get(p.work, set()):
            stats["span_page"] += 1
            continue
        if drop.get(k):
            stats["dropped:" + drop[k][0].split(":")[0]] += 1
            continue
        text = _rag._page_display_text(p)
        if len(text.strip()) < cfg.min_page_chars:
            stats["short_page"] += 1
            continue
        if quality_fn is not None and not quality_fn(p):
            stats["quality"] += 1
            continue
        a = acc.setdefault(p.work, {"keys": [], "texts": [], "cites": []})
        author, title, year = _rag.work_meta(dict(manifest or {}), p.work, p.part)
        a["keys"].append(k)
        a["texts"].append(text)
        a["cites"].append(_rag.citation(author, title, year, p.printed_page_num or p.printed_page))
        stats["eligible"] += 1
    loc = {p.page_key: (f"{p.work}/{p.part}", p.page_no) for p in pages if p.work in works}
    pools = {w: LitPool(work=w, keys=a["keys"], texts=a["texts"], cites=a["cites"], loc=loc,
                        index=BM25([bm25_tokens(t) for t in a["texts"]])) for w, a in acc.items()}
    return pools, stats


def select_block(pool: LitPool, query: str, exclude: Set[str], target: str, cfg: L2Config,
                 leak_fn: Optional[Callable[[str], bool]] = None) -> Tuple[List[Tuple[str, str, List[str]]],
                                                                            List[Dict[str, Any]], collections.Counter]:
    """Choose the literature block of a row: top BM25 pages for the input, leak-checked against the target.

    :param pool: The work's eligible pages.
    :param query: Input text (document + catalogue context; never the target).
    :param exclude: Page keys not allowed (the target span ±1).
    :param target: Target (for the 5 % block rule and the 60-character copy rule only).
    :param cfg: Build configuration (``k_max``, ``block_max_share``).
    :param leak_fn: Eval-reference leak test for page texts (None: skipped).
    :returns: ``(sources for rag_l3.render_block, lit_sources meta, counters)``.
    """
    from src.finetuning.text27b import contamination as _ct  # sibling module (numpy hashing)

    stats: collections.Counter = collections.Counter()
    scores = pool.index.scores(bm25_tokens(query))
    order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], pool.keys[i]))
    chosen: List[int] = []
    for i in order:
        if len(chosen) >= cfg.k_max:
            break
        k, text = pool.keys[i], pool.texts[i]
        if k in exclude:
            stats["block_page_in_window"] += 1
            continue
        if _tn.overlap60(target, text) or _ct.target_share(target, text) >= cfg.block_max_share:
            stats["block_page_target_overlap"] += 1
            continue
        if leak_fn is not None and leak_fn(text):
            stats["block_page_ngram_leak"] += 1
            continue
        chosen.append(i)
    while chosen and _ct.target_share(target, [pool.texts[i] for i in chosen]) >= cfg.block_max_share:
        chosen.pop()
        stats["block_union_trimmed"] += 1
    sources = [(pool.cites[i], pool.texts[i], []) for i in chosen]
    meta = [{"work": pool.work, "page_key": pool.keys[i], "page": pool.cites[i].rsplit("p. ", 1)[-1]
             if "p. " in pool.cites[i] else None, "bm25": round(scores[i], 2)} for i in chosen]
    return sources, meta, stats


# ============================================================================== tiers
@dataclass
class TierResult:
    """Outcome of the grounding tiers for one target."""

    tier: Optional[str]
    share_a: float
    candidates: int
    missing_a: List[str]
    share_b: Optional[float] = None
    share_c: Optional[float] = None
    extra: str = ""
    lit_sources: List[Dict[str, Any]] = field(default_factory=list)
    block_chars: List[int] = field(default_factory=list)
    stats: collections.Counter = field(default_factory=collections.Counter)

    @property
    def share(self) -> float:
        """Grounding share of the tier that passed (else the best share measured).

        :returns: Share.
        """
        if self.tier == TIER_C and self.share_c is not None:
            return self.share_c
        if self.tier == TIER_B and self.share_b is not None:
            return self.share_b
        if self.tier == TIER_A:
            return self.share_a
        return max(x for x in (self.share_a, self.share_b, self.share_c) if x is not None)


def decide_tier(target: str, input_text: str, ctx: str, scholar_names: Set[str], cfg: L2Config,
                translation: str = "", block_fn: Optional[Callable[[], Tuple[List[Tuple[str, str, List[str]]],
                                                                            List[Dict[str, Any]],
                                                                            collections.Counter]]] = None) -> TierResult:
    """Grounding tiers A′ (input + context ≥ 0.5), B (+ the book's translation ≥ 0.6, opt-in), C (+ literature block ≥ 0.6).

    :param target: Stripped target.
    :param input_text: Document text (the edition input).
    :param ctx: Catalogue context block.
    :param scholar_names: Folded scholar surnames (not grounding facts).
    :param cfg: Build configuration (``tiers`` and thresholds).
    :param translation: The book's translation of the document (tier B).
    :param block_fn: Builds the literature block (tier C); None skips tier C.
    :returns: The result; ``tier`` None when every tier failed.
    """
    share_a, n_c, miss = _tn.grounding(target, input_text + "\n" + ctx, scholar_names)
    res = TierResult(None, share_a, n_c, miss, extra=ctx)
    if TIER_A in cfg.tiers and share_a >= cfg.tier_a_pass:
        res.tier = TIER_A
        return res
    if len(translation) >= cfg.min_book_translation_chars:
        share_b, _, _ = _tn.grounding(target, input_text + "\n" + ctx + "\n" + translation, scholar_names)
        res.share_b = share_b
        if TIER_B in cfg.tiers and share_b >= cfg.tier_b_pass and not _tn.overlap60(target, translation):
            res.tier = TIER_B
            res.extra = (ctx + "\n\n" if ctx else "") + TRANSLATION_HEADER + translation
            return res
    if TIER_C in cfg.tiers and block_fn is not None:
        sources, meta, stats = block_fn()
        res.stats.update(stats)
        if not sources:
            res.stats["tier_c_no_block"] += 1
            return res
        block, chars = _rag.render_block(sources)
        share_c, _, _ = _tn.grounding(target, input_text + "\n" + ctx + "\n" + block, scholar_names)
        res.share_c = share_c
        if share_c >= cfg.tier_c_pass:
            res.tier = TIER_C
            res.extra = (ctx + "\n\n" if ctx else "") + block
            res.lit_sources = [dict(m, chars=c) for m, c in zip(meta, chars)]
            res.block_chars = chars
    return res


# ============================================================================== rows
def _paragraphs(target: str) -> List[str]:
    """Target paragraphs (the ``Notes`` header kept with its first paragraph).

    :param target: Target.
    :returns: Paragraphs.
    """
    parts = target.split("\n\n")
    out: List[str] = []
    for p in parts:
        if out and out[-1] == "Notes":
            out[-1] = out[-1] + "\n\n" + p
        else:
            out.append(p)
    return out


def fit_row(user_fn: Callable[[str], str], input_text: str, target: str, tok: TokenFns, cap: int,
            min_target_chars: int) -> Optional[Tuple[str, str, int, bool, bool]]:
    """Fit a row under the token cap: drop trailing target paragraphs, then truncate the input lines.

    :param user_fn: Input text → user message.
    :param input_text: Document text.
    :param target: Target.
    :param tok: Token counters.
    :param cap: Rendered-token cap.
    :param min_target_chars: Shortest target allowed after truncation.
    :returns: ``(user, target, tokens, target truncated, input truncated)`` or None when it cannot fit.
    """
    def n_of(u: str, t: str) -> int:
        """Rendered tokens.

        :param u: User message.
        :param t: Target.
        :returns: Tokens.
        """
        return tok.render_count(_rows.make_messages(u, t))

    user = user_fn(input_text)
    n = n_of(user, target)
    if n <= cap:
        return user, target, n, False, False
    paras = _paragraphs(target)
    t_trunc = False
    while len(paras) > 1 and n > cap:
        cand = "\n\n".join(paras[:-1]).strip()
        if len(cand) < min_target_chars:
            break
        paras.pop()
        t_trunc = True
        target = cand
        n = n_of(user, target)
    if n <= cap:
        return user, target, n, t_trunc, False
    lines = input_text.split("\n")
    lo, hi = 0, len(lines)
    best = None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        u = user_fn("\n".join(lines[:mid]).rstrip() + "\n[...]")
        m = n_of(u, target)
        if m <= cap:
            lo, best = mid, (u, m)
        else:
            hi = mid - 1
    if best is None or lo < max(3, len(lines) // 4):
        return None
    return best[0], target, best[1], t_trunc, True


@dataclass
class L2Inputs:
    """Everything :func:`build_rows` reads (injected in tests)."""

    candidates: List[Candidate]
    docs: DocIndex
    holdouts: Any
    scholar_names: Set[str]
    pools: Dict[str, LitPool] = field(default_factory=dict)
    ocr_by_key: Dict[str, Optional[str]] = field(default_factory=dict)
    page_loc: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    corpus_docs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    leak_fn: Optional[Callable[[str], bool]] = None
    excluded: Dict[str, List[str]] = field(default_factory=dict)


def _meta(cand: Candidate, dec: SplitDecision, doc_key: str, variant: Variant, lang: str, tr: TierResult,
          target: str, target_raw: str, ops: List[str], extras: Dict[str, Any]) -> Dict[str, Any]:
    """Meta block of one L2 row (v0_1 fields + v0_2 fields + L2 fields).

    :param cand: Candidate.
    :param dec: Split decision.
    :param doc_key: Document key.
    :param variant: Input variant.
    :param lang: Coarse language class.
    :param tr: Tier result.
    :param target: Final target.
    :param target_raw: Target before the v0_2 strips.
    :param ops: Strip ops.
    :param extras: Further fields (tokens, truncation flags, provenance numbers).
    :returns: Meta dict.
    """
    doc = dec.doc
    pgpid = cand.edition_pgpid or (doc.pgpid if doc else (cand.pgpids[0] if cand.pgpids else None))
    shelfmark = (doc.shelfmark if doc else None) or (cand.shelfmarks[0] if cand.shelfmarks else None)
    vm = variant.meta
    m: Dict[str, Any] = {
        "task": FAMILY, "family": FAMILY, "lang_class": lang, "condition": variant.condition, "pgpid": pgpid,
        "shelfmark": shelfmark, "shelfmark_group": (doc.group if doc else None) or f"k:{doc_key}", "split": dec.split,
        "held_out_vision_benchmark": bool(dec.held_out_vision_benchmark), "holdout_reasons": dec.holdout_reasons,
        "target_chars": len(target), "target_tokens": None, "tokens": None,
        "input_cer": vm.get("input_cer"), "input_source": vm.get("input_source"),
        "input_truncated": bool(vm.get("input_truncated")), "licence": cand.licence, "licence_class": cand.licence_class,
        "provenance": dict(cand.provenance, kind=cand.kind), "source": cand.source,
        "read_model": vm.get("read_model"), "read_coverage": vm.get("read_coverage"),
        "read_agreed_share": vm.get("read_agreed_share"), "edition_available": vm.get("edition_available"),
        "reader_exposed": vm.get("reader_exposed"), "exposure_level": vm.get("exposure_level"),
        "noise_band": vm.get("noise_band"), "noise_class": vm.get("noise_class"),
        "target_raw": target_raw if ops else None, "strip_ops": ops, "pgp_description_dup": False,
        "work": cand.work, "page_keys": list(cand.page_keys) + [s["page_key"] for s in tr.lit_sources],
        "lit_sources": tr.lit_sources, "page_range": _page_range(cand.printed_pages),
        "grounding_tier": tr.tier, "grounding_share": round(tr.share, 3), "grounding_share_a": round(tr.share_a, 3),
        "grounding_share_c": None if tr.share_c is None else round(tr.share_c, 3), "grounding_candidates": tr.candidates,
        "analysis_source": "l2", "l2_kind": cand.kind, "doc_key": doc_key, "split_reason": dec.reason,
        "target_lang": "he" if _tn.dominant_script(target) == "hebrew" else "en", "build": _rows.BUILD}
    if "noise_target_cer" in vm:
        m["noise_target_cer"] = vm["noise_target_cer"]
    for k in ("page_keys_source", "entry_pages", "page_keys_unmapped"):
        if cand.provenance.get(k) is not None:
            m[k] = cand.provenance[k]
    m.update(extras)
    m["row_id"] = _rows.row_id(FAMILY, pgpid or doc_key, variant.condition, vm.get("read_model"), "l2", cand.source,
                               cand.label)
    return m


def _page_range(printed: Sequence[Any]) -> Optional[str]:
    """``first–last`` of the printed pages.

    :param printed: Printed page numbers.
    :returns: Range string or None.
    """
    nums = [x for x in printed if isinstance(x, int)]
    if not nums:
        return None
    return str(min(nums)) if min(nums) == max(nums) else f"{min(nums)}-{max(nums)}"


def _lang_for(cand: Candidate, dec: SplitDecision, corpus_docs: Mapping[str, Dict[str, Any]], text: str) -> str:
    """Coarse language class of a candidate's document.

    :param cand: Candidate.
    :param dec: Split decision (its builder document wins).
    :param corpus_docs: PGP ``documents.csv`` rows by pgpid.
    :param text: Input text.
    :returns: Language class.
    """
    if dec.doc is not None and cand.kind == SPAN_PGP:
        return dec.doc.lang_class
    for p in cand.pgpids:
        row = corpus_docs.get(p)
        if row and row.get("languages_primary"):
            lc = _bv.lang_from_pgp(str(row["languages_primary"]))
            if lc not in ("OTHER", "UNKNOWN"):
                return lc
    return cand.lang_hint or _lang_of_text(text, cand.script)


def build_rows(inp: L2Inputs, cfg: L2Config, tok: TokenFns) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the gates and tiers over every candidate and emit the L2 rows.

    Candidates are processed by kind (PGP-edition spans, book-edition spans, book-edition entries);
    a later target of a document already kept is dropped when it shares a 60-character window with it.

    :param inp: Inputs.
    :param cfg: Build configuration.
    :param tok: Token counters.
    :returns: ``(rows, report)``; the report has per-candidate records under ``records``.
    """
    drops: collections.Counter = collections.Counter()
    per_kind_drops: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stats: collections.Counter = collections.Counter()
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    kept_targets: Dict[str, List[str]] = collections.defaultdict(list)
    seen_doc_source: Set[Tuple[str, str]] = set()
    by_loc = {v: k for k, v in inp.page_loc.items()}
    cands = sorted(inp.candidates, key=lambda c: (KIND_ORDER.index(c.kind), c.source, c.part, c.label))

    def drop(c: Candidate, reason: str, rec: Dict[str, Any]) -> None:
        """Count a dropped candidate.

        :param c: Candidate.
        :param reason: Reason.
        :param rec: Its record (updated).
        :returns: None.
        """
        drops[reason] += 1
        per_kind_drops[c.kind][reason] += 1
        rec["dropped"] = reason
        records.append(rec)

    for c in cands:
        doc_guess = inp.docs.get(c.edition_pgpid) if c.edition_pgpid else next(
            (inp.docs.get(p) for p in c.pgpids if inp.docs.get(p)), None)
        dkey = doc_key_of(c, doc_guess)
        dec = assign_split(c, inp.docs, inp.holdouts, dkey, cfg.split_buckets)
        rec: Dict[str, Any] = {"kind": c.kind, "source": c.source, "label": c.label, "doc_key": dkey, "split": dec.split,
                               "split_reason": dec.reason, "pgpid": c.edition_pgpid or (c.pgpids[0] if c.pgpids else None)}
        if (dkey, c.source) in seen_doc_source:
            drop(c, "duplicate_doc_in_source", rec)
            continue
        # ---- input
        if c.kind == SPAN_PGP:
            d = inp.docs.get(c.edition_pgpid)
            if d is None or d.edition is None:
                drop(c, "no_edition_text", rec)
                continue
            edition = d.edition
            ed_meta = next((v.meta for v in d.variants if v.condition == "edition"),
                           {"input_source": "pgp_digital_edition", "input_cer": None})
            variants = [Variant("edition", edition, {**ed_meta, "input_truncated": d.edition_truncated})]
            variants += [v for v in d.variants if v.condition != "edition" and v.condition in cfg.conditions]
            script = d.script or _bv.script_of(edition)
            catalogue = d.catalogue
        else:
            edition = c.edition_text or ""
            variants = [Variant("edition", edition, {"input_source": f"book_edition:{c.source.split(':', 1)[-1]}",
                                                     "input_cer": None, "input_truncated": False})]
            script = c.script or _bv.script_of(edition)
            catalogue = dec.doc.catalogue if (dec.doc is not None and c.kind != NAS) else None
            if catalogue is None and c.kind != NAS:
                for p in c.pgpids:
                    if p in inp.corpus_docs:
                        catalogue = catalogue_from_corpus(inp.corpus_docs[p])
                        if catalogue:
                            break
        # ---- target
        units: List[str] = []
        if c.kind == NAS:
            raw, acts = nas_target(str(c.intro), str(c.notes), edition, cfg)
        else:
            raw, acts, units = span_target(list(c.intro), list(c.notes), edition, cfg)
        stats.update({f"clean:{k}": v for k, v in acts.items()})
        if len(raw) < cfg.min_target_chars:
            drop(c, "target_short_after_cleaning", rec)
            continue
        if inp.excluded:
            classes = {r.split(":")[0] for k in c.page_keys for r in inp.excluded.get(k, [])}
            rec["span_page_exclusion_classes"] = sorted(classes)
            if cfg.page_level_exclusion and dec.split == "train" and classes - set(cfg.page_exclusion_ignore):
                drop(c, "span_page_excluded", rec)
                continue
        if c.kind != NAS:
            cont = ocr_containment(units, [inp.ocr_by_key.get(k) for k in c.page_keys])
            rec["ocr_containment"] = None if cont is None else round(cont, 3)
            if cont is None and not cfg.allow_missing_ocr:
                drop(c, "target_no_ocr", rec)
                continue
            if cont is not None and cont < cfg.min_ocr_containment:
                drop(c, "target_ocr_containment_low", rec)
                continue
        mentions = held_mentions(raw, inp.holdouts, c.shelfmarks + ([dec.doc.shelfmark] if dec.doc and dec.doc.shelfmark else []))
        rec["held_mentions"] = [r for _, r in mentions]
        if mentions and dec.split == "train":
            drop(c, "target_mentions_held_doc", rec)
            continue
        phys, pops = strip_physical(raw)
        target, ops = _bld.clean_target(phys, FAMILY)
        ops = pops + ops
        rec["strip_ops"] = ops
        if not target.strip():
            drop(c, "strip_emptied", rec)
            continue
        if len(target) < cfg.min_target_after_strip:
            drop(c, "target_short_after_strip", rec)
            continue
        if _tn.overlap60(target, edition):
            drop(c, "target_overlaps_input", rec)
            continue
        if any(_tn.overlap60(target, t) or _tn.overlap60(t, target) for t in kept_targets.get(dkey, [])):
            drop(c, "duplicate_target", rec)
            continue
        # ---- context and tiers
        hints = c.hints or ({"title": c.title} if c.title and c.title != "label-only" and len(c.title.split()) >= 2 else {})
        ctx, ctx_note = context_for(target, catalogue, hints, inp.leak_fn if dec.split == "train" else None)
        rec["context"] = ctx_note or "catalogue"
        block_fn = None
        pool = inp.pools.get(c.work)
        if pool is not None:
            exclude = _neighbour_keys(c.page_keys, inp.page_loc, by_loc, cfg.window)

            def block_fn(pool: LitPool = pool, exclude: Set[str] = exclude, query: str = edition + "\n" + ctx,
                         target: str = target) -> Tuple[List[Tuple[str, str, List[str]]], List[Dict[str, Any]],
                                                        collections.Counter]:
                """Literature block for this candidate.

                :param pool: The work's pool.
                :param exclude: Keys excluded around the span.
                :param query: Input text.
                :param target: Target.
                :returns: :func:`select_block` output.
                """
                return select_block(pool, query, exclude, target, cfg, inp.leak_fn)
        tr = decide_tier(target, edition, ctx, inp.scholar_names, cfg, c.translation_text, block_fn)
        stats.update(tr.stats)
        rec.update({"share_a": round(tr.share_a, 3), "share_b": None if tr.share_b is None else round(tr.share_b, 3),
                    "share_c": None if tr.share_c is None else round(tr.share_c, 3), "candidates": tr.candidates,
                    "missing_a": tr.missing_a[:8], "tier": tr.tier})
        if tr.tier is None:
            drop(c, "grounding_fail", rec)
            continue
        # ---- n-gram gate (train rows)
        if dec.split == "train" and inp.leak_fn is not None:
            if inp.leak_fn(target):
                drop(c, "target_ngram_leak", rec)
                continue
            if c.kind != SPAN_PGP and inp.leak_fn(edition):
                drop(c, "input_ngram_leak", rec)
                continue
        # ---- rows
        lang = _lang_for(c, dec, inp.corpus_docs, edition)
        n_rows = 0
        for v in variants:
            vscript = script if v.condition != "edition_rescript" else "hebrew"
            extra = tr.extra

            def user_fn(text: str, cond: str = v.condition, vscript: str = vscript, extra: str = extra) -> str:
                """User message for this variant.

                :param text: Input text.
                :param cond: Condition.
                :param vscript: Script.
                :param extra: Context / literature block.
                :returns: User message.
                """
                return _rows.user_prompt(FAMILY, cond, vscript, text, extra)

            fitted = fit_row(user_fn, v.text, target, tok, cfg.max_row_tokens, cfg.min_target_after_strip)
            if fitted is None:
                stats[f"over_cap:{v.condition}"] += 1
                continue
            user, t_final, n_tok, t_trunc, i_trunc = fitted
            msgs = _rows.make_messages(user, t_final)
            meta = _meta(c, dec, dkey, v, lang, tr, msgs[2]["content"], raw, ops,
                         {"tokens": n_tok, "target_tokens": tok.count(msgs[2]["content"]), "target_truncated": t_trunc,
                          "ocr_containment": rec.get("ocr_containment"), "context_note": ctx_note})
            meta["input_truncated"] = bool(meta["input_truncated"] or i_trunc)
            texts = [m["content"] for m in msgs]
            if any(_tn.has_source_names(t) for t in texts):
                stats["scrub_row_dropped"] += 1
                continue
            rows.append({"messages": msgs, "meta": meta})
            n_rows += 1
        if not n_rows:
            drop(c, "over_cap", rec)
            continue
        rec["rows"] = n_rows
        records.append(rec)
        kept_targets[dkey].append(target)
        seen_doc_source.add((dkey, c.source))
    report = {"drops": dict(drops), "drops_by_kind": {k: dict(v) for k, v in per_kind_drops.items()},
              "stats": dict(stats), "records": records}
    return rows, report


# ============================================================================== report
def _quantiles(xs: Sequence[float]) -> Dict[str, Any]:
    """Deciles and threshold counts of grounding shares.

    :param xs: Shares.
    :returns: ``{"n", "deciles", "ge"}``.
    """
    xs = list(xs)
    return {"n": len(xs), "deciles": [round(q, 3) for q in statistics.quantiles(xs, n=10)] if len(xs) >= 10 else
            sorted(round(x, 3) for x in xs), "ge": {str(t): sum(x >= t for x in xs) for t in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)}}


def summarise(rows: Sequence[Dict[str, Any]], report: Dict[str, Any], cand_drops: Mapping[str, int],
              build_args: Dict[str, Any], carried: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The manifest: counts, tiers, grounding distributions, drops, tokens, v0_1 coverage.

    :param rows: L2 rows.
    :param report: :func:`build_rows` report.
    :param cand_drops: Drops before the gates (candidate extraction).
    :param build_args: Arguments of the build.
    :param carried: The builder's carried v0_1 ``analyse_document`` rows (coverage comparison), if any.
    :returns: Manifest dict.
    """
    recs = [r for r in report["records"]]
    kept = [r for r in recs if not r.get("dropped")]
    tiers: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in kept:
        tiers[f"{r['kind']}"][f"{r['tier']}:{r['split']}"] += 1
    tier_docs = collections.Counter(r["tier"] for r in kept)
    tier_docs_train = collections.Counter(r["tier"] for r in kept if r["split"] == "train")
    evaluated = [r for r in recs if "share_a" in r]
    tok_by_split: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"rows": 0, "tokens": 0})
    tok_by_tier: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"rows": 0, "tokens": 0})
    for row in rows:
        m = row["meta"]
        for d, k in ((tok_by_split, m["split"]), (tok_by_tier, f"{m['grounding_tier']}:{m['split']}")):
            d[k]["rows"] += 1
            d[k]["tokens"] += int(m["tokens"] or 0)
    per_source: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        per_source[r["source"]]["kept" if not r.get("dropped") else "dropped:" + r["dropped"]] += 1
    docs_multi = collections.Counter(r["doc_key"] for r in kept)
    out: Dict[str, Any] = {
        "build": _rows.BUILD, "family": FAMILY, "file": f"families/{OUT_FILE}",
        "built": time.strftime("%Y-%m-%d %H:%M"), "build_args": build_args,
        "counts": _bld.count_table(rows),
        "documents": {"kept": len(kept), "kept_train": sum(1 for r in kept if r["split"] == "train"),
                      "by_split": dict(collections.Counter(r["split"] for r in kept)),
                      "by_kind": dict(collections.Counter(r["kind"] for r in kept)),
                      "by_kind_train": dict(collections.Counter(r["kind"] for r in kept if r["split"] == "train")),
                      "split_reasons": dict(collections.Counter(r["split_reason"].split(":")[0] for r in kept)),
                      "docs_with_several_sources": sum(1 for v in docs_multi.values() if v > 1)},
        "tiers": {"documents": dict(tier_docs), "documents_train": dict(tier_docs_train),
                  "by_kind": {k: dict(v) for k, v in tiers.items()}},
        "grounding": {"share_a_all_evaluated": _quantiles([r["share_a"] for r in evaluated]),
                      "share_a_by_kind": {k: _quantiles([r["share_a"] for r in evaluated if r["kind"] == k]) for k in KIND_ORDER
                                          if any(r["kind"] == k for r in evaluated)},
                      "share_c_tried": _quantiles([r["share_c"] for r in evaluated if r.get("share_c") is not None]),
                      "share_b_measured": _quantiles([r["share_b"] for r in evaluated if r.get("share_b") is not None]),
                      "tier_b_would_rescue": sum(1 for r in evaluated if r.get("tier") is None and (r.get("share_b") or 0) >= 0.6)},
        "drops": {"candidate_extraction": dict(cand_drops), "gates": report["drops"],
                  "gates_by_kind": report["drops_by_kind"]},
        "stats": report["stats"],
        "tokens": {"by_split": dict(tok_by_split), "by_tier_split": dict(tok_by_tier),
                   "total": {"rows": len(rows), "tokens": sum(int(r["meta"]["tokens"] or 0) for r in rows)},
                   "train": {"rows": tok_by_split.get("train", {}).get("rows", 0),
                             "tokens": tok_by_split.get("train", {}).get("tokens", 0)}},
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "ocr_containment": _quantiles([r["ocr_containment"] for r in recs if r.get("ocr_containment") is not None]),
        "fingerprint": _rows.fingerprint(rows)}
    if carried is not None:
        v01 = {str(r["meta"]["pgpid"]): r["meta"]["split"] for r in carried}
        v2 = {str(r["meta"]["pgpid"]) for r in rows if r["meta"].get("pgpid")}
        why = {str(r.get("pgpid")): r.get("dropped") for r in recs if r.get("dropped")}
        lost = sorted(p for p in v01 if p not in v2)
        out["v0_1_coverage"] = {"carried_docs": len(v01), "carried_train_docs": sum(1 for s in v01.values() if s == "train"),
                                "covered": len(v01) - len(lost), "lost": len(lost),
                                "lost_train": sum(1 for p in lost if v01[p] == "train"),
                                "lost_reasons": dict(collections.Counter(why.get(p) or "not_a_candidate" for p in lost))}
    return out


def audit_sample(rows: Sequence[Dict[str, Any]], records: Sequence[Dict[str, Any]], n: int = 30, seed: int = 3407
                 ) -> List[Dict[str, Any]]:
    """A stratified hand-audit sample: one edition-condition row per document, round-robin over (kind, tier).

    :param rows: L2 rows.
    :param records: Candidate records (missing entities, shares).
    :param n: Sample size.
    :param seed: Shuffle seed.
    :returns: Audit records (target before and after the strips, input head, context, block pages).
    """
    rec_by = {(r["source"], r["label"]): r for r in records if not r.get("dropped")}
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        m = row["meta"]
        if m["condition"] != "edition":
            continue
        strata[(m["l2_kind"], str(m["grounding_tier"]))].append(row)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    out: List[Dict[str, Any]] = []
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                row = strata[k].pop()
                m = row["meta"]
                user = row["messages"][1]["content"]
                label = str((m.get("provenance") or {}).get("doc_label") or (m.get("provenance") or {}).get("entry_no"))
                rec = rec_by.get((m["source"], label), {})
                out.append({"row_id": m["row_id"], "split": m["split"], "kind": m["l2_kind"], "source": m["source"],
                            "tier": m["grounding_tier"], "grounding_share": m["grounding_share"],
                            "share_a": m["grounding_share_a"], "share_c": m["grounding_share_c"],
                            "missing_a": rec.get("missing_a"), "ocr_containment": m.get("ocr_containment"),
                            "strip_ops": m["strip_ops"], "target_before_strip": m.get("target_raw"),
                            "target": row["messages"][2]["content"], "input_head": document_text(user)[:400],
                            "context_and_block": user[user.find("</document>") + len("</document>"):].strip()[:2500],
                            "lit_sources": m.get("lit_sources"), "tokens": m["tokens"]})
    return out


# ============================================================================== loading and CLI
def _scholar_names(p: _paths.Paths, local: Mapping[str, Any], manifest: Mapping[str, Dict[str, Any]],
                   books: Mapping[str, List[Dict[str, Any]]]) -> Set[str]:
    """Scholar surnames (PGP sources, local extras, literature-manifest authors, book-edition editors).

    :param p: Build paths.
    :param local: Local builder config.
    :param manifest: Literature manifest.
    :param books: Book-edition entries.
    :returns: Folded surnames.
    """
    extra = list(local.get("scholar_extra") or [])
    for e in manifest.values():
        a = str(e.get("author") or "")
        if a and a != "None":
            extra += [w for w in re.split(r"[,&;]|\band\b|\bet al\b", a) if w.strip()]
    for entries in books.values():
        for e in entries[:1]:
            for v in (str(e.get("editors") or ""), str(e.get("source") or "").split(",")[0]):
                v = re.sub(r"\([^)]*\)", " ", v)
                extra += [w for w in re.split(r"[&;]|\band\b", v) if w.strip()]
    surnames = [x.strip().split()[-1] for x in extra if x.strip() and x.strip().split()]
    path = Path(p.pgp_export) / "sources.csv"
    return _tn.scholar_surnames(path, surnames) if path.exists() else {_tn.strip_marks(x) for x in surnames}


def _nontrain_pages(index_rows: Sequence[Dict[str, Any]], holdouts: Any, page_loc: Mapping[str, Tuple[str, int]],
                    window: int) -> Dict[str, List[str]]:
    """Pages that mention a non-train document (any tier, any evidence) with a ±window, from ``rag_index.jsonl``.

    :param index_rows: Index rows.
    :param holdouts: Hold-out set.
    :param page_loc: page key → (work/part, page number).
    :param window: Window.
    :returns: ``{page_key: [reason]}``.
    """
    nt = _rag.NonTrain(holdouts=holdouts)
    by_loc = {v: k for k, v in page_loc.items()}
    hit: Dict[str, str] = {}
    seen: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = set()
    for r in index_rows:
        sig = (r["page_key"], tuple(r.get("pgpids") or []), tuple(r.get("merged_doc_ids") or []))
        if sig in seen:
            continue
        seen.add(sig)
        why = nt.reason(r.get("pgpids") or [], r.get("merged_doc_ids") or [])
        if why and r["page_key"] not in hit:
            hit[r["page_key"]] = why
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for k, why in hit.items():
        for kk in _neighbour_keys([k], page_loc, by_loc, window):
            out[kk].append(("mention:" if kk == k else "window:") + why)
    return dict(out)


def load_inputs(p: _paths.Paths, cfg: L2Config, families_dir: Path, use_spans: bool = True, use_nas: bool = True,
                use_ngram: bool = True, nas_books: Optional[Sequence[str]] = None) -> Tuple[L2Inputs, Dict[str, Any]]:
    """Load every input of a real build (read-only).

    :param p: Build paths.
    :param cfg: Build configuration (``excluded_books`` is filled from the local builder config when empty).
    :param families_dir: v0_2 builder ``families`` directory.
    :param use_spans: Build span candidates.
    :param use_nas: Build book-edition candidates.
    :param use_ngram: Load the eval-reference n-gram index.
    :param nas_books: Restrict the book-edition slugs.
    :returns: ``(inputs, info)``; info holds candidate-extraction drops, timings and input file names.
    """
    from src.finetuning.text27b import contamination as _ct
    from src.finetuning.text27b import holdouts as _ho
    from src.finetuning.text27b import literature_l1 as _l1

    t0 = time.time()
    info: Dict[str, Any] = {"seconds": {}, "inputs": {}}
    local = _bv.load_local_config(p)
    if not cfg.excluded_books:
        cfg.excluded_books = tuple(local.get("analysis_excluded_books") or [])
    manifest = _rag.load_literature_manifest(p)
    holdouts = _ho.Holdouts.load(p=p)
    corpus = _sm.load_corpus(p)
    info["seconds"]["holdouts_corpus"] = round(time.time() - t0, 1)
    docs0 = load_doc_index(families_dir, want=set(), conditions=cfg.conditions)
    cands: List[Candidate] = []
    cand_drops: collections.Counter = collections.Counter()
    pages: List[Any] = []
    span_keys: Dict[str, Set[str]] = collections.defaultdict(set)
    if use_spans:
        t1 = time.time()
        config = _rag.load_span_config(p=p)
        works = sorted(config.span_works())
        pages = _pg.iter_pages(root=p.academic_literature, works=works)
        span_rows = _rows.load_jsonl(p.out("rag_spans.jsonl"))
        full = attach_sections(span_rows, detect_full_spans(pages, corpus, config))
        for sp in full:
            span_keys[sp["book"]] |= set(sp.get("page_keys") or [])
        c1, d1 = span_candidates(full, docs0, cfg, local.get("licence_by_book") or {}, manifest)
        cands += c1
        cand_drops.update(d1)
        info["seconds"]["spans"] = round(time.time() - t1, 1)
        info["inputs"]["rag_spans"] = len(span_rows)
    books: Dict[str, List[Dict[str, Any]]] = {}
    if use_nas:
        from src.finetuning.text27b import book_pairs as _bp  # the shared entry → page mapping (F11)
        books = load_nas_books(p.book_editions_nas, nas_books)
        entry_pages = _bp.EntryPages.load(p, _bp.load_local_config(p)[0], extra_slugs=books.keys()) if books else None
        c2, d2 = nas_candidates(books, corpus, cfg, entry_pages)
        if entry_pages is not None:
            info["inputs"]["entry_page_works"] = {k: v for k, v in sorted(entry_pages.works.items()) if k in books}
        cands += c2
        cand_drops.update(d2)
        info["inputs"]["nas_books"] = {k: len(v) for k, v in books.items()}
    want = {c.edition_pgpid for c in cands if c.edition_pgpid} | {x for c in cands for x in c.pgpids}
    t2 = time.time()
    docs = load_doc_index(families_dir, want=want, conditions=cfg.conditions)
    info["seconds"]["doc_index"] = round(time.time() - t2, 1)
    page_loc = {pg.page_key: (f"{pg.work}/{pg.part}", pg.page_no) for pg in pages}
    excluded: Dict[str, List[str]] = {}
    ep = p.out(_ct.EXCLUDED_PAGES_FILE)
    if ep.exists():
        excluded.update(_ct.load_excluded_pages(ep))
        info["inputs"]["excluded_pages"] = len(excluded)
    else:
        info["inputs"]["excluded_pages"] = None
        LOG.warning("excluded_pages.json not found: blocks rely on the rag_index mentions and the extra list")
    extra_path = Path(p.source_file).parent / _ct.EXTRA_EXCLUDED_FILE if p.source_file else None
    if extra_path is not None and extra_path.exists() and pages:
        extra, unmatched = _ct.load_extra_excluded(extra_path, _ct.PageResolver(page_loc.keys()))
        for k, rs in extra.items():
            excluded.setdefault(k, []).extend(sorted(rs))
        info["inputs"]["extra_excluded_pages"] = {"pages": len(extra), "unmatched": unmatched}
    drop_pages: Dict[str, List[str]] = {k: list(v) for k, v in excluded.items()}
    ri = p.out("rag_index.jsonl")
    if ri.exists() and pages:
        for k, v in _nontrain_pages(_rows.load_jsonl(ri), holdouts, page_loc, cfg.window).items():
            drop_pages.setdefault(k, []).extend(v)
    leak_fn = None
    if use_ngram:
        t3 = time.time()
        index = _ct.build_reference_index(p, holdouts, corpus=corpus)
        leak_fn = index.is_leak
        info["seconds"]["ngram_index"] = round(time.time() - t3, 1)
        info["inputs"]["ngram_index"] = index.stats()
    pools: Dict[str, LitPool] = {}
    if pages and TIER_C in cfg.tiers:
        t4 = time.time()
        heads = _l1.running_heads_by_part(pages)
        pools, pstats = build_pools(pages, {c.work for c in cands if c.kind != NAS}, span_keys, drop_pages, cfg, manifest,
                                    quality_fn=lambda pg: _l1.quality_ok(pg, heads.get((pg.work, pg.part))))
        info["pool_pages"] = dict(pstats)
        info["seconds"]["pools"] = round(time.time() - t4, 1)
    inp = L2Inputs(candidates=cands, docs=docs, holdouts=holdouts,
                   scholar_names=_scholar_names(p, local, manifest, books), pools=pools,
                   ocr_by_key={pg.page_key: pg.ocr_text for pg in pages}, page_loc=page_loc,
                   corpus_docs={str(k): v for k, v in corpus.docs.items()}, leak_fn=leak_fn, excluded=excluded)
    info["candidate_drops"] = dict(cand_drops)
    info["candidates"] = dict(collections.Counter(c.kind for c in cands))
    info["seconds"]["load_total"] = round(time.time() - t0, 1)
    return inp, info


def run(p: Optional[_paths.Paths] = None, cfg: Optional[L2Config] = None, out_dir: Optional[Path] = None,
        families_dir: Optional[Path] = None, use_spans: bool = True, use_nas: bool = True, use_ngram: bool = True,
        nas_books: Optional[Sequence[str]] = None, audit_n: int = 30, tok: Optional[TokenFns] = None,
        build_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build and write ``families/analyse_document_v2.jsonl``, its manifest and the audit sample.

    :param p: Build paths.
    :param cfg: Build configuration.
    :param out_dir: Output root (default ``paths.out_dir``).
    :param families_dir: Builder families (default ``<out_dir>/families``).
    :param use_spans: Use the literature spans.
    :param use_nas: Use the book-edition entries.
    :param use_ngram: Apply the eval-reference n-gram gate.
    :param nas_books: Restrict the book-edition slugs.
    :param audit_n: Audit sample size (0 = none).
    :param tok: Token counters (default: the local tokenizer and template).
    :param build_args: Arguments recorded in the manifest.
    :returns: The manifest.
    """
    p = p or _paths.load_paths()
    cfg = cfg or L2Config()
    out = Path(out_dir) if out_dir is not None else p.out_dir
    fam_dir = Path(families_dir) if families_dir is not None else out / "families"
    t0 = time.time()
    inp, info = load_inputs(p, cfg, fam_dir, use_spans, use_nas, use_ngram, nas_books)
    t1 = time.time()
    rows, report = build_rows(inp, cfg, tok or default_token_fns(p.tokenizer_json, p.chat_template))
    rows.sort(key=lambda r: (_bld.SPLIT_ORDER.get(r["meta"]["split"], 99), r["meta"]["l2_kind"], r["meta"]["source"],
                             str(r["meta"]["doc_key"]), _bld.COND_ORDER.get(r["meta"]["condition"], 99),
                             r["meta"].get("read_model") or ""))
    t_build = round(time.time() - t1, 1)
    carried_path = fam_dir / f"{FAMILY}.jsonl"
    carried = _rows.load_jsonl(carried_path) if carried_path.exists() else None
    args = dict(build_args or {})
    args.update({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()},
                 "use_spans": use_spans, "use_nas": use_nas, "use_ngram": use_ngram})
    manifest = summarise(rows, report, info["candidate_drops"], args, carried)
    manifest["inputs"] = info["inputs"]
    manifest["candidates"] = info["candidates"]
    manifest["pool_pages"] = info.get("pool_pages")
    problems = collections.Counter(pr for r in rows for pr in _rows.validate_row(r))
    manifest["checks"] = {"schema_problems": dict(problems), "rows_over_cap": sum(1 for r in rows if r["meta"]["tokens"] > cfg.max_row_tokens),
                          "scrub_rows": sum(1 for r in rows if any(_tn.has_source_names(m["content"]) for m in r["messages"])),
                          "train_rows_benchmark_meta": sum(1 for r in rows if r["meta"]["split"] == "train" and inp.holdouts is not None
                                                           and inp.holdouts.is_benchmark(r["meta"]["pgpid"], r["meta"]["shelfmark"])),
                          "train_rows_eval_meta": sum(1 for r in rows if r["meta"]["split"] == "train" and inp.holdouts is not None
                                                      and inp.holdouts.is_eval_doc(r["meta"]["pgpid"], r["meta"]["shelfmark"])),
                          "duplicate_row_ids": len(rows) - len({r["meta"]["row_id"] for r in rows}),
                          "read_like_note_ok": all(_rows.READ_NOTE_V2 in r["messages"][1]["content"] for r in rows
                                                   if r["meta"]["condition"] in _rows.READ_LIKE_CONDITIONS)}
    n = _rows.write_jsonl(out / "families" / OUT_FILE, rows)
    if audit_n:
        sample = audit_sample(rows, report["records"], audit_n, cfg.seed)
        _rows.write_jsonl(out / "audit" / f"analyse_v2_sample_{audit_n}.jsonl", sample)
        manifest["audit_sample"] = {"file": f"audit/analyse_v2_sample_{audit_n}.jsonl", "rows": len(sample)}
    manifest["seconds"] = {**info["seconds"], "build_rows": t_build, "total": round(time.time() - t0, 1)}
    manifest["records_dropped_examples"] = [r for r in report["records"] if r.get("dropped") == "grounding_fail"][:12]
    mp = out / "families" / MANIFEST_FILE
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1, default=str)
    tmp.replace(mp)
    LOG.info("wrote %d rows to %s", n, out / "families" / OUT_FILE)
    return manifest


def _parse_tiers(s: str) -> Tuple[str, ...]:
    """Parse a ``--tiers`` value (``"A',C"``, ``A,B,C`` ...).

    :param s: Comma-separated tiers.
    :returns: Canonical tier names.
    :raises argparse.ArgumentTypeError: On an unknown tier.
    """
    out = []
    for t in (x.strip() for x in s.split(",") if x.strip()):
        if t not in TIER_ALIASES:
            raise argparse.ArgumentTypeError(f"unknown tier {t!r}")
        out.append(TIER_ALIASES[t])
    return tuple(dict.fromkeys(out))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (default ``sys.argv``).
    :returns: None.
    """
    ap = argparse.ArgumentParser(description="L2: widened analyse_document rows (text27b v0_2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build families/analyse_document_v2.jsonl + manifest + audit sample")
    b.add_argument("--out-dir", type=Path, default=None)
    b.add_argument("--families-dir", type=Path, default=None)
    b.add_argument("--tiers", type=_parse_tiers, default=(TIER_A, TIER_C))
    b.add_argument("--k-max", type=int, default=3)
    b.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    b.add_argument("--skip-pgp-overlap", action="append", default=[], metavar="SLUG",
                   help="book-edition slug whose PGP-matched entries are skipped (repeatable)")
    b.add_argument("--nas-books", nargs="*", default=None, help="restrict the book-edition slugs")
    b.add_argument("--min-ocr-containment", type=float, default=0.8)
    b.add_argument("--allow-missing-ocr", action="store_true")
    b.add_argument("--no-strict-non-span", action="store_true", help="let blocks use other documents' span pages")
    b.add_argument("--page-level-exclusion", dest="page_level_exclusion", action="store_true", default=True,
                   help="drop train targets whose span pages are in excluded_pages.json for any reason but "
                        "train_target_dup (default on)")
    b.add_argument("--no-page-level-exclusion", dest="page_level_exclusion", action="store_false",
                   help="text-level gates only (development)")
    b.add_argument("--no-ngram", action="store_true")
    b.add_argument("--no-nas", action="store_true")
    b.add_argument("--no-spans", action="store_true")
    b.add_argument("--audit-n", type=int, default=30)
    b.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = L2Config(tiers=args.tiers, k_max=args.k_max, conditions=tuple(x for x in args.conditions.split(",") if x),
                   skip_pgp_overlap=tuple(args.skip_pgp_overlap), min_ocr_containment=args.min_ocr_containment,
                   allow_missing_ocr=args.allow_missing_ocr, strict_non_span=not args.no_strict_non_span, seed=args.seed,
                   page_level_exclusion=args.page_level_exclusion)
    bargs = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    m = run(cfg=cfg, out_dir=args.out_dir, families_dir=args.families_dir, use_spans=not args.no_spans,
            use_nas=not args.no_nas, use_ngram=not args.no_ngram, nas_books=args.nas_books, audit_n=args.audit_n,
            build_args=bargs)
    print(json.dumps({k: m[k] for k in ("documents", "tiers", "tokens", "checks")}, indent=1, default=str))
    print(json.dumps({"drops": m["drops"], "seconds": m["seconds"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
