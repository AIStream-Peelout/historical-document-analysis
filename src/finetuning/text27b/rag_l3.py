# File name: rag_l3.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""L3 RAG-conditioned rows: the literature page index and the ``rag_context`` family (text27b).

Two stages (plan §9.2 L3, build brief module E):

``build_index`` (phase 1)
    Port of the 2026-10-18 inventory (``build_analysis.index_page`` / ``detect_spans`` /
    ``resolve_span`` / ``flag_boundaries``) and of the 2026-10-21 rebuild (``litjoin/build_index.py``):
    every shelfmark key a literature page carries (``shelf_marks_mentioned`` and ``transcriptions``
    keys) is matched to merged ids / pgpids and classified (``commentary``, ``translation``,
    ``edition_quote``, ``catalogue_entry``, ``footnote_only``, ``passing_mention``); document spans
    detected in document-centred works add rows for pages that never repeat the shelfmark. Each row
    gets a ``tier`` ∈ {discuss, catalogue, mention} and an ``evidence`` ∈ {located, clean_span,
    unclean_span, key_only} (``litjoin/join.py`` / ``promise.py``). Outputs under ``paths.out_dir``:
    ``rag_index.jsonl``, ``rag_spans.jsonl``, ``rag_index_report.json`` and ``rag_test_docs.json``
    (test documents with discuss pages, for the E6 with/without-literature eval).

``build_rows`` (phase 2)
    For train documents of the PGP families, join strong-evidence discuss pages (``located`` or
    ``clean_span``), drop pages that mention any non-train / held-out document, pages in
    ``excluded_pages.json`` and pages failing the L1 quality filter (``literature_l1.quality_ok``),
    apply the block-level 8-gram leak rule, render the site's source format and write
    ``families/rag_context.jsonl`` plus ``rag_context_manifest.json``.

No holdings in this file. Work-specific span rules (which work uses which heading layout, its
PGP source slugs, chapter heads ...) live in a git-ignored local JSON file
(``rag_span_config`` entry of ``paths.json``, default ``rag_span_config.json`` next to it); this
module only defines the generic layout styles that file names (:data:`HEADING_STYLES`,
:data:`RUNHEAD_STYLES`, :data:`MARKER_SETS`, :data:`LOCATION_LABEL_STYLES`,
:data:`KEY_LABEL_STYLES`). Author / title / year of a work come from the local literature manifest.

Public API:

Config
    :class:`WorkCfg`, :class:`SpanConfig`, :func:`load_span_config`, :func:`span_config_path`.
Spans and index
    :func:`sm_line`, :class:`Heading`, the heading / running-head styles, :func:`label_from_location`,
    :func:`build_crosswalk`, :func:`page_citations`, :class:`Span`, :func:`page_head`,
    :func:`classify_lines`, :func:`detect_spans`, :func:`resolve_span`, :func:`label_key`,
    :func:`flag_boundaries`, :func:`key_label`, :func:`index_page`, :func:`merge_span_rows`,
    :func:`tier_of`, :func:`evidence_of`, :func:`finalize_rows`, :func:`build_index`,
    :func:`index_report`, :func:`build_test_docs`, :func:`run_build_index`.
Rows
    :func:`words`, :func:`grams8`, :func:`share`, :func:`hebrew_char_ratio`, :func:`citation`,
    :func:`render_source`, :func:`render_block`, :class:`NonTrain`, :func:`load_literature_manifest`,
    :func:`build_rows`, :func:`rag_excluded_pages`, :func:`load_excluded_rows`, :func:`run_build_rows`.

CLI::

    .venv/bin/python -m src.finetuning.text27b.rag_l3 build_index [--works W ...] [--no-test-docs]
    .venv/bin/python -m src.finetuning.text27b.rag_l3 test_docs
    .venv/bin/python -m src.finetuning.text27b.rag_l3 build_rows [--seed 3407] [--families ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import pages as _pg
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as _tn

LOG = logging.getLogger(__name__)

Corpus = _sm.Corpus
PageRec = _pg.PageRec

#: Kinds that make a page a discussion of the document (``join.py``).
DISC_KINDS: frozenset = frozenset({"commentary", "translation", "edition_quote"})
TIERS: Tuple[str, ...] = ("discuss", "catalogue", "mention")
EVIDENCE: Tuple[str, ...] = ("located", "clean_span", "unclean_span", "key_only")
#: Evidence strong enough to join a page to a training row (``promise.py``).
STRONG_EVIDENCE: frozenset = frozenset({"located", "clean_span"})
ROMAN: Dict[str, int] = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

# ===================================================================================== config


@dataclass
class WorkCfg:
    """Document-span rules of one work (port of ``build_analysis.BookCfg`` plus the per-book branches).

    ``heading`` / ``runhead`` / ``markers`` / ``location_label`` / ``key_label`` name styles of this
    module; ``chapter_prefixed_labels`` prefixes labels with the running chapter head;
    ``runhead_is_set`` treats every running head as a set of labels; ``label_series`` ∈
    {default, roman_comma_number, roman} widens :func:`label_key`; ``trust_label_crosswalk`` takes
    the PGP shelfmark of a label-crosswalk join without locating it (works citing historic
    numbers); ``historic_label_re`` maps PGP ``shelfmarks_historic`` to labels; ``page_cites``
    False skips PGP page citations (volume page numbers collide).
    """

    name: str
    heading: Optional[str] = None
    runhead: Optional[str] = None
    markers: List[Tuple[str, str]] = field(default_factory=list)
    end_re: Optional[str] = None
    chapter_heads: Set[str] = field(default_factory=set)
    max_pages: int = 30
    hebrew_prose: bool = False
    latin_after_edition_is_translation: bool = False
    docness_translation: bool = False
    first_number_line_starts_edition: bool = False
    monotone: bool = True
    drop_frequent_titles: bool = False
    first_max: int = 100_000
    ja_he_split: bool = False
    chapter_prefixed_labels: bool = False
    runhead_is_set: bool = False
    pgp_source_slugs: List[str] = field(default_factory=list)
    page_cites: bool = True
    location_label: Optional[str] = None
    key_label: Optional[str] = None
    label_series: str = "default"
    trust_label_crosswalk: bool = False
    historic_label_re: Optional[str] = None


@dataclass
class SpanConfig:
    """All work rules plus the Hebrew-prose works (``HEBREW_BOOKS`` of the prototype)."""

    works: Dict[str, WorkCfg] = field(default_factory=dict)
    hebrew_works: Set[str] = field(default_factory=set)

    def get(self, work: str) -> Optional[WorkCfg]:
        """Rules of one work.

        :param work: Work directory name.
        :returns: The :class:`WorkCfg` or None.
        """
        return self.works.get(work)

    def span_works(self) -> Dict[str, WorkCfg]:
        """Works with a heading style (span detection runs on these only).

        :returns: ``{work: cfg}``.
        """
        return {w: c for w, c in self.works.items() if c.heading}


GENERIC_MARKERS: List[Tuple[str, str]] = [
    (r"^(\d\.0\.\s*)?(Discussion|General Description|Physical Description|Description|Introduction)$", "intro"),
    (r"^(\d\.0\.\s*)?(Texts?|TEXT|Main Text|Recto, Main Text|Transcription|Edition)$", "edition"),
    (r"^\{?(\d\.0\.\s*)?(Translations?|TRANSLATION|Partial Translation|תרגום)$", "translation"),
    (r"^(\d\.0\.\s*)?(Textual Notes?|Textual notes|Commentary|COMMENTS|Comments|General comments|Notes|NOTES|הערות)$",
     "notes"),
]
NUMBERED_SECTION_MARKERS: List[Tuple[str, str]] = [
    (r"^\d\.0\.\s*(Discussion)$", "intro"),
    (r"^\d\.0\.\s*Texts?$", "edition"),
    (r"^\d\.0\.\s*Translations?$", "translation"),
    (r"^\d\.0\.\s*(Textual Notes?|Notes|Commentary)$", "notes"),
]
HE_SIDE_MARKERS: List[Tuple[str, str]] = [(r"^\(?ע[\"״'][אב]\)?:?$", "edition")]
#: Section-marker sets a work config may name.
MARKER_SETS: Dict[str, List[Tuple[str, str]]] = {"generic": GENERIC_MARKERS, "numbered_section": NUMBERED_SECTION_MARKERS,
                                                 "he_side": HE_SIDE_MARKERS, "none": []}
GENERIC_END = (r"^PART [IVX]+\b|^(BIBLIOGRAPHY|Bibliography|INDEX|Index|INDICES|Indices|ABBREVIATIONS|Abbreviations)$"
               r"|^רשימת קיצורים|^ביבליוגרפיה|^מפתח")
_BOOL_FIELDS = ("hebrew_prose", "latin_after_edition_is_translation", "docness_translation",
                "first_number_line_starts_edition", "monotone", "drop_frequent_titles", "ja_he_split",
                "chapter_prefixed_labels", "runhead_is_set", "page_cites", "trust_label_crosswalk")


def span_config_path(p: Optional[_paths.Paths] = None) -> Path:
    """Location of the local span config.

    :param p: Build paths (default :func:`paths.load_paths`).
    :returns: ``paths.json``'s ``rag_span_config`` entry, else ``rag_span_config.json`` next to ``paths.json``.
    """
    p = p or _paths.load_paths()
    explicit = p.get("rag_span_config")
    if explicit:
        return Path(explicit)
    base = Path(p.source_file).parent if p.source_file else p.repo / _paths.DEFAULT_PATHS_FILE.parent
    return base / "rag_span_config.json"


def work_cfg_from_dict(name: str, d: Dict[str, Any]) -> WorkCfg:
    """Build and validate one :class:`WorkCfg` from its JSON entry.

    :param name: Work directory name.
    :param d: JSON entry.
    :returns: The config.
    :raises ValueError: On an unknown style name or field.
    """
    known = {f for f in WorkCfg.__dataclass_fields__} | {"extra_markers"}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{name}: unknown span-config fields {sorted(unknown)}")
    for fld, reg in (("heading", HEADING_STYLES), ("runhead", RUNHEAD_STYLES), ("location_label", LOCATION_LABEL_STYLES),
                     ("key_label", KEY_LABEL_STYLES)):
        v = d.get(fld)
        if v is not None and v not in reg:
            raise ValueError(f"{name}: unknown {fld} style {v!r}")
    markers_spec = d.get("markers", "generic")
    if isinstance(markers_spec, str):
        if markers_spec not in MARKER_SETS:
            raise ValueError(f"{name}: unknown marker set {markers_spec!r}")
        markers = list(MARKER_SETS[markers_spec])
    else:
        markers = [(str(a), str(b)) for a, b in markers_spec]
    markers += [(str(a), str(b)) for a, b in d.get("extra_markers", [])]
    if d.get("label_series", "default") not in ("default", "roman_comma_number", "roman"):
        raise ValueError(f"{name}: unknown label_series {d.get('label_series')!r}")
    cfg = WorkCfg(name=name, heading=d.get("heading"), runhead=d.get("runhead"), markers=markers,
                  end_re=d.get("end_re"), chapter_heads=set(d.get("chapter_heads", [])),
                  max_pages=int(d.get("max_pages", 30)), first_max=int(d.get("first_max", 100_000)),
                  pgp_source_slugs=list(d.get("pgp_source_slugs", [])), location_label=d.get("location_label"),
                  key_label=d.get("key_label"), label_series=d.get("label_series", "default"),
                  historic_label_re=d.get("historic_label_re"))
    for b in _BOOL_FIELDS:
        if b in d:
            setattr(cfg, b, bool(d[b]))
    return cfg


def load_span_config(path: Optional[Path] = None, p: Optional[_paths.Paths] = None) -> SpanConfig:
    """Load the local span config (an empty config when the file is absent: no spans, located joins only).

    :param path: Explicit config file.
    :param p: Build paths (used when ``path`` is None).
    :returns: The :class:`SpanConfig`.
    """
    path = Path(path) if path is not None else span_config_path(p)
    if not path.exists():
        LOG.warning("span config %s not found: no document spans will be detected", path)
        return SpanConfig()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    works = {w: work_cfg_from_dict(w, d) for w, d in (raw.get("works") or {}).items()}
    return SpanConfig(works=works, hebrew_works=set(raw.get("hebrew_works") or []))


# ===================================================================================== helpers


def roman_to_int(s: str) -> int:
    """Convert a Roman numeral.

    :param s: Numeral.
    :returns: Integer value (0 if invalid).
    """
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = ROMAN.get(ch, 0)
        if not v:
            return 0
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def page_number(p: Any) -> List[int]:
    """Printed page number(s) of a page.

    The prototype read ``extracted_page_number`` raw, so the structuring prompt's example label
    (:data:`pages.LEAKED_PAGE_LABELS`) produced false page citations. Here a raw value in that set is
    kept only when the neighbour-checked :attr:`PageRec.printed_page_num` confirms it, and that
    number is added when the raw field lacks it. Spreads keep both raw numbers.

    :param p: Page (``data`` and optionally ``printed_page_num``).
    :returns: List of ints.
    """
    v = p.data.get("extracted_page_number")
    vals = v if isinstance(v, list) else [v]
    raw: List[int] = []
    for x in vals:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            raw.append(x)
        elif isinstance(x, str) and x.strip().isdigit():
            raw.append(int(x.strip()))
    checked = getattr(p, "printed_page_num", None)
    has_check = hasattr(p, "printed_page_source")
    out = [x for x in raw if not (has_check and x in _pg.LEAKED_PAGE_LABELS and x != checked)]
    if has_check and checked is not None and checked not in out:
        out.insert(0, checked)
    return out


_SM_CACHE: Dict[int, Dict[str, Dict[str, Any]]] = {}


def _match_cached(part: str, corpus: Corpus) -> Dict[str, Any]:
    """:func:`shelfmarks.match_shelfmark` with a per-corpus cache.

    :param part: Shelfmark string.
    :param corpus: Corpus.
    :returns: Match dict.
    """
    cache = _SM_CACHE.setdefault(id(corpus), {})
    if part not in cache:
        cache[part] = _sm.match_shelfmark(part, corpus)
    return cache[part]


def clear_caches() -> None:
    """Drop the shelfmark-line match cache (tests build several corpora).

    :returns: None.
    """
    _SM_CACHE.clear()


def sm_line(line: str, corpus: Corpus) -> List[Dict[str, Any]]:
    """If a line is (mostly) a shelfmark citation, return its matches (split on ``+`` / ``;``).

    :param line: Text line.
    :param corpus: Corpus.
    :returns: Match dicts with a merged id or pgpid (empty if not a shelfmark line).
    """
    s = line.strip().rstrip(".")
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", s).strip()
    if not s or len(s) > 90 or not re.search(r"\d", s):
        return []
    s = re.sub(r"\(Cat\.?\s*\d+\)", "", s)
    if re.fullmatch(r"\{?[IVX]{1,3},?\s?\d+[a-z]?\}?|No\.?\s?\d+|\d+", s):
        return []  # a work's own document numbers / page numbers are not shelfmarks
    lower_words = [w for w in re.findall(r"[a-z]{4,}", s) if w not in {"heb", "misc", "arabic", "folio", "box"}]
    if len(lower_words) > 2:
        return []
    out = []
    for part in re.split(r"\s*\+\s*|;", s):
        part = part.strip(" ,.")
        if not part:
            continue
        m = _match_cached(part, corpus)
        if m["cids"] or m["pgpids"]:
            out.append(m)
    return out


# ===================================================================================== headings


@dataclass
class Heading:
    """A detected document heading on a page."""

    line: int
    label: str
    sms: List[Dict[str, Any]]
    title: str = ""


def _nonempty(lines: List[str]) -> List[Tuple[int, str]]:
    """Index non-empty stripped lines.

    :param lines: Raw lines.
    :returns: ``[(index, stripped line), ...]``.
    """
    return [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]


def h_roman_volume_number(lines: List[str], corpus: Corpus) -> List[Heading]:
    """``II, 30 Title`` (Roman volume, number, title), then a shelfmark line within 6 lines.

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings.
    """
    ne = _nonempty(lines)
    out = []
    for k, (i, ln) in enumerate(ne):
        m = re.match(r"^\{?([IVX]{1,3}), ?(\d{1,3}[a-z]?)\}?\s+(?![\-–]\s?\d)([A-Z{\[].{5,})$", ln)
        if not m:
            m2 = re.fullmatch(r"\{?([IVX]{1,3}), ?(\d{1,3}[a-z]?)\}?", ln)
            if m2 and k + 1 < len(ne):
                nxt = next((x for _, x in ne[k + 1:k + 4] if not x.startswith("SECTION") and not x.isupper()), "")
                m = re.match(r"^([IVX]{1,3}), ?(\d{1,3}[a-z]?)\s+(.*)$", f"{m2.group(1)}, {m2.group(2)} " + nxt)
        if not m or re.match(r"^[IVX]{1,3}, ?\d+[a-z]?\s?[\-–]", ln):
            continue
        for _, nx in ne[k + 1:k + 7]:
            sms = sm_line(nx, corpus)
            if sms:
                out.append(Heading(i, f"{m.group(1)}, {m.group(2)}", sms, m.group(3)[:80]))
                break
    return out


def h_number_material(lines: List[str], corpus: Corpus) -> List[Heading]:
    """An integer document-number line with a shelfmark line and a material / ``Text`` line in one window.

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings (``label-only`` title when no shelfmark line but a dated material line).
    """
    ne = _nonempty(lines)
    out: List[Heading] = []
    used: Set[int] = set()
    for k, (i, ln) in enumerate(ne):
        if not re.fullmatch(r"\d{1,3}", ln) or not 1 <= int(ln) <= 160:
            continue
        win = ne[max(0, k - 6):k + 7]
        smw = [(j, sm_line(x, corpus)) for j, x in win if j != i]
        smw = [(j, s) for j, s in smw if s]
        mat = any(re.search(r"(?i)^(paper|parchment|vellum)\b|\bcm\.|^text$", x) for _, x in ne[max(0, k - 4):k + 9])
        dated = any(re.search(r"A\.H\.|A\.D\.|century", x) for _, x in ne[k + 1:k + 4])
        if smw and mat and i not in used:
            j0 = min([i] + [j for j, _ in smw])
            out.append(Heading(j0, ln, smw[0][1], ""))
            used.add(i)
        elif mat and dated and i not in used:
            out.append(Heading(max(0, ne[max(0, k - 2)][0]), ln, [], "label-only"))
            used.add(i)
    return out


def h_document_n_colon(lines: List[str], corpus: Corpus) -> List[Heading]:
    """``DOCUMENT n: SHELFMARK`` (shelfmark may continue on the next line; ``AND`` joins several).

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings.
    """
    out = []
    ne = _nonempty(lines)
    for k, (i, ln) in enumerate(ne):
        m = re.match(r"^DOCUMENTS? (\d+(?:\s?[-–]\s?\d+)?): ?(.*)$", ln)
        if m:
            sm_txt = m.group(2)
            if (not sm_txt or sm_txt.rstrip().endswith("AND")) and k + 1 < len(ne):
                sm_txt = sm_txt + " " + ne[k + 1][1]
            sms: List[Dict[str, Any]] = []
            for piece in re.split(r"\s+AND\s+|,\s+(?=[A-Z])", sm_txt):
                sms += sm_line(piece, corpus)
            out.append(Heading(i, f"DOCUMENT {m.group(1).replace(' ', '')}", sms, ""))
    return out


def h_no_n_caps_title(lines: List[str], corpus: Corpus) -> List[Heading]:
    """``No. 1. PLACE, COUNTRY, DATE`` (an upper-case title after the number).

    :param lines: Page lines.
    :param corpus: Corpus (unused; uniform signature).
    :returns: Headings.
    """
    out = []
    for i, ln in _nonempty(lines):
        m = re.match(r"^No\.\s?(\d{1,3})\.\s+([A-ZĀ-ŽḤṢṬẒȘ][A-ZĀ-ŽḤṢṬẒȘ ,'’\-\.()0-9/&;:?]{6,})$", ln)
        if m:
            out.append(Heading(i, m.group(1), [], m.group(2)[:80]))
    return out


def h_number_shelfmark_dims_he(lines: List[str], corpus: Corpus) -> List[Heading]:
    """A three-digit number line, a shelfmark line, then a dimensions line with ``מ"מ``.

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings.
    """
    ne = _nonempty(lines)
    out = []
    for k, (i, ln) in enumerate(ne):
        if not re.fullmatch(r"\d{3}", ln):
            continue
        nxt = ne[k + 1:k + 3]
        for kk, (_, x) in enumerate(nxt):
            sms = sm_line(x, corpus)
            if sms and any('מ"מ' in y or "מ״מ" in y for _, y in ne[k + 1 + kk + 1:k + 1 + kk + 4]):
                out.append(Heading(i, ln, sms, ""))
                break
    return out


def h_number_caps_title(lines: List[str], corpus: Corpus) -> List[Heading]:
    """An integer line then an ALL-CAPS title line (or both on one line).

    :param lines: Page lines.
    :param corpus: Corpus (unused).
    :returns: Headings.
    """
    ne = _nonempty(lines)
    out = []
    for k, (i, ln) in enumerate(ne[:-1]):
        m1 = re.fullmatch(r"(\d{1,2})\s+([A-Z][A-Z ,.\-'’]{8,})", ln)
        if m1 and "SETTING" not in ln:
            out.append(Heading(i, m1.group(1), [], m1.group(2)[:80]))
            continue
        if not re.fullmatch(r"\d{1,2}", ln):
            continue
        t = ne[k + 1][1]
        letters = [c for c in t if c.isalpha()]
        if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.85 and "SETTING" not in t:
            out.append(Heading(i, ln, [], t[:80]))
    return out


def h_roman_caps_title(lines: List[str], corpus: Corpus) -> List[Heading]:
    """``XXII. A BUSINESS LETTER`` in the first six lines (also the running head).

    :param lines: Page lines.
    :param corpus: Corpus (unused).
    :returns: Headings.
    """
    out = []
    for i, ln in _nonempty(lines)[:6]:
        m = re.match(r"^([IVXL]{1,7})\.\s+([A-Z][A-Z ,\-'’]{3,}[A-Z])\s*\d*$", ln)
        if m and roman_to_int(m.group(1)):
            out.append(Heading(i, m.group(1), [], m.group(2)[:80]))
    return out


def h_series_code_title(lines: List[str], corpus: Corpus) -> List[Heading]:
    """``C.1.d: Title`` in the first eight lines, then a shelfmark line within 5 lines.

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings.
    """
    ne = _nonempty(lines)
    out = []
    for k, (i, ln) in enumerate(ne[:8]):
        m = re.match(r"^([CPWL]\.\d{1,2}(?:\.[a-z])?): (.+)$", ln)
        if not m:
            continue
        sms: List[Dict[str, Any]] = []
        for _, nx in ne[k + 1:k + 6]:
            sms = sm_line(nx, corpus)
            if sms:
                break
        out.append(Heading(i, m.group(1), sms, m.group(2)[:80]))
    return out


def h_number_alef_title_he(lines: List[str], corpus: Corpus) -> List[Heading]:
    """An integer next to an ``א`` series line, a Hebrew deed-type title and a shelfmark line within 6 lines.

    :param lines: Page lines.
    :param corpus: Corpus.
    :returns: Headings labelled ``A-<n>``.
    """
    ne = _nonempty(lines)
    out = []
    for k, (i, ln) in enumerate(ne):
        if not re.fullmatch(r"\d{1,3}", ln):
            continue
        adj = [x for _, x in ne[max(0, k - 1):k]] + [x for _, x in ne[k + 1:k + 2]]
        if not any(re.fullmatch(r"[-–]?\s?א\s?[-–]?", x) for x in adj):
            continue
        label = f"A-{ln}"
        win = ne[k + 1:k + 7]
        title = next((x for _, x in win if re.search(r"(הסכם|שטר|כתב|תנאי|שידוכין|אירוסין|קידושין)", x)), "")
        sms = next((s for _, x in win for s in [sm_line(x, corpus)] if s), [])
        if title and sms:
            out.append(Heading(i, label, sms, title[:60]))
    return out


#: Heading layouts a work config may name.
HEADING_STYLES: Dict[str, Callable[[List[str], Corpus], List[Heading]]] = {
    "roman_volume_number": h_roman_volume_number,
    "number_material": h_number_material,
    "document_n_colon": h_document_n_colon,
    "no_n_caps_title": h_no_n_caps_title,
    "number_shelfmark_dims_he": h_number_shelfmark_dims_he,
    "number_caps_title": h_number_caps_title,
    "roman_caps_title": h_roman_caps_title,
    "series_code_title": h_series_code_title,
    "number_alef_title_he": h_number_alef_title_he,
}


def rh_no_n(lines: List[str]) -> Optional[str]:
    """Running head ``No. N`` in the first 4 non-empty lines.

    :param lines: Page lines.
    :returns: Label or None.
    """
    for _, ln in _nonempty(lines)[:4]:
        m = re.fullmatch(r"[•·\s]*No\.\s?(\d{1,3})", ln)
        if m:
            return m.group(1)
    return None


def rh_he_series_numbers(lines: List[str]) -> Optional[str]:
    """Hebrew running head ``כתבי גניזה מס' 334-333`` → comma list of the document numbers on the spread.

    :param lines: Page lines.
    :returns: ``"n1,n2,..."`` or None.
    """
    for _, ln in _nonempty(lines)[:8]:
        m = re.search(r"כתבי?\s+גניזה\s+מס['׳]\s*([\d,\-–\s]+)", ln)
        if m:
            nums = [int(x) for x in re.findall(r"\d{3}", m.group(1))]
            if len(nums) == 2 and abs(nums[0] - nums[1]) < 6:
                nums = list(range(min(nums), max(nums) + 1))
            return ",".join(str(n) for n in nums) if nums else None
    return None


#: Running-head layouts a work config may name.
RUNHEAD_STYLES: Dict[str, Callable[[List[str]], Optional[str]]] = {"no_n": rh_no_n, "he_series_numbers": rh_he_series_numbers}


# ===================================================================================== crosswalk


def _loc_doc_n(loc: str) -> Optional[str]:
    """``Doc. 12`` → ``12``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"(?i)^doc\.?\s*(\d+)", loc)
    return m.group(1) if m else None


def _loc_hash_n(loc: str) -> Optional[str]:
    """``#012`` → ``12``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"^#0*(\d+)", loc)
    return m.group(1) if m else None


def _loc_doc_hash_n(loc: str) -> Optional[str]:
    """``Doc. #12`` / ``doc 12`` → ``12``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"(?i)^doc\.?\s*#?(\d+)", loc)
    return m.group(1) if m else None


def _loc_roman_dash_number(loc: str) -> Optional[str]:
    """``II-30a`` → ``II, 30a``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"^([IVX]{1,3})-(\d+[a-z]?)$", loc)
    return f"{m.group(1)}, {m.group(2)}" if m else None


def _loc_doc_roman(loc: str) -> Optional[str]:
    """``Doc. XXII`` → ``XXII``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"(?i)^doc\.?\s*([IVXL]+)$", loc)
    return m.group(1) if m else None


def _loc_doc_a_n(loc: str) -> Optional[str]:
    """``Doc. A-12`` → ``A-12``.

    :param loc: PGP footnote location.
    :returns: Label or None.
    """
    m = re.match(r"(?i)^doc\.?\s*A-(\d+)", loc)
    return f"A-{m.group(1)}" if m else None


#: Parsers of a PGP footnote location into a work's own document label.
LOCATION_LABEL_STYLES: Dict[str, Callable[[str], Optional[str]]] = {
    "doc_n": _loc_doc_n, "hash_n": _loc_hash_n, "doc_hash_n": _loc_doc_hash_n,
    "roman_dash_number": _loc_roman_dash_number, "doc_roman": _loc_doc_roman, "doc_a_n": _loc_doc_a_n,
}


def label_from_location(cfg: WorkCfg, loc: str) -> Optional[str]:
    """Parse a PGP footnote location into the work's own document label.

    :param cfg: Work rules.
    :param loc: PGP location string.
    :returns: Label or None (no parser configured / no match).
    """
    if not cfg.location_label:
        return None
    return LOCATION_LABEL_STYLES[cfg.location_label](loc.strip())


def build_crosswalk(corpus: Corpus, config: SpanConfig) -> Dict[str, Dict[str, Set[str]]]:
    """Map (work, own label) → pgpids from PGP footnote locations and historic shelfmarks.

    :param corpus: Corpus (``fn_rows``, ``fragments``).
    :param config: Span config.
    :returns: ``{work: {label: {pgpid}}}``.
    """
    cw: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    with_slugs = [(w, c) for w, c in config.works.items() if c.pgp_source_slugs]
    for r in corpus.fn_rows:
        for work, cfg in with_slugs:
            if any(r["slug"].startswith(s) for s in cfg.pgp_source_slugs):
                lab = label_from_location(cfg, r["location"])
                if lab:
                    cw[work][lab].add(r["pgpid"])
    hist = [(w, re.compile(c.historic_label_re)) for w, c in config.works.items() if c.historic_label_re]
    if hist:
        for f in corpus.fragments:
            for work, rx in hist:
                for m in rx.finditer(f.get("historic") or ""):
                    cw[work][m.group(1)] |= set(f["pgpids"])
    return cw


def page_citations(corpus: Corpus, config: SpanConfig) -> Dict[str, Dict[int, Set[str]]]:
    """Map (work, printed page) → pgpids whose PGP footnote cites that page of the work.

    :param corpus: Corpus.
    :param config: Span config (works with ``pgp_source_slugs`` and ``page_cites``).
    :returns: ``{work: {page: {pgpid}}}``.
    """
    out: Dict[str, Dict[int, Set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    with_slugs = [(w, c) for w, c in config.works.items() if c.pgp_source_slugs and c.page_cites]
    for r in corpus.fn_rows:
        for work, cfg in with_slugs:
            if not any(r["slug"].startswith(s) for s in cfg.pgp_source_slugs):
                continue
            loc = re.sub(r"(?i)^(doc\.?\s*#?[\w\-]+|#\s?\d+[a-z]?|[IVX]{1,3}-\d+[a-z]?(-\d+)?),?\s*", "",
                         r["location"].strip())
            if not loc:
                continue
            for m in re.finditer(r"(\d{1,3})(?:\s?[\-–]\s?(\d{1,3}))?", re.sub(r"n\.?\s?\d+", "", loc)):
                a = int(m.group(1))
                b_raw = m.group(2)
                if b_raw:
                    b = int(b_raw)
                    if b < a:
                        b = int(str(a)[: len(str(a)) - len(str(b))] + str(b))
                else:
                    b = a
                if 0 < a <= b <= a + 30:
                    for pnum in range(a, b + 1):
                        out[work][pnum].add(r["pgpid"])
    return out


# ===================================================================================== spans


@dataclass
class Span:
    """A document treatment across consecutive pages."""

    book: str
    part: str
    label: str
    title: str
    heading_sms: List[Dict[str, Any]]
    pages: List[str] = field(default_factory=list)
    page_nums: List[int] = field(default_factory=list)
    chars: collections.Counter = field(default_factory=collections.Counter)
    langs: collections.Counter = field(default_factory=collections.Counter)
    state: str = "intro"
    seen_edition: bool = False
    chapter_head: str = ""
    per_page: Dict[str, collections.Counter] = field(default_factory=dict)
    start_line: int = 0
    text_head: str = ""
    heads: collections.Counter = field(default_factory=collections.Counter)
    samples: Dict[str, List[str]] = field(default_factory=lambda: collections.defaultdict(list))
    first_footnote: str = ""


def page_head(lines: List[str]) -> str:
    """First non-numeric short line (running head) of a page.

    :param lines: Page lines.
    :returns: Head or ''.
    """
    for _, ln in _nonempty(lines)[:3]:
        if not re.fullmatch(r"[\dxivlXIVL ]+", ln) and len(ln) < 60:
            return ln
    return ""


def classify_lines(span: Span, cfg: WorkCfg, page: Any, lines: List[str], a: int, b: int, body_end_line: int) -> None:
    """Add a page slice's characters to the span's section counters (intro / edition / translation / notes).

    :param span: Span to update.
    :param cfg: Work rules.
    :param page: Page.
    :param lines: Page lines.
    :param a: First line index (inclusive).
    :param b: Last line index (exclusive).
    :param body_end_line: First line of the footnote region.
    :returns: None.
    """
    pp = span.per_page.setdefault(page.page_file, collections.Counter())
    buf_latin: List[str] = []
    head = page_head(lines)
    for i in range(a, b):
        ln = lines[i].strip()
        if not ln:
            continue
        if i < a + 3 and (re.fullmatch(r"[\dxivlXIVL{}\[\] ]+", ln) or ln == head or re.fullmatch(r"[•·\s]*No\.\s?\d+", ln)):
            continue
        new_state = None
        for rx, st in cfg.markers:
            if re.match(rx, ln):
                new_state = st
                break
        if new_state:
            span.state = new_state
            continue
        if i >= body_end_line:
            sec = "notes"
        elif _pg.is_rtl_line(ln):
            if cfg.ja_he_split:
                # regest (Hebrew) before the first side marker; then JA edition vs Hebrew translation per line
                h, ar, _ = _pg.letter_counts(ln)
                kind = _pg.ja_or_he(ln)
                if span.state == "intro":
                    sec = "edition" if ar > h else "intro"
                elif kind == "he" or (kind == "?" and re.match(r"^\(?\d{1,2}\)", ln)):
                    sec = "translation"
                else:
                    sec = "edition"
                    span.seen_edition = True
            elif cfg.hebrew_prose:
                kind = _pg.ja_or_he(ln)
                h, ar, _ = _pg.letter_counts(ln)
                if span.state == "translation":
                    sec = "translation"
                elif ar > h or kind == "ja" or span.state == "edition":
                    sec = "edition"
                    span.seen_edition = True
                elif span.seen_edition:
                    sec = "translation"
                else:
                    sec = "intro"
            else:
                sec = "edition"
                span.seen_edition = True
        else:
            if cfg.first_number_line_starts_edition and span.state == "intro" and re.fullmatch(r"1", ln):
                span.state = "edition"
                continue
            if span.state == "edition" and not cfg.hebrew_prose and cfg.latin_after_edition_is_translation and len(ln) > 25:
                sec = "translation"
            elif span.state == "edition":
                wl = re.findall(r"[a-z]+", ln.lower())
                sec = "notes" if sum(w in _pg.EN_SW for w in wl) >= 2 else "edition"
            elif cfg.latin_after_edition_is_translation and span.seen_edition and span.state == "intro":
                sec = "translation"
            else:
                sec = span.state
            if cfg.docness_translation and sec == "intro":
                buf_latin.append(ln)
                continue
        span.chars[sec] += len(ln)
        pp[sec] += len(ln)
        if sum(len(x) for x in span.samples[sec]) < 1500:
            span.samples[sec].append(ln)
    if not span.first_footnote:
        span.first_footnote = next((f for f in _pg.footnote_texts(page.data)
                                    if len(f) >= 60 and len(re.findall(r"\(\d{1,2}\)", f)) < 2), "")
    if buf_latin:
        # no 'Translation' marker: decide per block of Latin lines by docness
        blocks = _pg.segment_blocks("\n".join(buf_latin))
        for bl in blocks:
            sec = "translation" if (span.chars["intro"] > 300 and _pg.docness(bl.text) >= 1.0) else "intro"
            span.chars[sec] += len(bl.text.replace("\n", ""))
            pp[sec] += len(bl.text.replace("\n", ""))
            if sum(len(x) for x in span.samples[sec]) < 1500:
                span.samples[sec].extend(bl.lines)
    span.langs[_pg.lang_of("\n".join(lines[a:b]))] += 1


def _label_value(label: str) -> int:
    """Numeric value of a label for the monotone rule (Roman numerals or the digits it contains).

    :param label: Document label.
    :returns: Integer (0 when it has no digits).
    """
    return roman_to_int(label) if re.fullmatch(r"[IVXL]+", label) else int(re.sub(r"\D", "", label) or 0)


def detect_spans(pages: Sequence[Any], corpus: Corpus, config: SpanConfig) -> List[Span]:
    """Detect document spans in the works that have a heading style.

    :param pages: All pages (``book``, ``part``, ``page_file``, ``idx``, ``text``, ``data``).
    :param corpus: Corpus.
    :param config: Span config.
    :returns: Spans in detection order.
    """
    spans: List[Span] = []
    cfgs = config.span_works()
    by_part: Dict[Tuple[str, str], List[Any]] = collections.defaultdict(list)
    for p in pages:
        if p.book in cfgs:
            by_part[(p.book, p.part)].append(p)
    for (book, part), plist in sorted(by_part.items()):
        cfg = cfgs[book]
        heading_fn = HEADING_STYLES[cfg.heading]  # type: ignore[index]
        runhead_fn = RUNHEAD_STYLES[cfg.runhead] if cfg.runhead else None
        end_rx = GENERIC_END + ("|" + cfg.end_re if cfg.end_re else "")
        cur: Optional[Span] = None
        last_num = -1
        seen_labels: Set[str] = set()
        last_chapter = ""
        freq: collections.Counter = collections.Counter()
        if cfg.drop_frequent_titles:
            for q in plist:
                for ln in {x.strip() for x in q.text.split("\n") if x.strip()}:
                    freq[ln] += 1

        def close() -> None:
            """Close the current span (prefixing its label with its dominant chapter head when configured).

            :returns: None.
            """
            nonlocal cur
            if cur is not None and cur.pages:
                if cfg.chapter_prefixed_labels and cur.heads:
                    cur.label = f"{cur.heads.most_common(1)[0][0]}: {cur.label.split(': ')[-1]}"
                spans.append(cur)
            cur = None

        for p in sorted(plist, key=lambda x: x.idx):
            text = p.text
            lines = text.split("\n")
            offs = _pg.line_offsets(text)
            fstart = _pg.footnote_region_start(text, p.data)
            body_end_line = len(lines)
            if fstart is not None:
                body_end_line = next((i for i, o in enumerate(offs) if o >= fstart), len(lines))
            hs = heading_fn(lines, corpus)
            if sum(1 for ln in lines if re.search(r"\.{4,}\s*\d+\s*$", ln)) >= 2:
                hs = []  # table of contents page
            hs = [h for h in hs if not re.search(r"\.{4,}", lines[h.line])]
            if cfg.drop_frequent_titles:
                hs = [h for h in hs if freq.get(h.title.strip(), 0) < 3]
            head = page_head(lines)
            if head in cfg.chapter_heads:
                last_chapter = head
            if cfg.chapter_prefixed_labels:
                for h in hs:
                    h.label = f"{last_chapter}: {h.label}"
            hs = [h for h in hs if not (h.label in seen_labels and (cur is None or h.label != cur.label))]
            if cfg.monotone:
                keep = []
                for h in hs:
                    v = _label_value(h.label)
                    if cur is not None and h.label == cur.label:
                        continue  # running head repeating the current label
                    if h.title == "label-only" and v != last_num + 1:
                        continue
                    if v > last_num and (v - last_num <= 40 if last_num >= 0 else v <= cfg.first_max):
                        keep.append(h)
                        last_num = v
                hs = keep
            else:
                hs = [h for h in hs if not (cur is not None and h.label == cur.label and not h.sms)]
            # running-head label switch (no explicit heading on this page)
            if runhead_fn is not None and cur is not None:
                rl = runhead_fn(lines)
                if rl and ("," in rl or cfg.runhead_is_set):
                    if cur.label not in rl.split(",") and not hs:
                        close()
                elif not hs and rl and rl != cur.label:
                    close()
            # chapter change / page cap
            if cur is not None:
                if cfg.chapter_heads and head in cfg.chapter_heads and cur.chapter_head and head != cur.chapter_head:
                    close()
                elif len(cur.pages) >= cfg.max_pages:
                    close()
            end_line = None
            for i, ln in _nonempty(lines):
                if re.match(end_rx, ln) and (not hs or i < hs[0].line):
                    end_line = i
                    break
            first_h = hs[0].line if hs else len(lines)
            if cur is not None:
                stop = min(first_h, end_line if end_line is not None else len(lines))
                classify_lines(cur, cfg, p, lines, 0, stop, body_end_line if not hs else max(body_end_line, len(lines)))
                cur.pages.append(p.page_file)
                cur.page_nums.extend(page_number(p))
                if head in cfg.chapter_heads and stop > 3:
                    cur.heads[head] += 1
                if end_line is not None and end_line < first_h:
                    close()
            for k, h in enumerate(hs):
                if cur is not None and h.label == cur.label:
                    continue
                seen_labels.add(h.label)
                close()
                nxt = hs[k + 1].line if k + 1 < len(hs) else len(lines)
                cur = Span(book, part, h.label, h.title, h.sms, chapter_head=head if head in cfg.chapter_heads else "",
                           start_line=h.line)
                cur.text_head = "\n".join(lines[h.line:h.line + 60])
                classify_lines(cur, cfg, p, lines, h.line, nxt, body_end_line if k + 1 == len(hs) else len(lines))
                cur.pages.append(p.page_file)
                cur.page_nums.extend(page_number(p))
                if head in cfg.chapter_heads:
                    cur.heads[head] += 1
                if cur.chapter_head == "" and cfg.chapter_heads:
                    cur.chapter_head = head if head in cfg.chapter_heads else ""
        close()
    return spans


def _int_key(x: str) -> int:
    """Sort key for pgpids (non-digits last).

    :param x: pgpid.
    :returns: Integer.
    """
    return int(x) if str(x).isdigit() else 1 << 62


def resolve_span(sp: Span, cw: Dict[str, Dict[str, Set[str]]], pages_by_key: Dict[Tuple[str, str, str], Any],
                 corpus: Corpus, cites: Dict[str, Dict[int, Set[str]]], config: SpanConfig) -> Dict[str, Any]:
    """Resolve a span's shelfmarks / pgpids / merged ids and its join flags.

    :param sp: Span.
    :param cw: Label crosswalk (:func:`build_crosswalk`).
    :param pages_by_key: ``(book, part, page_file)`` → page.
    :param corpus: Corpus.
    :param cites: Page citation map (:func:`page_citations`).
    :param config: Span config.
    :returns: Span row (``document_treatments`` schema; ``read_models`` replaces the per-reader flags).
    """
    cfg = config.get(sp.book) or WorkCfg(sp.book)
    sm_pg: Set[str] = set()
    sm_cids: Set[str] = set()
    sms: List[str] = []
    for m in sp.heading_sms:
        sms.append(m["norm"])
        sm_pg |= set(m["pgpids"])
        sm_cids |= set(m["cids"])
    lab_pg = set(cw.get(sp.book, {}).get(sp.label, set()))
    basis = "heading_shelfmark" if sms else "none"
    verified: Optional[bool] = None
    if sms and lab_pg:
        conflict = not (sm_pg & lab_pg)
        verified = not conflict
        pg = (sm_pg & lab_pg) or sm_pg
        basis = "heading_shelfmark+label_crosswalk" if not conflict else "heading_shelfmark(label_crosswalk_conflict)"
    elif sms:
        pg = sm_pg
    elif lab_pg:
        pg = lab_pg
        basis = "label_crosswalk"
        # verify: one of the PGP shelfmarks is located on the span's pages
        first_txt = "\n".join(pages_by_key[(sp.book, sp.part, pf)].text for pf in sp.pages)
        hit = False
        for p in lab_pg:
            psm = corpus.docs.get(p, {}).get("shelfmark", "")
            for piece in re.split(r"\s\+\s", psm):
                if piece and _sm.locate(piece, first_txt):
                    hit = True
                    sms.append(_sm.prep_shelfmark(piece))
                    sm_cids |= set(_sm.match_shelfmark(piece, corpus)["cids"])
        verified = None if cfg.trust_label_crosswalk else hit
        if cfg.trust_label_crosswalk:
            for p in lab_pg:
                psm = corpus.docs.get(p, {}).get("shelfmark", "")
                sms.append(psm)
                for piece in re.split(r"\s\+\s", psm):
                    sm_cids |= set(_sm.match_shelfmark(piece, corpus)["cids"])
    else:
        pg = set()
    for p in pg:
        sm_cids |= corpus.pgp_to_cids.get(p, set())
    flags = _sm.pgp_flags(pg, sm_cids, corpus)
    cited: Set[str] = set()
    for pn in sp.page_nums:
        cited |= cites.get(sp.book, {}).get(pn, set())
    page_cite_agrees = (bool(set(pg) & cited) if cited and pg else None)
    ch = sp.chars
    return {
        "book": sp.book, "part": sp.part, "doc_label": sp.label, "title": sp.title,
        "shelfmarks": sorted(set(x for x in sms if x)), "pgpids": sorted(pg, key=_int_key), "merged_doc_ids": sorted(sm_cids),
        "join_basis": basis, "label_crosswalk_pgpids": sorted(lab_pg, key=_int_key),
        "label_crosswalk_agrees": verified, "pgp_page_citation_agrees": page_cite_agrees, "pages": list(sp.pages),
        "printed_pages": sorted(set(sp.page_nums)),
        "intro_chars": ch["intro"], "translation_chars": ch["translation"], "notes_chars": ch["notes"],
        "edition_chars": ch["edition"],
        "langs": [k for k, _ in sp.langs.most_common() if k != "other"] or ["other"],
        **flags,
        "read_models": sorted(f for f, ids in corpus.read_ids_by_model.items() if any(c in ids for c in sm_cids)),
        "per_page_chars": {k: dict(v) for k, v in sp.per_page.items()},
        "head_excerpt": sp.text_head[:600],
        "section_excerpts": {k: " ".join(v)[:1200] for k, v in sp.samples.items()},
        "first_footnote": sp.first_footnote[:800],
    }


def label_key(cfg: WorkCfg, label: str) -> Optional[Tuple[str, int, str, int]]:
    """Split a document label into ``(series, first number, suffix, last number)`` for gap checks.

    :param cfg: Work rules (``label_series`` widens the accepted forms).
    :param label: Document label.
    :returns: The tuple, or None when labels are not sequential numbers.
    """
    m = re.fullmatch(r"(?:(.*): )?DOCUMENTS? (\d+)(?:[-–](\d+))?", label)
    if m:
        return (m.group(1) or "", int(m.group(2)), "", int(m.group(3) or m.group(2)))
    m = re.fullmatch(r"([IVX]{1,3}), (\d+)([a-z]?)", label)
    if m and cfg.label_series == "roman_comma_number":
        return (m.group(1), int(m.group(2)), m.group(3), int(m.group(2)))
    m = re.fullmatch(r"A-(\d+)", label)
    if m:
        return ("A", int(m.group(1)), "", int(m.group(1)))
    if re.fullmatch(r"[IVXL]+", label) and cfg.label_series == "roman":
        return ("", roman_to_int(label), "", roman_to_int(label))
    if re.fullmatch(r"\d+", label):
        return ("", int(label), "", int(label))
    return None


def flag_boundaries(rows: List[Dict[str, Any]], config: SpanConfig) -> None:
    """Flag spans whose successor label skips a number (an undetected heading was probably absorbed) or that hit the page cap.

    Sets ``hit_page_cap``, ``next_label_gap`` and ``boundary_clean`` on every row.

    :param rows: Span rows in detection order (mutated).
    :param config: Span config.
    :returns: None.
    """
    for i, r in enumerate(rows):
        cfg = config.get(r["book"]) or WorkCfg(r["book"])
        r["hit_page_cap"] = len(r["pages"]) >= cfg.max_pages
        r["next_label_gap"] = None
        k = label_key(cfg, r["doc_label"])
        nxt = rows[i + 1] if i + 1 < len(rows) and rows[i + 1]["book"] == r["book"] and rows[i + 1]["part"] == r["part"] else None
        if k is None or nxt is None:
            continue
        k2 = label_key(cfg, nxt["doc_label"])
        if k2 is None or k2[0] != k[0]:
            continue
        r["next_label_gap"] = (k2[1] - k[3]) > 1
    for r in rows:
        r["boundary_clean"] = not r["hit_page_cap"] and not r["next_label_gap"]


# ===================================================================================== page index


def _kl_roman_comma_number(s: str) -> Optional[str]:
    """``II, 30`` / ``II30`` → ``II, 30``.

    :param s: Cleaned key.
    :returns: Label or None.
    """
    m = re.fullmatch(r"([IVX]{1,3}),?\s?(\d{1,3}[a-z]?)", s)
    return f"{m.group(1)}, {m.group(2)}" if m else None


def _kl_doc_or_no_n(s: str) -> Optional[str]:
    """``Doc 12`` / ``No. 12r`` / ``document #012`` → ``12``.

    :param s: Cleaned key.
    :returns: Label or None.
    """
    m = re.fullmatch(r"(?i)(?:doc(?:ument)?|no)\.?\s?#?(\d{1,3})[rv]?", s)
    return str(int(m.group(1))) if m else None


def _kl_no_or_doc_roman(s: str) -> Optional[str]:
    """``No. xxii`` / ``Doc XXII`` → ``XXII``.

    :param s: Cleaned key.
    :returns: Label or None.
    """
    m = re.fullmatch(r"(?i)(?:no|doc)\.?\s?([IVXL]+)", s)
    return m.group(1).upper() if m else None


#: Parsers of a ``shelf_marks_mentioned`` key that is really a work's own document number.
KEY_LABEL_STYLES: Dict[str, Callable[[str], Optional[str]]] = {
    "roman_comma_number": _kl_roman_comma_number, "doc_or_no_n": _kl_doc_or_no_n, "no_or_doc_roman": _kl_no_or_doc_roman,
}


def key_label(cfg: Optional[WorkCfg], raw: str) -> Optional[str]:
    """Parse a shelfmark key that is really the work's own document number.

    :param cfg: Work rules (None → no parser).
    :param raw: Raw key.
    :returns: Label usable with the crosswalk, or None.
    """
    if cfg is None or not cfg.key_label:
        return None
    s = re.sub(r"^[uxv_]+(?=[A-Z0-9])", "", raw.strip()).replace("_", " ").strip()
    return KEY_LABEL_STYLES[cfg.key_label](s)


def index_page(p: Any, corpus: Corpus, cites: Dict[str, Dict[int, Set[str]]], stats: collections.Counter,
               cw: Dict[str, Dict[str, Set[str]]], config: SpanConfig) -> List[Dict[str, Any]]:
    """Emit index rows (one per distinct shelfmark key) for one page.

    :param p: Page.
    :param corpus: Corpus.
    :param cites: Page citation map.
    :param stats: Counters to update.
    :param cw: Label crosswalk.
    :param config: Span config.
    :returns: Rows (prototype schema; :func:`finalize_rows` adds key, tier and evidence).
    """
    d = p.data
    text = p.text
    toks = _sm.tokens_with_pos(text)
    toks2 = _sm.strip_paren_tokens(toks, text)
    fstart = _pg.footnote_region_start(text, d)
    body_end = fstart if fstart is not None else len(text) + 1
    blocks = _pg.segment_blocks(text, fstart)
    is_idx = _pg.index_like(text) or sum(1 for ln in text.split("\n") if re.search(r"\.{4,}\s*\d+\s*$", ln)) >= 2
    cls = d.get("classification")
    hebrew_book = p.book in config.hebrew_works
    cfg = config.get(p.book)
    fns = _pg.footnote_texts(d)
    trk = _pg.tr_keys(d)
    trk_cores = {k: _sm.match_shelfmark(k, corpus)["core"] for k in trk}
    keys: List[str] = []
    seen: Set[str] = set()
    for k in _pg.sm_keys(d) + trk:
        pk = _sm.prep_shelfmark(k)
        if pk.lower() in seen:
            continue
        seen.add(pk.lower())
        keys.append(k)
    occ_all: Dict[str, List[int]] = {k: _sm.locate(k, text, toks, toks2) for k in keys}
    pnums = page_number(p)
    rows = []
    for k in keys:
        stats["sm_strings"] += 1
        m = _sm.match_shelfmark(k, corpus)
        occ = occ_all[k]
        if not m["cids"] and not m["pgpids"]:
            lab = key_label(cfg, k)
            if lab and cw.get(p.book, {}).get(lab):
                pg = sorted(cw[p.book][lab], key=_int_key)
                cids = sorted({c for x in pg for c in corpus.pgp_to_cids.get(x, set())})
                m = {"norm": f"{p.book}:{lab}", "core": "", "how": "book_label", "cids": cids, "pgpids": pg}
        if m["how"] == "junk" or re.search(r"u[0-9a-f]{4}", k):
            stats["junk_keys"] += 1
            continue
        if m["core"] in _sm.LEAKED_CORES and not occ:
            stats["leaked_dropped"] += 1
            continue
        stats[f"match_{m['how']}"] += 1
        body_occ = [o for o in occ if o < body_end]
        fn_occ = [o for o in occ if o >= body_end]
        fn_dict_hit = [f for f in fns if _sm.locate(k, f)]
        pblocks: List[int] = []
        for o in body_occ:
            for bi, bl in enumerate(blocks):
                if bl.start <= o <= bl.end + 1 and bi not in pblocks:
                    pblocks.append(bi)
        ptxt = "\n".join(blocks[bi].text for bi in pblocks)
        passage_chars = sum(len(blocks[bi].text) for bi in pblocks)
        kinds: Set[str] = set()
        # edition_quote
        # (the prototype also matched an empty core, i.e. a book-label key against any junk transcription key)
        same = [t for t in trk if (m["core"] and trk_cores.get(t) == m["core"])
                or _sm.prep_shelfmark(t).lower() == m["norm"].lower()]
        tr_hit = bool(same) and any(_pg.tr_chars(d, t) >= 20 for t in same)
        follow: List[str] = []
        for bi in pblocks:
            follow.extend(blocks[bi].lines)
            for bj in range(bi + 1, min(bi + 3, len(blocks))):
                if blocks[bj].start < body_end:
                    follow.extend(blocks[bj].lines)
        if tr_hit or _pg.rtl_run_chars(follow, need_ja=hebrew_book) >= 120:
            kinds.add("edition_quote")
        # translation: mention near a block head, followed by document-like prose
        if body_occ and not hebrew_book:
            for bi in pblocks:
                bl = blocks[bi]
                near_head = len(bl.lines) <= 2 or any(bl.start <= o <= bl.start + 160 for o in body_occ)
                if not near_head:
                    continue
                cand = [blocks[bj] for bj in range(bi, min(bi + 4, len(blocks))) if blocks[bj].start < body_end]
                marker = any(re.match(r"^\{?(Translation|TRANSLATION)$", c.text.strip()) for c in cand)
                if marker or any(_pg.docness(c.text) >= 1.0 and _pg.lang_of(c.text) in ("en", "fr", "de") for c in cand):
                    kinds.add("translation")
                    break
        # catalogue entry
        if not is_idx and (cls == "catalog" or (pblocks and _pg.catalogue_like(follow[:12]))):
            kinds.add("catalogue_entry")
        # commentary
        if body_occ and not is_idx and cls != "catalog":
            for bi in pblocks:
                bt = blocks[bi].text
                lg = _pg.lang_of(bt)
                prose_chars = len(bt) if lg in ("en", "fr", "de", "he") else 0
                if prose_chars >= 300 and _pg.docness(bt) < 1.0 and not _pg.catalogue_like(blocks[bi].lines):
                    kinds.add("commentary")
                    break
        if not body_occ and (fn_occ or fn_dict_hit):
            kinds.add("footnote_only")
            src = fn_dict_hit[0] if fn_dict_hit else text[fn_occ[0]:fn_occ[0] + 600]
            ptxt = ptxt or src
            passage_chars = passage_chars or len(src)
        if not kinds:
            kinds.add("passing_mention")
        others = 0
        if pblocks:
            for k2, o2 in occ_all.items():
                if k2 != k and any(blocks[bi].start <= o <= blocks[bi].end for bi in pblocks for o in o2):
                    others += 1
        cited = sorted({pg for pn in pnums for pg in cites.get(p.book, {}).get(pn, set())} & set(m["pgpids"]), key=_int_key)
        rows.append({
            "book": p.book, "part": p.part, "page_file": p.page_file, "extracted_page_number": d.get("extracted_page_number"),
            "classification": cls, "shelfmark_raw": k, "shelfmark_norm": m["norm"], "shelfmark_core": m["core"],
            "match_how": m["how"], "pgpids": list(m["pgpids"]), "merged_doc_ids": list(m["cids"]),
            "kinds": sorted(kinds), "passage_chars": passage_chars,
            "passage_lang": _pg.lang_of(ptxt) if ptxt else "other",
            "located_in_text": bool(occ), "mention_count": len(occ), "passage_other_shelfmarks": others,
            "index_page": is_idx, "pgp_cites_this_page": cited,
            "snippet": re.sub(r"\s+", " ", ptxt)[:200], "via_span": None, "span_clean": None,
        })
        stats["located" if occ else "not_located"] += 1
    return rows


def merge_span_rows(rows: List[Dict[str, Any]], span_rows: Sequence[Dict[str, Any]],
                    pages_by_key: Dict[Tuple[str, str, str], Any], stats: collections.Counter) -> None:
    """Merge span evidence into the index (``build_analysis.main`` / ``litjoin/build_index.py``).

    A span page with ≥ 300 intro+notes chars counts as ``commentary``, ≥ 150 translation chars as
    ``translation``, ≥ 120 edition chars as ``edition_quote``. Index rows on that page naming the
    same document get the kinds (``passing_mention`` / ``footnote_only`` replaced) and ``via_span``;
    pages without such a row get a new ``span_label`` row. ``span_clean`` carries the span's
    ``boundary_clean``.

    :param rows: Index rows (mutated / extended).
    :param span_rows: Resolved span rows (after :func:`flag_boundaries`).
    :param pages_by_key: ``(book, part, page_file)`` → page.
    :param stats: Counters (``span_rows_added`` / ``span_rows_merged``).
    :returns: None.
    """
    by_page: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_page[(r["book"], r["part"], r["page_file"])].append(r)
    added = merged = 0
    for sr in span_rows:
        if not sr["pgpids"] and not sr["merged_doc_ids"]:
            continue
        for pf, pc in sr["per_page_chars"].items():
            key = (sr["book"], sr["part"], pf)
            kinds: Set[str] = set()
            if pc.get("intro", 0) + pc.get("notes", 0) >= 300:
                kinds.add("commentary")
            if pc.get("translation", 0) >= 150:
                kinds.add("translation")
            if pc.get("edition", 0) >= 120:
                kinds.add("edition_quote")
            if not kinds:
                continue
            hit = [r for r in by_page.get(key, [])
                   if set(r["pgpids"]) & set(sr["pgpids"]) or set(r["merged_doc_ids"]) & set(sr["merged_doc_ids"])]
            if hit:
                for r in hit:
                    before = set(r["kinds"])
                    r["kinds"] = sorted((before - {"passing_mention", "footnote_only"}) | kinds) if kinds - before else r["kinds"]
                    r["via_span"] = sr["doc_label"]
                    r["span_clean"] = bool(sr.get("boundary_clean")) or bool(r.get("span_clean"))
                    merged += 1
                continue
            p = pages_by_key[key]
            new = {
                "book": sr["book"], "part": sr["part"], "page_file": pf,
                "extracted_page_number": p.data.get("extracted_page_number"),
                "classification": p.data.get("classification"), "shelfmark_raw": (sr["shelfmarks"] or [""])[0],
                "shelfmark_norm": (sr["shelfmarks"] or [""])[0], "shelfmark_core": "", "match_how": "span_label",
                "pgpids": list(sr["pgpids"]), "merged_doc_ids": list(sr["merged_doc_ids"]), "kinds": sorted(kinds),
                "passage_chars": pc.get("intro", 0) + pc.get("notes", 0) + pc.get("translation", 0),
                "passage_lang": sr["langs"][0], "located_in_text": False, "mention_count": 0,
                "passage_other_shelfmarks": 0, "index_page": False, "pgp_cites_this_page": [],
                "snippet": re.sub(r"\s+", " ", p.text)[:200], "via_span": sr["doc_label"],
                "span_clean": bool(sr.get("boundary_clean")),
            }
            rows.append(new)
            by_page[key].append(new)
            added += 1
    stats["span_rows_added"] = added
    stats["span_rows_merged"] = merged


def tier_of(row: Dict[str, Any]) -> str:
    """Tier of an index row: ``discuss`` (commentary / translation / edition quote), ``catalogue`` or ``mention``.

    Index pages (back-of-book indexes, contents pages) are always ``mention`` (``join.py`` called
    them ``index``; the row keeps ``index_page`` True).

    :param row: Index row.
    :returns: Tier.
    """
    if row.get("index_page"):
        return "mention"
    k = set(row.get("kinds") or [])
    if k & DISC_KINDS:
        return "discuss"
    if "catalogue_entry" in k:
        return "catalogue"
    return "mention"


def evidence_of(row: Dict[str, Any]) -> str:
    """Join evidence of an index row: ``located`` > ``clean_span`` > ``unclean_span`` > ``key_only``.

    :param row: Index row.
    :returns: Evidence label.
    """
    if row.get("located_in_text"):
        return "located"
    if row.get("via_span"):
        return "clean_span" if row.get("span_clean") else "unclean_span"
    return "key_only"


def finalize_rows(rows: List[Dict[str, Any]], pages_by_key: Dict[Tuple[str, str, str], Any]) -> None:
    """Add ``page_key``, ``work``, printed page, ``tier`` and ``evidence`` to index rows (in place).

    :param rows: Index rows.
    :param pages_by_key: ``(book, part, page_file)`` → page.
    :returns: None.
    """
    for r in rows:
        p = pages_by_key.get((r["book"], r["part"], r["page_file"]))
        r["page_key"] = _pg.page_key(r["book"], r["part"], r["page_file"])
        r["work"] = r["book"]
        r["printed_page"] = getattr(p, "printed_page_num", None) if p is not None else None
        r["printed_page_source"] = getattr(p, "printed_page_source", None) if p is not None else None
        r["tier"] = tier_of(r)
        r["evidence"] = evidence_of(r)


def build_index(pages: Sequence[Any], corpus: Corpus, config: SpanConfig
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], collections.Counter]:
    """Build the literature page index over the given pages.

    :param pages: Pages (:func:`pages.iter_pages`).
    :param corpus: Corpus (:func:`shelfmarks.load_corpus`).
    :param config: Span config.
    :returns: ``(index rows, span rows, stats)``.
    """
    clear_caches()
    t0 = time.time()
    pages_by_key = {(p.book, p.part, p.page_file): p for p in pages}
    cw = build_crosswalk(corpus, config)
    cites = page_citations(corpus, config)
    spans = detect_spans(pages, corpus, config)
    span_rows = [resolve_span(s, cw, pages_by_key, corpus, cites, config) for s in spans]
    flag_boundaries(span_rows, config)
    LOG.info("spans: %d in %.1fs", len(span_rows), time.time() - t0)
    stats: collections.Counter = collections.Counter()
    rows: List[Dict[str, Any]] = []
    for p in pages:
        rows.extend(index_page(p, corpus, cites, stats, cw, config))
    merge_span_rows(rows, span_rows, pages_by_key, stats)
    finalize_rows(rows, pages_by_key)
    for sr in span_rows:
        sr["page_keys"] = [_pg.page_key(sr["book"], sr["part"], pf) for pf in sr["pages"]]
    LOG.info("index: %d rows over %d pages in %.1fs", len(rows), len(pages), time.time() - t0)
    return rows, span_rows, stats


def index_report(rows: Sequence[Dict[str, Any]], span_rows: Sequence[Dict[str, Any]], stats: collections.Counter,
                 n_pages: int) -> Dict[str, Any]:
    """Summary counts of an index build.

    :param rows: Index rows (finalised).
    :param span_rows: Span rows.
    :param stats: Build counters.
    :param n_pages: Pages scanned.
    :returns: Report dict (pages, resolved strings by method, tiers, evidence, spans).
    """
    resolved = [r for r in rows if r["pgpids"] or r["merged_doc_ids"]]
    by_how = collections.Counter(r["match_how"] for r in rows)
    by_how_resolved = collections.Counter(r["match_how"] for r in resolved)
    tiers = {t: {"rows": 0, "pages": set(), "pgpids": set(), "merged_ids": set()} for t in TIERS}
    for r in resolved:
        t = tiers[r["tier"]]
        t["rows"] += 1
        t["pages"].add(r["page_key"])
        t["pgpids"].update(r["pgpids"])
        t["merged_ids"].update(r["merged_doc_ids"])
    ev = collections.Counter(r["evidence"] for r in resolved)
    ev_disc = collections.Counter(r["evidence"] for r in resolved if r["tier"] == "discuss")
    strong_pairs = {(p, r["page_key"]) for r in resolved if r["tier"] == "discuss" and r["evidence"] in STRONG_EVIDENCE
                    for p in r["pgpids"]}
    return {
        "pages": n_pages,
        "pages_with_rows": len({r["page_key"] for r in rows}),
        "pages_with_resolved_mention": len({r["page_key"] for r in resolved}),
        "rows": len(rows), "rows_resolved": len(resolved),
        "resolved_pgpids": len({p for r in resolved for p in r["pgpids"]}),
        "resolved_merged_ids": len({c for r in resolved for c in r["merged_doc_ids"]}),
        "strings_by_method": dict(by_how.most_common()),
        "resolved_strings_by_method": dict(by_how_resolved.most_common()),
        "tiers": {k: {"rows": v["rows"], "pages": len(v["pages"]), "pgpids": len(v["pgpids"]),
                      "merged_ids": len(v["merged_ids"])} for k, v in tiers.items()},
        "evidence_resolved_rows": dict(ev), "evidence_discuss_rows": dict(ev_disc),
        "strong_discuss_doc_page_pairs": len(strong_pairs),
        "strong_discuss_pgpids": len({p for p, _ in strong_pairs}),
        "index_page_rows": sum(1 for r in rows if r.get("index_page")),
        "spans": {"total": len(span_rows), "resolved": sum(1 for s in span_rows if s["pgpids"] or s["merged_doc_ids"]),
                  "boundary_clean": sum(1 for s in span_rows if s.get("boundary_clean")),
                  "by_work": dict(collections.Counter(s["book"] for s in span_rows).most_common())},
        "stats": dict(stats),
    }


# ===================================================================================== leak helpers


def words(s: str) -> List[str]:
    """Lower-cased word tokens (``\\w+``; Hebrew / Arabic letters included).

    :param s: Text.
    :returns: Tokens.
    """
    return re.findall(r"\w+", (s or "").lower())


def grams8(s: str, n: int = 8) -> Set[Tuple[str, ...]]:
    """Word n-grams of a text.

    :param s: Text.
    :param n: n-gram length (8 for the leak rule).
    :returns: Set of n-gram tuples.
    """
    ws = words(s)
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def share(target_grams: Set[Tuple[str, ...]], context_grams: Set[Tuple[str, ...]]) -> float:
    """Share of the target's n-grams present in a context.

    :param target_grams: Target n-grams.
    :param context_grams: Context n-grams.
    :returns: Share in [0, 1] (0 for an empty target).
    """
    if not target_grams:
        return 0.0
    return len(target_grams & context_grams) / len(target_grams)


# ===================================================================================== test docs


#: Families of the E6 with/without-literature eval.
E6_FAMILIES: Tuple[str, ...] = ("describe_pgp", "catalogue_json", "translate_en")


def _iter_eval_rows(p: _paths.Paths, families_dir: Optional[Path] = None) -> Tuple[str, Iterable[Dict[str, Any]]]:
    """Rows of the test documents: v0_2 family files (eval splits) once every E6 family file exists, else v0_1 ``test.jsonl``.

    :param p: Build paths.
    :param families_dir: v0_2 ``families`` directory (default ``out_dir/families``).
    :returns: ``(source description, row iterator)``.
    """
    fam = families_dir or p.out("families")
    files = [fam / f"{f}.jsonl" for f in _rows.PGP_FAMILIES if (fam / f"{f}.jsonl").exists()]
    if all((fam / f"{f}.jsonl").exists() for f in E6_FAMILIES):
        def gen() -> Iterable[Dict[str, Any]]:
            """Stream eval-split rows of the v0_2 family files.

            :returns: Row iterator.
            """
            for f in files:
                for r in _rows.read_jsonl(f):
                    if (r.get("meta") or {}).get("split") in ("test_indist", "heldout_vision", "heldout"):
                        yield r
        return "v0_2 families (test_indist/heldout_vision/heldout)", gen()
    return "v0_1 test.jsonl", _rows.read_jsonl(p.text_ft_v0_1 / "test.jsonl")


def build_test_docs(index_rows: Sequence[Dict[str, Any]], eval_rows: Iterable[Dict[str, Any]],
                    page_text: Callable[[str], str], families: Sequence[str] = E6_FAMILIES,
                    leak_max: float = 0.05, source: str = "") -> Dict[str, Any]:
    """Test documents with discuss pages, for the E6 with/without-literature eval.

    Per document: split, vision flag, families present, every discuss page (evidence, kinds), the
    strong ones (``located`` / ``clean_span``), per family the target's 8-gram share on each strong
    page and the leak-safe strong pages (share < ``leak_max``; the edition-condition target is used).

    :param index_rows: Finalised index rows.
    :param eval_rows: Test rows (v0_1 test or v0_2 eval splits).
    :param page_text: Page key → page text.
    :param families: Families E6 pairs with literature.
    :param leak_max: Leak threshold.
    :param source: Description of the test-row source.
    :returns: JSON-ready dict ``{"source", "families", "counts", "docs": [...]}``.
    """
    disc: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for r in index_rows:
        if r["tier"] != "discuss":
            continue
        for pid in r["pgpids"]:
            cur = disc[pid].get(r["page_key"])
            rank = EVIDENCE.index(r["evidence"])
            if cur is None or rank < EVIDENCE.index(cur["evidence"]):
                disc[pid][r["page_key"]] = {"page_key": r["page_key"], "work": r["work"], "printed_page": r.get("printed_page"),
                                            "evidence": r["evidence"], "kinds": sorted(set(r["kinds"]))}
            else:
                cur["kinds"] = sorted(set(cur["kinds"]) | set(r["kinds"]))
    docs: Dict[str, Dict[str, Any]] = {}
    targets: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for row in eval_rows:
        m = row.get("meta") or {}
        pid = str(m.get("pgpid") or "")
        if not pid or pid not in disc:
            continue
        d = docs.setdefault(pid, {"pgpid": pid, "shelfmark": m.get("shelfmark"), "splits": set(), "families": set(),
                                  "held_out_vision_benchmark": False})
        d["splits"].add(m.get("split"))
        d["families"].add(m.get("task"))
        d["held_out_vision_benchmark"] = d["held_out_vision_benchmark"] or bool(m.get("held_out_vision_benchmark"))
        task = m.get("task")
        if task in families:
            pri = 0 if m.get("condition") == "edition" else 1
            tgt = row["messages"][-1]["content"]
            if (task, pid) not in targets or pri < targets[(task, pid)][0]:
                targets[(task, pid)] = (pri, tgt)
    gram_cache: Dict[str, Set[Tuple[str, ...]]] = {}
    out_docs = []
    counts: collections.Counter = collections.Counter()
    for pid in sorted(docs, key=_int_key):
        d = docs[pid]
        pages = sorted(disc[pid].values(), key=lambda x: x["page_key"])
        strong = [x["page_key"] for x in pages if x["evidence"] in STRONG_EVIDENCE]
        shares: Dict[str, Dict[str, float]] = {}
        safe: Dict[str, List[str]] = {}
        for fam in families:
            if (fam, pid) not in targets:
                continue
            tg = grams8(targets[(fam, pid)][1])
            sh = {}
            for k in strong:
                if k not in gram_cache:
                    gram_cache[k] = grams8(page_text(k))
                sh[k] = round(share(tg, gram_cache[k]), 4)
            shares[fam] = sh
            safe[fam] = [k for k in strong if sh[k] < leak_max]
            if safe[fam]:
                counts[f"docs_leak_safe_{fam}"] += 1
        counts["docs_with_discuss"] += 1
        counts["docs_with_strong_discuss"] += bool(strong)
        counts["discuss_pages"] += len(pages)
        counts["strong_pages"] += len(strong)
        out_docs.append({"pgpid": pid, "shelfmark": d["shelfmark"], "splits": sorted(x for x in d["splits"] if x),
                         "held_out_vision_benchmark": d["held_out_vision_benchmark"],
                         "families": sorted(x for x in d["families"] if x), "discuss_pages": pages,
                         "strong_pages": strong, "leak_share": shares, "leak_safe_pages": safe})
    counts["docs_any_leak_safe"] = sum(1 for d in out_docs if any(d["leak_safe_pages"].values()))
    return {"source": source, "families": list(families), "leak_max": leak_max, "counts": dict(counts), "docs": out_docs}


# ===================================================================================== CLI: build_index


def run_build_index(p: Optional[_paths.Paths] = None, works: Optional[Sequence[str]] = None,
                    span_config: Optional[Path] = None, test_docs: bool = True,
                    out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build and write ``rag_index.jsonl``, ``rag_spans.jsonl``, ``rag_index_report.json`` (and ``rag_test_docs.json``).

    :param p: Build paths.
    :param works: Restrict to these works (sampling).
    :param span_config: Explicit span config file.
    :param test_docs: Also write ``rag_test_docs.json``.
    :param out_dir: Output directory (default ``paths.out_dir``).
    :returns: The report dict.
    """
    p = p or _paths.load_paths()
    out = Path(out_dir) if out_dir is not None else p.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    config = load_span_config(span_config, p)
    pages = _pg.iter_pages(root=p.academic_literature, works=works)
    corpus = _sm.load_corpus(p)
    t_load = time.time() - t0
    rows, span_rows, stats = build_index(pages, corpus, config)
    report = index_report(rows, span_rows, stats, len(pages))
    cfg_file = span_config_path(p) if span_config is None else span_config
    report["build_args"] = {"works": list(works) if works else None, "span_config": Path(cfg_file).name,
                            "span_works": len(config.span_works()), "build": _rows.BUILD}
    report["seconds"] = {"load": round(t_load, 1), "total": round(time.time() - t0, 1)}
    n_idx = _rows.write_jsonl(out / "rag_index.jsonl", rows)
    _rows.write_jsonl(out / "rag_spans.jsonl", span_rows)
    if test_docs:
        texts = {pg.page_key: pg.text for pg in pages}
        src, eval_rows = _iter_eval_rows(p)
        td = build_test_docs(rows, eval_rows, lambda k: texts.get(k, ""), source=src)
        _write_json(out / "rag_test_docs.json", td)
        report["test_docs"] = td["counts"]
    _write_json(out / "rag_index_report.json", report)
    LOG.info("wrote %d index rows, %d spans to %s", n_idx, len(span_rows), out)
    return report


def _write_json(path: Path, obj: Any) -> None:
    """Write a JSON file atomically (UTF-8, indented).

    :param path: Destination.
    :param obj: JSON-serialisable object.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=sorted)
    tmp.replace(path)


def run_test_docs(p: Optional[_paths.Paths] = None, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Re-write ``rag_test_docs.json`` from an existing ``rag_index.jsonl`` (e.g. once the v0_2 families exist).

    :param p: Build paths.
    :param out_dir: Directory holding ``rag_index.jsonl`` (default ``paths.out_dir``).
    :returns: The counts block.
    """
    p = p or _paths.load_paths()
    out = Path(out_dir) if out_dir is not None else p.out_dir
    rows = _rows.load_jsonl(out / "rag_index.jsonl")
    wanted = {r["work"] for r in rows if r["tier"] == "discuss"}
    texts = {pg.page_key: pg.text for pg in _pg.iter_pages(root=p.academic_literature, works=wanted, with_ocr=False)}
    src, eval_rows = _iter_eval_rows(p)
    td = build_test_docs(rows, eval_rows, lambda k: texts.get(k, ""), source=src)
    _write_json(out / "rag_test_docs.json", td)
    return td["counts"]


# ===================================================================================== rows: rendering


def hebrew_char_ratio(text: str) -> float:
    """Share of alphabetic characters in the Hebrew block (the site's Hebrew-dominance test).

    :param text: Any text.
    :returns: Ratio in [0, 1] (0 for no letters).
    """
    alphabetic = [c for c in text or "" if c.isalpha()]
    if not alphabetic:
        return 0.0
    return sum(1 for c in alphabetic if "֐" <= c <= "׿") / len(alphabetic)


#: Page-text caps of the site (``min(1800, …)``; Hebrew-dominant pages ``max(500, cap // 3)``).
QUOTE_CAP = 1800
HEBREW_QUOTE_CAP = max(500, QUOTE_CAP // 3)
MAX_SHELF_MARKS = 10
LIT_HEADER = "Literature:"


def citation(author: Optional[str], title: Optional[str], year: Any = None, page: Any = None) -> str:
    """Citation stub ``{author}, *{title}*, {year}, p. {page}`` (year / page omitted when unknown).

    :param author: Author(s) (``Unknown`` when empty, as on the site).
    :param title: Title (``Untitled`` when empty).
    :param year: Publication year.
    :param page: Printed page.
    :returns: Citation text.
    """
    out = f"{(author or '').strip() or 'Unknown'}, *{(title or '').strip() or 'Untitled'}*"
    if year not in (None, "", 0):
        out += f", {year}"
    if page not in (None, ""):
        out += f", p. {page}"
    return out


def render_source(n: int, cite: str, page_text: str, shelf_marks: Sequence[str]) -> Tuple[str, int]:
    """One source in the site's literature-source layout (without its generated-summary line).

    :param n: Source number (1-based).
    :param cite: Citation stub (:func:`citation`).
    :param page_text: Page text (project-name-stripped by the caller); capped here.
    :param shelf_marks: Located shelfmarks (≤ 10 shown).
    :returns: ``(prompt text, quoteable chars used)``.
    """
    text = (page_text or "").strip()
    cap = HEBREW_QUOTE_CAP if hebrew_char_ratio(text) > 0.5 else QUOTE_CAP
    quote = text[:cap]
    parts = [f"[SOURCE {n}] {cite}"]
    if quote:
        parts.append(f"Original page text (quoteable): {quote}")
    if shelf_marks:
        parts.append("Shelf marks cited in this source: " + ", ".join(str(s) for s in list(shelf_marks)[:MAX_SHELF_MARKS]))
    return "\n".join(parts), len(quote)


def render_block(sources: Sequence[Tuple[str, str, Sequence[str]]]) -> Tuple[str, List[int]]:
    """The ``Literature:`` block: numbered sources joined by newlines (the site joins prompt texts with ``\\n``).

    :param sources: ``[(citation, page text, shelf marks), ...]`` in display order.
    :returns: ``(block text, quoteable chars per source)``.
    """
    texts, chars = [], []
    for i, (cite, text, sms) in enumerate(sources, start=1):
        t, c = render_source(i, cite, text, sms)
        texts.append(t)
        chars.append(c)
    return LIT_HEADER + "\n" + "\n".join(texts), chars


# ===================================================================================== rows: inputs


def load_literature_manifest(p: Optional[_paths.Paths] = None, path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Work metadata (author, title, year) from the local literature manifest (module D).

    Accepts a list of entries, ``{"works": [...]}`` or ``{work_dir: entry}``; each entry carries
    ``work_dir`` (or is keyed by it), ``author``, ``title``, ``year`` and optionally ``parts`` →
    per-part overrides.

    :param p: Build paths.
    :param path: Explicit manifest file (default ``paths.json``'s ``literature_manifest`` or the file next to it).
    :returns: ``{work_dir: entry}`` (empty when the manifest is absent).
    """
    if path is None:
        p = p or _paths.load_paths()
        explicit = p.get("literature_manifest")
        base = Path(p.source_file).parent if p.source_file else p.repo / _paths.DEFAULT_PATHS_FILE.parent
        path = Path(explicit) if explicit else base / "literature_manifest.json"
    if not Path(path).exists():
        LOG.warning("literature manifest %s not found: citations fall back to Unknown/Untitled", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw.get("works", raw) if isinstance(raw, dict) else raw
    if isinstance(items, dict):
        return {k: dict(v, work_dir=v.get("work_dir", k)) for k, v in items.items() if isinstance(v, dict)}
    return {e["work_dir"]: e for e in items if isinstance(e, dict) and e.get("work_dir")}


def work_meta(manifest: Dict[str, Dict[str, Any]], work: str, part: str = "") -> Tuple[Optional[str], Optional[str], Any]:
    """Author, title and year of a work (part override first).

    :param manifest: :func:`load_literature_manifest` output.
    :param work: Work directory.
    :param part: Part path.
    :returns: ``(author, title, year)``.
    """
    e = manifest.get(work) or {}
    pe = (e.get("parts") or {}).get(part) or {}
    return (pe.get("author") or e.get("author"), pe.get("title") or e.get("title"), pe.get("year") or e.get("year"))


@dataclass
class NonTrain:
    """Which documents a training page must not mention (dev / test / held-out / probe).

    ``pgpids`` and ``canonical_ids`` are direct sets; ``holdouts`` is an optional
    ``holdouts.Holdouts`` object asked through ``is_benchmark`` / ``is_eval_doc``.
    """

    pgpids: Set[str] = field(default_factory=set)
    canonical_ids: Set[str] = field(default_factory=set)
    holdouts: Any = None
    displays: Dict[str, str] = field(default_factory=dict)

    def reason(self, pgpids: Iterable[str], cids: Iterable[str]) -> Optional[str]:
        """Why a mention is non-train, or None.

        :param pgpids: The mention's pgpids.
        :param cids: The mention's merged ids.
        :returns: Reason string or None.
        """
        pgpids = [str(x) for x in pgpids]
        cids = list(cids)
        if set(pgpids) & self.pgpids:
            return "nontrain_pgpid"
        if set(cids) & self.canonical_ids:
            return "nontrain_canonical"
        h = self.holdouts
        if h is None:
            return None
        for meth in ("is_benchmark", "is_eval_doc"):
            fn = getattr(h, meth, None)
            if fn is None:
                continue
            for pid in pgpids:
                r = fn(pid, None, None)
                if r:
                    return f"{meth}:{r}" if isinstance(r, str) else meth
            for c in cids:
                r = fn(None, self.displays.get(c), c)
                if r:
                    return f"{meth}:{r}" if isinstance(r, str) else meth
        return None


def _load_holdouts(p: Optional[_paths.Paths] = None) -> Any:
    """``holdouts.Holdouts.load()`` when module A and its ``holdouts.json`` are available.

    :param p: Build paths (``holdouts.json`` sits in ``out_dir``).
    :returns: The object, or None.
    """
    try:
        from src.finetuning.text27b import holdouts as _ho  # noqa: WPS433 (optional sibling module)
    except ImportError:
        LOG.warning("holdouts module not importable: only family-split non-train pgpids are used")
        return None
    try:
        return _ho.Holdouts.load(p=p)
    except FileNotFoundError:
        LOG.warning("holdouts.json not found: only family-split non-train pgpids are used")
        return None


def _load_quality_fn(pages: Sequence[Any] = ()) -> Optional[Callable[[Any], Any]]:
    """``literature_l1.quality_ok`` (with each part's running heads) when module D is available.

    :param pages: Pages of the joined works (running heads are computed per part from them).
    :returns: Page → result function, or None.
    """
    try:
        from src.finetuning.text27b import literature_l1 as _l1  # noqa: WPS433 (optional sibling module)
    except ImportError:
        LOG.warning("literature_l1 not importable: the L1 quality filter is skipped")
        return None
    fn = getattr(_l1, "quality_ok", None)
    heads_fn = getattr(_l1, "running_heads_by_part", None)
    if fn is None or heads_fn is None or not pages:
        return fn
    heads = heads_fn(pages)

    def ok(page: Any) -> Any:
        """Quality test of one page with its part's running heads.

        :param page: Page.
        :returns: ``quality_ok`` result.
        """
        return fn(page, heads.get((page.work, page.part)))

    return ok


def _quality_pass(fn: Optional[Callable[[Any], Any]], page: Any) -> bool:
    """Coerce a quality function's result (bool, ``(ok, reasons)`` or ``{"ok": ...}``) to a bool.

    :param fn: Quality function or None (always passes).
    :param page: Page.
    :returns: True when the page passes.
    """
    if fn is None:
        return True
    r = fn(page)
    if isinstance(r, tuple):
        return bool(r[0])
    if isinstance(r, dict):
        return bool(r.get("ok"))
    return bool(r)


def load_excluded_pages(path: Path) -> Dict[str, List[str]]:
    """``excluded_pages.json`` of the contamination module (``{page_key: [reasons]}``, a ``pages`` / ``excluded`` wrapper or a key list; empty when absent).

    :param path: File.
    :returns: Mapping.
    """
    if not path.exists():
        LOG.warning("%s not found: page exclusion relies on the index mentions only", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        for k in ("pages", "excluded"):
            if isinstance(raw.get(k), (dict, list)):
                raw = raw[k]
                break
    if isinstance(raw, list):
        return {str(k): ["listed"] for k in raw}
    return {k: (v if isinstance(v, list) else [str(v)]) for k, v in raw.items()}


# ===================================================================================== rows: build


#: Families joined to literature (analyse_document is dropped in run 1: 5 leak-safe documents).
RAG_FAMILIES: Tuple[str, ...] = ("catalogue_json", "describe_pgp", "extract_relations", "translate_en", "translate_he")
#: Input conditions a rag row may use (the document is an edition or a real read).
INPUT_CONDITIONS: Tuple[str, ...] = ("edition", "read", "read_hard", "read_partial")
#: Families sharing one per-document budget (B5).
GROUPS: Dict[str, str] = {"catalogue_json": "catalogue_describe", "describe_pgp": "catalogue_describe"}


@dataclass
class RagInputs:
    """Everything :func:`build_rows` reads (injected in tests)."""

    index_rows: List[Dict[str, Any]]
    pages: Dict[str, Any]
    base_rows: List[Dict[str, Any]]
    nontrain: NonTrain
    manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    excluded: Dict[str, List[str]] = field(default_factory=dict)
    quality_fn: Optional[Callable[[Any], Any]] = None
    span_pages: Dict[str, Set[str]] = field(default_factory=dict)


def _page_display_text(page: Any) -> str:
    """Quoteable text of a page: the structured ``full_main_text`` (what the site serves), project-name strings stripped.

    :param page: Page.
    :returns: Text.
    """
    text, _ = _tn.strip_source_names(page.text or "")
    return text


def _located_marks(page: Any) -> List[str]:
    """Located shelfmarks of a page, cleaned for display (fake / prompt-example keys drop out).

    :param page: Page (``shelf_marks`` or ``data``).
    :returns: Cleaned shelfmark strings, in page order, without duplicates.
    """
    raw = getattr(page, "shelf_marks", None)
    if raw is None:
        raw = _pg.sm_keys(page.data)
    out: List[str] = []
    for k in _sm.located_keys(raw, page.text or ""):
        c = _sm.prep_shelfmark(k)
        if c and c not in out:
            out.append(c)
    return out


def page_drop_reasons(inputs: RagInputs) -> Dict[str, List[str]]:
    """Pages a rag row may not use, with reasons.

    Reasons: ``excluded_pages`` (contamination manifest), ``nontrain_*`` / ``is_*`` (the page
    mentions a non-train document, any tier or evidence), ``quality`` (L1 filter), ``missing_page``.

    :param inputs: Inputs.
    :returns: ``{page_key: [reasons]}``.
    """
    reasons: Dict[str, List[str]] = collections.defaultdict(list)
    for k in inputs.excluded:
        reasons[k].append("excluded_pages")
    seen: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = set()
    for r in inputs.index_rows:
        sig = (r["page_key"], tuple(r["pgpids"]), tuple(r["merged_doc_ids"]))
        if sig in seen:
            continue
        seen.add(sig)
        why = inputs.nontrain.reason(r["pgpids"], r["merged_doc_ids"])
        if why and why not in reasons[r["page_key"]]:
            reasons[r["page_key"]].append(why)
    checked: Set[str] = set()
    for r in inputs.index_rows:
        k = r["page_key"]
        if r["tier"] != "discuss" or k in checked:
            continue
        checked.add(k)
        page = inputs.pages.get(k)
        if page is None:
            reasons[k].append("missing_page")
        elif not _quality_pass(inputs.quality_fn, page):
            reasons[k].append("quality")
    return {k: v for k, v in reasons.items() if v}


def strong_pages(index_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Strong-evidence discuss pages per pgpid (``located`` or ``clean_span``, never index pages).

    :param index_rows: Index rows.
    :returns: ``{pgpid: {page_key: best evidence}}``.
    """
    out: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for r in index_rows:
        if r["tier"] != "discuss" or r["evidence"] not in STRONG_EVIDENCE:
            continue
        for pid in r["pgpids"]:
            cur = out[pid].get(r["page_key"])
            if cur is None or EVIDENCE.index(r["evidence"]) < EVIDENCE.index(cur):
                out[pid][r["page_key"]] = r["evidence"]
    return out


def _choose_base(cands: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    """Pick the base row of a document group: edition or a real read with equal odds, then uniformly.

    :param cands: Candidate train rows (one document, one group, allowed conditions).
    :param rng: Seeded generator.
    :returns: The chosen row.
    """
    ed = [r for r in cands if r["meta"]["condition"] == "edition"]
    rd = [r for r in cands if r["meta"]["condition"] != "edition"]
    pool = ed if (ed and (not rd or rng.random() < 0.5)) else rd
    pool = sorted(pool, key=lambda r: (r["meta"].get("task"), r["meta"].get("condition"), r["meta"].get("read_model") or "",
                                       r["meta"].get("row_id") or ""))
    return pool[rng.randrange(len(pool))]


def build_rows(inputs: RagInputs, seed: int = 3407, k_max: int = 4, distractor_rate: float = 0.25,
               leak_max: float = 0.05, max_tokens: int = 6144, families: Sequence[str] = RAG_FAMILIES,
               token_fn: Optional[Callable[[Dict[str, Any]], int]] = None
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the ``rag_context`` rows.

    Per train document and family group (catalogue_json ∪ describe_pgp is one group, B5): one base
    row (edition or real read), up to ``k_max`` strong-evidence pages that are allowed
    (:func:`page_drop_reasons`), each holding < ``leak_max`` of the target's word 8-grams and their
    union too; analyse rows also skip the document's own span pages. ``distractor_rate`` of the
    rows (exact, seeded) get one extra source: a strong page of another train document in another
    work, under the same leak rule. Sources are shuffled, rendered in the site's format, appended
    to the base user turn; the target is the base row's human target. Rows over ``max_tokens`` lose
    their last non-distractor source until they fit (or are skipped).

    :param inputs: Inputs.
    :param seed: Seed.
    :param k_max: Maximum sources from the document's own pages.
    :param distractor_rate: Share of rows with a distractor source.
    :param leak_max: Leak threshold (single page and union).
    :param max_tokens: Rendered length cap.
    :param families: Families joined.
    :param token_fn: Row → rendered tokens (default :func:`rows.tokens_of_row`).
    :returns: ``(rows, manifest dict)``.
    """
    token_fn = token_fn or _rows.tokens_of_row
    drops = page_drop_reasons(inputs)
    strong = strong_pages(inputs.index_rows)
    # every pgpid each page mentions (for distractor safety)
    page_docs: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in inputs.index_rows:
        page_docs[r["page_key"]].update(r["pgpids"])
    by_doc_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    train_docs: Set[str] = set()
    for r in inputs.base_rows:
        m = r.get("meta") or {}
        if m.get("split") != "train" or m.get("task") not in families:
            continue
        pid = str(m.get("pgpid") or "")
        if not pid:
            continue
        train_docs.add(pid)
        if m.get("condition") in INPUT_CONDITIONS:
            by_doc_group[(pid, GROUPS.get(m["task"], m["task"]))].append(r)
    stats: collections.Counter = collections.Counter()
    page_gram_cache: Dict[str, Set[Tuple[str, ...]]] = {}
    text_cache: Dict[str, str] = {}

    def ptext(k: str) -> str:
        """Display text of a page (cached).

        :param k: Page key.
        :returns: Text.
        """
        if k not in text_cache:
            text_cache[k] = _page_display_text(inputs.pages[k])
        return text_cache[k]

    def pgrams(k: str) -> Set[Tuple[str, ...]]:
        """8-grams of a page's full text (cached).

        :param k: Page key.
        :returns: n-grams.
        """
        if k not in page_gram_cache:
            page_gram_cache[k] = grams8(inputs.pages[k].text)
        return page_gram_cache[k]

    allowed_strong: Dict[str, List[Tuple[str, str]]] = {}
    for pid, pages in strong.items():
        if pid not in train_docs:
            continue
        keep = [(k, ev) for k, ev in sorted(pages.items()) if k not in drops]
        stats["doc_pages_strong"] += len(pages)
        stats["doc_pages_dropped_page_rules"] += len(pages) - len(keep)
        if keep:
            allowed_strong[pid] = keep
    pending: List[Dict[str, Any]] = []
    for (pid, group) in sorted(by_doc_group, key=lambda x: (_int_key(x[0]), x[1])):
        if pid not in allowed_strong:
            continue
        rng = random.Random(f"{seed}|{pid}|{group}")
        base = _choose_base(by_doc_group[(pid, group)], rng)
        task = base["meta"]["task"]
        tg = grams8(base["messages"][-1]["content"])
        own_span = inputs.span_pages.get(pid, set()) if task == "analyse_document" else set()
        cands = [(k, ev) for k, ev in allowed_strong[pid] if k not in own_span]
        cands.sort(key=lambda x: (EVIDENCE.index(x[1]), rng.random()))
        chosen: List[Tuple[str, str]] = []
        union: Set[Tuple[str, ...]] = set()
        for k, ev in cands:
            if len(chosen) >= k_max:
                break
            sk = share(tg, pgrams(k))
            if sk >= leak_max:
                stats["leak_drop_page"] += 1
                continue
            u2 = union | (tg & pgrams(k))
            if tg and len(u2) / len(tg) >= leak_max:
                stats["leak_drop_union"] += 1
                continue
            chosen.append((k, ev))
            union = u2
        if not chosen:
            stats["groups_no_safe_page"] += 1
            continue
        pending.append({"pid": pid, "group": group, "base": base, "chosen": chosen, "union": union, "tg": tg,
                        "rng": rng})
    # distractors: exact share of rows, seeded
    pick_rng = random.Random(f"{seed}|distractors")
    n_dis = round(distractor_rate * len(pending))
    dis_idx = set(pick_rng.sample(range(len(pending)), n_dis)) if pending else set()
    pool = sorted({(k, pid) for pid, ks in allowed_strong.items() for k, _ in ks})
    out_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(pending):
        rng: random.Random = item["rng"]
        sources: List[Tuple[str, str, bool]] = [(k, ev, False) for k, ev in item["chosen"]]
        if i in dis_idx and pool:
            works_used = {inputs.pages[k].book for k, _ in item["chosen"]}
            options = [(k, op) for k, op in pool if op != item["pid"] and item["pid"] not in page_docs.get(k, set())
                       and inputs.pages[k].book not in works_used and k not in {s[0] for s in sources}]
            rng.shuffle(options)
            for k, _ in options[:50]:
                if share(item["tg"], pgrams(k)) >= leak_max:
                    continue
                if item["tg"] and len(item["union"] | (item["tg"] & pgrams(k))) / len(item["tg"]) >= leak_max:
                    continue
                sources.append((k, "distractor", True))
                break
            else:
                stats["distractor_not_found"] += 1
        rng.shuffle(sources)
        row = _render_row(item, sources, inputs, ptext, token_fn, max_tokens, stats)
        if row is not None:
            out_rows.append(row)
    manifest = _rows_manifest(out_rows, stats, drops, seed, k_max, distractor_rate, leak_max, max_tokens, families)
    return out_rows, manifest


def _render_row(item: Dict[str, Any], sources: List[Tuple[str, str, bool]], inputs: RagInputs,
                ptext: Callable[[str], str], token_fn: Callable[[Dict[str, Any]], int], max_tokens: int,
                stats: collections.Counter) -> Optional[Dict[str, Any]]:
    """Render one rag row, trimming own sources until it fits ``max_tokens``.

    :param item: Pending row (base row, chosen pages, rng; ``split``, default ``train``, is the row's split).
    :param sources: ``[(page_key, evidence, is_distractor)]`` in display order.
    :param inputs: Inputs (pages, manifest).
    :param ptext: Page key → display text.
    :param token_fn: Row → tokens.
    :param max_tokens: Length cap.
    :param stats: Counters.
    :returns: The row, or None when no own source fits.
    """
    base = item["base"]
    bm = base["meta"]
    srcs = list(sources)
    while True:
        if not any(not d for _, _, d in srcs):
            stats["rows_skipped_length"] += 1
            return None
        rendered = []
        for k, _, _ in srcs:
            pg = inputs.pages[k]
            author, title, year = work_meta(inputs.manifest, pg.book, pg.part)
            page_no = getattr(pg, "printed_page_num", None)
            rendered.append((citation(author, title, year, page_no), ptext(k), _located_marks(pg)))
        block, chars = render_block(rendered)
        user = base["messages"][1]["content"] + "\n\n" + block
        messages = _rows.make_messages(user, base["messages"][2]["content"], base["messages"][0]["content"])
        meta = dict(bm)
        meta.update({
            "family": "rag_context", "task": bm["task"], "condition": "literature",
            "input_condition": bm["condition"], "split": item.get("split", "train"), "build": _rows.BUILD,
            "row_id": _rows.row_id("rag_context", item["pid"], "literature", bm.get("read_model"), bm["task"]),
            "twin_row_id": bm.get("row_id"),
            "lit_sources": [{"work": inputs.pages[k].book, "page": getattr(inputs.pages[k], "printed_page_num", None),
                             "chars": c, "page_key": k, "evidence": ev, "distractor": dis}
                            for (k, ev, dis), c in zip(srcs, chars)],
            "lit_block_chars": len(block),
            "lit_leak_share": round(len(item["union"]) / len(item["tg"]), 4) if item["tg"] else 0.0,
        })
        row = {"messages": messages, "meta": meta}
        n = token_fn(row)
        if n <= max_tokens:
            meta["tokens"] = n
            stats["rows"] += 1
            return row
        stats["sources_trimmed_length"] += 1
        own = [j for j, (_, _, d) in enumerate(srcs) if not d]
        srcs.pop(own[-1])


# ===================================================================================== dev + eval rows (prep pass)
#: Excluded-page reason tiers a dev-monitor rag row may ignore (prep pass 2026-10-26): mentions of dev documents (the
#: row's own document is one), their ±1 windows and treatments, the L1-only train-target dedupe, and n-gram hits on the
#: row's own document (the 5 % leak rule then applies to it). Benchmark / test / probe / vision / extra hold-out
#: reasons still bind.
DEV_OK_REASON_TIERS: Tuple[str, ...] = ("dev",)
#: Output of the E6 eval rows (test documents with their leak-safe pages; never a family file, never trained).
E6_ROWS_FILE = "rag_e6_rows.jsonl"


def dev_page_ok(reasons: Sequence[str], pid: str) -> bool:
    """Whether a page's ``excluded_pages.json`` reasons allow it as context of a DEV-monitor row of ``pid``.

    :param reasons: The page's exclusion reasons (empty = not excluded).
    :param pid: The dev document's pgpid.
    :returns: True when every reason is one :data:`DEV_OK_REASON_TIERS` (or the train-target dedupe, or an n-gram hit
        on the document itself) explains.
    """
    for r in reasons:
        base = r[len("window:"):] if r.startswith("window:") else r
        if base.startswith("train_target_dup"):
            continue
        if base.startswith(("mention:", "treatment:")) and base.rsplit(":", 1)[-1] in DEV_OK_REASON_TIERS:
            continue
        if base.startswith("ngram:") and re.search(r"[:]" + re.escape(pid) + r"#", base):
            continue
        return False
    return True


def _choose_pages(cands: Sequence[Tuple[str, str]], tg: Set[Tuple[str, ...]], pgrams: Callable[[str], Set[Tuple[str, ...]]],
                  k_max: int, leak_max: float, stats: collections.Counter, prefix: str = ""
                  ) -> Tuple[List[Tuple[str, str]], Set[Tuple[str, ...]]]:
    """Up to ``k_max`` pages under the leak rule (each page and their union hold < ``leak_max`` of the target's 8-grams).

    :param cands: ``(page_key, evidence)`` in preference order.
    :param tg: Target 8-grams.
    :param pgrams: Page key → page 8-grams.
    :param k_max: Page cap.
    :param leak_max: Leak threshold.
    :param stats: Counters (``<prefix>leak_drop_page`` / ``<prefix>leak_drop_union``).
    :param prefix: Counter prefix.
    :returns: ``(chosen pages, union of shared 8-grams)``.
    """
    chosen: List[Tuple[str, str]] = []
    union: Set[Tuple[str, ...]] = set()
    for k, ev in cands:
        if len(chosen) >= k_max:
            break
        if share(tg, pgrams(k)) >= leak_max:
            stats[f"{prefix}leak_drop_page"] += 1
            continue
        u2 = union | (tg & pgrams(k))
        if tg and len(u2) / len(tg) >= leak_max:
            stats[f"{prefix}leak_drop_union"] += 1
            continue
        chosen.append((k, ev))
        union = u2
    return chosen, union


def build_dev_rows(inputs: RagInputs, dev_rows: Iterable[Dict[str, Any]], train_page_keys: Set[str],
                   excluded_all: Mapping[str, Sequence[str]], dev_pids: Set[str], seed: int = 3407, k_max: int = 4,
                   leak_max: float = 0.05, max_tokens: int = 6144, families: Sequence[str] = RAG_FAMILIES,
                   token_fn: Optional[Callable[[Dict[str, Any]], int]] = None
                   ) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """``rag_context`` rows of the DEV documents, for the in-run dev monitor (prep pass 2026-10-26; never trained).

    Per dev document and family group: one base row (edition or an unexposed real read), up to ``k_max`` of the
    document's strong-evidence discuss pages that (a) no train row of any family uses (``train_page_keys``: L1 /
    L2 / L5 page keys and this build's train rag sources), (b) pass the L1 quality filter, (c) carry only
    :func:`dev_page_ok` exclusion reasons and (d) mention no non-train document other than dev documents and no
    benchmark document; the same leak rule as train rows; no distractor.

    :param inputs: Inputs (pages, index, quality filter, hold-outs).
    :param dev_rows: Dev rows of the PGP families.
    :param train_page_keys: Page keys used by train rows.
    :param excluded_all: The full ``excluded_pages.json`` mapping.
    :param dev_pids: pgpids of every dev document.
    :param seed: Seed.
    :param k_max: Page cap.
    :param leak_max: Leak threshold.
    :param max_tokens: Length cap.
    :param families: Families joined.
    :param token_fn: Row → rendered tokens.
    :returns: ``(rows with split dev, counters)``.
    """
    token_fn = token_fn or _rows.tokens_of_row
    strong = strong_pages(inputs.index_rows)
    page_index: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in inputs.index_rows:
        page_index[r["page_key"]].append(r)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in dev_rows:
        m = r.get("meta") or {}
        pid = str(m.get("pgpid") or "")
        if m.get("split") != "dev" or m.get("task") not in families or not pid:
            continue
        if m.get("condition") in INPUT_CONDITIONS and m.get("reader_exposed") is not True:
            groups[(pid, GROUPS.get(m["task"], m["task"]))].append(r)
    stats: collections.Counter = collections.Counter()
    gram_cache: Dict[str, Set[Tuple[str, ...]]] = {}

    def pgrams(k: str) -> Set[Tuple[str, ...]]:
        """8-grams of a page (cached).

        :param k: Page key.
        :returns: n-grams.
        """
        if k not in gram_cache:
            gram_cache[k] = grams8(inputs.pages[k].text)
        return gram_cache[k]

    def drop_reason(k: str, pid: str) -> Optional[str]:
        """Why a page may not serve a dev row of ``pid`` (None = usable).

        :param k: Page key.
        :param pid: Dev document.
        :returns: Reason or None.
        """
        if k in train_page_keys:
            return "train_page"
        page = inputs.pages.get(k)
        if page is None:
            return "missing_page"
        if not _quality_pass(inputs.quality_fn, page):
            return "quality"
        if not dev_page_ok(excluded_all.get(k, ()), pid):
            return "excluded_pages"
        h = inputs.nontrain.holdouts
        for ir in page_index.get(k, ()):
            others = [str(x) for x in ir["pgpids"] if str(x) != pid and str(x) not in dev_pids]
            if others and inputs.nontrain.reason(others, []):
                return "mentions_nontrain"
            if h is not None and hasattr(h, "is_benchmark"):
                if any(h.is_benchmark(x, None, None) for x in others):
                    return "mentions_benchmark"
                for c in ir.get("merged_doc_ids") or []:
                    if h.is_benchmark(None, inputs.nontrain.displays.get(c), c):
                        return "mentions_benchmark"
        return None

    out: List[Dict[str, Any]] = []
    for (pid, group) in sorted(groups, key=lambda x: (_int_key(x[0]), x[1])):
        if pid not in strong:
            stats["dev_groups_no_strong_page"] += 1
            continue
        rng = random.Random(f"{seed}|dev|{pid}|{group}")
        base = _choose_base(groups[(pid, group)], rng)
        tg = grams8(base["messages"][-1]["content"])
        cands = []
        for k, ev in sorted(strong[pid].items()):
            why = drop_reason(k, pid)
            if why:
                stats[f"dev_page_drop:{why}"] += 1
                continue
            cands.append((k, ev))
        cands.sort(key=lambda x: (EVIDENCE.index(x[1]), rng.random()))
        chosen, union = _choose_pages(cands, tg, pgrams, k_max, leak_max, stats, "dev_")
        if not chosen:
            stats["dev_groups_no_safe_page"] += 1
            continue
        item = {"pid": pid, "group": group, "base": base, "chosen": chosen, "union": union, "tg": tg, "rng": rng,
                "split": "dev"}
        row = _render_row(item, [(k, ev, False) for k, ev in chosen], inputs,
                          lambda k: _page_display_text(inputs.pages[k]), token_fn, max_tokens, stats)
        if row is not None:
            out.append(row)
            stats["dev_rows"] += 1
    return out, stats


def build_eval_rows(inputs: RagInputs, test_docs: Mapping[str, Any], eval_rows: Iterable[Dict[str, Any]],
                    k_max: int = 4, max_tokens: int = 6144, families: Sequence[str] = E6_FAMILIES,
                    token_fn: Optional[Callable[[Dict[str, Any]], int]] = None
                    ) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """The E6 "with literature" prompts materialised: per test document of ``rag_test_docs.json`` and family, the
    document's edition row with up to ``k_max`` of its leak-safe strong pages in the site format (an eval file, never a
    family file; E9 draws its RAG items from it).

    :param inputs: Inputs (pages, manifest).
    :param test_docs: Parsed ``rag_test_docs.json``.
    :param eval_rows: Rows of the eval splits of the E6 families.
    :param k_max: Page cap.
    :param max_tokens: Length cap.
    :param families: E6 families.
    :param token_fn: Row → rendered tokens.
    :returns: ``(rows, counters)``; rows keep the base row's split and carry ``meta.e6 = True``.
    """
    token_fn = token_fn or _rows.tokens_of_row
    base: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in eval_rows:
        m = r.get("meta") or {}
        if m.get("condition") != "edition" or m.get("task") not in families or m.get("split") == "train":
            continue
        key = (str(m.get("pgpid") or ""), str(m.get("task")))
        if key not in base or str(m.get("row_id")) < str(base[key]["meta"].get("row_id")):
            base[key] = r
    stats: collections.Counter = collections.Counter()
    out: List[Dict[str, Any]] = []
    for d in test_docs.get("docs") or []:
        pid = str(d.get("pgpid"))
        ev_of = {x["page_key"]: x["evidence"] for x in d.get("discuss_pages") or []}
        for fam in families:
            safe = [k for k in (d.get("leak_safe_pages") or {}).get(fam) or [] if k in inputs.pages]
            if not safe or (pid, fam) not in base:
                stats[f"e6_skipped:{fam}"] += 1
                continue
            b = base[(pid, fam)]
            tg = grams8(b["messages"][-1]["content"])
            chosen = sorted(safe, key=lambda k: (EVIDENCE.index(ev_of.get(k, "key_only")), k))[:k_max]
            union: Set[Tuple[str, ...]] = set()
            for k in chosen:
                union |= tg & grams8(inputs.pages[k].text)
            item = {"pid": pid, "group": fam, "base": b, "chosen": [(k, ev_of.get(k, "")) for k in chosen],
                    "union": union, "tg": tg, "split": b["meta"]["split"]}
            row = _render_row(item, [(k, ev_of.get(k, ""), False) for k in chosen], inputs,
                              lambda k: _page_display_text(inputs.pages[k]), token_fn, max_tokens, stats)
            if row is not None:
                row["meta"]["e6"] = True
                out.append(row)
                stats[f"e6_rows:{fam}"] += 1
    return out, stats


def _rows_manifest(rows: Sequence[Dict[str, Any]], stats: collections.Counter, drops: Dict[str, List[str]], seed: int,
                   k_max: int, distractor_rate: float, leak_max: float, max_tokens: int,
                   families: Sequence[str]) -> Dict[str, Any]:
    """Manifest of the rag rows.

    :param rows: Built rows.
    :param stats: Build counters.
    :param drops: Page drop reasons.
    :param seed: Seed.
    :param k_max: Source cap.
    :param distractor_rate: Distractor share.
    :param leak_max: Leak threshold.
    :param max_tokens: Length cap.
    :param families: Families joined.
    :returns: Manifest dict.
    """
    counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"rows": 0, "tokens": 0})
    by_task: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {"rows": 0, "docs": set(), "pairs": 0, "tokens": 0,
                                                                          "works": set(), "distractor_rows": 0})
    for r in rows:
        m = r["meta"]
        key = "|".join(str(x) for x in (m["family"], m["condition"], m.get("lang_class"), m["split"]))
        counts[key]["rows"] += 1
        counts[key]["tokens"] += int(m.get("tokens") or 0)
        t = by_task[m["task"]]
        t["rows"] += 1
        t["docs"].add(m.get("pgpid"))
        own = [s for s in m["lit_sources"] if not s["distractor"]]
        t["pairs"] += len(own)
        t["works"].update(s["work"] for s in own)
        t["tokens"] += int(m.get("tokens") or 0)
        t["distractor_rows"] += any(s["distractor"] for s in m["lit_sources"])
    reason_counts = collections.Counter(x for v in drops.values() for x in v)
    return {
        "build": _rows.BUILD, "family": "rag_context",
        "counts": dict(counts),
        "by_task": {k: {"rows": v["rows"], "docs": len(v["docs"]), "doc_page_pairs": v["pairs"], "works": len(v["works"]),
                        "tokens": v["tokens"], "distractor_rows": v["distractor_rows"]} for k, v in sorted(by_task.items())},
        "input_conditions": dict(collections.Counter(r["meta"]["input_condition"] for r in rows)),
        "docs": len({r["meta"].get("pgpid") for r in rows}),
        "rows": len(rows), "tokens": sum(int(r["meta"].get("tokens") or 0) for r in rows),
        "page_drops": {"pages": len(drops), "by_reason": dict(reason_counts.most_common())},
        "stats": dict(stats),
        "fingerprint": _rows.fingerprint(rows),
        "build_args": {"seed": seed, "k_max": k_max, "distractor_rate": distractor_rate, "leak_max": leak_max,
                       "max_tokens": max_tokens, "families": list(families)},
    }


def _block_check(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Re-check every row's literature block with ``contamination.block_ok`` (the rescan's definition) when available.

    :param rows: Built rag rows.
    :returns: ``(rows kept, rows dropped)``; all rows are kept when the contamination module is absent.
    """
    try:
        from src.finetuning.text27b import contamination as _ct  # noqa: WPS433 (optional sibling module)
    except ImportError:
        LOG.warning("contamination not importable: block rule checked with the local 8-gram helper only")
        return rows, 0
    fn = getattr(_ct, "block_ok", None)
    if fn is None:
        return rows, 0
    kept = []
    for r in rows:
        user = r["messages"][1]["content"]
        block = user[user.rindex(LIT_HEADER):]
        if fn(r["messages"][2]["content"], block):
            kept.append(r)
    return kept, len(rows) - len(kept)


def rag_excluded_pages(excluded: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The page exclusions that bind RAG rows: pages whose only reasons are L1-only (the train-target
    dedupe, :data:`contamination.L1_ONLY_REASONS`) stay usable as context.

    :param excluded: ``excluded_pages.json`` mapping.
    :returns: The binding subset.
    """
    from src.finetuning.text27b import contamination as _ct  # sibling module; imported here to keep load light
    return {k: v for k, v in excluded.items() if _ct.excluded_for_family(v, "rag_context")}


def load_excluded_rows(path: Path) -> Set[str]:
    """Row ids of ``excluded_rows.json`` (contamination module, decision D5: train rows whose target
    leaks an eval reference); a mapping or a list is accepted, a missing file gives an empty set.

    :param path: File.
    :returns: Excluded row ids.
    """
    if not path.exists():
        LOG.warning("%s not found: no base rows excluded", path.name)
        return set()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k) for k in raw}


def train_page_keys(families_dir: Path, rag_train_rows: Iterable[Dict[str, Any]] = (),
                    stems: Sequence[str] = ("domain_reading", "book_pairs", "analyse_document_v2", "analyse_document")
                    ) -> Set[str]:
    """Literature page keys that train rows use (``meta.page_keys``, ``provenance.page_keys``, ``lit_sources``).

    :param families_dir: ``out_dir/families``.
    :param rag_train_rows: This build's train rag rows (their sources count too).
    :param stems: Family files carrying page keys.
    :returns: Page keys.
    """
    keys: Set[str] = set()

    def add(m: Mapping[str, Any]) -> None:
        """Collect the page keys of one meta block.

        :param m: Row meta.
        :returns: None.
        """
        keys.update(str(k) for k in m.get("page_keys") or [])
        keys.update(str(k) for k in (m.get("provenance") or {}).get("page_keys") or [] if isinstance(m.get("provenance"), dict))
        keys.update(str(s.get("page_key")) for s in m.get("lit_sources") or [] if s.get("page_key"))

    for stem in stems:
        f = Path(families_dir) / f"{stem}.jsonl"
        if f.exists():
            for r in _rows.read_jsonl(f):
                if (r.get("meta") or {}).get("split") == "train":
                    add(r["meta"])
    for r in rag_train_rows:
        if (r.get("meta") or {}).get("split") == "train":
            add(r["meta"])
    return keys


def run_build_rows(p: Optional[_paths.Paths] = None, seed: int = 3407, families: Sequence[str] = RAG_FAMILIES,
                   k_max: int = 4, distractor_rate: float = 0.25, leak_max: float = 0.05, max_tokens: int = 6144,
                   excluded_pages: Optional[Path] = None, literature_manifest: Optional[Path] = None,
                   excluded_rows: Optional[Path] = None, dev_rows: bool = True, e6_rows: bool = True) -> Dict[str, Any]:
    """Phase-2 CLI: read the index, pages, v0_2 family files, hold-outs and exclusions; write ``families/rag_context.jsonl``.

    Base rows are the ``train`` rows of ``families/<family>.jsonl`` minus the rows listed in
    ``excluded_rows.json`` (their target leaks an eval reference, so a rag row on them would too);
    every pgpid seen in another split is non-train, as is everything ``holdouts.json`` names
    (benchmark and eval tiers).

    :param p: Build paths.
    :param seed: Seed.
    :param families: Families joined.
    :param k_max: Source cap.
    :param distractor_rate: Distractor share.
    :param leak_max: Leak threshold.
    :param max_tokens: Length cap.
    :param excluded_pages: ``excluded_pages.json`` (default ``out_dir/excluded_pages.json``).
    :param literature_manifest: Literature manifest (default: next to ``paths.json``).
    :param excluded_rows: ``excluded_rows.json`` (default ``out_dir/excluded_rows.json``; missing → none excluded).
    :param dev_rows: Also build the dev-monitor rows of the dev documents (:func:`build_dev_rows`; prep pass
        2026-10-26), written to the same family file with ``split = dev``.
    :param e6_rows: Also write the E6 eval rows (:func:`build_eval_rows`) to ``out_dir/rag_e6_rows.jsonl``.
    :returns: The manifest.
    :raises FileNotFoundError: When ``rag_index.jsonl`` or no family file exists.
    """
    p = p or _paths.load_paths()
    index_rows = _rows.load_jsonl(p.out("rag_index.jsonl"))
    spans = _rows.load_jsonl(p.out("rag_spans.jsonl")) if p.out("rag_spans.jsonl").exists() else []
    fam_files = [p.out("families", f"{f}.jsonl") for f in _rows.PGP_FAMILIES if p.out("families", f"{f}.jsonl").exists()]
    if not fam_files:
        raise FileNotFoundError(f"no family files under {p.out('families')}: run the builder first")
    base_rows: List[Dict[str, Any]] = []
    dev_base: List[Dict[str, Any]] = []
    eval_base: List[Dict[str, Any]] = []
    nontrain_pg: Set[str] = set()
    dev_pids: Set[str] = set()
    ex_rows = load_excluded_rows(Path(excluded_rows) if excluded_rows else p.out("excluded_rows.json"))
    n_ex_rows = 0
    for f in fam_files:
        for r in _rows.read_jsonl(f):
            m = r.get("meta") or {}
            if m.get("split") != "train":
                if m.get("pgpid"):
                    nontrain_pg.add(str(m["pgpid"]))
                if m.get("split") == "dev" and m.get("pgpid"):
                    dev_pids.add(str(m["pgpid"]))
                    if dev_rows and m.get("task") in families and str(m.get("row_id")) not in ex_rows:
                        dev_base.append(r)
                elif e6_rows and m.get("task") in E6_FAMILIES and m.get("condition") == "edition":
                    eval_base.append(r)
                continue
            if m.get("task") in families:
                if str(m.get("row_id")) in ex_rows:
                    n_ex_rows += 1
                    continue
                base_rows.append(r)
    corpus = _sm.load_corpus(p)
    works = {r["work"] for r in index_rows if r["tier"] == "discuss"}
    page_list = _pg.iter_pages(root=p.academic_literature, works=works)
    pages = {pg.page_key: pg for pg in page_list}
    span_pages: Dict[str, Set[str]] = collections.defaultdict(set)
    for s in spans:
        for pid in s["pgpids"]:
            span_pages[pid].update(s.get("page_keys") or [])
    nontrain = NonTrain(pgpids=nontrain_pg, holdouts=_load_holdouts(p),
                        displays={c: v.get("disp", "") for c, v in corpus.merged.items()})
    excluded_all = load_excluded_pages(Path(excluded_pages) if excluded_pages else p.out("excluded_pages.json"))
    inputs = RagInputs(index_rows=index_rows, pages=pages, base_rows=base_rows, nontrain=nontrain,
                       manifest=load_literature_manifest(p, literature_manifest),
                       excluded=rag_excluded_pages(excluded_all),
                       quality_fn=_load_quality_fn(page_list), span_pages=span_pages)
    out_rows, manifest = build_rows(inputs, seed=seed, k_max=k_max, distractor_rate=distractor_rate, leak_max=leak_max,
                                    max_tokens=max_tokens, families=families)
    out_rows, n_block = _block_check(out_rows)
    if n_block:
        LOG.warning("%d rows failed contamination.block_ok and were dropped", n_block)
        manifest = _rows_manifest(out_rows, collections.Counter({**manifest["stats"], "block_check_dropped": n_block}),
                                  page_drop_reasons(inputs), seed, k_max, distractor_rate, leak_max, max_tokens, families)
    dev_out: List[Dict[str, Any]] = []
    if dev_rows:
        train_keys = train_page_keys(p.out("families"), out_rows)
        dev_out, dev_stats = build_dev_rows(inputs, dev_base, train_keys, excluded_all, dev_pids, seed=seed, k_max=k_max,
                                            leak_max=leak_max, max_tokens=max_tokens, families=families)
        dev_out, n_dev_block = _block_check(dev_out)
        manifest["dev"] = {"rows": len(dev_out), "docs": len({r["meta"]["pgpid"] for r in dev_out}),
                           "base_rows": len(dev_base), "train_page_keys": len(train_keys),
                           "block_check_dropped": n_dev_block, "stats": dict(sorted(dev_stats.items())),
                           "by_task": dict(collections.Counter(r["meta"]["task"] for r in dev_out))}
    if e6_rows and p.out("rag_test_docs.json").exists():
        e6, e6_stats = build_eval_rows(inputs, json.loads(p.out("rag_test_docs.json").read_text(encoding="utf-8")),
                                       eval_base, k_max=k_max, max_tokens=max_tokens)
        _rows.write_jsonl(p.out(E6_ROWS_FILE), e6)
        manifest["e6_rows"] = {"file": E6_ROWS_FILE, "rows": len(e6), "stats": dict(sorted(e6_stats.items()))}
    out_rows = out_rows + dev_out
    for r in out_rows:
        problems = _rows.validate_row(r)
        if problems:
            raise ValueError(f"invalid rag row {r['meta']['row_id']}: {problems}")
    _rows.write_jsonl(p.out("families", "rag_context.jsonl", mkdir=True), out_rows)
    manifest["inputs"] = {"family_files": [f.name for f in fam_files], "holdouts": nontrain.holdouts is not None,
                          "quality_filter": inputs.quality_fn is not None, "excluded_pages": len(inputs.excluded),
                          "literature_manifest_works": len(inputs.manifest), "block_check_dropped": n_block,
                          "excluded_rows_listed": len(ex_rows), "base_rows_excluded": n_ex_rows}
    _write_json(p.out("families", "rag_context_manifest.json"), manifest)
    LOG.info("rag_context: %d rows, %d tokens", manifest["rows"], manifest["tokens"])
    return manifest


# ===================================================================================== CLI


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point (``build_index`` / ``test_docs`` / ``build_rows``).

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: None.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bi = sub.add_parser("build_index", help="literature page index + report + test docs")
    bi.add_argument("--works", nargs="*", default=None, help="restrict to these work directories (sampling)")
    bi.add_argument("--span-config", type=Path, default=None)
    bi.add_argument("--out-dir", type=Path, default=None)
    bi.add_argument("--no-test-docs", action="store_true")
    td = sub.add_parser("test_docs", help="re-write rag_test_docs.json from rag_index.jsonl")
    td.add_argument("--out-dir", type=Path, default=None)
    br = sub.add_parser("build_rows", help="families/rag_context.jsonl (phase 2)")
    br.add_argument("--seed", type=int, default=3407)
    br.add_argument("--families", nargs="*", default=list(RAG_FAMILIES))
    br.add_argument("--k-max", type=int, default=4)
    br.add_argument("--distractor-rate", type=float, default=0.25)
    br.add_argument("--leak-max", type=float, default=0.05)
    br.add_argument("--max-tokens", type=int, default=6144)
    br.add_argument("--excluded-pages", type=Path, default=None)
    br.add_argument("--literature-manifest", type=Path, default=None)
    br.add_argument("--excluded-rows", type=Path, default=None, help="default <out_dir>/excluded_rows.json (D5)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "build_index":
        rep = run_build_index(works=args.works, span_config=args.span_config, test_docs=not args.no_test_docs,
                              out_dir=args.out_dir)
        print(json.dumps({k: rep[k] for k in ("pages", "rows", "rows_resolved", "tiers", "evidence_discuss_rows",
                                              "strong_discuss_pgpids", "spans", "seconds")}, indent=1, default=str))
        if "test_docs" in rep:
            print(json.dumps({"test_docs": rep["test_docs"]}, indent=1))
    elif args.cmd == "test_docs":
        print(json.dumps(run_test_docs(out_dir=args.out_dir), indent=1))
    else:
        man = run_build_rows(seed=args.seed, families=args.families, k_max=args.k_max, distractor_rate=args.distractor_rate,
                             leak_max=args.leak_max, max_tokens=args.max_tokens, excluded_pages=args.excluded_pages,
                             literature_manifest=args.literature_manifest, excluded_rows=args.excluded_rows)
        print(json.dumps({k: man[k] for k in ("rows", "docs", "tokens", "by_task", "page_drops", "stats")}, indent=1))


if __name__ == "__main__":
    main()
