# File name: book_pairs.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""L5 ``book_pairs``: translation pairs and editor-introduction rows from book editions (text27b v0_2).

Plan 2026-10-21 §9.2 (L5) and the phase-2 brief (decisions D14, D15). Two kinds of sources, both
named only in local, git-ignored configuration (no holdings in this file):

**Parsed edition entries** (``nas_books`` in the local config): one ``entries.jsonl`` per book under
``paths.book_editions_nas`` (read-only), written by the book-edition parsers. Layout ``lines``:
``transcription_lines`` / ``translation_lines`` as ``{n, side, part, text}``; layout ``paragraphs``:
``transcription_lines`` with a per-line ``script`` and ``translation_paragraphs`` with ``(N)`` / ``N.``
line markers. The edition becomes the input, the human translation the target; each target line is
copied from the entry, never generated.

**Located spans** (``span_sources``, from the local spans file): book pages under
``paths.academic_literature`` whose translation / transcription / introduction spans the agent placed
by reading the pages (D15: LLM-located spans are approved). A span is a list of pieces, each a line
range of ONE page's OCR text (:attr:`pages.PageRec.ocr_text`, never the structurer's
``full_main_text``) with its first / last characters and a sha1 of the cut text; :func:`cut_piece`
re-cuts the piece and rejects it unless all three match, so every raw target is a verified verbatim
substring of the OCR. Normalisation (:data:`NORMALISERS`: re-joining wrapped lines, removing footnote
reference digits, line-number remnants, editorial section headings) only deletes characters or joins
lines; :func:`words_subset` asserts that every word of the final target occurs in the raw span. Inputs
are either the book's own transcription span or the PGP digital edition of the joined document
(``input: pgp_edition``).

Second pass on every pair (D15): the F1 length band of the builder (0.5–2.5 English, 0.5–2.0 Hebrew,
on :func:`textnorm.normalize_for_cer` lengths) and a pairing check (:func:`pair_check`): the target's
overlap with its own edition (English: entity candidates grounded in the edition by consonant skeleton;
Hebrew: Hebrew words shared with the edition) must not be beaten by another document of the same source.

Dedupe and hold-outs (D14, plan §9.4): a document is matched to PGP ids by the source's PGP edition
slug and number (``doc. N``) and by normalised shelfmark key. It is dropped when PGP already carries a
translation into the same language (``dup_pgp_translation``).
Split: any match with the benchmark tier of :class:`holdouts.Holdouts` or with its eval tier (v0_1
dev / test documents, shelfmark-keyed) → ``heldout``; a PGP id that v0_1 trains on → ``train``;
otherwise sha1 of the document's shelfmark key (else ``book:<slug>:<doc>``) → 80/10/10 ``train`` /
``dev`` / ``test_indist``.

Rows (schema of :mod:`rows`, ``build = v0_2``): ``translate_en`` / ``translate_he`` rows with
``condition`` ``edition``; ``noised_edition`` for Hebrew-script editions in train (the builder's noise
model, calibrated as the builder does; none for Arabic script, B8); ``edition_rescript`` for
Arabic-script Arabic documents (B9, ``translate_en`` only). ``analyse_document`` rows from located
introductions (span kind ``intro``), gated like v0's tier A′/B: grounding of the stripped target in the
input + catalogue context ≥ 0.5, or ≥ 0.6 with the book translation appended; targets that copy the input
are dropped. Commentary rows from the parsed entries (``discussion`` / ``commentary``) are built by L2
(``analyse_l2``); ``--with-entry-commentary`` adds them here instead (off by default, to avoid double
supervision). Every row carries ``source = book:<slug>``, ``doc_id``, ``licence`` / ``licence_class``.

Literature pages (fix passes F2, F11): every row carries ``meta.page_keys``, the literature pages it draws on -- the
located span pages of a span document, and for a parsed entry every page of the entry's PDF range in the book's work
directory (:class:`EntryPages`: ``work`` in the local config, the entry's ``pdf_page`` up to the next entry's start,
widened to the end page the entry records), with ``page_keys_source = nas_entry``; an entry that cannot be mapped is
flagged ``page_keys_unmapped`` and its train rows are excluded by :mod:`contamination` -- so the page rule applies to
these rows like any other.

Outputs under ``paths.out_dir``: ``families/book_pairs.jsonl`` (all splits), ``families/
book_pairs_manifest.json`` (counts per family × condition × lang_class × split, per-source pair counts,
drops by reason, the documents removed as hold-outs, build args) and ``audit/book_pairs_sample_30.jsonl``
(source / target pairs for a hand check).

Local configuration (``paths.json`` keys ``book_pairs_config`` / ``book_pairs_spans``, defaulting to
``logs/text_finetune_27b/book_pairs_local.json`` and ``book_pairs_spans.json``):

* config: ``{"nas_books": [{"slug", "dir", "layout": "lines"|"paragraphs", "licence", "licence_class",
  "pgp_slugs": [...], "target_lang": "en", "work": <literature work directory>}], "priority": [slug, ...]}``;
* spans: ``{"sources": [{"slug", "work", "target_lang", "input": "span"|"pgp_edition", "licence",
  "licence_class", "pgp_slugs": [...], "docs": [{"doc", "shelfmarks", "pgpids", "lang_class",
  "spans": {kind: {"pieces": [{"page", "lines": [i, j], "head", "tail", "sha1", "cols": [a, b]}],
  "normalise": [op, ...]}}}]}]}``.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.book_pairs                # full build
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.book_pairs --dry-run      # counts only
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.book_pairs --show-page KEY  # numbered OCR lines

Public API: :data:`NORMALISERS`, :func:`label_of`, :func:`render_blocks`, :func:`group_lines`,
:func:`rejoin_indented`, :func:`rejoin_blank_separated`, :func:`prose_paragraphs`,
:func:`drop_footnote_refs`, :func:`drop_digit_remnants`, :func:`drop_bracket_headings`,
:func:`side_labels`, :func:`drop_blank_lines`, :func:`normalise`, :func:`paragraphs_to_lines`,
:func:`words_subset`, :class:`Piece`, :func:`piece_from_lines`, :func:`cut_piece`, :func:`cut_span`,
:data:`NAS_PAGE_SOURCE`, :func:`entry_start_page`, :func:`entry_last_page`, :func:`entry_page_ranges`,
:class:`EntryPages`, :class:`BookDoc`, :func:`doc_from_lines_entry`, :func:`doc_from_paragraph_entry`,
:func:`doc_from_spans`, :data:`ROW_SPAN_KINDS`, :func:`row_page_keys`, :class:`PgpIndex`, :func:`match_pgpids`, :func:`hash_split`,
:func:`assign_split`, :func:`overlap_score`, :func:`pair_check`, :class:`BuildContext`,
:func:`translate_rows`, :func:`analyse_rows`, :func:`build`, :func:`write_outputs`, :func:`main`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import builder as _bd
from src.finetuning.text27b import builder_v0 as _bv
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)
BUILD = _rows.BUILD
FAMILY_FILE = "book_pairs"
MANIFEST_NAME = "book_pairs_manifest.json"
AUDIT_NAME = "book_pairs_sample_30.jsonl"
AUDIT_SIZE = 30
DEFAULT_CONFIG = Path("logs") / "text_finetune_27b" / "book_pairs_local.json"
DEFAULT_SPANS = Path("logs") / "text_finetune_27b" / "book_pairs_spans.json"
#: D14: hash split shares (train, dev, test_indist) in percent.
SPLIT_SHARES: Tuple[int, int, int] = (80, 10, 10)
#: Analysis gates (v0 tiers: A′ input + catalogue, B with the book translation appended).
GROUNDING_A = 0.5
GROUNDING_B = 0.6
ANALYSIS_MIN_CHARS = _bd.ANALYSIS_MIN_CHARS
MAX_ROW_TOKENS = _bd.MAX_ROW_TOKENS
#: Pairing check: below this many entity candidates an English pair check is uninformative.
MIN_CANDIDATES = 3
#: Hebrew pairing check floor (share of the translation's Hebrew words found in the edition).
HE_OVERLAP_FLOOR = 0.05
#: English pairing check floor (grounded share of the translation's entity candidates).
EN_OVERLAP_FLOOR = 0.1
#: Edition-text dedupe: word n-gram length and the share of a book edition's n-grams that one PGP
#: edition must hold for the two to count as the same text (formulaic openings stay below 0.3).
EDITION_GRAM = 4
EDITION_MATCH_SHARE = 0.4
_RTL_LETTER = re.compile(r"[א-ת؀-ۿ]")
_HE_WORD = re.compile(r"[א-ת][א-ת׳'\"״]*[א-ת]")
_MARKER_PAREN = re.compile(r"\((\d{1,3})\)")


# ============================================================================== configuration
def load_local_config(p: Optional[_paths.Paths] = None, config: Optional[Path] = None,
                      spans: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    """Read the local (git-ignored) book configuration and the located-spans file.

    :param p: Build paths (``book_pairs_config`` / ``book_pairs_spans`` keys override the defaults).
    :param config: Explicit config file.
    :param spans: Explicit spans file.
    :returns: ``(config, spans, config path, spans path)``; a missing file gives an empty dict.
    """
    p = p or _paths.load_paths()
    cpath = Path(config) if config else Path(p.get("book_pairs_config") or (p.repo / DEFAULT_CONFIG))
    spath = Path(spans) if spans else Path(p.get("book_pairs_spans") or (p.repo / DEFAULT_SPANS))
    cfg = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    spn = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    return cfg, spn, cpath, spath


# ============================================================================== text helpers
def label_of(side: Optional[str], part: Optional[str]) -> str:
    """Side / part label of a block (``recto``, ``verso, address``; '' when neither is given).

    :param side: Side name or None.
    :param part: Part name or None.
    :returns: The label text without the ``§`` sign.
    """
    return ", ".join(str(x).strip() for x in (side, part) if x and str(x).strip())


def group_lines(lines: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]
                ) -> List[Tuple[str, List[str]]]:
    """Group line records into blocks by a key, in order of first appearance.

    :param lines: Line records with a ``text`` field.
    :param key: Block key of a record (its label).
    :returns: ``[(label, [line texts])]``.
    """
    order: List[str] = []
    blocks: Dict[str, List[str]] = {}
    for ln in lines:
        k = key(ln)
        if k not in blocks:
            order.append(k)
            blocks[k] = []
        text = str(ln.get("text") or "").strip()
        if text:
            blocks[k].append(text)
    return [(k, blocks[k]) for k in order if blocks[k]]


def render_blocks(blocks: Sequence[Tuple[str, Sequence[str]]]) -> str:
    """Render blocks in the PGP edition style: ``§ label`` lines between blocks when there are several.

    :param blocks: ``[(label, lines)]``.
    :returns: Text (single unlabelled block: just its lines).
    """
    blocks = [(lab, list(lines)) for lab, lines in blocks if lines]
    if not blocks:
        return ""
    if len(blocks) == 1:
        return "\n".join(blocks[0][1])
    parts = []
    for lab, lines in blocks:
        head = [f"§ {lab}"] if lab else []
        parts.append("\n".join(head + lines))
    return "\n\n".join(parts)


def rendered_labels(blocks: Sequence[Tuple[str, Sequence[str]]]) -> List[str]:
    """Labels :func:`render_blocks` writes for these blocks (none for a single block).

    :param blocks: ``[(label, lines)]``.
    :returns: Labels in order.
    """
    kept = [lab for lab, lines in blocks if lines]
    return kept if len(kept) > 1 else []


def realign_parts(ed_lines: Sequence[Mapping[str, Any]], tr_lines: Sequence[Mapping[str, Any]]
                  ) -> Optional[List[Dict[str, Any]]]:
    """Copy the edition's part labels onto the translation lines when the parser mislabelled them.

    The entry parsers sometimes give every translation line of a side the first part label they met
    (all lines ``witness clauses``). When each side has as many translation lines as edition lines,
    the lines correspond one to one and the edition's ``(side, part)`` is copied positionally.

    :param ed_lines: Edition line records (``side``, ``part``).
    :param tr_lines: Translation line records.
    :returns: Relabelled copies of the translation lines, or None when the per-side counts differ.
    """
    def by_side(lines: Sequence[Mapping[str, Any]]) -> Dict[Any, List[Mapping[str, Any]]]:
        """Group records by side, keeping order.

        :param lines: Records.
        :returns: side → records.
        """
        out: Dict[Any, List[Mapping[str, Any]]] = {}
        for ln in lines:
            out.setdefault(ln.get("side"), []).append(ln)
        return out

    es, ts = by_side(ed_lines), by_side(tr_lines)
    if list(es) != list(ts) or any(len(es[k]) != len(ts[k]) for k in es):
        return None
    out: List[Dict[str, Any]] = []
    for side in ts:
        for e, t in zip(es[side], ts[side]):
            out.append({**t, "side": e.get("side"), "part": e.get("part")})
    return out


def reconcile_blocks(eb: List[Tuple[str, List[str]]], tb: List[Tuple[str, List[str]]]
                     ) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """Reconcile translation blocks with the edition's labels (paragraph layout).

    Equal rendered labels pass; a translation with one unlabelled block passes as printed
    (``translation_unlabelled``); a short leading unlabelled block (a chapter heading such as an
    editor's title) before otherwise matching labels is dropped (``heading_block_dropped``); anything
    else is flagged ``block_keys_differ``.

    :param eb: Edition blocks.
    :param tb: Translation blocks.
    :returns: ``(translation blocks, flags)``.
    """
    if rendered_labels(eb) == rendered_labels(tb):
        return tb, []
    if not rendered_labels(tb):
        return tb, ["translation_unlabelled"]
    if tb and tb[0][0] == "" and len(tb[0][1]) == 1 and len(tb[0][1][0]) <= 120 and \
            rendered_labels(tb[1:]) == rendered_labels(eb):
        return tb[1:], ["heading_block_dropped"]
    return tb, ["block_keys_differ"]


def rtl_letters(text: str) -> int:
    """Number of Hebrew / Arabic letters in a text.

    :param text: Any text.
    :returns: Count.
    """
    return len(_RTL_LETTER.findall(text or ""))


def join_hyphenated(left: str, right: str) -> str:
    """Join a line ending in a hyphen with the next line's text.

    The hyphen is dropped (``contrib-`` + ``uted``) unless both fragments are dictionary words and their
    concatenation is not (``fifty-`` + ``four`` keeps ``fifty-four``); without a word list it is dropped.

    :param left: Line ending with ``-``.
    :param right: Continuation text.
    :returns: Joined text.
    """
    a = re.search(r"([A-Za-zÀ-ɏḀ-ỿ]+)-$", left)
    b = re.match(r"([A-Za-zÀ-ɏḀ-ỿ]+)", right)
    words = tn.common_words()
    if a and b and words:
        wa, wb = a.group(1).lower(), b.group(1).lower()
        if wa in words and wb in words and (wa + wb) not in words:
            return left + right
    return left[:-1] + right


def rejoin_indented(text: str) -> Tuple[str, int]:
    """Re-join printed lines whose continuation lines lost the indent (line numbers set as spaces).

    A line that starts with whitespace opens a new edition line; a line that starts flush left
    continues the previous one (joined with a space, or by :func:`join_hyphenated` after a hyphen
    between letters). ``§`` label lines stand alone and the line after a label opens a new line.
    Whitespace-only lines are dropped.

    :param text: Raw span text.
    :returns: ``(text, number of joins)``.
    """
    out: List[str] = []
    joins = 0
    after_label = True
    for ln in text.split("\n"):
        if not ln.strip():
            continue
        if ln.strip().startswith("§"):
            out.append(ln.strip())
            after_label = True
            continue
        if out and not after_label and not ln[:1].isspace():
            prev = out[-1]
            if re.search(r"[A-Za-zÀ-ɏḀ-ỿ]-$", prev):
                out[-1] = join_hyphenated(prev, ln.strip())
            else:
                out[-1] = prev.rstrip() + " " + ln.strip()
            joins += 1
        else:
            out.append(ln.strip())
        after_label = False
    return "\n".join(out), joins


def rejoin_blank_separated(text: str) -> Tuple[str, int]:
    """Re-join printed lines that a text layer splits into several rows, with blank rows between lines.

    Consecutive non-blank rows form one printed line (joined with a space); a whitespace-only row ends
    it; a ``§`` label row stands alone.

    :param text: Raw span text.
    :returns: ``(text, number of joins)``.
    """
    out: List[str] = []
    cur: List[str] = []
    joins = 0
    for ln in text.split("\n") + [""]:
        label = ln.strip().startswith("§")
        if ln.strip() and not label:
            cur.append(ln.strip())
            continue
        if cur:
            joins += len(cur) - 1
            out.append(" ".join(cur))
            cur = []
        if label:
            out.append(ln.strip())
    return "\n".join(out), joins


def prose_paragraphs(text: str) -> Tuple[str, int]:
    """Turn wrapped prose lines into paragraphs (hyphenated breaks re-joined).

    A paragraph ends at a blank line, or after a line that ends a sentence and is clearly shorter than
    the typical line (< 75 % of the median width of lines over 20 characters).

    :param text: Raw prose span.
    :returns: ``(text, number of joins)``.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    widths = [len(x) for x in lines if len(x) > 20] or [60]
    med = statistics.median(widths)
    paras: List[str] = []
    cur = ""
    joins = 0
    for i, ln in enumerate(lines):
        if not ln:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        if not cur:
            cur = ln
        elif re.search(r"[A-Za-zÀ-ɏḀ-ỿ]-$", cur) and ln[:1].islower():
            cur = join_hyphenated(cur, ln)
            joins += 1
        else:
            cur = cur + " " + ln
            joins += 1
        if re.search(r"[.!?:”\"’)]$", ln) and len(ln) < 0.75 * med:
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    return "\n\n".join(paras), joins


_FOOTREF = re.compile(r"(?<![\d][.,:;/])(?<=[A-Za-zÀ-ɏḀ-ỿʿʾ’”\"'.,;:)\]?!])\d{1,3}(?![\d.,:/]\d)(?=[\s,.;:)\]—–-]|$)")
_FOOTREF_HE = re.compile(r"(?<![\d][.,:;/])(?<=[א-ת׳'\"”’.,;:)\]?!])\d{1,3}(?![\d.,:/]\d)(?=[\s,.;:)\]—–-]|$)")
_FOOTREF_LEAD = re.compile(r"(?<=\s)\d{1,3}(?=[א-ת])")


def drop_footnote_refs(text: str) -> Tuple[str, int]:
    """Remove footnote reference numbers glued to a word or punctuation mark (``Fustāt.37``, ``מנגו’9``).

    Only 1–3 digit runs directly after a letter or a closing punctuation mark (no space between) and
    followed by a space, punctuation or the line end are removed, plus a digit run glued to the front of
    a Hebrew word (``6אנוס``, a superscript the text layer put before the word). Numbers standing alone
    (amounts, years, ``l. 11``) and the parts of a number (``3.5``, ``3:2``, ``11/2``) are kept.

    :param text: Span text.
    :returns: ``(text, number of references removed)``.
    """
    n = 0
    for pat in (_FOOTREF, _FOOTREF_HE, _FOOTREF_LEAD):
        text, k = pat.subn("", text)
        n += k
    return text, n


def drop_digit_remnants(text: str) -> Tuple[str, int]:
    """Remove line-number remnants at the start of a line (``0 they finish``: the text layer kept the
    last digit of ``10``) and lines holding nothing but digits.

    :param text: Span text.
    :returns: ``(text, number of remnants removed)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        s = ln.strip()
        if re.fullmatch(r"\d{1,3}", s):
            n += 1
            continue
        m = re.match(r"^\s*\d{1,3}\s+(?=\S)", ln)
        if m and not re.match(r"^\s*\d{1,3}\s+(?:dinars?|dirhams?|qīrāṭs?|years?|days?|months?|ra?ṭls?)\b", ln, re.I):
            ln = " " + ln[m.end():]  # the remnant marked a new printed line: keep it indented
            n += 1
        out.append(ln)
    return "\n".join(out), n


_BRACKET_HEADING = re.compile(r"^\s*\{?[\[(]\s*[א-ת]{1,2}[.׳']\s[^\]})]{2,140}[\]})]\s*")


def drop_bracket_headings(text: str) -> Tuple[str, int]:
    """Remove editorial section headings set in brackets (``[א. קבלת מכתבו של הנמען]``) at the start of a
    line (a heading the OCR ran into the following text is cut off; a line holding only it is dropped).

    :param text: Span text.
    :returns: ``(text, number of headings removed)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        m = _BRACKET_HEADING.match(ln)
        if m:
            n += 1
            ln = ln[m.end():]
            if not ln.strip():
                continue
        out.append(ln)
    return "\n".join(out), n


_SIDE_LINE = re.compile(r"^\s*(recto|verso|margin|address|right margin|left margin|top margin|bottom margin|"
                        r"ע\"א|ע\"ב|ע״א|ע״ב|שוליים)\s*[:.]?\s*$", re.I)
_SIDE_HE = {"ע\"א": "recto", "ע״א": "recto", "ע\"ב": "verso", "ע״ב": "verso", "שוליים": "margin", "השוליים": "margin",
            "כתובת": "address"}


def side_labels(text: str) -> Tuple[str, int]:
    """Turn bare side / margin heading lines (``Recto``, ``Verso``, ``[ע"ב]``) into ``§`` labels.

    :param text: Span text.
    :returns: ``(text, number of labels)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        s = ln.strip().strip("[]")
        if _SIDE_LINE.match(s):
            word = _SIDE_HE.get(s.strip(":. "), s.strip(":. ").lower())
            out.append(f"§ {word}")
            n += 1
        else:
            out.append(ln)
    return "\n".join(out), n


_LATIN_RUN = re.compile(r"[^\u0590-\u06FF\n]*[A-Za-zÀ-ɏḀ-ỿ][^\u0590-\u06FF\n]*")


def strip_latin_runs(line: str) -> Tuple[str, int]:
    """Remove Latin-script runs (editor's notes, lead-ins, ``MS ... fol. 7r``) from a Hebrew / Arabic-script line.

    A run is a maximal stretch without Hebrew / Arabic characters that holds three or more Latin letters;
    shorter runs (``p.``, single sigla) and digits / punctuation between RTL words are kept.

    :param line: One edition line.
    :returns: ``(line, number of runs removed)``.
    """
    n = 0

    def sub(m: re.Match) -> str:
        """Replace a qualifying run by a space.

        :param m: Match of a non-RTL run.
        :returns: Replacement.
        """
        nonlocal n
        if len(re.findall(r"[A-Za-zÀ-ɏḀ-ỿ]", m.group(0))) >= 3:
            n += 1
            return " "
        return m.group(0)

    out = _LATIN_RUN.sub(sub, line)
    return (re.sub(r"[ \t]{2,}", " ", out).strip() if n else line), n


_HEADING_LABEL = re.compile(r"^\s*(?:(?:Transcription|Translation) of (?P<sm>.+?)|(?P<cont>Continuation of [^:]{1,80}"
                            r"|(?:Text|Written|Continuing|Continued)\b[^:]{0,80}\b(?:margin|degrees|upside down)\b[^:]{0,40}):?"
                            r"(?P<rest>.*))\s*$")


def heading_labels(text: str) -> Tuple[str, int]:
    """Turn sub-document headings into ``§`` labels (``Transcription of TS 16.23`` → ``§ TS 16.23``;
    ``Continuation of Document on Verso`` → ``§ Continuation of Document on Verso``, text after a
    colon kept on the next line).

    :param text: Span text.
    :returns: ``(text, number of labels)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        m = _HEADING_LABEL.match(ln)
        if not m:
            out.append(ln)
            continue
        n += 1
        if m.group("sm"):
            out.append(f"§ {m.group('sm').strip()}")
        else:
            out.append(f"§ {m.group('cont').strip()}")
            if m.group("rest").strip():
                out.append(" " + m.group("rest").strip())
    return "\n".join(out), n


def latin_labels(text: str) -> Tuple[str, int]:
    """In a Hebrew / Arabic-script edition span, turn editorial Latin-only lines into ``§`` labels.

    :param text: Edition span text.
    :returns: ``(text, number of labels)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        s = ln.strip()
        if s and not s.startswith("§") and not rtl_letters(s) and len(re.findall(r"[A-Za-zÀ-ɏḀ-ỿ]", s)) >= 3:
            out.append(f"§ {s.rstrip(':')}")
            n += 1
        else:
            out.append(ln)
    return "\n".join(out), n


_LATIN_BRACKET = re.compile(r"[\[{(][^\[\]{}()א-ת؀-ۿ\n]*[A-Za-z£][^\[\]{}()א-ת؀-ۿ\n]*[\]})]")


def drop_latin_brackets(text: str) -> Tuple[str, int]:
    """Remove bracketed Latin-script sigla inside Hebrew / Arabic-script prose (``[TS 1 J 2, f. 3]``,
    OCR-reversed ``[3.2,f J 1 TS]``): groups in ``[]``, ``{}`` or ``()`` holding Latin letters and no
    Hebrew / Arabic letters.

    :param text: Span text.
    :returns: ``(text, number of groups removed)``.
    """
    out, n = _LATIN_BRACKET.subn(" ", text)
    return (re.sub(r"[ \t]{2,}", " ", out) if n else text), n


_BARE_NUMBER = re.compile(r"(?<=\s)\d{1,2}(?=\s|$|[.,;:])")


_OCR_REFS = (
    re.compile(r"[\[{]{1,2}\s*I?[א-ת]?\d{1,3}[א-ת]?\s*[\]}]{0,2}(?=\s|$|[.,;:])"),    # [61], [[51], [ג6א], {Iג15
    re.compile(r"[\[{]\s*[א-ת]\s*[\]}]"),                                         # [ג]
    re.compile(r"(?<=\s)[*“”]\d{1,3}[′’]?(?=\s|$|[.,;:])"),                         # *2  “4
    re.compile(r"(?<=[^\s\d(])[*/“”]?\d{1,3}[′’]?(?=\s|$|[.,;:\]}])"),             # טַלְחִי',23  ה'מחודש20′  /18
    re.compile(r"(?<=[א-ת'\"”’)])[*]\d?(?=\s|$|[.,;:])"),                           # מזה*  לשמור"*
)


def drop_ocr_refs(text: str) -> Tuple[str, int]:
    """Remove OCR residue of footnote references and page / fragment sigla inside Hebrew-script prose:
    bracketed page numbers (``[61]``), fragment sigla (``[ג6א]``), digits glued to a word with an
    asterisk, slash or prime (``*2``, ``/18``, ``מחודש20′``). Only lines with Hebrew letters are touched.

    :param text: Span text.
    :returns: ``(text, number of removals)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        if re.search(r"[א-ת]", ln):
            for pat in _OCR_REFS:
                ln, k = pat.subn("", ln)
                n += k
            ln = re.sub(r"[ \t]{2,}", " ", ln)
        out.append(ln)
    return "\n".join(out), n


def drop_bare_numbers(text: str) -> Tuple[str, int]:
    """Remove free-standing 1–2 digit numbers from Hebrew-script translation lines (footnote references the
    OCR set apart from the word; the translation writes quantities in letters and line markers in
    parentheses, which are kept).

    :param text: Span text.
    :returns: ``(text, number of numbers removed)``.
    """
    out: List[str] = []
    n = 0
    for ln in text.split("\n"):
        if re.search(r"[א-ת]", ln):
            ln, k = _BARE_NUMBER.subn("", ln)
            n += k
            if k:
                ln = re.sub(r"[ \t]{2,}", " ", ln).replace(" .", ".").replace(" ,", ",")
        out.append(ln)
    return "\n".join(out), n


_INLINE_SIDE = re.compile(r"[\[{(]\s*(ע[\"״]א|ע[\"״]ב|שוליים|השוליים|כתובת|recto|verso|margin)\s*[\]})]", re.I)


def split_at_markers(text: str) -> Tuple[str, int]:
    """Re-line a translation printed as prose with inline line markers ``(N)``.

    Lines are joined (a line ending in a maqaf ``־`` joins without a space), bracketed side labels
    (``[ע"ב]``, ``[שוליים]``) become ``§`` label lines, and with two or more markers the text is split
    into one line per marker, the markers removed; with fewer markers each label-delimited part stays
    one line.

    :param text: Span text.
    :returns: ``(text, number of markers removed)``.
    """
    joined = ""
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        joined = s if not joined else (joined + s if joined.endswith("־") else joined + " " + s)
    joined = _INLINE_SIDE.sub(lambda m: "\x00" + _SIDE_HE.get(m.group(1), m.group(1).lower()) + "\x00", joined)
    n_markers = len(_MARKER_PAREN.findall(joined))
    split = n_markers >= 2
    out: List[str] = []
    for part in re.split(r"(\x00[^\x00]*\x00)", joined):
        if part.startswith("\x00"):
            out.append(f"§ {part.strip(chr(0)).strip()}")
            continue
        segs = _MARKER_PAREN.split(part)[::2] if split else [part]
        out.extend(seg.strip() for seg in segs if seg.strip())
    return "\n".join(out), (n_markers if split else 0)


def drop_blank_lines(text: str) -> Tuple[str, int]:
    """Remove whitespace-only lines and trailing spaces.

    :param text: Span text.
    :returns: ``(text, number of lines removed)``.
    """
    lines = text.split("\n")
    out = [ln.rstrip() for ln in lines if ln.strip()]
    return "\n".join(out), len(lines) - len(out)


#: Normalisation ops a span may name (each deletes characters or joins lines; none adds words).
NORMALISERS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "rejoin_indented": rejoin_indented,
    "rejoin_blank_separated": rejoin_blank_separated,
    "prose_paragraphs": prose_paragraphs,
    "drop_footnote_refs": drop_footnote_refs,
    "drop_digit_remnants": drop_digit_remnants,
    "drop_bracket_headings": drop_bracket_headings,
    "side_labels": side_labels,
    "heading_labels": heading_labels,
    "latin_labels": latin_labels,
    "drop_latin_brackets": drop_latin_brackets,
    "drop_bare_numbers": drop_bare_numbers,
    "drop_ocr_refs": drop_ocr_refs,
    "split_at_markers": split_at_markers,
    "drop_blank_lines": drop_blank_lines,
}


def normalise(text: str, ops: Sequence[str]) -> Tuple[str, List[str]]:
    """Apply named normalisation ops in order.

    :param text: Raw span text.
    :param ops: Names from :data:`NORMALISERS`.
    :returns: ``(text, applied op names prefixed L5:)`` (an op that changed nothing is not listed).
    :raises KeyError: On an unknown op.
    """
    done: List[str] = []
    for op in ops:
        text, n = NORMALISERS[op](text)
        if n:
            done.append(f"L5:{op}")
    return re.sub(r"[ \t]{2,}", " ", text).strip(), done


def _words(text: str) -> List[str]:
    """Word tokens (letters only, lower-cased) of a text.

    :param text: Any text.
    :returns: Tokens.
    """
    return [w.lower() for w in re.findall(r"[^\W\d_]+", text or "")]


def words_subset(target: str, raw: str) -> Tuple[bool, List[str]]:
    """Check that normalisation added no words: every word of the target occurs in the raw text.

    Words are compared after joining hyphenated breaks of the raw text, and a target word may also be
    the concatenation of two raw words across a removed hyphen / line break. ``§`` label lines are
    structure derived from the source's side / part markers (``[ע"ב]`` → ``§ verso``) and are not checked.

    :param target: Final target.
    :param raw: Verbatim raw span (or entry text).
    :returns: ``(ok, up to 10 missing words)``.
    """
    joined = re.sub(r"([^\W\d_])-\s*\n\s*([^\W\d_])", r"\1\2", raw or "")
    pool = set(_words(joined)) | set(_words(raw))
    squeezed = re.sub(r"\s+", "", joined.lower())
    body = "\n".join(ln for ln in (target or "").split("\n") if not ln.strip().startswith("§"))
    missing = [w for w in dict.fromkeys(_words(body)) if w not in pool and w not in squeezed]
    return not missing, missing[:10]


def paragraphs_to_lines(paragraphs: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Turn translation paragraphs with line markers into translation lines (markers removed).

    A paragraph starting with ``N.`` opens a line; a paragraph with two or more inline ``(N)`` markers
    in mostly ascending order is split at them; an unnumbered paragraph that starts in lower case
    continues the previous line; a paragraph that is one bracketed editorial note (``[Written in the
    hand of ...]``) is dropped.

    :param paragraphs: Paragraph texts in order.
    :returns: ``(lines, ops)``; ops name what was done (``split_paren_markers``, ``numbered_paragraphs``,
        ``continuation_joined``, ``editorial_note_dropped``).
    """
    lines: List[str] = []
    ops: List[str] = []

    def note(op: str) -> None:
        """Record an op once.

        :param op: Op name.
        """
        if op not in ops:
            ops.append(op)

    for para in paragraphs:
        p = re.sub(r"\s+", " ", str(para or "")).strip()
        if not p:
            continue
        if re.fullmatch(r"\[[^\[\]]{12,}\]", p) and not _MARKER_PAREN.search(p) and len(p.split()) >= 3 \
                and not re.fullmatch(r"\[[.…\s]*\]", p):
            note("editorial_note_dropped")
            continue
        nums = [int(m.group(1)) for m in _MARKER_PAREN.finditer(p)]
        if len(nums) >= 2 and sum(1 for a, b in zip(nums, nums[1:]) if b > a) >= 0.6 * (len(nums) - 1):
            pieces = _MARKER_PAREN.split(p)
            head = pieces[0].strip()
            if head:
                if lines and head[:1].islower():
                    lines[-1] = lines[-1] + " " + head
                else:
                    lines.append(head)
            for i in range(1, len(pieces), 2):
                seg = pieces[i + 1].strip() if i + 1 < len(pieces) else ""
                if seg:
                    lines.append(seg)
            note("split_paren_markers")
            continue
        m = re.match(r"^(\d{1,3})\.\s+(?=\S)", p)
        if m:
            lines.append(p[m.end():])
            note("numbered_paragraphs")
            continue
        if lines and p[:1].islower():
            lines[-1] = lines[-1] + " " + p
            note("continuation_joined")
            continue
        lines.append(p)
    return lines, ops


# ============================================================================== located spans
@dataclass
class Piece:
    """One verbatim piece of a span: a line range of one page's OCR text.

    ``lines`` is inclusive ``[first, last]`` over ``ocr_text.split("\\n")``; ``cols`` optionally trims
    the first line from column ``cols[0]`` and the last line to column ``cols[1]`` (None = whole line).
    ``head`` / ``tail`` are the first / last 24 characters of the cut text and ``sha1`` its digest.
    """

    page: str
    lines: Tuple[int, int]
    head: str = ""
    tail: str = ""
    sha1: str = ""
    cols: Tuple[Optional[int], Optional[int]] = (None, None)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Piece":
        """Parse a piece from the spans file.

        :param d: Piece dict.
        :returns: The piece.
        """
        cols = d.get("cols") or [None, None]
        return cls(page=str(d["page"]), lines=(int(d["lines"][0]), int(d["lines"][1])), head=str(d.get("head", "")),
                   tail=str(d.get("tail", "")), sha1=str(d.get("sha1", "")), cols=(cols[0], cols[1]))

    def to_dict(self) -> Dict[str, Any]:
        """Spans-file form of the piece.

        :returns: Dict (``cols`` omitted when unused).
        """
        d: Dict[str, Any] = {"page": self.page, "lines": list(self.lines), "head": self.head, "tail": self.tail,
                             "sha1": self.sha1}
        if self.cols != (None, None):
            d["cols"] = list(self.cols)
        return d


def _sha1(text: str) -> str:
    """sha1 hex digest of a text.

    :param text: Text.
    :returns: 40 hex characters.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cut(page_text: str, lines: Tuple[int, int], cols: Tuple[Optional[int], Optional[int]]) -> str:
    """Cut a line range (with optional first / last line columns) out of a page text.

    :param page_text: Full OCR text of the page.
    :param lines: Inclusive line range.
    :param cols: Column trims of the first / last line.
    :returns: The cut text.
    :raises ValueError: When the range lies outside the page.
    """
    ls = page_text.split("\n")
    a, b = lines
    if not (0 <= a <= b < len(ls)):
        raise ValueError(f"line range {lines} outside page of {len(ls)} lines")
    seg = ls[a:b + 1]
    if cols[1] is not None:
        seg[-1] = seg[-1][:cols[1]]
    if cols[0] is not None:
        seg[0] = seg[0][cols[0]:]
    return "\n".join(seg)


def piece_from_lines(page: str, page_text: str, first: int, last: int,
                     cols: Tuple[Optional[int], Optional[int]] = (None, None)) -> Piece:
    """Make a piece (with head, tail and sha1) for a line range of a page.

    :param page: Page key.
    :param page_text: Full OCR text of the page.
    :param first: First line index.
    :param last: Last line index (inclusive).
    :param cols: Optional column trims.
    :returns: The piece.
    """
    text = _cut(page_text, (first, last), cols)
    return Piece(page=page, lines=(first, last), head=text[:24], tail=text[-24:], sha1=_sha1(text), cols=cols)


def cut_piece(piece: Piece, page_text: Optional[str]) -> Tuple[Optional[str], str]:
    """Re-cut a piece from its page and verify it (D15: verbatim substring of the OCR text).

    :param piece: The piece.
    :param page_text: The page's OCR text (None when the page or its OCR is missing).
    :returns: ``(text, "ok")`` or ``(None, reason)`` with reason ``page_missing``, ``range``,
        ``head_mismatch``, ``tail_mismatch`` or ``sha1_mismatch``.
    """
    if page_text is None:
        return None, "page_missing"
    try:
        text = _cut(page_text, piece.lines, piece.cols)
    except ValueError:
        return None, "range"
    if piece.head and not text.startswith(piece.head):
        return None, "head_mismatch"
    if piece.tail and not text.endswith(piece.tail):
        return None, "tail_mismatch"
    if piece.sha1 and _sha1(text) != piece.sha1:
        return None, "sha1_mismatch"
    if text not in page_text:
        return None, "not_verbatim"
    return text, "ok"


def cut_span(span: Mapping[str, Any], page_texts: Mapping[str, Optional[str]]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Cut every piece of a span and join them (one newline between pieces).

    :param span: ``{"pieces": [...], "normalise": [...]}``.
    :param page_texts: page key → OCR text.
    :returns: ``(raw text, info)``; raw is None when any piece fails (``info["fail"]`` names it).
    """
    texts: List[str] = []
    pages: List[str] = []
    for d in span.get("pieces") or []:
        pc = Piece.from_dict(d)
        text, why = cut_piece(pc, page_texts.get(pc.page))
        if text is None:
            return None, {"fail": why, "page": pc.page, "lines": list(pc.lines)}
        texts.append(text)
        if pc.page not in pages:
            pages.append(pc.page)
    if not texts:
        return None, {"fail": "no_pieces"}
    return "\n".join(texts), {"pages": pages, "pieces": len(texts)}


# ============================================================================== entry pages (fix pass F11)
#: Pages the last entry of a book runs past its start page when the entry records no end page.
ENTRY_TAIL_PAGES = 2
#: ``meta.page_keys_source`` of rows whose page keys come from the entry → PDF page-range mapping (F11).
NAS_PAGE_SOURCE = "nas_entry"


def _page_int(value: Any) -> Optional[int]:
    """An integer page number of a raw entry field (``"32"`` / ``32``), else None.

    :param value: Raw value.
    :returns: The number or None.
    """
    return int(value) if value is not None and str(value).strip().isdigit() else None


def entry_start_page(entry: Mapping[str, Any]) -> Optional[int]:
    """PDF start page of a parsed edition entry: ``pdf_page``, else the printed ``page`` (the INT rule).

    :param entry: Entry record (or a document provenance with the same fields).
    :returns: The page or None.
    """
    nums = [n for n in (_page_int(entry.get("page")), _page_int(entry.get("pdf_page"))) if n is not None]
    return nums[-1] if nums else None


def entry_last_page(entry: Mapping[str, Any]) -> Optional[int]:
    """Last PDF page an entry records itself: ``pdf_pages`` (``end`` and every listed page) and its printed
    ``pages`` moved by the entry's printed → PDF offset.

    :param entry: Entry record.
    :returns: The largest such page or None.
    """
    found: List[int] = []
    pp = entry.get("pdf_pages")
    if isinstance(pp, Mapping):
        for v in pp.values():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                n = _page_int(x)
                if n is not None:
                    found.append(n)
    elif isinstance(pp, (list, tuple)):
        found += [n for n in (_page_int(x) for x in pp) if n is not None]
    page, pdf = _page_int(entry.get("page")), _page_int(entry.get("pdf_page"))
    if page is not None and pdf is not None:
        found += [n + pdf - page for n in (_page_int(x) for x in entry.get("pages") or []) if n is not None]
    return max(found) if found else None


def entry_page_ranges(entries: Sequence[Mapping[str, Any]], tail: int = ENTRY_TAIL_PAGES) -> Dict[str, Tuple[int, int]]:
    """PDF page range of every entry of one book (the mapping of ``int_nontrain_pages``, F11): from its start page
    up to the next entry's start page inclusive (an entry ends on the page where the next begins), widened to the
    last page the entry records itself; the last entry without a recorded end runs ``tail`` pages.

    :param entries: All entries of one book (any order).
    :param tail: Pages after the start of the last entry when it records no end.
    :returns: ``entry_no -> (first, last)``; entries without a start page are left out.
    """
    starts = sorted((st, str(e.get("entry_no"))) for e in entries
                    if (st := entry_start_page(e)) is not None and e.get("entry_no") not in (None, ""))
    own = {str(e.get("entry_no")): entry_last_page(e) for e in entries}
    out: Dict[str, Tuple[int, int]] = {}
    for i, (st, no) in enumerate(starts):
        nxt = next((s for s, _ in starts[i + 1:] if s > st), None)
        hi = nxt if nxt is not None else st + tail
        if own.get(no) is not None:
            hi = max(hi, own[no])
        out[no] = (st, max(st, hi))
    return out


@dataclass
class EntryPages:
    """Literature page keys of parsed edition entries (F11): a book's work directory (local configuration) and the
    entry's PDF page range (:func:`entry_page_ranges`) give the pages of that work with those page numbers.

    :ivar works: Book slug / directory → literature work directory (None: not mapped).
    :ivar page_index: ``(work, page number)`` → page keys.
    :ivar ranges: Book slug → entry_no → ``(first, last)`` PDF pages.
    """

    works: Dict[str, Optional[str]]
    page_index: Dict[Tuple[str, int], List[str]]
    ranges: Dict[str, Dict[str, Tuple[int, int]]] = field(default_factory=dict)

    @staticmethod
    def work_for(slug: str, configured: Optional[str], known_works: Iterable[str]) -> Optional[str]:
        """Work directory of a book: the configured one, else the unique known work whose name contains the slug.

        :param slug: Book slug or directory.
        :param configured: Work named in the local configuration (may be None).
        :param known_works: Literature work directories.
        :returns: The work or None.
        """
        known = set(known_works)
        if configured:
            return configured if configured in known else None
        cand = [w for w in known if slug and slug in w]
        return cand[0] if len(cand) == 1 else None

    @classmethod
    def from_pages(cls, books: Mapping[str, Optional[str]], pages: Iterable[Any]) -> "EntryPages":
        """Index literature pages for a set of books.

        :param books: Book slug / directory → configured work (None: derive from the slug).
        :param pages: Page records with ``work``, ``page_no`` and ``page_key`` (:class:`pages.PageRec`).
        :returns: The mapper.
        """
        index: Dict[Tuple[str, int], List[str]] = collections.defaultdict(list)
        for pg in pages:
            index[(str(pg.work), int(pg.page_no))].append(str(pg.page_key))
        known = {w for w, _ in index}
        works = {slug: cls.work_for(slug, conf, known) for slug, conf in books.items()}
        return cls(works=works, page_index={k: sorted(v) for k, v in index.items()})

    @classmethod
    def load(cls, p: _paths.Paths, cfg: Mapping[str, Any], extra_slugs: Iterable[str] = ()) -> "EntryPages":
        """Mapper for the configured books (``nas_books``: ``slug`` / ``dir`` / optional ``work``) plus further book
        directories (L2 reads every directory holding ``entries.jsonl``); pages are read from the works found.

        :param p: Build paths.
        :param cfg: Local book configuration.
        :param extra_slugs: Further book slugs / directories (work derived from the name).
        :returns: The mapper.
        """
        from src.finetuning.text27b import pages as _pages  # heavy import only when entries exist
        books: Dict[str, Optional[str]] = {}
        for b in cfg.get("nas_books") or []:
            for name in {str(b.get("slug") or ""), str(b.get("dir") or "")} - {""}:
                books[name] = b.get("work")
        for s in extra_slugs:
            books.setdefault(str(s), None)
        root = Path(p.academic_literature)
        known = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
        wanted = sorted({w for s, c in books.items() if (w := cls.work_for(s, c, known))})
        pgs = _pages.iter_pages(root, works=wanted, with_ocr=False) if wanted else []
        mapper = cls.from_pages(books, pgs)
        LOG.info("entry pages: %d books, works %s", len(books), sorted({w for w in mapper.works.values() if w}))
        return mapper

    def add_book(self, slug: str, entries: Sequence[Mapping[str, Any]]) -> None:
        """Register the page ranges of a book's entries.

        :param slug: Book slug / directory.
        :param entries: All its entries.
        :returns: None.
        """
        self.ranges[str(slug)] = entry_page_ranges(entries)

    def keys(self, slug: str, entry_no: Any) -> Tuple[List[str], Optional[str]]:
        """Page keys of one entry.

        :param slug: Book slug / directory (registered with :meth:`add_book`).
        :param entry_no: Entry number.
        :returns: ``(keys, None)``, or ``([], reason)`` when the entry cannot be mapped (``no_work``,
            ``no_start_page``, ``no_pages``).
        """
        work = self.works.get(str(slug))
        if not work:
            return [], "no_work"
        rng = self.ranges.get(str(slug), {}).get(str(entry_no))
        if rng is None:
            return [], "no_start_page"
        keys = sorted({k for n in range(rng[0], rng[1] + 1) for k in self.page_index.get((work, n), [])})
        return (keys, None) if keys else ([], "no_pages")

    def describe(self, slug: str, entry_no: Any) -> Dict[str, Any]:
        """Work and PDF range of an entry, for ``meta.entry_pages``.

        :param slug: Book slug / directory.
        :param entry_no: Entry number.
        :returns: ``{"work", "first", "last"}`` (None values when unknown).
        """
        rng = self.ranges.get(str(slug), {}).get(str(entry_no))
        return {"work": self.works.get(str(slug)), "first": rng[0] if rng else None, "last": rng[1] if rng else None}


# ============================================================================== documents
@dataclass
class BookDoc:
    """One book document with its edition (input) and human translation (target).

    ``edition_blocks`` / ``translation_blocks`` keep the per-side / per-part structure for splitting
    over-long documents; ``translation_raw`` is the text the target was cut from (entry lines or the
    verbatim span) for the word-subset check.
    """

    source: str
    doc: str
    shelfmarks: List[str]
    target_lang: str
    edition: str
    translation: str
    translation_raw: str
    script: str
    lang_class: str
    input_source: str
    licence: Optional[str]
    licence_class: Optional[str]
    edition_blocks: List[Tuple[str, List[str]]] = field(default_factory=list)
    translation_blocks: List[Tuple[str, List[str]]] = field(default_factory=list)
    intro: Optional[str] = None
    intro_raw: Optional[str] = None
    commentary: Optional[str] = None
    pgp_hints: List[str] = field(default_factory=list)
    pgp_number: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    norm_ops: List[str] = field(default_factory=list)
    located: bool = False
    entry_page_keys: Optional[List[str]] = None
    entry_pages_unmapped: Optional[str] = None

    @property
    def doc_id(self) -> str:
        """Stable document id ``book:<source>:<doc>``.

        :returns: The id.
        """
        return f"book:{self.source}:{self.doc}"

    @property
    def keys(self) -> List[str]:
        """Shelfmark keys of all shelfmarks of the document.

        :returns: Keys (order kept, no duplicates).
        """
        out: List[str] = []
        for s in self.shelfmarks:
            for k in _sm.keys(s):
                if k not in out:
                    out.append(k)
        return out


def _lang_class(script: str, languages: Sequence[str] = ()) -> str:
    """lang_class of a book document from its edition script and stated languages.

    :param script: Dominant script of the edition (``hebrew`` / ``arabic`` / ...).
    :param languages: Languages named by the book (e.g. ``["Judaeo-Arabic"]``).
    :returns: ``AR`` for Arabic script, ``HE`` for Hebrew-language Hebrew script, else ``JA``.
    """
    if script == "arabic":
        return "AR"
    langs = " ".join(languages).lower()
    if "hebrew" in langs and "arabic" not in langs:
        return "HE"
    if "aramaic" in langs and "arabic" not in langs:
        return "ARAM"
    return "JA"


def _book_meta(book: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Licence string and class of a configured book.

    :param book: Book config entry.
    :returns: ``(licence, licence_class)``.
    """
    return book.get("licence"), book.get("licence_class")


def doc_from_lines_entry(entry: Mapping[str, Any], book: Mapping[str, Any]) -> BookDoc:
    """BookDoc from a ``lines``-layout entry (``transcription_lines`` / ``translation_lines``).

    Edition lines without a Hebrew / Arabic letter (editorial rows such as ``CHECK MARK``) are left out.

    :param entry: One record of ``entries.jsonl``.
    :param book: Book config entry (``slug``, ``target_lang``, licence fields).
    :returns: The document.
    """
    def key(ln: Mapping[str, Any]) -> str:
        """Block label of a line.

        :param ln: Line record.
        :returns: Label.
        """
        return label_of(ln.get("side"), ln.get("part"))

    all_ed = list(entry.get("transcription_lines") or [])
    ed_lines = [ln for ln in all_ed if rtl_letters(str(ln.get("text") or ""))]
    tr_lines = list(entry.get("translation_lines") or [])
    eb, tb = group_lines(ed_lines, key), group_lines(tr_lines, key)
    flags: List[str] = []
    if rendered_labels(eb) != rendered_labels(tb):
        fixed = realign_parts(all_ed, tr_lines) or realign_parts(ed_lines, tr_lines)
        if fixed is not None and rendered_labels(group_lines(fixed, key)) == rendered_labels(eb):
            tr_lines, tb = fixed, group_lines(fixed, key)
            flags.append("translation_parts_realigned")
        else:
            flags.append("block_keys_differ")
    edition, translation = render_blocks(eb), render_blocks(tb)
    script = tn.dominant_script(edition)
    lic, lcl = _book_meta(book)
    pgp = (entry.get("pgp") or {}).get("pgpids") or []
    return BookDoc(source=str(book["slug"]), doc=str(entry.get("entry_no")),
                   shelfmarks=[s for s in (entry.get("shelfmarks_pgp") or entry.get("shelfmarks") or
                                           [entry.get("shelfmark")]) if s],
                   target_lang=str(book.get("target_lang", "en")), edition=edition, translation=translation,
                   translation_raw=translation + "\n" + "\n".join(str(x.get("text") or "") for x in tr_lines),
                   script=script,
                   lang_class=_lang_class(script, entry.get("languages") or []),
                   input_source=f"book_edition:{book['slug']}", licence=lic, licence_class=lcl,
                   edition_blocks=eb, translation_blocks=tb,
                   commentary="\n\n".join(x for x in (entry.get("discussion"), entry.get("textual_notes")) if x) or None,
                   pgp_hints=[str(x) for x in pgp], pgp_number=str(entry.get("entry_no")),
                   provenance={"entry_no": entry.get("entry_no"), "page": entry.get("page"),
                               "pdf_page": entry.get("pdf_page"), "quality_flags": entry.get("quality_flags") or []},
                   flags=flags)


def doc_from_paragraph_entry(entry: Mapping[str, Any], book: Mapping[str, Any],
                             scripts: Sequence[str] = ("arabic", "hebrew")) -> BookDoc:
    """BookDoc from a ``paragraphs``-layout entry (per-line scripts; translation paragraphs with markers).

    Edition: transcription lines in ``scripts`` only (Latin transliteration rows, Syriac, Mandaic left
    out), grouped by part. Target: :func:`paragraphs_to_lines` per part of ``translation_paragraphs``
    (falling back to ``translation_lines`` when there are no paragraphs).

    :param entry: One record of ``entries.jsonl``.
    :param book: Book config entry.
    :param scripts: Line scripts kept in the edition.
    :returns: The document (``script`` reflects the kept lines; ``none`` when nothing is kept).
    """
    def key(ln: Mapping[str, Any]) -> str:
        """Block label of a line or paragraph.

        :param ln: Record.
        :returns: Label.
        """
        return label_of(ln.get("side"), ln.get("part"))

    ed_lines = []
    n_latin = 0
    for ln in entry.get("transcription_lines") or []:
        if ln.get("script") not in scripts:
            continue
        text, k = strip_latin_runs(str(ln.get("text") or ""))
        n_latin += k
        if rtl_letters(text):
            ed_lines.append({**ln, "text": text})
    eb = group_lines(ed_lines, key)
    paras = entry.get("translation_paragraphs") or []
    ops: List[str] = []
    if paras:
        tb = []
        for lab, texts in group_lines(paras, key):
            lines, o = paragraphs_to_lines(texts)
            ops += [x for x in o if x not in ops]
            if lines:
                tb.append((lab, lines))
        raw = "\n".join(str(p.get("text") or "") for p in paras)
    else:
        tb = group_lines(entry.get("translation_lines") or [], key)
        raw = "\n".join(str(p.get("text") or "") for p in entry.get("translation_lines") or [])
    tb, flags = reconcile_blocks(eb, tb)
    edition, translation = render_blocks(eb), render_blocks(tb)
    script = tn.dominant_script(edition) if edition else "none"
    raw = raw + "\n" + "\n".join(f"§ {lab}" for lab in rendered_labels(tb))
    lic, lcl = _book_meta(book)
    return BookDoc(source=str(book["slug"]), doc=str(entry.get("entry_no")),
                   shelfmarks=[s for s in (entry.get("shelfmarks") or []) if s and re.search(r"\d", s)],
                   target_lang=str(book.get("target_lang", "en")), edition=edition, translation=translation,
                   translation_raw=raw, script=script, lang_class=_lang_class(script, entry.get("languages") or []),
                   input_source=f"book_edition:{book['slug']}", licence=lic, licence_class=lcl,
                   edition_blocks=eb, translation_blocks=tb, commentary=entry.get("commentary") or None,
                   pgp_number=str(entry.get("entry_no")),
                   provenance={"entry_no": entry.get("entry_no"), "pdf_page": entry.get("pdf_page"),
                               "entry_script": entry.get("script"), "quality_flags": entry.get("quality_flags") or [],
                               "edition_latin_runs_removed": n_latin},
                   flags=flags, norm_ops=[f"L5:{o}" for o in ops])


def doc_from_spans(src: Mapping[str, Any], d: Mapping[str, Any], page_texts: Mapping[str, Optional[str]],
                   pgp_edition: Optional[Callable[[Sequence[str], Sequence[str]], Optional[Tuple[str, str, str]]]] = None
                   ) -> Tuple[Optional[BookDoc], Dict[str, Any]]:
    """BookDoc from located spans (D15): cut, verify and normalise the translation / edition / intro spans.

    :param src: Span source config (``slug``, ``target_lang``, ``input``, licence fields).
    :param d: One ``docs`` entry (``doc``, ``shelfmarks``, ``pgpids``, ``lang_class``, ``spans``).
    :param page_texts: page key → OCR text.
    :param pgp_edition: For ``input: pgp_edition``: callable ``(pgpids, shelfmarks) -> (edition, slug,
        pgpid)`` or None when PGP has no edition.
    :returns: ``(doc or None, info)``; ``info["drop"]`` holds the reason when no document results.
    """
    spans = d.get("spans") or {}
    info: Dict[str, Any] = {"doc": d.get("doc")}
    out: Dict[str, Tuple[str, str, List[str]]] = {}
    for kind, span in spans.items():
        raw, ci = cut_span(span, page_texts)
        if raw is None:
            info.setdefault("span_failures", {})[kind] = ci
            continue
        text, ops = normalise(raw, span.get("normalise") or [])
        ok, missing = words_subset(text, raw)
        if not ok:
            info.setdefault("span_failures", {})[kind] = {"fail": "words_added", "missing": missing}
            continue
        out[kind] = (text, raw, ops)
        info.setdefault("pages", {})[kind] = ci.get("pages")
    if "translation" not in out and "intro" not in out:
        info["drop"] = "span_cut_failed" if info.get("span_failures") else "no_spans"
        return None, info
    shelfmarks = [str(s) for s in d.get("shelfmarks") or []]
    hints = [str(x) for x in d.get("pgpids") or []]
    edition, input_source, pg_used = "", f"book_edition:{src['slug']}", None
    if src.get("input") == "pgp_edition":
        got = pgp_edition(hints, shelfmarks) if pgp_edition else None
        if got is None:
            info["drop"] = "no_pgp_edition"
            return None, info
        edition, slug, pg_used = got
        input_source = f"pgp_digital_edition:{slug}"
    elif "edition" in out:
        edition = out["edition"][0]
    if not edition:
        info["drop"] = "no_edition"
        return None, info
    script = tn.dominant_script(edition)
    tr, tr_raw, tr_ops = out.get("translation", ("", "", []))
    intro = out.get("intro")
    ed_ops = out.get("edition", ("", "", []))[2]
    doc = BookDoc(source=str(src["slug"]), doc=str(d["doc"]), shelfmarks=shelfmarks,
                  target_lang=str(src.get("target_lang", "en")), edition=edition, translation=tr,
                  translation_raw=tr_raw, script=script,
                  lang_class=str(d.get("lang_class") or _lang_class(script)), input_source=input_source,
                  licence=src.get("licence"), licence_class=src.get("licence_class"),
                  intro=intro[0] if intro else None, intro_raw=intro[1] if intro else None,
                  pgp_hints=hints + ([pg_used] if pg_used and pg_used not in hints else []),
                  provenance={"work_pages": info.get("pages"), "located_by": "agent_read_pages_D15",
                              "edition_ops": ed_ops, "intro_ops": intro[2] if intro else []},
                  norm_ops=tr_ops, located=True)
    return doc, info


# ============================================================================== PGP / v0_1 lookups
@dataclass
class PgpIndex:
    """PGP pairs (``pgp_parallel``) and catalogue lookups used for dedupe, splits and inputs.

    ``docs``: pgpid → pairs row (subset of fields); ``by_key``: shelfmark key → pgpids;
    ``by_slug_no``: (edition / translation source slug, number) → pgpids; ``v01``: pgpid → v0_1 summary
    (:class:`holdouts.V01Doc`); ``catalogue``: pgpid → documents.csv fields (type, description).
    """

    docs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_key: Dict[str, Set[str]] = field(default_factory=dict)
    by_slug_no: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    v01: Dict[str, Any] = field(default_factory=dict)
    catalogue: Dict[str, Dict[str, str]] = field(default_factory=dict)
    corpus: Any = None
    grams: Dict[str, Set[str]] = field(default_factory=dict)

    @staticmethod
    def word_grams(text: str, n: int = EDITION_GRAM) -> Set[str]:
        """Hebrew / Arabic-script word n-grams of an edition (harakat, tatweel and editorial signs ignored).

        :param text: Edition text.
        :param n: n-gram length in words.
        :returns: Set of space-joined n-grams.
        """
        words = re.findall(r"[א-ת؀-ۿ]{2,}", re.sub(r"[ً-ٰٟـ]", "", text or ""))
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    def edition_matches(self, text: str, min_share: float = EDITION_MATCH_SHARE) -> List[Tuple[str, float]]:
        """PGP documents whose edition holds the same text (share of the book edition's word n-grams).

        :param text: Book edition text.
        :param min_share: Minimum share of the book edition's n-grams found in one PGP edition.
        :returns: ``[(pgpid, share)]``, best first.
        """
        g = self.word_grams(text)
        if not g or not self.grams:
            return []
        c: collections.Counter = collections.Counter()
        for x in g:
            for pid in self.grams.get(x, ()):
                c[pid] += 1
        return [(pid, round(k / len(g), 3)) for pid, k in c.most_common() if k / len(g) >= min_share]

    @staticmethod
    def location_number(location: Optional[str]) -> Optional[str]:
        """Document number of a PGP location (``doc. 12`` → ``12``; ``III-10a`` → ``III-10a``).

        :param location: Location string.
        :returns: Normalised number or None.
        """
        s = str(location or "").strip()
        if not s:
            return None
        m = re.match(r"^(?:doc(?:ument)?\.?|no\.?|#)\s*(\d+[a-z]?)\b", s, re.I)
        if m:
            return m.group(1)
        return re.sub(r"\s+", "", s)

    @classmethod
    def build(cls, pairs_path: Path, v01: Optional[Mapping[str, Any]] = None, corpus: Any = None,
              with_grams: bool = True) -> "PgpIndex":
        """Index the PGP pairs file.

        :param pairs_path: ``pgp_parallel`` JSONL.
        :param v01: Output of :func:`holdouts.scan_v01` (optional).
        :param corpus: :class:`shelfmarks.Corpus` for catalogue lookups (optional).
        :param with_grams: Build the edition word n-gram index (:meth:`edition_matches`).
        :returns: The index.
        """
        ix = cls(v01=dict(v01 or {}), corpus=corpus)
        for r in _rows.read_jsonl(pairs_path):
            p = str(r["pgpid"])
            ix.docs[p] = {k: r.get(k) for k in ("pgpid", "shelfmark", "split", "translation_en", "translation_he",
                                                "edition_text", "edition_source_slug", "edition_script", "editions",
                                                "translations", "lang_class", "type", "description")}
            for k in _sm.keys(r.get("shelfmark")):
                ix.by_key.setdefault(k, set()).add(p)
            for e in (r.get("editions") or []) + (r.get("translations") or []):
                no = cls.location_number(e.get("location"))
                if e.get("source_slug") and no:
                    ix.by_slug_no.setdefault((e["source_slug"], no), set()).add(p)
            if with_grams:
                for t in [r.get("edition_text") or ""] + [str(e.get("raw") or "") for e in r.get("editions") or []]:
                    for gram in cls.word_grams(t):
                        ix.grams.setdefault(gram, set()).add(p)
        if corpus is not None:
            ix.catalogue = {pid: dict(v) for pid, v in corpus.docs.items()}
            for pid, v in corpus.docs.items():
                for k in _sm.keys(v.get("shelfmark")):
                    ix.by_key.setdefault(k, set()).add(pid)
        return ix

    def has_translation(self, pgpid: str, lang: str) -> bool:
        """Whether PGP carries a translation of the document into ``lang``.

        :param pgpid: PGP id.
        :param lang: ``en`` / ``he``.
        :returns: True when the pairs row has ``translation_<lang>`` or v0_1 has translate rows in it.
        """
        d = self.docs.get(pgpid) or {}
        if d.get(f"translation_{lang}"):
            return True
        v = self.v01.get(pgpid)
        tasks = getattr(v, "tasks", {}) if v is not None else {}
        return bool(tasks.get(f"translate_{lang}"))

    def v01_split(self, pgpid: str) -> Optional[str]:
        """The v0_1 split a pgpid's rows sit in (train wins when several).

        :param pgpid: PGP id.
        :returns: ``train`` / ``dev`` / ``test`` or None.
        """
        v = self.v01.get(pgpid)
        splits = getattr(v, "splits", {}) if v is not None else {}
        for s in ("train", "dev", "test"):
            if splits.get(s):
                return s
        return None


def match_pgpids(doc: BookDoc, ix: PgpIndex, pgp_slugs: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """PGP ids of a book document: source hints, ``(slug, number)`` of the book's PGP edition, shelfmark keys.

    :param doc: The document.
    :param ix: PGP index.
    :param pgp_slugs: PGP source slugs of this book (local config).
    :returns: ``(pgpids, how)`` with ``how`` naming the routes that matched.
    """
    found: List[str] = []
    how: List[str] = []

    def add(ids: Iterable[str], route: str) -> None:
        """Add matched ids.

        :param ids: PGP ids.
        :param route: Route name.
        """
        new = [i for i in ids if i not in found]
        if new:
            found.extend(sorted(new, key=lambda x: int(x) if x.isdigit() else 0))
            if route not in how:
                how.append(route)

    add(doc.pgp_hints, "source_hint")
    if doc.pgp_number:
        for slug in pgp_slugs:
            add(ix.by_slug_no.get((slug, doc.pgp_number), set()), "pgp_slug_number")
    for k in doc.keys:
        add(ix.by_key.get(k, set()), "shelfmark_key")
    if not doc.input_source.startswith("pgp_digital_edition"):
        add([pid for pid, _ in ix.edition_matches(doc.edition)], "edition_text")
    return found, how


def hash_split(key: str, shares: Tuple[int, int, int] = SPLIT_SHARES) -> str:
    """D14 hash split of a new-book document.

    :param key: Document key (shelfmark key, else the doc id).
    :param shares: Percent shares of train / dev / test_indist.
    :returns: Split name.
    """
    b = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100
    if b < shares[0]:
        return "train"
    if b < shares[0] + shares[1]:
        return "dev"
    return "test_indist"


def assign_split(doc: BookDoc, pgpids: Sequence[str], holdouts: Any, ix: PgpIndex) -> Tuple[str, str, Optional[str]]:
    """D14 split of a book document.

    :param doc: The document.
    :param pgpids: Its matched PGP ids.
    :param holdouts: :class:`holdouts.Holdouts` (``is_benchmark`` / ``is_eval_doc``).
    :param ix: PGP index (v0_1 placement).
    :returns: ``(split, split_reason, holdout_reason)``.
    """
    for test, tier in ((holdouts.is_benchmark, "benchmark"), (holdouts.is_eval_doc, "eval_doc")):
        for p in pgpids:
            r = test(pgpid=p)
            if r:
                return "heldout", f"{tier}:pgpid", r
        for s in doc.shelfmarks:
            r = test(shelfmark=s)
            if r:
                return "heldout", f"{tier}:shelfmark", r
    for p in pgpids:
        if ix.v01_split(p) == "train":
            return "train", "v0_1_train_group", None
    key = doc.keys[0] if doc.keys else doc.doc_id
    return hash_split(key), "hash", None


# ============================================================================== pair checks
def _he_words(text: str) -> Set[str]:
    """Distinct Hebrew-script words of length ≥ 3 (vowel points removed; with a one-letter prefix-stripped variant).

    :param text: Text.
    :returns: Word set.
    """
    out: Set[str] = set()
    for w in _HE_WORD.findall(re.sub(r"[\u0591-\u05C7]", "", text or "")):
        w = re.sub(r"[׳'\"״]", "", w)
        if len(w) >= 3:
            out.add(w)
            if w[0] in "ובלהמשכ" and len(w) >= 4:
                out.add(w[1:])
    return out


def overlap_score(edition: str, target: str, lang: str, scholar_names: Iterable[str] = (),
                  stop: Optional[Set[str]] = None) -> Tuple[float, int]:
    """Character-level overlap of a translation with an edition (the D15 pairing sanity score).

    English: :func:`textnorm.grounding` (entity candidates of the target matched in the edition, also by
    consonant skeleton against Hebrew / Arabic-script words). Hebrew: share of the target's Hebrew words
    (≥ 3 letters, one prefix letter stripped as a variant) that occur among the edition's words, words in
    ``stop`` (formula words common to many editions of the source) left out.

    :param edition: Edition text.
    :param target: Translation.
    :param lang: ``en`` / ``he``.
    :param scholar_names: Folded scholar surnames (ignored as entities).
    :param stop: Hebrew words to ignore (see :func:`common_edition_words`).
    :returns: ``(score, number of candidates)``.
    """
    if lang == "he":
        tall = _he_words(target)
        tw = {w for w in tall if not (w[0] in "ובלהמשכ" and w[1:] in tall and len(w) >= 4)} - (stop or set())
        if not tw:
            return 0.0, 0
        ew = _he_words(edition)
        return sum(1 for w in tw if w in ew) / len(tw), len(tw)
    share, n, _ = tn.grounding(target, edition, scholar_names)
    return (share if n else 0.0), n


def common_edition_words(editions: Sequence[str], share: float = 0.25, min_docs: int = 3) -> Set[str]:
    """Hebrew words found in many editions of one source (letter formulae: ``אדוני``, ``השיך`` ...).

    :param editions: Edition texts of the source's documents.
    :param share: Document-frequency share from which a word counts as common.
    :param min_docs: Minimum document frequency for a common word.
    :returns: Word set.
    """
    df: collections.Counter = collections.Counter()
    for e in editions:
        df.update(_he_words(e))
    cut = max(min_docs, share * len(editions))
    return {w for w, k in df.items() if k >= cut}


def pair_check(doc: BookDoc, target: str, others: Sequence[BookDoc], scholar_names: Iterable[str] = ()
               ) -> Dict[str, Any]:
    """D15 second pass on a pair: F1 length band and the pairing (overlap rank) check.

    The pair passes the overlap test when its own score is at least the best score of the target
    against any other document's edition of the same source, and (when informative) at or above the
    language floor. English targets with fewer than :data:`MIN_CANDIDATES` entity candidates are
    judged on the length band alone (``overlap: "uninformative"``). The overlap test is enforced for
    located spans (``doc.located``, D15); for parsed entries, whose pairing is structural (one entry
    record), it is recorded (``enforced: False``) and only the band decides.

    :param doc: The document.
    :param target: Final translation target.
    :param others: Other documents of the same source (their editions are the decoys).
    :param scholar_names: Folded scholar surnames.
    :returns: ``{"ok", "reason", "length_ratio", "own", "best_other", "candidates", "overlap"}``.
    """
    lang = doc.target_lang
    band, ratio = _bd.band_ok(doc.edition, target, lang)
    decoys = [o for o in others if o is not doc and o.doc != doc.doc and o.edition]
    stop = common_edition_words([doc.edition] + [o.edition for o in decoys]) if lang == "he" else None
    own, n = overlap_score(doc.edition, target, lang, scholar_names, stop)
    best = 0.0
    for o in decoys:
        best = max(best, overlap_score(o.edition, target, lang, scholar_names, stop)[0])
    info: Dict[str, Any] = {"length_ratio": round(ratio, 3), "own": round(own, 3), "best_other": round(best, 3),
                            "candidates": n, "decoys": len(decoys),
                            "enforced": bool(doc.located)}
    if not band:
        return {**info, "ok": False, "reason": "length_ratio_F1", "overlap": "n/a"}
    informative = n >= (MIN_CANDIDATES if lang == "en" else 5)
    if not informative:
        return {**info, "ok": True, "reason": "ok", "overlap": "uninformative"}
    floor = HE_OVERLAP_FLOOR if lang == "he" else EN_OVERLAP_FLOOR
    verdict = "beaten" if own + 1e-9 < best else "below_floor" if own < floor else "pass"
    if verdict != "pass" and doc.located:
        return {**info, "ok": False, "reason": f"pair_check_{'rank' if verdict == 'beaten' else 'floor'}",
                "overlap": verdict}
    return {**info, "ok": True, "reason": "ok", "overlap": verdict}


# ============================================================================== rows
@dataclass
class BuildContext:
    """Shared state of one build: tokens, noise, PGP lookups, hold-outs, counters."""

    holdouts: Any
    ix: PgpIndex
    noiser: Any = None
    scholar_names: Set[str] = field(default_factory=set)
    drops: collections.Counter = field(default_factory=collections.Counter)
    with_entry_commentary: bool = False
    tc: Any = None


#: Located-span kinds a row of each family draws on (its target, its input when the book prints the edition, the
#: book translation an analysis row's context may append).
ROW_SPAN_KINDS: Dict[str, Tuple[str, ...]] = {"translate_en": ("translation", "edition"),
                                              "translate_he": ("translation", "edition"),
                                              "analyse_document": ("intro", "edition", "translation")}


def row_page_keys(doc: BookDoc, family: str) -> List[str]:
    """Literature pages a row draws on: for a located-span document ``provenance.work_pages`` of the kinds in
    :data:`ROW_SPAN_KINDS` (the ``edition`` pages only when the input is the book's own edition).

    Parsed entries (F11) give every page of their entry's PDF range (:class:`EntryPages`), whatever the family.

    :param doc: The document.
    :param family: Row family.
    :returns: Sorted page keys (empty for a parsed entry that could not be mapped).
    """
    if doc.entry_page_keys is not None:
        return sorted(set(doc.entry_page_keys))
    work_pages = doc.provenance.get("work_pages") if isinstance(doc.provenance.get("work_pages"), dict) else {}
    kinds = [k for k in ROW_SPAN_KINDS.get(family, tuple(work_pages))
             if not (k == "edition" and doc.input_source.startswith("pgp_digital_edition"))]
    return sorted({k for kind in kinds for k in (work_pages.get(kind) or []) if k})


def _base_meta(doc: BookDoc, family: str, condition: str, split: str, pgpids: Sequence[str],
               split_reason: str, hold_reason: Optional[str]) -> Dict[str, Any]:
    """Meta fields shared by every row of a document (v0_1 fields + v0_2 fields + L5 provenance).

    :param doc: The document.
    :param family: Row family.
    :param condition: Input condition.
    :param split: Split.
    :param pgpids: Matched PGP ids.
    :param split_reason: Why the split was chosen.
    :param hold_reason: Hold-out reason (None when not held out).
    :returns: Meta dict (row id filled in later); ``page_keys`` lists the literature pages the row draws on
        (:func:`row_page_keys`), so the page exclusions and the re-scan see them (fix pass F2); a parsed-entry row
        also carries ``page_keys_source`` = :data:`NAS_PAGE_SOURCE`, ``entry_pages`` (work and PDF range) and, when
        its entry could not be mapped, ``page_keys_unmapped`` (the row is then excluded from training, F11).
    """
    key = doc.keys[0] if doc.keys else None
    page_keys = row_page_keys(doc, family)
    nas: Dict[str, Any] = {}
    if doc.entry_page_keys is not None:
        nas = {"page_keys_source": NAS_PAGE_SOURCE, "entry_pages": doc.provenance.get("entry_pages")}
        if doc.entry_pages_unmapped:
            nas["page_keys_unmapped"] = doc.entry_pages_unmapped
    return {**nas, "task": family, "family": family, "lang_class": doc.lang_class, "condition": condition,
            "pgpid": pgpids[0] if pgpids else None, "pgpids": list(pgpids), "doc_id": doc.doc_id,
            "shelfmark": " + ".join(doc.shelfmarks) if doc.shelfmarks else None,
            "shelfmark_group": f"bk:{key}" if key else doc.doc_id, "split": split, "split_reason": split_reason,
            "holdout_reason": hold_reason,
            "held_out_vision_benchmark": bool(hold_reason and _bd.BENCH_REASONS & {_base_reason(hold_reason)}),
            "target_chars": 0, "target_tokens": 0, "tokens": 0, "input_cer": None, "input_source": doc.input_source,
            "input_truncated": False, "licence": doc.licence, "licence_class": doc.licence_class,
            "provenance": {"source": doc.source, "doc": doc.doc, **doc.provenance}, "source": f"book:{doc.source}",
            "read_model": None, "read_coverage": None, "read_agreed_share": None, "noise_band": None,
            "noise_class": None, "reader_exposed": False, "exposure_level": None, "target_raw": None, "strip_ops": [],
            "pgp_description_dup": False, "build": BUILD, "book_doc": doc.doc, "edition_script": doc.script,
            "page_keys": page_keys}


def _base_reason(reason: str) -> str:
    """Hold-out reason without ``key:`` / ``group:`` prefixes.

    :param reason: Reason string.
    :returns: Base reason.
    """
    return reason.split(":")[-1]


def _row(family: str, condition: str, doc: BookDoc, user: str, target: str, meta: Dict[str, Any],
         side: Optional[str] = None) -> Dict[str, Any]:
    """Assemble one row with its id.

    :param family: Row family.
    :param condition: Input condition.
    :param doc: The document.
    :param user: User prompt.
    :param target: Assistant target.
    :param meta: Meta (copied).
    :param side: Block label when the row covers one block of the document.
    :returns: Row.
    """
    m = dict(meta)
    m["condition"], m["side"] = condition, side
    m["row_id"] = _rows.row_id(family, doc.doc_id, condition, None, *([side] if side else []))
    msgs = _rows.make_messages(user, target)
    m["target_chars"] = len(msgs[2]["content"])
    return {"messages": msgs, "meta": m}


def _input_variants(doc: BookDoc, edition: str, family: str, split: str, ctx: BuildContext, side: Optional[str]
                    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Input variants of a document: edition, noised edition (Hebrew script, train), re-scripted edition (B9).

    :param doc: The document.
    :param edition: Edition text of the row (whole document or one block).
    :param family: Row family.
    :param split: Split.
    :param ctx: Build context (noiser).
    :param side: Block label or None.
    :returns: ``[(condition, text, prompt script, meta extras)]``.
    """
    out: List[Tuple[str, str, str, Dict[str, Any]]] = [("edition", edition, doc.script, {})]
    if split == "train" and doc.script == "hebrew" and ctx.noiser is not None and ctx.noiser.available("hebrew"):
        rng = random.Random(_bv.stable_int(_bv.SEED, doc.doc_id, family, side or ""))
        noised, target_cer = ctx.noiser.apply(edition, "hebrew", rng)
        cer = round(tn.cer(noised, edition), 4)
        out.append(("noised_edition", noised, "hebrew",
                    {"input_source": f"noised:{doc.input_source}", "noise_target_cer": round(target_cer, 3),
                     "input_cer": cer, "noise_band": _bd.noise_band(cer), "noise_class": "synthetic"}))
    if family == "translate_en" and doc.script == "arabic" and doc.lang_class == "AR":
        out.append(("edition_rescript", tn.transliterate_ar_to_ja_plain(edition), "hebrew",
                    {"input_source": f"rescript_ar_to_ja_plain:{doc.input_source}", "input_script": "hebrew",
                     "rescript": "transliterate_ar_to_ja_plain"}))
    return out


def _token_len(user: str, target: str, ctx: BuildContext) -> int:
    """Approximate rendered length of a row (exact counts come from :func:`builder.finalize_tokens`).

    :param user: User prompt.
    :param target: Target.
    :param ctx: Build context (token counter).
    :returns: Tokens.
    """
    if ctx.tc is None:
        return (len(user) + len(target)) // 3
    return ctx.tc.count(_rows.SYSTEM) + ctx.tc.count(user) + ctx.tc.count(target) + 3 * _bv.CHAT_OVERHEAD


def translate_rows(doc: BookDoc, split: str, pgpids: Sequence[str], split_reason: str, hold_reason: Optional[str],
                   others: Sequence[BookDoc], ctx: BuildContext) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Translate rows of one document (whole document; per block when over the token cap).

    :param doc: The document.
    :param split: Split.
    :param pgpids: Matched PGP ids.
    :param split_reason: Why the split was chosen.
    :param hold_reason: Hold-out reason.
    :param others: Other documents of the same source (pairing-check decoys).
    :param ctx: Build context.
    :returns: ``(rows, info)``; ``info["drop"]`` names the reason when no row results.
    """
    family = f"translate_{doc.target_lang}"
    info: Dict[str, Any] = {}
    if doc.script not in ("hebrew", "arabic"):
        return [], {"drop": "script_not_supported"}
    if "block_keys_differ" in doc.flags:
        return [], {"drop": "block_labels_mismatch", "edition_labels": rendered_labels(doc.edition_blocks),
                    "translation_labels": rendered_labels(doc.translation_blocks)}
    target, ops = _bd.clean_target(doc.translation, family)
    ops = doc.norm_ops + ops
    if not target:
        return [], {"drop": "target_empty_after_strip"}
    ok, missing = words_subset(target, doc.translation_raw)
    if not ok:
        return [], {"drop": "target_words_not_in_source", "missing": missing}
    if tn.overlap60(target, doc.edition):
        return [], {"drop": "target_overlaps_edition"}
    chk = pair_check(doc, target, others, ctx.scholar_names)
    info["pair_check"] = chk
    if not chk["ok"]:
        return [], {**info, "drop": chk["reason"]}
    base = _base_meta(doc, family, "edition", split, pgpids, split_reason, hold_reason)
    base.update({"target_lang": doc.target_lang, "length_ratio": chk["length_ratio"], "pair_check": chk,
                 "target_raw": (doc.translation_raw if doc.located else doc.translation) if ops else None,
                 "strip_ops": ops, "doc_flags": doc.flags})
    units: List[Tuple[Optional[str], str, str]] = [(None, doc.edition, target)]
    probe_user = _rows.user_prompt(family, "edition", doc.script, doc.edition)
    if _token_len(probe_user, target, ctx) > MAX_ROW_TOKENS - 64:
        units = _block_units(doc, family, ctx)
        if not units:
            return [], {**info, "drop": "over_cap_no_block_alignment"}
        info["split_into_blocks"] = len(units)
    rows: List[Dict[str, Any]] = []
    for side, ed, tgt in units:
        for cond, text, script, mx in _input_variants(doc, ed, family, split, ctx, side):
            user = _rows.user_prompt(family, cond, script, text)
            if _token_len(user, tgt, ctx) > MAX_ROW_TOKENS - 16:
                ctx.drops[f"{family}:{cond}:over_cap"] += 1
                continue
            meta = {**base, **mx}
            if side:
                meta["length_ratio"] = round(_bd.band_ok(ed, tgt, doc.target_lang)[1], 3)
            rows.append(_row(family, cond, doc, user, tgt, meta, side))
    if not rows:
        return [], {**info, "drop": "over_cap"}
    return rows, info


def _block_units(doc: BookDoc, family: str, ctx: BuildContext) -> List[Tuple[Optional[str], str, str]]:
    """Per-block (side / part) edition–translation units of an over-long document.

    Blocks are paired only when both sides list the same labels in the same order; each unit must pass
    the F1 band and hold ≥ 40 normalised characters on both sides.

    :param doc: The document.
    :param family: Row family.
    :param ctx: Build context (drop counter).
    :returns: ``[(label, edition, target)]`` (empty when the blocks do not align).
    """
    eb, tb = doc.edition_blocks, doc.translation_blocks
    if not eb or [k for k, _ in eb] != [k for k, _ in tb] or len(eb) < 2:
        return []
    out = []
    for (lab, el), (_, tl) in zip(eb, tb):
        ed = "\n".join(el)
        tgt, _ = _bd.clean_target("\n".join(tl), family)
        ok, _ = _bd.band_ok(ed, tgt, doc.target_lang)
        if len(tn.normalize_for_cer(ed)) < 40 or len(tn.normalize_for_cer(tgt)) < 40 or not ok:
            ctx.drops[f"{family}:block_band_or_short"] += 1
            continue
        out.append((lab or None, ed, tgt))
    return out


def _context_block(pgpids: Sequence[str], ix: PgpIndex) -> str:
    """``Known catalogue context`` block from the PGP catalogue (type + B4-stripped description).

    :param pgpids: Matched PGP ids.
    :param ix: PGP index.
    :returns: Block text ('' when no catalogue record is known).
    """
    for p in pgpids:
        cat = ix.catalogue.get(p) or {}
        typ = cat.get("type") or (ix.docs.get(p) or {}).get("type")
        desc = cat.get("description") or (ix.docs.get(p) or {}).get("description")
        if not (typ or desc):
            continue
        desc_s, _ = _bd.clean_target(desc or "", "describe_pgp") if desc else ("", [])
        lines = [f"Type: {typ}" if typ else "", f"Catalogue description: {desc_s}" if desc_s else ""]
        return "Known catalogue context:\n" + "\n".join(x for x in lines if x)
    return ""


def analyse_rows(doc: BookDoc, text: Optional[str], raw: Optional[str], origin: str, split: str,
                 pgpids: Sequence[str], split_reason: str, hold_reason: Optional[str], ctx: BuildContext
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``analyse_document`` rows from an editor's introduction / commentary (tier A′ / B gate).

    :param doc: The document.
    :param text: Normalised introduction (target before strips).
    :param raw: Raw introduction (word-subset reference).
    :param origin: ``intro`` or ``entry_commentary``.
    :param split: Split.
    :param pgpids: Matched PGP ids.
    :param split_reason: Why the split was chosen.
    :param hold_reason: Hold-out reason.
    :param ctx: Build context.
    :returns: ``(rows, info)``.
    """
    if not text:
        return [], {"drop": "no_intro"}
    if doc.script not in ("hebrew", "arabic"):
        return [], {"drop": "script_not_supported"}
    target, ops = _bd.clean_target(text, "analyse_document")
    ops = list(doc.provenance.get("intro_ops") or []) + ops if origin == "intro" else ops
    if len(target) < ANALYSIS_MIN_CHARS:
        return [], {"drop": "analysis_short_after_strip"}
    ok, missing = words_subset(target, raw or text)
    if not ok:
        return [], {"drop": "target_words_not_in_source", "missing": missing}
    if tn.overlap60(target, doc.edition):
        return [], {"drop": "target_overlaps_edition"}
    ctx_block = _context_block(pgpids, ctx.ix)
    if ctx_block and tn.overlap60(target, ctx_block):
        ctx_block = ""
    share_a, n_c, miss = tn.grounding(target, doc.edition + "\n" + ctx_block, ctx.scholar_names)
    tier, extra, share = None, ctx_block, share_a
    if share_a >= GROUNDING_A:
        tier = "A_prime"
    elif doc.translation and len(doc.translation) >= 150 and not tn.overlap60(target, doc.translation):
        tr_block = "Translation (from the edition):\n" + doc.translation
        share_b, _, miss = tn.grounding(target, doc.edition + "\n" + ctx_block + "\n" + doc.translation,
                                        ctx.scholar_names)
        share = share_b
        if share_b >= GROUNDING_B:
            tier = "B"
            extra = (ctx_block + "\n\n" if ctx_block else "") + tr_block
    info = {"grounding_share": round(share, 3), "grounding_candidates": n_c, "grounding_tier": tier,
            "ungrounded": miss[:8]}
    if tier is None:
        return [], {**info, "drop": "analysis_grounding_fail"}
    base = _base_meta(doc, "analyse_document", "edition", split, pgpids, split_reason, hold_reason)
    base.update({"target_lang": "en" if re.search(r"[A-Za-z]{3}", target) else "he", "grounding_share": round(share, 3),
                 "grounding_candidates": n_c, "grounding_tier": tier, "analysis_source": origin,
                 "target_raw": (raw if doc.located and raw else text) if ops else None, "strip_ops": ops})
    rows = []
    for cond, inp, script, mx in _input_variants(doc, doc.edition, "analyse_document", split, ctx, None):
        if cond == "edition_rescript":
            continue
        user = _rows.user_prompt("analyse_document", cond, script, inp, extra)
        if _token_len(user, target, ctx) > MAX_ROW_TOKENS - 16:
            ctx.drops[f"analyse_document:{cond}:over_cap"] += 1
            continue
        rows.append(_row("analyse_document", cond, doc, user, target, {**base, **mx}))
    if not rows:
        return [], {**info, "drop": "over_cap"}
    return rows, info


# ============================================================================== noise
def make_noiser(p: _paths.Paths, pairs_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """The builder's noise model, calibrated the way the builder calibrates it.

    Uses ``out_dir/noise_model.json`` (v0_2 builder) when present, else ``text_ft_v0_1/noise_model.json``;
    calibration samples train Hebrew / Arabic editions of ≥ 300 characters from the pairs file.

    :param p: Build paths.
    :param pairs_path: ``pgp_parallel`` JSONL.
    :returns: ``(noiser or None, info)``.
    """
    for path in (p.out("noise_model.json"), p.text_ft_v0_1 / "noise_model.json"):
        if path.exists():
            model = json.loads(path.read_text(encoding="utf-8"))
            break
    else:
        return None, {"noise_model": None}
    noiser = _bv.Noiser(model)
    pool: Dict[str, List[str]] = collections.defaultdict(list)
    for r in _rows.read_jsonl(pairs_path):
        if r.get("edition_script") in ("hebrew", "arabic") and r.get("split") == "train" and \
                len(r.get("edition_text") or "") >= 300:
            pool[r["edition_script"]].append(r["edition_text"])
    calib = noiser.calibrate(pool)
    return noiser, {"noise_model": str(path.relative_to(p.repo)) if path.is_relative_to(p.repo) else path.name,
                    "calibration": {k: round(v, 3) for k, v in calib.items()}}


# ============================================================================== build
def load_nas_docs(cfg: Mapping[str, Any], root: Path, drops: collections.Counter,
                  entry_pages: Optional[EntryPages] = None) -> Tuple[List[BookDoc], Dict[str, int]]:
    """Read every configured edition-entries book (read-only) into documents.

    With ``entry_pages`` (F11) every document gets the literature page keys of its entry
    (:attr:`BookDoc.entry_page_keys`, an empty list and :attr:`BookDoc.entry_pages_unmapped` when the entry cannot
    be mapped); without it the documents carry no page information (``entry_page_keys`` None).

    :param cfg: Local config (``nas_books``).
    :param root: ``paths.book_editions_nas``.
    :param drops: Drop counter (``<slug>:entries_missing``).
    :param entry_pages: Entry → page-key mapper.
    :returns: ``(documents, entries per book)``.
    """
    docs: List[BookDoc] = []
    seen: Dict[str, int] = {}
    for book in cfg.get("nas_books") or []:
        path = root / str(book["dir"]) / "entries.jsonl"
        if not path.exists():
            drops[f"{book['slug']}:entries_missing"] += 1
            LOG.warning("entries missing for %s", book["slug"])
            continue
        entries = list(_rows.read_jsonl(path))
        seen[str(book["slug"])] = len(entries)
        if entry_pages is not None:
            entry_pages.add_book(str(book["slug"]), entries)
        for e in entries:
            if book.get("layout") == "paragraphs":
                doc = doc_from_paragraph_entry(e, book)
            else:
                doc = doc_from_lines_entry(e, book)
            if entry_pages is not None:
                keys, why = entry_pages.keys(str(book["slug"]), e.get("entry_no"))
                doc.entry_page_keys, doc.entry_pages_unmapped = keys, why
                doc.provenance["entry_pages"] = entry_pages.describe(str(book["slug"]), e.get("entry_no"))
            docs.append(doc)
    return docs, seen


def load_span_docs(spans: Mapping[str, Any], p: _paths.Paths, ix: PgpIndex, drops: collections.Counter,
                   page_texts: Optional[Mapping[str, Optional[str]]] = None) -> Tuple[List[BookDoc], List[Dict[str, Any]]]:
    """Cut and verify every located span (D15) into documents.

    :param spans: Spans file (``sources``).
    :param p: Build paths (literature root).
    :param ix: PGP index (PGP editions for ``input: pgp_edition``).
    :param drops: Drop counter.
    :param page_texts: Pre-loaded page key → OCR text (default: read the works named in the file).
    :returns: ``(documents, per-document info records)``.
    """
    sources = spans.get("sources") or []
    if page_texts is None:
        from src.finetuning.text27b import pages as _pages  # heavy import only when spans exist
        works = sorted({str(s["work"]) for s in sources if s.get("work")})
        page_texts = {pg.page_key: pg.ocr_text for pg in _pages.iter_pages(p.academic_literature, works=works)} \
            if works else {}
    docs: List[BookDoc] = []
    infos: List[Dict[str, Any]] = []
    for src in sources:
        slugs = list(src.get("pgp_slugs") or [])

        def pgp_edition(hints: Sequence[str], shelfmarks: Sequence[str], _slugs: Sequence[str] = slugs
                        ) -> Optional[Tuple[str, str, str]]:
            """PGP digital edition of a document (the source's slug preferred).

            :param hints: PGP ids named in the spans file.
            :param shelfmarks: Shelfmarks of the document.
            :param _slugs: Preferred edition slugs.
            :returns: ``(edition text, slug, pgpid)`` or None.
            """
            cands = list(hints)
            for s in shelfmarks:
                for k in _sm.keys(s):
                    cands += sorted(ix.by_key.get(k, set()))
            for pid in dict.fromkeys(cands):
                d = ix.docs.get(pid)
                if not d or not d.get("edition_text"):
                    continue
                for e in d.get("editions") or []:
                    if e.get("source_slug") in _slugs and e.get("source_slug") == d.get("edition_source_slug"):
                        return tn.strip_latin_lines(d["edition_text"])[0], e["source_slug"], pid
                for e in d.get("editions") or []:
                    if e.get("source_slug") in _slugs and e.get("raw"):
                        return tn.strip_latin_lines(str(e["raw"]).strip())[0], e["source_slug"], pid
                return tn.strip_latin_lines(d["edition_text"])[0], str(d.get("edition_source_slug")), pid
            return None

        for d in src.get("docs") or []:
            doc, info = doc_from_spans(src, d, page_texts, pgp_edition)
            info["source"] = src["slug"]
            infos.append(info)
            if doc is None:
                drops[f"{src['slug']}:{info.get('drop')}"] += 1
                continue
            docs.append(doc)
    return docs, infos


def build(p: Optional[_paths.Paths] = None, cfg: Optional[Mapping[str, Any]] = None,
          spans: Optional[Mapping[str, Any]] = None, holdouts: Any = None, ix: Optional[PgpIndex] = None,
          noiser: Any = None, tc: Any = None, with_entry_commentary: bool = False,
          page_texts: Optional[Mapping[str, Optional[str]]] = None,
          entry_pages: Optional[EntryPages] = None) -> Dict[str, Any]:
    """Build the L5 rows from the configured sources.

    :param p: Build paths.
    :param cfg: Local config (default: :func:`load_local_config`).
    :param spans: Spans file content (default: :func:`load_local_config`).
    :param holdouts: :class:`holdouts.Holdouts` (default ``Holdouts.load()``).
    :param ix: PGP index (default: built from ``paths.pgp_parallel`` + v0_1 + corpus).
    :param noiser: Noise model (default: :func:`make_noiser`; pass False to disable).
    :param tc: Token counter (default :class:`builder_v0.TokenCounter`).
    :param with_entry_commentary: Also emit analyse rows from entry ``discussion`` / ``commentary``.
    :param page_texts: Pre-loaded page texts for the span sources (tests).
    :param entry_pages: Entry → page-key mapper of the parsed entries (default :meth:`EntryPages.load` over the
        configured books; F11).
    :returns: ``{"rows", "docs", "report"}``.
    """
    t0 = time.time()
    p = p or _paths.load_paths()
    if cfg is None or spans is None:
        c2, s2, _, _ = load_local_config(p)
        cfg = c2 if cfg is None else cfg
        spans = s2 if spans is None else spans
    if holdouts is None:
        from src.finetuning.text27b.holdouts import Holdouts
        holdouts = Holdouts.load(p=p)
    noise_info: Dict[str, Any] = {}
    if ix is None:
        from src.finetuning.text27b import holdouts as _ho
        v01 = _ho.scan_v01(p.text_ft_v0_1)
        corpus = _sm.load_corpus(p)
        ix = PgpIndex.build(p.pgp_parallel, v01, corpus)
    if noiser is None:
        noiser, noise_info = make_noiser(p, p.pgp_parallel)
    elif noiser is False:
        noiser = None
    ctx = BuildContext(holdouts=holdouts, ix=ix, noiser=noiser, with_entry_commentary=with_entry_commentary,
                       tc=tc if tc is not None else _bv.TokenCounter(p.tokenizer_json))
    pgp_ex = p.pgp_export / "sources.csv"
    if pgp_ex.exists():
        extra = list(_bv.load_local_config(p).get("scholar_extra") or []) + list(cfg.get("scholar_extra") or [])
        ctx.scholar_names = tn.scholar_surnames(pgp_ex, extra)
    LOG.info("[%5.1fs] lookups ready", time.time() - t0)
    drops: collections.Counter = collections.Counter()
    if entry_pages is None and cfg.get("nas_books"):
        entry_pages = EntryPages.load(p, cfg)
    nas_docs, entries_seen = load_nas_docs(cfg, p.book_editions_nas, drops, entry_pages)
    span_docs, span_info = load_span_docs(spans, p, ix, drops, page_texts)
    LOG.info("[%5.1fs] %d entry docs, %d span docs", time.time() - t0, len(nas_docs), len(span_docs))
    pgp_slugs: Dict[str, List[str]] = {str(b["slug"]): list(b.get("pgp_slugs") or []) for b in cfg.get("nas_books") or []}
    for s in spans.get("sources") or []:
        pgp_slugs[str(s["slug"])] = list(s.get("pgp_slugs") or [])
    priority = list(cfg.get("priority") or [])
    all_docs = nas_docs + span_docs
    all_docs.sort(key=lambda d: (priority.index(d.source) if d.source in priority else len(priority), d.source,
                                 int(d.doc) if d.doc.isdigit() else 10 ** 6, d.doc))
    by_source: Dict[str, List[BookDoc]] = collections.defaultdict(list)
    for d in all_docs:
        by_source[d.source].append(d)
    rows: List[Dict[str, Any]] = []
    doc_records: List[Dict[str, Any]] = []
    seen_keys: Dict[Tuple[str, str], str] = {}
    held_docs: List[Dict[str, Any]] = []
    for doc in all_docs:
        pgpids, how = match_pgpids(doc, ix, pgp_slugs.get(doc.source, []))
        split, split_reason, hold = assign_split(doc, pgpids, holdouts, ix)
        rec: Dict[str, Any] = {"source": doc.source, "doc": doc.doc, "doc_id": doc.doc_id, "shelfmarks": doc.shelfmarks,
                               "keys": doc.keys, "pgpids": pgpids, "match": how, "split": split,
                               "split_reason": split_reason, "holdout_reason": hold, "script": doc.script,
                               "lang_class": doc.lang_class, "flags": doc.flags, "translate": None, "analyse": None,
                               "page_keys": sorted(set(doc.entry_page_keys)) if doc.entry_page_keys is not None else
                               sorted({k for v in (doc.provenance.get("work_pages") or {}).values() for k in v or []}),
                               "page_keys_source": NAS_PAGE_SOURCE if doc.entry_page_keys is not None else
                               ("work_pages" if doc.provenance.get("work_pages") else None),
                               "page_keys_unmapped": doc.entry_pages_unmapped,
                               "book_pages": [doc.provenance.get(k) for k in ("page", "pdf_page") if doc.provenance.get(k)]}
        if split == "heldout":
            held_docs.append({k: rec[k] for k in ("source", "doc", "shelfmarks", "pgpids", "holdout_reason",
                                                  "split_reason")})
        fam = f"translate_{doc.target_lang}"
        if doc.translation:
            dup_pg = [x for x in pgpids if ix.has_translation(x, doc.target_lang)]
            dup_l5 = next((seen_keys[(fam, k)] for k in doc.keys if (fam, k) in seen_keys
                           and not seen_keys[(fam, k)].startswith(f"book:{doc.source}:")), None)
            if dup_pg:
                rec["translate"] = {"drop": "dup_pgp_translation", "pgpids": dup_pg}
            elif dup_l5:
                rec["translate"] = {"drop": "dup_within_l5", "of": dup_l5}
            else:
                r, info = translate_rows(doc, split, pgpids, split_reason, hold, by_source[doc.source], ctx)
                rec["translate"] = {**info, "rows": len(r)}
                if r:
                    rows.extend(r)
                    for k in doc.keys:
                        seen_keys.setdefault((fam, k), doc.doc_id)
            if rec["translate"].get("drop"):
                drops[f"{doc.source}:translate:{rec['translate']['drop']}"] += 1
        intro = (doc.intro, doc.intro_raw, "intro") if doc.intro else \
            ((doc.commentary, doc.commentary, "entry_commentary") if with_entry_commentary and doc.commentary else None)
        if intro:
            r, info = analyse_rows(doc, intro[0], intro[1], intro[2], split, pgpids, split_reason, hold, ctx)
            rec["analyse"] = {**info, "rows": len(r)}
            rows.extend(r)
            if info.get("drop"):
                drops[f"{doc.source}:analyse:{info['drop']}"] += 1
        doc_records.append(rec)
    LOG.info("[%5.1fs] %d rows before exact token counts", time.time() - t0, len(rows))
    non_train = [r for r in doc_records if r["split"] != "train" and ((r.get("translate") or {}).get("rows")
                                                                     or (r.get("analyse") or {}).get("rows"))]
    report = {"entries_per_book": entries_seen, "span_docs": span_info, "noise": noise_info,
              "non_train_docs": [{k: r[k] for k in ("source", "doc", "split", "shelfmarks", "page_keys", "page_keys_source",
                                                    "book_pages")}
                                 for r in non_train],
              "drops": dict(sorted((drops + ctx.drops).items())), "heldout_docs": held_docs,
              "elapsed_s": round(time.time() - t0, 1)}
    return {"rows": rows, "docs": doc_records, "report": report, "doc_objs": all_docs}


def _per_source(rows: Sequence[Dict[str, Any]], docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-source pair / row counts.

    :param rows: Final rows.
    :param docs: Document records of :func:`build`.
    :returns: source → {docs, translate_docs, analyse_docs, rows, tokens, by_split, drops}.
    """
    out: Dict[str, Any] = {}
    for d in docs:
        s = out.setdefault(d["source"], {"docs": 0, "translate_docs": 0, "analyse_docs": 0, "rows": 0, "tokens": 0,
                                         "pair_docs_by_split": collections.Counter(), "drops": collections.Counter()})
        s["docs"] += 1
        t, a = d.get("translate") or {}, d.get("analyse") or {}
        if t.get("rows"):
            s["translate_docs"] += 1
            s["pair_docs_by_split"][d["split"]] += 1
        elif t.get("drop"):
            s["drops"][f"translate:{t['drop']}"] += 1
        if a.get("rows"):
            s["analyse_docs"] += 1
        elif a.get("drop"):
            s["drops"][f"analyse:{a['drop']}"] += 1
    for r in rows:
        src = r["meta"]["source"].split(":", 1)[1]
        if src in out:
            out[src]["rows"] += 1
            out[src]["tokens"] += int(r["meta"].get("tokens") or 0)
    for s in out.values():
        s["pair_docs_by_split"] = dict(s["pair_docs_by_split"])
        s["drops"] = dict(s["drops"])
    return out


def audit_sample(rows: Sequence[Dict[str, Any]], n: int = AUDIT_SIZE, seed: int = 3407) -> List[Dict[str, Any]]:
    """Stratified hand-check sample: edition-condition rows, round-robin over sources, seeded.

    :param rows: Final rows.
    :param n: Sample size.
    :param seed: Seed.
    :returns: ``[{row_id, source, doc, family, split, source_text, target, target_raw, strip_ops, pair_check}]``.
    """
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["meta"]["condition"] == "edition":
            pools[r["meta"]["source"]].append(r)
    for v in pools.values():
        rng.shuffle(v)
    out: List[Dict[str, Any]] = []
    while len(out) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(out) < n:
                r = pools[k].pop()
                m = r["meta"]
                user = r["messages"][1]["content"]
                a, b = user.find("<document>\n"), user.find("\n</document>")
                out.append({"row_id": m["row_id"], "source": m["source"], "doc": m.get("book_doc"),
                            "family": m["family"], "split": m["split"], "pgpid": m.get("pgpid"),
                            "shelfmark": m.get("shelfmark"), "source_text": user[a + 11:b] if a >= 0 else user,
                            "target": r["messages"][2]["content"], "target_raw": m.get("target_raw"),
                            "strip_ops": m.get("strip_ops"), "pair_check": m.get("pair_check"),
                            "grounding_tier": m.get("grounding_tier")})
    return out


def write_outputs(result: Dict[str, Any], p: _paths.Paths, out_dir: Optional[Path] = None,
                  build_args: Optional[Dict[str, Any]] = None, exact_tokens: bool = True) -> Dict[str, Any]:
    """Exact token counts, family file, manifest and audit sample.

    :param result: Output of :func:`build`.
    :param p: Build paths.
    :param out_dir: Output root (default ``paths.out_dir``).
    :param build_args: CLI arguments recorded in the manifest.
    :param exact_tokens: Render and tokenize every row (False keeps the estimates; tests).
    :returns: The manifest.
    """
    out = Path(out_dir) if out_dir else p.out_dir
    rows = result["rows"]
    over: collections.Counter = collections.Counter()
    if exact_tokens:
        rows, over = _bd.finalize_tokens(rows, MAX_ROW_TOKENS)
    else:
        for r in rows:
            m = r["meta"]
            m["target_tokens"] = len(r["messages"][2]["content"]) // 3
            m["tokens"] = sum(len(x["content"]) for x in r["messages"]) // 3
    problems = collections.Counter()
    for r in rows:
        for prob in _rows.validate_row(r):
            problems[prob] += 1
    rows.sort(key=lambda r: (r["meta"]["source"], r["meta"]["doc_id"], r["meta"]["family"], r["meta"]["condition"],
                             r["meta"].get("side") or ""))
    fam_path = out / "families" / f"{FAMILY_FILE}.jsonl"
    fam_path.parent.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(fam_path, rows)
    audit = audit_sample(rows)
    audit_path = out / "audit" / AUDIT_NAME
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(audit_path, audit)
    ids = [r["meta"]["row_id"] for r in rows]
    checks = {
        "rows": len(rows), "duplicate_row_ids": len(ids) - len(set(ids)), "schema_problems": dict(problems),
        "heldout_rows_in_train": sum(1 for r in rows if r["meta"]["split"] == "train" and r["meta"].get("holdout_reason")),
        "scrub_rows": sum(1 for r in rows if tn.has_source_names(r["messages"][2]["content"])),
        "ar_noised_rows": sum(1 for r in rows if r["meta"]["condition"] == "noised_edition" and
                              (r["meta"]["lang_class"] == "AR" or r["meta"].get("edition_script") == "arabic")),
        "marker_rows": sum(1 for r in rows if r["meta"]["family"].startswith("translate") and
                           tn.strip_line_markers(r["messages"][2]["content"])[1]),
        "dropped_over_cap_exact": dict(over),
        "rows_by_page_keys_source": dict(collections.Counter(str(r["meta"].get("page_keys_source") or
                                                                 ("work_pages" if r["meta"].get("page_keys") else "none"))
                                                             for r in rows)),
        "rows_without_page_keys": sum(1 for r in rows if not r["meta"].get("page_keys")),
        "rows_unmapped_entry_pages": sum(1 for r in rows if r["meta"].get("page_keys_unmapped")),
    }
    manifest = {
        "build": BUILD, "family_file": f"families/{FAMILY_FILE}.jsonl", "counts": _bd.count_table(rows),
        "per_source": _per_source(rows, result["docs"]), "checks": checks, "report": result["report"],
        "documents": result["docs"], "fingerprint": _rows.fingerprint(rows), "build_args": build_args or {},
    }
    (out / "families" / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    return manifest


def show_page(key: str, p: Optional[_paths.Paths] = None, width: int = 110) -> str:
    """Numbered OCR lines of one literature page (the view used to place span pieces).

    :param key: Page key ``<work>/<part>/<page_file>``.
    :param p: Build paths.
    :param width: Characters shown per line.
    :returns: Printable text.
    """
    from src.finetuning.text27b import pages as _pages
    p = p or _paths.load_paths()
    work = key.split("/", 1)[0]
    for pg in _pages.iter_pages(p.academic_literature, works=[work]):
        if pg.page_key == key:
            return "\n".join(f"{i:3d} {ln[:width]}" for i, ln in enumerate((pg.ocr_text or "").split("\n")))
    return f"page not found: {key}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: build ``families/book_pairs.jsonl`` + manifest + audit sample.

    :param argv: Arguments (default ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None, help="local book config (default from paths.json)")
    ap.add_argument("--spans", type=Path, default=None, help="located-spans file (default from paths.json)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output root (default paths.out_dir)")
    ap.add_argument("--with-entry-commentary", action="store_true",
                    help="also emit analyse rows from entry discussion / commentary (L2 builds these by default)")
    ap.add_argument("--no-noise", action="store_true", help="no noised_edition rows")
    ap.add_argument("--dry-run", action="store_true", help="build and print counts; write nothing")
    ap.add_argument("--show-page", default=None, help="print a page's numbered OCR lines and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = _paths.load_paths()
    if args.show_page:
        print(show_page(args.show_page, p))
        return
    t0 = time.time()
    cfg, spans, cpath, spath = load_local_config(p, args.config, args.spans)
    res = build(p, cfg, spans, noiser=False if args.no_noise else None,
                with_entry_commentary=args.with_entry_commentary)
    LOG.info("[%5.1fs] built %d rows", time.time() - t0, len(res["rows"]))
    if args.dry_run:
        print(json.dumps({"rows": len(res["rows"]), "drops": res["report"]["drops"]}, indent=1, ensure_ascii=False))
        return
    build_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    build_args.update({"config_file": cpath.name, "spans_file": spath.name})
    man = write_outputs(res, p, args.out_dir, build_args)
    LOG.info("[%5.1fs] wrote %d rows", time.time() - t0, man["checks"]["rows"])
    print(json.dumps({"total": man["counts"]["total"], "checks": man["checks"],
                      "per_source": {k: {kk: v[kk] for kk in ("docs", "translate_docs", "analyse_docs", "rows", "tokens")}
                                     for k, v in man["per_source"].items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
