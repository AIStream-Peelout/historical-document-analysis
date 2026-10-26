# File name: web_pairs.py
# Date: 10/26/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Web translation pairs: human translations from open-access article / web sources (text27b v0_2, 2026-10-26).

The open-access fetch of 2026-10-26 stored article- and post-length texts under ``paths.academic_literature``
in a flat layout (no page structuring), one directory per source::

    <dir>/metadata.json                    # source-level fields, ``licence``
    <dir>/items.jsonl                      # one line per item (slug, title, authors, year, licence_terms, ...)
    <dir>/text/<slug>.txt                  # extracted plain text of the item
    <dir>/metadata/<slug>_metadata.json    # shelfmarks, PGP ids, hold-out reason and an UNVERIFIED ``review`` block

Which directories are read, and which items are skipped (not Genizah, low value), comes from the local, git-ignored
config (``paths.json`` key ``web_pairs_config``, default ``logs/text_finetune_27b/web_pairs_local.json``:
``{"dirs": [...], "skip_items": {dir: [slug, ...]}, "priority": [dir, ...]}``); no holdings are named here.

Pairs. For an item that translates a document, the translation span is located by reading the item (D15:
LLM-located spans are approved) and recorded in the local spans file (``web_pairs_spans`` key, default
``logs/text_finetune_27b/web_pairs_spans.json``). Each span is a list of pieces -- an inclusive line range of the
item text with its first / last 24 characters and a sha1 -- re-cut and verified at build time
(:func:`book_pairs.cut_span`), so every raw target is a verbatim substring of the item text; normalisation ops
(:data:`WEB_NORMALISERS`) only delete characters or join lines, and :func:`book_pairs.words_subset` asserts that
no word was added. The input is

* the PGP digital edition of the document (``input: pgp_edition``; the default when PGP has an edition) -- the
  whole edition, or a verified line range of it (``pgp_lines``) when the item translates part of the document; or
* the item's own transcription span (``input: span``; only when PGP has no edition of the document), marked
  ``meta.edition_source = "web"``.

Second pass (D15), per pair, as in :mod:`book_pairs`: the F1 length band (0.5–2.5 English, 0.5–2.0 Hebrew) and the
overlap-rank sanity check against the editions of every other document of the same source directory
(:func:`book_pairs.pair_check`); the independence screen against the PGP digital translations of that language
(word 5-gram containment ≥ 0.5 or character ratio > 0.9 → ``independence_dup_pgp_translation``,
:func:`book_pairs.independence_check`).

Dedupe and splits. A document is matched to PGP ids by the spans file's hints, its shelfmark keys and (item
transcriptions) edition n-grams (:func:`book_pairs.match_pgpids`). It is dropped when PGP already carries a
translation into the target language (``dup_pgp_translation``), when an L5 book pair already translates it into
that language (``dup_book_pairs``) or when an earlier web item does (``dup_within_web``). Hold-outs
(:class:`holdouts.Holdouts`): an item whose text or metadata names ANY held document (benchmark or eval tier,
:func:`contamination.text_mention_reason`) is an eval-only reference -- all its pairs go to ``heldout``
(``split_reason = item_mention``); otherwise the document's own benchmark / eval-tier match → ``heldout``, a PGP
id v0_1 trains on → ``train``, else the D14 hash split 80/10/10 (:func:`book_pairs.assign_split`).

Rows (schema of :mod:`rows`, ``build = v0_2``): ``translate_en`` / ``translate_he`` with condition ``edition``;
``noised_edition`` for Hebrew-script editions in train; an ``edition_rescript`` twin for Arabic-script Arabic
documents (``translate_en``). ``meta.source = web:<dir>:<slug>``, ``meta.doc_id = web:<dir>:<slug>:<doc>``,
``meta.translator`` (spans file, else the item's review block, else ``unknown``) with ``translator_source``,
``meta.edition_source`` (``pgp`` / ``web``), ``meta.licence`` / ``licence_class`` (item terms, else the
directory's), ``meta.page_keys`` = the item key ``<dir>/text/<slug>.txt``.

Commentary passages (an author's discussion of a specific document) are NOT built in this pass; they are counted
(:func:`commentary_counts`) for the next one.

Outputs under ``paths.out_dir``: ``families/web_pairs.jsonl`` (all splits), ``families/web_pairs_manifest.json``
(counts per family × condition × lang_class × split, per-source pairs by language × split, translators, drops by
reason, independence drops, commentary counts, document records, build args) and ``audit/web_pairs_sample_30.jsonl``.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs                       # full build
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --dry-run             # counts only
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --candidates OUT.json # items to read
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --show DIR/SLUG       # numbered item lines
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --show-pgp PGPID      # numbered PGP edition
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --find-pgp "T-S 12.537"  # PGP ids by shelfmark
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --grep-pgp "PHRASE"   # PGP ids by edition text
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --fill-specs SPECS.json [SPECS2.json ...]
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --check               # per-document verdicts
    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.web_pairs --check-specs SPECS.json  # verdicts, nothing written

Public API: :data:`WEB_NORMALISERS`, :func:`drop_bracket_refs`, :func:`drop_inline_line_numbers`,
:func:`web_normalise`, :func:`item_key`, :class:`WebItem`, :func:`load_local_config`, :func:`load_items`,
:func:`licence_of`, :func:`translators_of`, :class:`WebDoc`, :func:`pgp_edition_text`, :func:`pgp_line_piece`,
:func:`cut_pgp_lines`, :func:`doc_from_spec`, :func:`spec_to_span_doc`, :func:`fill_specs`,
:func:`item_holdout_reason`, :func:`assign_web_split`, :func:`book_pair_keys`, :func:`commentary_counts`,
:func:`candidates`, :func:`build`, :func:`scrub_rows`, :func:`per_source`, :func:`audit_sample`, :func:`write_outputs`,
:func:`show_item`, :func:`show_pgp`, :func:`find_pgp`, :func:`main`.
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.finetuning.text27b import book_pairs as _bp
from src.finetuning.text27b import builder as _bd
from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import shelfmarks as _sm
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)
BUILD = _rows.BUILD
FAMILY_FILE = "web_pairs"
MANIFEST_NAME = "web_pairs_manifest.json"
AUDIT_NAME = "web_pairs_sample_30.jsonl"
AUDIT_SIZE = 30
DEFAULT_CONFIG = Path("logs") / "text_finetune_27b" / "web_pairs_local.json"
DEFAULT_SPANS = Path("logs") / "text_finetune_27b" / "web_pairs_spans.json"
#: Review translation levels that make an item a translation candidate (``excerpts`` / ``none`` are not).
CANDIDATE_LEVELS: Tuple[str, ...] = ("full", "partial")
#: Target languages with a translate family.
TARGET_LANGS: Tuple[str, ...] = ("en", "he")
#: ``meta.translator`` when neither the spans file nor the review names one.
UNKNOWN_TRANSLATOR = "unknown"
#: Split reason of every pair of an item that names a held document (eval-only reference).
ITEM_MENTION = "item_mention"

_BRACKET_REF = re.compile(r"[ \t]?\[\d{1,3}\]")
_INLINE_PAREN_NUM = re.compile(r"(?:(?<=\s)|^)\((\d{1,3})\)(?=\s)", re.M)


# ============================================================================== normalisers
def drop_bracket_refs(text: str) -> Tuple[str, int]:
    """Remove bracketed footnote references (``word[2]``, ``line [12]``); editorial brackets with letters stay.

    :param text: Span text.
    :returns: ``(text, number removed)``.
    """
    n = len(_BRACKET_REF.findall(text))
    return _BRACKET_REF.sub("", text), n


def drop_inline_line_numbers(text: str) -> Tuple[str, int]:
    """Remove ``(N)`` line markers written inline in a translation (``… God. (3) And …``) when at least two occur in
    mostly ascending order (a lone ``(1)`` may be a list item and stays).

    :param text: Span text.
    :returns: ``(text, number removed)``.
    """
    nums = [int(m.group(1)) for m in _INLINE_PAREN_NUM.finditer(text)]
    if len(nums) < 2 or sum(1 for a, b in zip(nums, nums[1:]) if b > a) < 0.6 * (len(nums) - 1):
        return text, 0
    out = _INLINE_PAREN_NUM.sub("", text)
    return "\n".join(re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in out.split("\n")), len(nums)


#: Normalisers a spans file may name: :data:`book_pairs.NORMALISERS` plus the web-specific ones.
WEB_NORMALISERS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    **_bp.NORMALISERS,
    "drop_bracket_refs": drop_bracket_refs,
    "drop_inline_line_numbers": drop_inline_line_numbers,
}


def web_normalise(text: str, ops: Sequence[str]) -> Tuple[str, List[str]]:
    """Apply named normalisation ops in order (:data:`WEB_NORMALISERS`).

    :param text: Raw span text.
    :param ops: Op names.
    :returns: ``(text, applied op names prefixed W:)`` (an op that changed nothing is not listed).
    :raises KeyError: On an unknown op.
    """
    done: List[str] = []
    for op in ops:
        text, n = WEB_NORMALISERS[op](text)
        if n:
            done.append(f"W:{op}")
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip(), done


# ============================================================================== items
def item_key(dir_name: str, slug: str) -> str:
    """Literature key of a web item (its text file relative to the literature root).

    :param dir_name: Source directory.
    :param slug: Item slug.
    :returns: ``<dir>/text/<slug>.txt``.
    """
    return f"{dir_name}/text/{slug}.txt"


@dataclass
class WebItem:
    """One fetched item with the fields the build uses (metadata and review are taken as given, review unverified)."""

    dir: str
    slug: str
    title: str
    authors: List[str]
    year: Any
    licence: Optional[str]
    licence_class: str
    review: Dict[str, Any] = field(default_factory=dict)
    meta_holdout: Optional[str] = None
    shelfmarks: List[str] = field(default_factory=list)
    pgpids: List[str] = field(default_factory=list)
    heuristic_commentary: bool = False
    text_path: Optional[Path] = None

    @property
    def key(self) -> str:
        """Literature key (:func:`item_key`).

        :returns: The key.
        """
        return item_key(self.dir, self.slug)

    @property
    def level(self) -> Optional[str]:
        """Review translation level (``full`` / ``partial`` / ``excerpts`` / ``none``).

        :returns: Level or None.
        """
        return self.review.get("translation_level")

    @property
    def has_commentary(self) -> bool:
        """Whether the item discusses specific documents (review flag, else the fetch heuristic).

        :returns: Flag.
        """
        v = self.review.get("has_commentary")
        return bool(self.heuristic_commentary if v is None else v)

    def text(self) -> str:
        """The item text ('' when the file is missing).

        :returns: Text.
        """
        return self.text_path.read_text(encoding="utf-8") if self.text_path and self.text_path.exists() else ""


def load_local_config(p: Optional[_paths.Paths] = None, config: Optional[Path] = None,
                      spans: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    """Read the local (git-ignored) web-source configuration and the located-spans file.

    :param p: Build paths (``web_pairs_config`` / ``web_pairs_spans`` keys override the defaults).
    :param config: Explicit config file.
    :param spans: Explicit spans file.
    :returns: ``(config, spans, config path, spans path)``; a missing file gives an empty dict.
    """
    p = p or _paths.load_paths()
    cpath = Path(config) if config else Path(p.get("web_pairs_config") or (p.repo / DEFAULT_CONFIG))
    spath = Path(spans) if spans else Path(p.get("web_pairs_spans") or (p.repo / DEFAULT_SPANS))
    cfg = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    spn = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    return cfg, spn, cpath, spath


def licence_of(item_terms: Optional[str], dir_licence: Optional[str]) -> Tuple[Optional[str], str]:
    """Licence string and class of an item (its own terms first, else the directory's statement).

    :param item_terms: The item's ``licence_terms``.
    :param dir_licence: The directory metadata's ``licence``.
    :returns: ``(licence, licence_class)`` with the class from :func:`literature_l1.licence_class`.
    """
    from src.finetuning.text27b.literature_l1 import licence_class
    lic = item_terms or dir_licence
    return lic, licence_class(lic)


def translators_of(review: Mapping[str, Any]) -> List[str]:
    """Translator names of an item's review block (empty when it names none).

    :param review: Review block.
    :returns: Names, order kept, no duplicates.
    """
    out: List[str] = []
    for t in review.get("translators") or []:
        s = re.sub(r"\s+", " ", str(t or "")).strip()
        if s and s not in out:
            out.append(s)
    return out


def load_items(root: Path, dirs: Sequence[str], skip: Optional[Mapping[str, Sequence[str]]] = None
               ) -> Dict[str, WebItem]:
    """Read the items of the configured web-source directories (duplicates of another item left out).

    :param root: ``paths.academic_literature``.
    :param dirs: Directory names.
    :param skip: ``{dir: [slug, ...]}`` items to leave out (local config).
    :returns: item key → item.
    """
    skip = skip or {}
    out: Dict[str, WebItem] = {}
    for d in dirs:
        base = Path(root) / d
        if not (base / "items.jsonl").exists():
            LOG.warning("no items.jsonl in %s", d)
            continue
        dmeta = json.loads((base / "metadata.json").read_text(encoding="utf-8")) if (base / "metadata.json").exists() else {}
        skipped = set(skip.get(d) or [])
        for it in _rows.read_jsonl(base / "items.jsonl"):
            slug = str(it.get("slug") or "")
            if not slug or slug in skipped or it.get("duplicate_of"):
                continue
            mpath = base / "metadata" / f"{slug}_metadata.json"
            m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
            if m.get("duplicate_of"):
                continue
            lic, lcls = licence_of(m.get("licence_terms") or it.get("licence_terms"), dmeta.get("licence"))
            authors = m.get("authors") or it.get("authors") or []
            out[item_key(d, slug)] = WebItem(
                dir=d, slug=slug, title=str(m.get("title") or it.get("title") or ""),
                authors=[str(a) for a in (authors if isinstance(authors, list) else [authors])],
                year=m.get("year") or it.get("year"), licence=lic, licence_class=lcls,
                review=dict(m.get("review") or it.get("review") or {}),
                meta_holdout=m.get("holdout_reason") or it.get("holdout_reason"),
                shelfmarks=[str(s) for s in m.get("shelfmarks_mentioned") or it.get("shelfmarks_mentioned") or []],
                pgpids=[str(x) for x in m.get("pgpids_resolved") or it.get("pgpids_resolved") or []],
                heuristic_commentary=bool(m.get("has_commentary") or it.get("has_commentary")),
                text_path=base / "text" / f"{slug}.txt")
    return out


# ============================================================================== documents
@dataclass
class WebDoc(_bp.BookDoc):
    """A web-item document: :class:`book_pairs.BookDoc` with the web source fields.

    ``source`` is the directory, ``doc`` is ``<slug>:<label>`` (unique over the directory, so the pair check's decoys
    are every other document of the directory).
    """

    item_slug: str = ""
    translator: str = UNKNOWN_TRANSLATOR
    translator_source: str = "none"
    edition_source: str = "pgp"
    pgp_edition_pgpid: Optional[str] = None
    pgp_lines: Optional[List[int]] = None

    @property
    def doc_id(self) -> str:
        """Stable document id ``web:<dir>:<slug>:<label>``.

        :returns: The id.
        """
        return f"web:{self.source}:{self.doc}"


def pgp_edition_text(ix: _bp.PgpIndex, pgpid: str) -> Optional[Tuple[str, str]]:
    """The PGP digital edition of a document as the builder renders inputs (Latin-only lines removed).

    :param ix: PGP index.
    :param pgpid: PGP id.
    :returns: ``(edition, edition source slug)`` or None when PGP has no edition text.
    """
    d = ix.docs.get(str(pgpid)) or {}
    if not d.get("edition_text"):
        return None
    return tn.strip_latin_lines(str(d["edition_text"]))[0], str(d.get("edition_source_slug") or "")


def pgp_line_piece(edition: str, first: int, last: int) -> Dict[str, Any]:
    """Spans-file record of a line range of a PGP edition (head, tail and sha1 filled in).

    :param edition: Edition text (:func:`pgp_edition_text`).
    :param first: First line index.
    :param last: Last line index (inclusive).
    :returns: ``{"lines", "head", "tail", "sha1"}``.
    """
    pc = _bp.piece_from_lines("pgp", edition, first, last)
    return {"lines": [first, last], "head": pc.head, "tail": pc.tail, "sha1": pc.sha1}


def cut_pgp_lines(edition: str, spec: Optional[Mapping[str, Any]]) -> Tuple[Optional[str], str]:
    """Cut and verify a line range of a PGP edition (the partial-translation input).

    :param edition: Edition text.
    :param spec: ``{"lines": [a, b], "head", "tail", "sha1"}`` or None for the whole edition.
    :returns: ``(text, "ok")`` or ``(None, reason)`` (reasons of :func:`book_pairs.cut_piece`).
    """
    if not spec:
        return edition, "ok"
    pc = _bp.Piece(page="pgp", lines=(int(spec["lines"][0]), int(spec["lines"][1])), head=str(spec.get("head", "")),
                   tail=str(spec.get("tail", "")), sha1=str(spec.get("sha1", "")))
    return _bp.cut_piece(pc, edition)


def _cut_kind(span: Mapping[str, Any], page_texts: Mapping[str, Optional[str]]
              ) -> Tuple[Optional[Tuple[str, str, List[str]]], Dict[str, Any]]:
    """Cut, normalise and word-check one span.

    :param span: ``{"pieces": [...], "normalise": [...]}``.
    :param page_texts: item key → text.
    :returns: ``((text, raw, ops) or None, info)``.
    """
    raw, ci = _bp.cut_span(span, page_texts)
    if raw is None:
        return None, ci
    text, ops = web_normalise(raw, span.get("normalise") or [])
    ok, missing = _bp.words_subset(text, raw)
    if not ok:
        return None, {"fail": "words_added", "missing": missing}
    return (text, raw, ops), ci


def doc_from_spec(item: WebItem, d: Mapping[str, Any], text: str, ix: _bp.PgpIndex
                  ) -> Tuple[Optional[WebDoc], Dict[str, Any]]:
    """WebDoc from one spans-file document: cut and verify the spans, resolve the input.

    :param item: The item.
    :param d: Spans-file document (``doc``, ``shelfmarks``, ``pgpids``, ``target_lang``, ``lang_class``, ``input``,
        ``pgpid`` (edition to use), ``pgp_lines``, ``translator``, ``spans``; or ``not_placed``).
    :param text: The item text.
    :param ix: PGP index.
    :returns: ``(doc or None, info)``; ``info["drop"]`` names the reason when no document results.
    """
    info: Dict[str, Any] = {"doc": d.get("doc")}
    spans = d.get("spans") or {}
    if d.get("not_placed") and not spans:
        info["drop"] = f"not_placed:{d['not_placed']}"
        return None, info
    lang = str(d.get("target_lang") or "")
    if lang not in TARGET_LANGS:
        info["drop"] = f"target_lang_unsupported:{lang or 'none'}"
        return None, info
    page_texts = {item.key: text}
    got: Dict[str, Tuple[str, str, List[str]]] = {}
    for kind in ("translation", "edition"):
        if kind in spans:
            res, ci = _cut_kind(spans[kind], page_texts)
            if res is None:
                info.setdefault("span_failures", {})[kind] = ci
                continue
            got[kind] = res
    if "translation" not in got:
        info["drop"] = "span_cut_failed" if info.get("span_failures") else "no_translation_span"
        return None, info
    hints = [str(x) for x in d.get("pgpids") or []]
    mode = str(d.get("input") or ("span" if "edition" in spans else "pgp_edition"))
    pg_used: Optional[str] = None
    if mode == "pgp_edition":
        cands = ([str(d["pgpid"])] if d.get("pgpid") else []) + hints
        for s in d.get("shelfmarks") or []:
            for k in _sm.keys(s):
                cands += sorted(ix.by_key.get(k, set()))
        found = next(((pid, e) for pid in dict.fromkeys(cands) for e in [pgp_edition_text(ix, pid)] if e), None)
        if found is None:
            info["drop"] = "no_pgp_edition"
            return None, info
        pg_used, (edition, slug) = found[0], found[1]
        edition, why = cut_pgp_lines(edition, d.get("pgp_lines"))
        if edition is None:
            info["drop"] = f"pgp_lines_{why}"
            return None, info
        input_source, ed_source, ed_ops = f"pgp_digital_edition:{slug}", "pgp", []
    else:
        if "edition" not in got:
            info["drop"] = "span_cut_failed" if "edition" in (info.get("span_failures") or {}) else "no_edition_span"
            return None, info
        edition, _, ed_ops = got["edition"]
        input_source, ed_source = f"web_edition:{item.dir}:{item.slug}", "web"
    if not edition.strip():
        info["drop"] = "no_edition"
        return None, info
    script = tn.dominant_script(edition)
    tr_text, tr_raw, tr_ops = got["translation"]
    names = translators_of(item.review)
    if d.get("translator"):
        translator, t_src = str(d["translator"]), "spans_file"
    elif names:
        translator, t_src = "; ".join(names), "review"
    else:
        translator, t_src = UNKNOWN_TRANSLATOR, "none"
    pages = {"translation": [item.key]}
    if ed_source == "web":
        pages["edition"] = [item.key]
    doc = WebDoc(source=item.dir, doc=f"{item.slug}:{d['doc']}", shelfmarks=[str(s) for s in d.get("shelfmarks") or []],
                 target_lang=lang, edition=edition, translation=tr_text, translation_raw=tr_raw, script=script,
                 lang_class=str(d.get("lang_class") or _bp._lang_class(script)), input_source=input_source,
                 licence=item.licence, licence_class=item.licence_class,
                 pgp_hints=hints + ([pg_used] if pg_used and pg_used not in hints else []),
                 provenance={"work_pages": pages, "located_by": "agent_read_item_D15", "edition_ops": ed_ops,
                             "item": item.key, "note": d.get("note")},
                 norm_ops=tr_ops, located=True, item_slug=item.slug, translator=translator,
                 translator_source=t_src, edition_source=ed_source, pgp_edition_pgpid=pg_used,
                 pgp_lines=list(d["pgp_lines"]["lines"]) if d.get("pgp_lines") else None)
    info["pages"] = pages
    return doc, info


# ============================================================================== spans file
def _piece_specs(ranges: Sequence[Sequence[Any]], key: str, text: str) -> List[Dict[str, Any]]:
    """Pieces for agent line ranges ``[first, last]`` or ``[first, last, col_start, col_end]``.

    :param ranges: Ranges (inclusive line indices of ``text.split("\\n")``; columns may be null).
    :param key: Item key.
    :param text: Item text.
    :returns: Piece dicts.
    """
    out = []
    for r in ranges:
        cols = (r[2], r[3]) if len(r) >= 4 else (None, None)
        out.append(_bp.piece_from_lines(key, text, int(r[0]), int(r[1]), cols).to_dict())
    return out


def spec_to_span_doc(spec: Mapping[str, Any], key: str, text: str, ix: Optional[_bp.PgpIndex] = None
                     ) -> Dict[str, Any]:
    """Spans-file document from an agent spec (line ranges → verified pieces).

    :param spec: ``{"doc", "shelfmarks", "pgpids", "pgpid", "pgp_lines": [a, b], "input", "target_lang",
        "lang_class", "translator", "translation": [[a, b], ...], "edition": [[a, b], ...], "normalise": [...],
        "edition_normalise": [...], "not_placed", "note"}``.
    :param key: Item key.
    :param text: Item text.
    :param ix: PGP index (needed for ``pgp_lines``).
    :returns: Spans-file document.
    :raises ValueError: When ``pgp_lines`` is given without a resolvable PGP edition.
    """
    out: Dict[str, Any] = {k: spec[k] for k in ("doc", "shelfmarks", "pgpids", "pgpid", "input", "target_lang",
                                                 "lang_class", "translator", "not_placed", "note") if spec.get(k)}
    spans: Dict[str, Any] = {}
    if spec.get("translation"):
        spans["translation"] = {"pieces": _piece_specs(spec["translation"], key, text),
                                "normalise": list(spec.get("normalise") or [])}
    if spec.get("edition"):
        spans["edition"] = {"pieces": _piece_specs(spec["edition"], key, text),
                            "normalise": list(spec.get("edition_normalise") or [])}
    if spans:
        out["spans"] = spans
    if spec.get("pgp_lines"):
        ed = pgp_edition_text(ix, str(spec.get("pgpid"))) if ix is not None and spec.get("pgpid") else None
        if ed is None:
            raise ValueError(f"pgp_lines need a pgpid with a PGP edition: {key} {spec.get('doc')}")
        a, b = spec["pgp_lines"][:2]
        out["pgp_lines"] = pgp_line_piece(ed[0], int(a), int(b))
    return out


def fill_specs(spec_files: Sequence[Path], root: Path, ix: Optional[_bp.PgpIndex] = None,
               existing: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge agent spec files into a spans file (later files replace earlier records of the same item).

    :param spec_files: JSON files holding ``{"items": [{"dir", "slug", "docs": [spec, ...], "commentary": ...}]}``
        or a bare list of such items.
    :param root: Literature root.
    :param ix: PGP index (for ``pgp_lines``).
    :param existing: Current spans file content (kept for items the specs do not mention).
    :returns: ``{"items": [...]}`` sorted by dir and slug; errors are recorded per item under ``errors``.
    """
    items: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in (existing or {}).get("items") or []:
        items[(it["dir"], it["slug"])] = dict(it)
    for f in spec_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for it in data["items"] if isinstance(data, dict) else data:
            key = item_key(it["dir"], it["slug"])
            text = (Path(root) / key).read_text(encoding="utf-8")
            rec: Dict[str, Any] = {"dir": it["dir"], "slug": it["slug"], "docs": []}
            for k in ("commentary", "commentary_docs", "note"):
                if k in it:
                    rec[k] = it[k]
            for spec in it.get("docs") or []:
                try:
                    rec["docs"].append(spec_to_span_doc(spec, key, text, ix))
                except (ValueError, IndexError) as e:  # a bad line range is recorded, not fatal
                    rec.setdefault("errors", []).append({"doc": spec.get("doc"), "error": str(e)})
            items[(it["dir"], it["slug"])] = rec
    return {"items": [items[k] for k in sorted(items)]}


# ============================================================================== hold-outs, splits, dedupe
def item_holdout_reason(item: WebItem, text: str, holdouts: Any, mention_reason: Optional[Callable[..., Any]] = None
                        ) -> Optional[str]:
    """Whether an item names any held document (either tier): its text, its metadata hold-out reason, its mentioned
    shelfmarks and resolved PGP ids.

    :param item: The item.
    :param text: Item text.
    :param holdouts: :class:`holdouts.Holdouts` (``is_benchmark`` / ``is_eval_doc``).
    :param mention_reason: ``(texts, holdouts) -> reason | None`` (default
        :func:`contamination.text_mention_reason`).
    :returns: The reason (``text:<r>``, ``meta:<r>``, ``shelfmark:<r>``, ``pgpid:<r>``) or None.
    """
    if mention_reason is None:
        from src.finetuning.text27b.contamination import text_mention_reason as mention_reason
    r = mention_reason([text], holdouts)
    if r:
        return f"text:{r}"
    for s in item.shelfmarks:
        r = holdouts.is_benchmark(shelfmark=s) or holdouts.is_eval_doc(shelfmark=s)
        if r:
            return f"shelfmark:{r}"
    for pid in item.pgpids:
        r = holdouts.is_benchmark(pgpid=pid) or holdouts.is_eval_doc(pgpid=pid)
        if r:
            return f"pgpid:{r}"
    if item.meta_holdout:
        return f"meta:{item.meta_holdout}"
    return None


def assign_web_split(doc: WebDoc, pgpids: Sequence[str], item_reason: Optional[str], holdouts: Any,
                     ix: _bp.PgpIndex) -> Tuple[str, str, Optional[str]]:
    """Split of a web document: an item naming a held document → ``heldout``; else :func:`book_pairs.assign_split`.

    :param doc: The document.
    :param pgpids: Matched PGP ids.
    :param item_reason: :func:`item_holdout_reason` of its item.
    :param holdouts: Hold-out set.
    :param ix: PGP index.
    :returns: ``(split, split_reason, holdout_reason)``; an item-mention hold-out without a reason of the document's
        own carries ``item_mention:<item reason>``.
    """
    own = _bp.assign_split(doc, pgpids, holdouts, ix)
    if item_reason:
        return "heldout", ITEM_MENTION, own[2] or f"{ITEM_MENTION}:{item_reason}"
    return own


def book_pair_keys(path: Path) -> Dict[Tuple[str, str], str]:
    """Documents the L5 book pairs already translate: ``(family, shelfmark key or pgpid:<id>) → doc_id``.

    :param path: ``families/book_pairs.jsonl`` (missing: empty).
    :returns: The map.
    """
    out: Dict[Tuple[str, str], str] = {}
    if not Path(path).exists():
        return out
    for r in _rows.read_jsonl(path):
        m = r.get("meta") or {}
        fam = str(m.get("family") or "")
        if not fam.startswith("translate_"):
            continue
        ids = [k for s in str(m.get("shelfmark") or "").split(" + ") for k in _sm.keys(s)]
        ids += [f"pgpid:{x}" for x in m.get("pgpids") or ([m["pgpid"]] if m.get("pgpid") else [])]
        for k in ids:
            out.setdefault((fam, k), str(m.get("doc_id")))
    return out


def _doc_ids(doc: WebDoc, pgpids: Sequence[str]) -> List[str]:
    """Dedupe identities of a document: shelfmark keys and ``pgpid:<id>``.

    :param doc: The document.
    :param pgpids: Matched PGP ids.
    :returns: Identities.
    """
    return list(doc.keys) + [f"pgpid:{x}" for x in pgpids]


# ============================================================================== commentary (counted only)
def commentary_counts(items: Mapping[str, WebItem], item_reasons: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """Items with commentary on specific documents (built in the next pass; counted now).

    :param items: item key → item.
    :param item_reasons: item key → :func:`item_holdout_reason` (only items present are classified).
    :returns: ``{"items", "eval_only", "train_eligible", "by_dir": {dir: {...}}, "translation_items_with_commentary"}``.
    """
    by_dir: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for k, it in items.items():
        if not (it.has_commentary and (it.shelfmarks or it.review.get("shelfmarks_translated"))):
            continue
        c = by_dir[it.dir]
        c["items"] += 1
        c["eval_only" if item_reasons.get(k) else "train_eligible"] += 1
        if it.level in CANDIDATE_LEVELS:
            c["translation_items"] += 1
    tot = collections.Counter()
    for c in by_dir.values():
        tot.update(c)
    return {"items": tot["items"], "eval_only": tot["eval_only"], "train_eligible": tot["train_eligible"],
            "translation_items_with_commentary": tot["translation_items"],
            "by_dir": {d: dict(c) for d, c in sorted(by_dir.items())}}


# ============================================================================== candidates (for span placement)
def candidates(items: Mapping[str, WebItem], ix: _bp.PgpIndex, holdouts: Any,
               item_reasons: Optional[Mapping[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
    """Translation items with their translated documents resolved against PGP (the reading list for span placement).

    :param items: item key → item.
    :param ix: PGP index.
    :param holdouts: Hold-out set.
    :param item_reasons: Pre-computed item hold-out reasons (computed when absent).
    :returns: One record per candidate item: key, title, level, languages, translators, item hold-out reason and per
        translated shelfmark the PGP ids with ``edition`` / ``translation_<lang>`` flags and a suggested input.
    """
    out = []
    for k, it in sorted(items.items()):
        if it.level not in CANDIDATE_LEVELS:
            continue
        langs = [str(x) for x in it.review.get("translation_lang") or []]
        reason = (item_reasons or {}).get(k) if item_reasons is not None else item_holdout_reason(it, it.text(), holdouts)
        docs = []
        for s in it.review.get("shelfmarks_translated") or []:
            pids = sorted({x for kk in _sm.keys(s) for x in ix.by_key.get(kk, set())}, key=lambda x: int(x) if x.isdigit() else 0)
            pg = [{"pgpid": x, "edition": bool((ix.docs.get(x) or {}).get("edition_text")),
                   **{f"translation_{lg}": ix.has_translation(x, lg) for lg in TARGET_LANGS}} for x in pids]
            lang = langs[0] if langs else None
            if any(x.get(f"translation_{lang}") for x in pg):
                hint = "skip:pgp_has_translation"
            elif any(x["edition"] for x in pg):
                hint = "pgp_edition"
            else:
                hint = "span_if_item_prints_transcription"
            docs.append({"shelfmark": s, "pgp": pg, "suggested_input": hint})
        out.append({"key": k, "dir": it.dir, "slug": it.slug, "title": it.title, "level": it.level, "langs": langs,
                    "translators": translators_of(it.review), "item_holdout": reason,
                    "review_notes": it.review.get("notes"), "docs": docs})
    return out


# ============================================================================== build
def _patch_meta(row: Dict[str, Any], doc: WebDoc, item: WebItem, item_reason: Optional[str]) -> Dict[str, Any]:
    """Web fields on a row built by :func:`book_pairs.translate_rows`.

    :param row: Row (modified in place).
    :param doc: The document.
    :param item: Its item.
    :param item_reason: Item hold-out reason.
    :returns: The row.
    """
    m = row["meta"]
    m["source"] = f"web:{item.dir}:{item.slug}"
    m["translator"], m["translator_source"] = doc.translator, doc.translator_source
    m["edition_source"] = doc.edition_source
    m["web_item"] = {"dir": item.dir, "slug": item.slug, "title": item.title, "authors": item.authors,
                     "year": item.year}
    m["item_holdout_reason"] = item_reason
    own = str(m.get("holdout_reason") or "")
    m["held_out_vision_benchmark"] = bool(own and not own.startswith(f"{ITEM_MENTION}:")
                                          and _bd.BENCH_REASONS & {own.split(":")[-1]})
    m["licence"], m["licence_class"] = item.licence, item.licence_class
    m["web_doc"] = doc.doc
    m.pop("book_doc", None)
    if doc.pgp_edition_pgpid:
        m["pgp_edition_pgpid"] = doc.pgp_edition_pgpid
    if doc.pgp_lines:
        m["pgp_lines"] = doc.pgp_lines
    m["provenance"] = {**(m.get("provenance") or {}), "source": item.dir, "item": item.slug}
    return row


def build(p: Optional[_paths.Paths] = None, cfg: Optional[Mapping[str, Any]] = None,
          spans: Optional[Mapping[str, Any]] = None, holdouts: Any = None, ix: Optional[_bp.PgpIndex] = None,
          noiser: Any = None, tc: Any = None, items: Optional[Mapping[str, WebItem]] = None,
          texts: Optional[Mapping[str, str]] = None, independence: bool = True,
          book_keys: Optional[Mapping[Tuple[str, str], str]] = None,
          mention_reason: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Build the web translation rows from the spans file.

    :param p: Build paths.
    :param cfg: Local config (default :func:`load_local_config`).
    :param spans: Spans file content (default :func:`load_local_config`).
    :param holdouts: :class:`holdouts.Holdouts` (default ``Holdouts.load()``).
    :param ix: PGP index (default: ``paths.pgp_parallel`` + v0_1 + corpus).
    :param noiser: Noise model (default :func:`book_pairs.make_noiser`; False disables).
    :param tc: Token counter (default :class:`builder_v0.TokenCounter`).
    :param items: Pre-loaded items (tests; default :func:`load_items` over the configured dirs).
    :param texts: Pre-loaded item texts (tests; default read from disk).
    :param independence: Apply the independence screen.
    :param book_keys: Documents the L5 pairs translate (default :func:`book_pair_keys` of the families dir).
    :param mention_reason: Text-mention rule (tests; default :func:`contamination.text_mention_reason`).
    :returns: ``{"rows", "docs", "report", "items"}``.
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
    if ix is None:
        from src.finetuning.text27b import holdouts as _ho
        ix = _bp.PgpIndex.build(p.pgp_parallel, _ho.scan_v01(p.text_ft_v0_1), _sm.load_corpus(p))
    noise_info: Dict[str, Any] = {}
    if noiser is None:
        noiser, noise_info = _bp.make_noiser(p, p.pgp_parallel)
    elif noiser is False:
        noiser = None
    from src.finetuning.text27b import builder_v0 as _bv
    ctx = _bp.BuildContext(holdouts=holdouts, ix=ix, noiser=noiser, tc=tc if tc is not None else _bv.TokenCounter(p.tokenizer_json),
                           translations=_bp.PgpTranslations.build(ix) if independence else None)
    pgp_ex = p.pgp_export / "sources.csv"
    if pgp_ex.exists():
        ctx.scholar_names = tn.scholar_surnames(pgp_ex, list(_bv.load_local_config(p).get("scholar_extra") or []))
    if items is None:
        items = load_items(p.academic_literature, list(cfg.get("dirs") or []), cfg.get("skip_items") or {})
    if book_keys is None:
        book_keys = book_pair_keys(p.out("families", "book_pairs.jsonl"))
    texts = dict(texts or {})
    for k, it in items.items():
        if k not in texts:
            texts[k] = it.text()
    item_reasons = {k: item_holdout_reason(it, texts.get(k, ""), holdouts, mention_reason) for k, it in items.items()}
    LOG.info("[%5.1fs] %d items, %d eval-only", time.time() - t0, len(items), sum(1 for v in item_reasons.values() if v))
    drops: collections.Counter = collections.Counter()
    docs: List[Tuple[WebDoc, WebItem]] = []
    span_info: List[Dict[str, Any]] = []
    for rec in spans.get("items") or []:
        key = item_key(rec["dir"], rec["slug"])
        it = items.get(key)
        if it is None:
            drops["item_not_configured"] += 1
            span_info.append({"item": key, "drop": "item_not_configured"})
            continue
        for d in rec.get("docs") or []:
            doc, info = doc_from_spec(it, d, texts.get(key, ""), ix)
            info["item"] = key
            span_info.append(info)
            if doc is None:
                drops[f"{it.dir}:{info['drop']}"] += 1
                continue
            docs.append((doc, it))
    priority = list(cfg.get("priority") or [])
    docs.sort(key=lambda x: (priority.index(x[0].source) if x[0].source in priority else len(priority), x[0].source,
                             x[0].doc))
    by_dir: Dict[str, List[WebDoc]] = collections.defaultdict(list)
    for doc, _ in docs:
        by_dir[doc.source].append(doc)
    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], str] = {}
    for doc, it in docs:
        pgpids, how = _bp.match_pgpids(doc, ix)
        if doc.pgp_edition_pgpid and doc.pgp_edition_pgpid not in pgpids:
            pgpids = [doc.pgp_edition_pgpid] + pgpids
        reason = item_reasons.get(it.key)
        split, split_reason, hold = assign_web_split(doc, pgpids, reason, holdouts, ix)
        fam = f"translate_{doc.target_lang}"
        rec: Dict[str, Any] = {"source": f"web:{it.dir}:{it.slug}", "dir": it.dir, "item": it.slug, "doc": doc.doc,
                               "doc_id": doc.doc_id, "shelfmarks": doc.shelfmarks, "keys": doc.keys, "pgpids": pgpids,
                               "match": how, "family": fam, "split": split, "split_reason": split_reason,
                               "holdout_reason": hold, "item_holdout_reason": reason, "script": doc.script,
                               "lang_class": doc.lang_class, "edition_source": doc.edition_source,
                               "translator": doc.translator, "translate": None}
        ids = _doc_ids(doc, pgpids)
        dup_pg = [x for x in pgpids if ix.has_translation(x, doc.target_lang)]
        dup_book = next((book_keys[(fam, k)] for k in ids if (fam, k) in book_keys), None)
        dup_web = next((seen[(fam, k)] for k in ids if (fam, k) in seen and seen[(fam, k)] != doc.doc_id), None)
        if dup_pg:
            rec["translate"] = {"drop": "dup_pgp_translation", "pgpids": dup_pg}
        elif dup_book:
            rec["translate"] = {"drop": "dup_book_pairs", "of": dup_book}
        elif dup_web:
            rec["translate"] = {"drop": "dup_within_web", "of": dup_web}
        else:
            r, info = _bp.translate_rows(doc, split, pgpids, split_reason, hold, by_dir[doc.source], ctx)
            rec["translate"] = {**info, "rows": len(r)}
            for row in r:
                rows.append(_patch_meta(row, doc, it, reason))
            if r:
                for k in ids:
                    seen.setdefault((fam, k), doc.doc_id)
        if rec["translate"].get("drop"):
            drops[f"{it.dir}:translate:{rec['translate']['drop']}"] += 1
        records.append(rec)
    report = {"items": len(items), "items_eval_only": sum(1 for v in item_reasons.values() if v),
              "span_docs": span_info, "noise": noise_info, "drops": dict(sorted((drops + ctx.drops).items())),
              "commentary": commentary_counts(items, item_reasons),
              "commentary_flagged_by_span_agents": {
                  "items": sum(1 for rec in spans.get("items") or [] if rec.get("commentary")),
                  "documents": sum(len(rec.get("commentary_docs") or []) for rec in spans.get("items") or []
                                   if rec.get("commentary")),
                  "eval_only_items": sum(1 for rec in spans.get("items") or [] if rec.get("commentary")
                                         and item_reasons.get(item_key(rec["dir"], rec["slug"])))},
              "not_placed": dict(collections.Counter(str(s.get("drop", "")).split(":", 1)[1]
                                                     for s in span_info if str(s.get("drop", "")).startswith("not_placed:"))),
              "candidate_items": sum(1 for it in items.values() if it.level in CANDIDATE_LEVELS),
              "spans_items": len(spans.get("items") or []), "elapsed_s": round(time.time() - t0, 1)}
    LOG.info("[%5.1fs] %d rows from %d documents", time.time() - t0, len(rows), len(docs))
    return {"rows": rows, "docs": records, "report": report, "items": items, "item_reasons": item_reasons}


def scrub_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Project-name hygiene (F4/F12): rows whose messages name a scrubbed project are dropped; meta strings are scrubbed.

    :param rows: Rows.
    :returns: ``(kept rows, number dropped)``.
    """
    def strip(t: str) -> str:
        """Scrub one string.

        :param t: String.
        :returns: Scrubbed string.
        """
        return tn.strip_source_names(t)[0] if tn.has_source_names(t) else t

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for r in rows:
        if any(tn.has_source_names(str(x.get("content") or "")) for x in r["messages"]):
            dropped += 1
            continue
        r["meta"] = _rows._scrub_value(r["meta"], strip)
        kept.append(r)
    return kept, dropped


def _split_group(split: str) -> str:
    """Report group of a split: ``train``, ``heldout`` or ``dev_test``.

    :param split: Split.
    :returns: Group name.
    """
    return split if split in ("train", "heldout") else "dev_test"


def per_source(rows: Sequence[Dict[str, Any]], docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-directory pair counts: documents with pairs by language × split group, rows, tokens, drops.

    :param rows: Final rows.
    :param docs: Document records of :func:`build`.
    :returns: dir → {"pairs": {lang: {train, heldout, dev_test}}, "rows", "tokens", "drops", "translators"}.
    """
    out: Dict[str, Any] = {}
    kept = {r["meta"]["doc_id"] for r in rows if r["meta"]["condition"] == "edition"}
    for d in docs:
        s = out.setdefault(d["dir"], {"pairs": {lg: collections.Counter() for lg in TARGET_LANGS}, "rows": 0,
                                      "tokens": 0, "drops": collections.Counter(), "translators": collections.Counter()})
        t = d.get("translate") or {}
        if d["doc_id"] in kept:
            s["pairs"][d["family"].split("_", 1)[1]][_split_group(d["split"])] += 1
            s["translators"][d.get("translator") or UNKNOWN_TRANSLATOR] += 1
        elif t.get("drop"):
            s["drops"][t["drop"]] += 1
    for r in rows:
        d = str(r["meta"]["source"]).split(":")[1]
        if d in out:
            out[d]["rows"] += 1
            out[d]["tokens"] += int(r["meta"].get("tokens") or 0)
    for s in out.values():
        s["pairs"] = {lg: dict(c) for lg, c in s["pairs"].items()}
        s["drops"] = dict(s["drops"])
        s["translators"] = dict(s["translators"].most_common())
    return out


def audit_sample(rows: Sequence[Dict[str, Any]], n: int = AUDIT_SIZE, seed: int = 3407) -> List[Dict[str, Any]]:
    """Hand-check sample: edition-condition rows, round-robin over source directories, seeded.

    :param rows: Final rows.
    :param n: Sample size.
    :param seed: Seed.
    :returns: ``[{row_id, source, doc, family, split, translator, edition_source, source_text, target, target_raw,
        strip_ops, pair_check, independence}]``.
    """
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r["meta"]["condition"] == "edition":
            pools[str(r["meta"]["source"]).split(":")[1]].append(r)
    for k in sorted(pools):
        pools[k].sort(key=lambda r: r["meta"]["row_id"])
        rng.shuffle(pools[k])
    out: List[Dict[str, Any]] = []
    while len(out) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(out) < n:
                r = pools[k].pop()
                m = r["meta"]
                user = r["messages"][1]["content"]
                a, b = user.find("<document>\n"), user.find("\n</document>")
                out.append({"row_id": m["row_id"], "source": m["source"], "doc": m.get("web_doc"), "family": m["family"],
                            "split": m["split"], "pgpid": m.get("pgpid"), "shelfmark": m.get("shelfmark"),
                            "translator": m.get("translator"), "edition_source": m.get("edition_source"),
                            "source_text": user[a + 11:b] if a >= 0 else user, "target": r["messages"][2]["content"],
                            "target_raw": m.get("target_raw"), "strip_ops": m.get("strip_ops"),
                            "pair_check": m.get("pair_check"), "independence": m.get("independence")})
    return out


def write_outputs(result: Dict[str, Any], p: _paths.Paths, out_dir: Optional[Path] = None,
                  build_args: Optional[Dict[str, Any]] = None, exact_tokens: bool = True) -> Dict[str, Any]:
    """Exact token counts, family file, manifest and audit sample.

    :param result: Output of :func:`build`.
    :param p: Build paths.
    :param out_dir: Output root (default ``paths.out_dir``).
    :param build_args: CLI arguments recorded in the manifest.
    :param exact_tokens: Render and tokenize every row (False keeps estimates; tests).
    :returns: The manifest.
    """
    out = Path(out_dir) if out_dir else p.out_dir
    rows, n_scrub = scrub_rows(result["rows"])
    over: collections.Counter = collections.Counter()
    if exact_tokens:
        rows, over = _bd.finalize_tokens(rows, _bp.MAX_ROW_TOKENS)
    else:
        for r in rows:
            m = r["meta"]
            m["target_tokens"] = len(r["messages"][2]["content"]) // 3
            m["tokens"] = sum(len(x["content"]) for x in r["messages"]) // 3
    problems: collections.Counter = collections.Counter()
    for r in rows:
        for prob in _rows.validate_row(r):
            problems[prob] += 1
    rows.sort(key=lambda r: (r["meta"]["source"], r["meta"]["doc_id"], r["meta"]["family"], r["meta"]["condition"],
                             r["meta"].get("side") or ""))
    fam_path = out / "families" / f"{FAMILY_FILE}.jsonl"
    fam_path.parent.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(fam_path, rows)
    audit_path = out / "audit" / AUDIT_NAME
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(audit_path, audit_sample(rows))
    ids = [r["meta"]["row_id"] for r in rows]
    docs = result["docs"]
    kept_docs = [d for d in docs if (d.get("translate") or {}).get("rows")]
    indep = [d for d in docs if (d.get("translate") or {}).get("drop") == "independence_dup_pgp_translation"]
    translators: collections.Counter = collections.Counter(d.get("translator") or UNKNOWN_TRANSLATOR for d in kept_docs)
    pairs: Dict[str, Dict[str, int]] = {lg: dict(collections.Counter(_split_group(d["split"]) for d in kept_docs
                                                                     if d["family"] == f"translate_{lg}"))
                                        for lg in TARGET_LANGS}
    checks = {
        "rows": len(rows), "duplicate_row_ids": len(ids) - len(set(ids)), "schema_problems": dict(problems),
        "heldout_rows_in_train": sum(1 for r in rows if r["meta"]["split"] == "train" and
                                     (r["meta"].get("holdout_reason") or r["meta"].get("item_holdout_reason"))),
        "scrub_rows": sum(1 for r in rows if tn.has_source_names(json.dumps(r, ensure_ascii=False))),
        "ar_noised_rows": sum(1 for r in rows if r["meta"]["condition"] == "noised_edition" and
                              (r["meta"]["lang_class"] == "AR" or r["meta"].get("edition_script") == "arabic")),
        "marker_rows": sum(1 for r in rows if tn.strip_line_markers(r["messages"][2]["content"])[1]),
        "non_nfc_rows": sum(1 for r in rows if _rows.non_nfc_roles(r)),
        "target_not_verbatim": sum(1 for r in rows if not _bp.words_subset(r["messages"][2]["content"],
                                                                           r["meta"].get("target_raw") or
                                                                           r["messages"][2]["content"])[0]),
        "dropped_over_cap_exact": dict(over), "dropped_source_names": n_scrub,
        "rows_without_page_keys": sum(1 for r in rows if not r["meta"].get("page_keys")),
    }
    manifest = {
        "build": BUILD, "family_file": f"families/{FAMILY_FILE}.jsonl", "counts": _bd.count_table(rows),
        "pairs_by_lang_split": pairs, "documents_with_pairs": len(kept_docs),
        "translators": dict(translators.most_common()),
        "translator_sources": dict(collections.Counter(r["meta"].get("translator_source") for r in rows
                                                        if r["meta"]["condition"] == "edition")),
        "edition_sources": dict(collections.Counter(d.get("edition_source") for d in kept_docs)),
        "independence_drops": [{k: d.get(k) for k in ("doc_id", "pgpids")} | {"independence": d["translate"].get("independence")}
                               for d in indep],
        "per_source": per_source(rows, docs), "checks": checks, "report": result["report"], "documents": docs,
        "fingerprint": _rows.fingerprint(rows), "build_args": build_args or {},
    }
    (out / "families" / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    return manifest


# ============================================================================== CLI helpers
def show_item(key: str, root: Path, width: int = 220) -> str:
    """Numbered lines of an item text (indices as the spans file uses them).

    :param key: ``<dir>/<slug>`` or the item key.
    :param root: Literature root.
    :param width: Characters shown per line.
    :returns: Text.
    """
    d, _, slug = key.partition("/")
    slug = slug.replace("text/", "").removesuffix(".txt")
    text = (Path(root) / item_key(d, slug)).read_text(encoding="utf-8")
    return "\n".join(f"{i:4d}| {ln[:width]}" for i, ln in enumerate(text.split("\n")))


def show_pgp(pgpid: str, ix: _bp.PgpIndex) -> str:
    """Numbered lines of a PGP edition as the build renders it, with the translation flags.

    :param pgpid: PGP id.
    :param ix: PGP index.
    :returns: Text.
    """
    d = ix.docs.get(str(pgpid))
    if not d:
        return f"pgpid {pgpid}: not in the PGP pairs file (no edition)"
    ed = pgp_edition_text(ix, str(pgpid))
    head = (f"pgpid {pgpid} | {d.get('shelfmark')} | {d.get('lang_class')} | edition {d.get('edition_source_slug')} | "
            f"translation_en={ix.has_translation(str(pgpid), 'en')} translation_he={ix.has_translation(str(pgpid), 'he')}")
    if not ed:
        return head + "\n(no edition text)"
    return head + "\n" + "\n".join(f"{i:4d}| {ln}" for i, ln in enumerate(ed[0].split("\n")))


def find_pgp(query: str, ix: _bp.PgpIndex, grep: bool = False, limit: int = 20) -> str:
    """PGP documents for a shelfmark (by key) or, with ``grep``, whose edition contains a phrase.

    :param query: Shelfmark or phrase.
    :param ix: PGP index.
    :param grep: Substring search over the edition texts (brackets and vowel points ignored) instead of the key lookup.
    :param limit: Maximum documents listed.
    :returns: One line per document: pgpid, shelfmark, lang_class, edition / translation flags.
    """
    if grep:
        def flat(t: str) -> str:
            """Comparison form of an edition.

            :param t: Text.
            :returns: Text without brackets, vowel points and doubled spaces.
            """
            return re.sub(r"\s+", " ", re.sub(r"[\[\]()<>{}|\u0591-\u05C7]", "", t or ""))
        q = flat(query).strip()
        ids = [pid for pid, d in ix.docs.items() if q and q in flat(str(d.get("edition_text") or ""))]
    else:
        ids = sorted({x for k in _sm.keys(query) for x in ix.by_key.get(k, set())})
    lines = []
    for pid in sorted(ids, key=lambda x: int(x) if x.isdigit() else 0)[:limit]:
        d = ix.docs.get(pid) or {}
        cat = ix.catalogue.get(pid) or {}
        lines.append(f"{pid}\t{d.get('shelfmark') or cat.get('shelfmark')}\t{d.get('lang_class')}\t"
                     f"edition={bool(d.get('edition_text'))}\ttranslation_en={ix.has_translation(pid, 'en')}\t"
                     f"translation_he={ix.has_translation(pid, 'he')}")
    return "\n".join(lines) or f"no PGP document for {query!r}"


def _index(p: _paths.Paths, with_grams: bool = True) -> _bp.PgpIndex:
    """PGP index over the pairs file, v0_1 and the corpus.

    :param p: Build paths.
    :param with_grams: Build the edition n-gram index.
    :returns: The index.
    """
    from src.finetuning.text27b import holdouts as _ho
    return _bp.PgpIndex.build(p.pgp_parallel, _ho.scan_v01(p.text_ft_v0_1), _sm.load_corpus(p), with_grams=with_grams)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command line (see the module docstring).

    :param argv: Arguments (default ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--spans", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="print per-document verdicts (no files written)")
    ap.add_argument("--show", default=None, help="DIR/SLUG: numbered item lines")
    ap.add_argument("--show-pgp", default=None, help="PGPID: numbered PGP edition lines")
    ap.add_argument("--find-pgp", default=None, help="SHELFMARK: PGP documents by shelfmark key")
    ap.add_argument("--grep-pgp", default=None, help="PHRASE: PGP documents whose edition contains the phrase")
    ap.add_argument("--candidates", type=Path, default=None, help="write the candidate reading list (JSON)")
    ap.add_argument("--fill-specs", type=Path, nargs="+", default=None, help="agent spec files → spans file")
    ap.add_argument("--check-specs", type=Path, nargs="+", default=None,
                    help="agent spec files: build them in memory (spans file untouched) and print the verdicts")
    ap.add_argument("--no-independence", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = _paths.load_paths()
    if args.show:
        print(show_item(args.show, p.academic_literature))
        return
    if args.show_pgp:
        print(show_pgp(args.show_pgp, _index(p, with_grams=False)))
        return
    if args.find_pgp or args.grep_pgp:
        print(find_pgp(args.find_pgp or args.grep_pgp, _index(p, with_grams=False), grep=bool(args.grep_pgp)))
        return
    cfg, spans, _, spath = load_local_config(p, args.config, args.spans)
    if args.fill_specs:
        merged = fill_specs(args.fill_specs, p.academic_literature, _index(p, with_grams=False), spans)
        spath.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        errs = [(it["slug"], e) for it in merged["items"] for e in it.get("errors") or []]
        print(f"spans file: {len(merged['items'])} items, {sum(len(it['docs']) for it in merged['items'])} docs, "
              f"{len(errs)} errors")
        for e in errs:
            print("  ERROR", e)
        return
    if args.candidates:
        from src.finetuning.text27b.holdouts import Holdouts
        items = load_items(p.academic_literature, list(cfg.get("dirs") or []), cfg.get("skip_items") or {})
        recs = candidates(items, _index(p, with_grams=False), Holdouts.load(p=p))
        args.candidates.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(recs)} candidate items → {args.candidates}")
        return
    if args.check_specs:
        ix = _index(p)
        spans = fill_specs(args.check_specs, p.academic_literature, ix)
        for it in spans["items"]:
            for e in it.get("errors") or []:
                print("SPEC ERROR", it["slug"], e)
        res = build(p, cfg, spans, ix=ix, independence=not args.no_independence)
    else:
        res = build(p, cfg, spans, independence=not args.no_independence)
    if args.check or args.dry_run or args.check_specs:
        for d in res["docs"]:
            t = d.get("translate") or {}
            pc = t.get("pair_check") or {}
            ind = t.get("independence") or {}
            print(f"{d['doc_id'][:70]:70s} {d['split']:11s} {d['family']:12s} rows={t.get('rows', 0):<2} "
                  f"drop={t.get('drop', '-')} ratio={pc.get('length_ratio')} own={pc.get('own')} "
                  f"best_other={pc.get('best_other')} overlap={pc.get('overlap')} "
                  f"indep={ind.get('containment')}/{ind.get('ratio')} pgp={','.join(d['pgpids'][:3])}")
        for s in res["report"]["span_docs"]:
            if s.get("drop"):
                print("SPAN", s["item"], s.get("doc"), s["drop"], json.dumps(s.get("span_failures"), ensure_ascii=False)[:200])
        print(json.dumps(res["report"]["drops"], indent=1))
        return
    man = write_outputs(res, p, args.out_dir, build_args={k: (str(v) if isinstance(v, Path) else v)
                                                         for k, v in vars(args).items()})
    print(json.dumps({"rows": man["checks"]["rows"], "pairs": man["pairs_by_lang_split"],
                      "translators": dict(list(man["translators"].items())[:10]), "checks": man["checks"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
