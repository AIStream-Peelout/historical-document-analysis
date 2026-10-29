"""LLM-free ingest of web-fetched literature into the bibliography pipeline layout.

A web fetch leaves, per source directory under ``academic_literature/``, an ``items.jsonl`` (one record per
post / article / PDF), a ``fetch_log.jsonl`` (one record per HTTP request), the fetched bytes in ``raw/`` and
extracted text in ``text/``. This module turns every usable item into one *work* laid out like the PDF works the
pipeline already indexes, so the stock discovery (``index_all_bibliography.discover_indexing_tasks``), the
stage ledger (``academic_pipeline_status``) and ``BibliographyDocument.from_structured_json`` read it unchanged::

    <source_dir>/<slug>/<slug>.<pdf|html|txt>         copy of the fetched file (original bytes, never moved)
    <source_dir>/<slug>/<slug>.txt                    copy of the fetcher's text when pages were built from it
    <source_dir>/<slug>/<slug>_metadata.json          title, authors, publication, source (licence, fetch record)
    <source_dir>/<slug>/<slug>/<slug>_ocr_results.json             PDF works only: the text layer, per PDF page
    <source_dir>/<slug>/<slug>/<slug>_structured_webtext/page_NNN_structured.json

Pages: one per PDF page for PDFs (text layer; the fetcher's form-feed-separated text when it aligns with the PDF
pages, which carries its legacy-font remap; words re-joined for one-word-per-line layers), and ~3,000-character
chunks cut on paragraph boundaries for HTML / plain-text items. ``full_main_text`` is a verbatim slice; ``summary``
is empty; shelfmarks are located with the text27b helpers (``textnorm.find_shelfmarks`` candidates plus the
fetcher's item-level list, kept only when ``shelfmarks.locate`` finds them on the page) and canonicalised with
``shelfmarks.match_shelfmark`` where it resolves to one merged id. No model of any kind is called.

Per-source settings (publication type, licence string from the fetch report, OCR reliability per script, per-item
overrides) live in ``academic_literature/web_items_sources.json`` next to the data, so the corpus folder alone is
enough to rebuild every work dir.

Dedup (read-only): every item is compared with the local works (titles, DOIs, sampled 8-word shingles of their
page text) and, with ``--es-index``, with a scan of that Elasticsearch index (book_uuids, source urls, DOIs,
titles, shingles). Only items judged ``new`` are materialised; ``--allow-duplicate`` overrides one slug.

Usage::

    python -m src.datasets.indexing.bibliography.web_items_ingest --dry-run --es-index <index> \\
        --manifest-dir <dir>
    python -m src.datasets.indexing.bibliography.web_items_ingest --es-index <index> --manifest-dir <dir>
    python -m src.datasets.indexing.bibliography.web_items_ingest --source <source_dir> --rebuild

Elasticsearch credentials are read from the repo ``.env`` inside this process and never printed.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import logging
import re
import shutil
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import fitz
from bs4 import BeautifulSoup

from src.datasets.indexing.bibliography.academic_pipeline_status import DEFAULT_ROOT

LOG = logging.getLogger(__name__)

ROUTE = "webtext_llm_free"
INGEST_VERSION = 1
VARIANT_SUFFIX = "_structured_webtext"
CONFIG_NAME = "web_items_sources.json"
#: Target page size for HTML / plain-text items (characters).
CHUNK_CHARS = 3000
#: Pages with less stripped text than this are blank (plates, separators); the ledger expects no page for them.
MIN_PAGE_CHARS = 50
#: Item kinds that are works. ``landing_page_only`` items become works once their PDF is found in ``raw/``.
ELIGIBLE_KINDS = {"item", "post"}
LANDING_KIND = "landing_page_only"
SHINGLE_N = 8
SHINGLE_MOD = 16
#: Share of an item's sampled shingles found in one other work that makes it a duplicate of that work.
CONTAINMENT_DUP = 0.5
#: Fewer sampled shingles than this: containment is not used for the verdict.
MIN_SHINGLES = 8
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_LINE = re.compile(r"^https?://\S+$")
HEB_LETTER = re.compile(r"[א-ת]")
YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
WP_PUBLISHED = re.compile(r"Post published:\s*[A-Z][a-z]+\s+\d{1,2},\s*(\d{4})")
#: Fonts whose PDF text layer encodes Hebrew through a pre-Unicode Macintosh code page.
HEBREW_FONT_RE = re.compile(r"(?i)narkis|frank-?ruehl|david|guttman|hadassah|miriam|aharoni|koren|drugulin|vilna|keter")
#: Whole pages that are a distributor's cover sheet, not the work (``not_this_work`` keeps them out of the index).
BOILERPLATE_PAGES = {"Early Journal Content on JSTOR, Free to Anyone in the World": "jstor_early_journal_content_cover"}
LANGUAGE_NAMES = {"en": "English", "he": "Hebrew", "fr": "French", "es": "Spanish", "de": "German", "ar": "Arabic"}


# ============================================================================== configuration
@dataclass
class SourceConfig:
    """Settings of one fetched source directory.

    :param name: Directory name under ``academic_literature``.
    :param publication_type: Publication type written to the metadata (``journal article`` ...).
    :param licence_report: Licence / terms string as the fetch report states it for the directory.
    :param ocr_reliability: Script -> reliability label, plus ``method``.
    :param item_overrides: Slug -> ``{"ocr_reliability": {...}, "title": ..., "authors": [...], "year": ...}``:
        labels overriding the source default and catalogue fields the fetch record lacks (filled by a person; they
        survive ``--rebuild``).
    :param publisher: Publisher, when one applies to the whole source.
    :param series: Series / journal name, when one applies to the whole source.
    """

    name: str
    publication_type: Optional[str] = None
    licence_report: Optional[str] = None
    ocr_reliability: Dict[str, str] = field(default_factory=dict)
    item_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    publisher: Optional[str] = None
    series: Optional[str] = None


def load_sources_config(root: Path, path: Optional[Path] = None) -> Dict[str, SourceConfig]:
    """Read the per-source settings file.

    :param root: ``academic_literature`` root (default location of the file).
    :param path: Explicit settings file.
    :returns: Source name -> settings, in file order.
    """
    data = json.loads((path or root / CONFIG_NAME).read_text(encoding="utf-8"))
    out: Dict[str, SourceConfig] = {}
    for name, raw in data["sources"].items():
        known = {k: v for k, v in raw.items() if k in SourceConfig.__dataclass_fields__ and k != "name"}
        out[name] = SourceConfig(name=name, **known)
    return out


# ============================================================================== small helpers
def read_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """Read a JSONL file keeping 1-based line numbers.

    :param path: File.
    :returns: ``[(line_no, record), ...]`` (blank lines skipped).
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                out.append((n, json.loads(line)))
    return out


def sha256_file(path: Path) -> str:
    """SHA-256 of a file.

    :param path: File.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strip_fetch_header(text: str) -> str:
    """Drop the ``title / url / blank`` header the fetcher writes at the top of ``text/<slug>.txt``.

    :param text: Fetched text.
    :returns: Text without the header (unchanged when there is none).
    """
    lines = text.split("\n", 3)
    if len(lines) >= 2 and URL_LINE.match(lines[1].strip()):
        rest = lines[2:]
        if rest and not rest[0].strip():
            rest = rest[1:]
        return "\n".join(rest)
    return text


def chunk_text(text: str, target: int = CHUNK_CHARS) -> List[str]:
    """Cut text into ~``target``-character pages on paragraph boundaries (verbatim slices).

    The cut is the last blank-line break between 0.5× and 1.25× ``target`` after the page start, else the last line
    break, else the last space. A tail of up to 1.25× ``target`` stays on the last page.

    :param text: Text.
    :param target: Target page length in characters.
    :returns: Non-empty slices of ``text`` with surrounding whitespace stripped.
    """
    pieces: List[str] = []
    start, n = 0, len(text)
    while start < n:
        if n - start <= int(target * 1.25):
            pieces.append(text[start:])
            break
        lo, hi = start + target // 2, start + int(target * 1.25)
        window = text[lo:hi]
        cut = None
        for pattern in (r"\n[ \t]*\n", r"\n", r" "):
            hits = [m.end() for m in re.finditer(pattern, window)]
            if hits:
                cut = lo + hits[-1]
                break
        cut = cut or hi
        pieces.append(text[start:cut])
        start = cut
    return [p.strip() for p in pieces if p.strip()]


def first_year(*values: Any) -> Optional[str]:
    """First four-digit year (1500-2099) found in the values, in order.

    :param values: Strings / numbers / None.
    :returns: The year as a string, or None.
    """
    for v in values:
        if v is None:
            continue
        m = YEAR_RE.search(str(v))
        if m:
            return m.group(1)
    return None


def title_key(title: Optional[str]) -> str:
    """Normalised title for matching (case-folded, accents dropped, letters and digits only).

    :param title: Title.
    :returns: Space-joined tokens (empty for no title).
    """
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()
    return " ".join(re.findall(r"[a-z0-9א-תؠ-ي]+", t))


def common_title_suffix(titles: Iterable[str]) -> List[str]:
    """Site-name suffixes (`` – Site``) each shared by at least half (and at least two) of a source's titles.

    :param titles: Raw titles of one source.
    :returns: Suffixes including their separator, longest first (empty when none).
    """
    titles = [t for t in titles if t]
    counts: collections.Counter = collections.Counter()
    for t in titles:
        counts.update({t[m.start():] for m in re.finditer(r"\s+[-–|]\s+", t)})
    need = max(2, len(titles) / 2)
    return sorted((s for s, n in counts.items() if n >= need), key=len, reverse=True)


def html_title(path: Path) -> Optional[str]:
    """``og:title`` (else ``<title>``) of a saved HTML page, parsed as data.

    :param path: HTML file.
    :returns: Title text or None.
    """
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content", "").strip():
        return og["content"].strip()
    t = soup.find("title")
    return t.get_text(" ", strip=True) if t and t.get_text(strip=True) else None


# ============================================================================== PDF text
def _remap_mac_hebrew(text: str) -> str:
    """Map characters of a legacy Mac Hebrew font back to Hebrew (Mac Roman bytes read as Windows-1255).

    :param text: Span text in the legacy encoding.
    :returns: Remapped text (characters outside Mac Roman are kept).
    """
    out = []
    for ch in text:
        try:
            out.append(ch.encode("mac_roman").decode("cp1255"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            out.append(ch)
    return "".join(out)


def _unreverse_hebrew_tokens(text: str) -> str:
    """Reverse every whitespace token that holds a Hebrew letter (legacy layers store words right-to-left).

    :param text: Text.
    :returns: Text with Hebrew tokens in logical order (line order is left as stored).
    """
    return re.sub(r"\S+", lambda m: m.group(0)[::-1] if HEB_LETTER.search(m.group(0)) else m.group(0), text)


def legacy_page_text(page: "fitz.Page") -> str:
    """Text of one PDF page whose Hebrew is set in a legacy (pre-Unicode) font.

    Spans in a :data:`HEBREW_FONT_RE` font are remapped; Latin spans are left as they are.

    :param page: PyMuPDF page.
    :returns: Page text.
    """
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            lines.append("".join(_remap_mac_hebrew(s["text"]) if HEBREW_FONT_RE.search(s["font"]) else s["text"]
                                 for s in line["spans"]))
    return _unreverse_hebrew_tokens("\n".join(lines) + ("\n" if lines else ""))


def has_legacy_hebrew_font(path: Path, min_chars: int = 50) -> bool:
    """Whether a PDF sets Hebrew in a legacy font (Hebrew-font spans with no Hebrew code points).

    :param path: PDF.
    :param min_chars: Least non-space characters in Hebrew-font spans (all pages) for a positive answer.
    :returns: True when the Hebrew-font spans hold at least ``min_chars`` characters and under 5 % Hebrew letters.
    """
    chars: List[str] = []
    with fitz.open(path) as doc:
        if not any(HEBREW_FONT_RE.search(str(f[3])) for i in range(doc.page_count) for f in doc.get_page_fonts(i)):
            return False
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    chars += [s["text"] for s in line["spans"] if HEBREW_FONT_RE.search(s["font"])]
    text = re.sub(r"\s", "", "".join(chars))
    return len(text) >= min_chars and len(HEB_LETTER.findall(text)) < 0.05 * len(text)


def pdf_page_texts(path: Path, mode: str = "verbatim") -> List[str]:
    """Text of every page of a PDF.

    :param path: PDF.
    :param mode: ``verbatim`` (text layer as stored), ``words_joined`` (whitespace runs collapsed to one space, for
        layers with one word per line) or ``legacy_remap`` (:func:`legacy_page_text`).
    :returns: One string per page.
    """
    with fitz.open(path) as doc:
        if mode == "legacy_remap":
            return [legacy_page_text(p) for p in doc]
        texts = [p.get_text() for p in doc]
    if mode == "words_joined":
        return [" ".join(t.split()) for t in texts]
    return texts


def is_one_word_per_line(texts: Sequence[str]) -> bool:
    """Whether a text layer stores about one word per line.

    :param texts: Page texts.
    :returns: True when there are at least 200 lines averaging under 8 characters.
    """
    lines = [ln for t in texts for ln in t.split("\n")]
    return len(lines) >= 200 and sum(len(ln) for ln in lines) / len(lines) < 8


# ============================================================================== OCR-garble flags
_SYMBOLS = re.compile(r"[\^~|<>{}\\_*=+@#$%]")


def garbled_line_spans(text: str, skip: Sequence[Tuple[int, int]] = ()) -> List[List[int]]:
    """Character spans of lines that look like OCR of a non-Latin script read as Latin (flag, never drop).

    A line is garbled when fewer than half of its tokens are plain words (letters with a vowel or at most two
    letters, numbers) or it holds OCR symbol noise; lines overlapping ``skip`` (e.g. shelfmarks) are kept.
    Adjacent garbled lines merge into one span.

    :param text: Page text.
    :param skip: Spans never flagged.
    :returns: ``[[start, end], ...]``.
    """
    spans: List[List[int]] = []
    pos = 0
    for line in text.split("\n"):
        a, b = pos, pos + len(line)
        pos = b + 1
        toks = line.split()
        if not toks or any(s < b and a < e for s, e in skip):
            continue
        good = 0
        for t in toks:
            core = re.sub(r"['’\-]|'s$", "", t.strip(".,;:!?'\"()[]—–"))
            if core.isdigit() or (core.isalpha() and core.isascii()
                                  and (re.search(r"[aeiouyAEIOUY]", core) or len(core) <= 2)
                                  and (core.islower() or core.istitle() or core.isupper())):
                good += 1
        if good / len(toks) < 0.5 or len(_SYMBOLS.findall(line)) >= 2:
            if spans and a - spans[-1][1] <= 1:
                spans[-1][1] = b
            else:
                spans.append([a, b])
    return spans


# ============================================================================== shelfmarks
class ShelfmarkResolver:
    """Locate shelfmarks on a page and canonicalise them against the merged corpus (text27b helpers)."""

    def __init__(self, corpus: Any = None) -> None:
        """Wrap a loaded corpus.

        :param corpus: ``shelfmarks.Corpus`` or None (no canonicalisation: normalised raw strings are kept).
        """
        from src.finetuning.text27b import shelfmarks as sm
        from src.finetuning.text27b.textnorm import find_shelfmarks

        self.sm = sm
        self.find = find_shelfmarks
        self.corpus = corpus
        self._cache: Dict[str, Dict[str, Any]] = {}

    def resolve(self, raw: str) -> Dict[str, Any]:
        """Match one raw shelfmark.

        :param raw: Raw string.
        :returns: ``{"norm", "key", "how", "canonical_id", "pgpids"}``; ``canonical_id`` only for a unique match.
        """
        if raw not in self._cache:
            norm = self.sm.prep_shelfmark(raw)
            out = {"norm": norm, "key": self.sm.key(raw), "how": "no_corpus", "canonical_id": None, "pgpids": []}
            if self.corpus is not None:
                m = self.sm.match_shelfmark(raw, self.corpus)
                out.update(how=m["how"], pgpids=m["pgpids"][:10],
                           canonical_id=m["cids"][0] if len(m["cids"]) == 1 else None)
            self._cache[raw] = out
        return self._cache[raw]

    def page_marks(self, text: str, item_candidates: Sequence[str] = ()) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Shelfmarks located on a page.

        :param text: Page text.
        :param item_candidates: The fetcher's item-level shelfmark list (kept only where located on this page).
        :returns: ``(shelf_marks_mentioned, detail)``: the mention list (canonical id when resolved, else the
            normalised string) and one detail record per mark (raw, offsets, key, match).
        """
        toks = self.sm.tokens_with_pos(text)
        toks2 = self.sm.strip_paren_tokens(toks, text)
        found: Dict[str, Dict[str, Any]] = {}
        cands = [text[a:b] for a, b in self.find(text)] + [str(c) for c in item_candidates if c]
        for raw in cands:
            raw = " ".join(raw.split())
            k = self.sm.key(raw)
            if not k or k in found:
                continue
            offsets = self.sm.locate(raw, text, toks, toks2)
            if not offsets:
                continue
            found[k] = {"raw": raw, "offsets": offsets[:20], **self.resolve(raw)}
        detail = sorted(found.values(), key=lambda d: d["offsets"][0])
        mentioned: List[str] = []
        for d in detail:
            label = d["canonical_id"] or d["norm"]
            if label not in mentioned:
                mentioned.append(label)
        return mentioned, detail


# ============================================================================== planning
@dataclass
class PageText:
    """One page of a work.

    :param seq: File sequence number (``page_NNN``): the PDF page for PDFs, the chunk index otherwise.
    :param text: Verbatim page text.
    :param pdf_page: PDF page number, when the page is a PDF page.
    :param not_this_work: Reason the page is not part of the work (kept out of the index), or None.
    """

    seq: int
    text: str
    pdf_page: Optional[int] = None
    not_this_work: Optional[str] = None


@dataclass
class WorkPlan:
    """Everything needed to materialise (or skip) one item.

    :param source: Source directory name.
    :param slug: Item slug (= work dir name = book name).
    :param item: The ``items.jsonl`` record.
    :param items_line: Its line number in ``items.jsonl``.
    :param fetch_rows: ``(line_no, record)`` of the matching ``fetch_log.jsonl`` rows.
    :param skip_reason: Why the item is not a work (None for works).
    :param primary: Fetched file the work is built from (copied as ``<slug>.<ext>``).
    :param extras: Other fetched files of the item (copied under their own names).
    :param text_file: Fetcher's text file the pages were cut from (copied as ``<slug>.txt``), if used.
    :param page_source: How pages were built.
    :param pdf_pages: Page count of the primary PDF (None for non-PDF works).
    :param pdf_page_texts: Text of every PDF page (for the OCR-results file).
    :param pages: The pages.
    :param metadata: Work metadata (``<slug>_metadata.json``).
    :param ocr_reliability: Reliability labels for this item.
    :param flags: Notes for the manifest.
    """

    source: str
    slug: str
    item: Dict[str, Any]
    items_line: int
    fetch_rows: List[Tuple[int, Dict[str, Any]]] = field(default_factory=list)
    skip_reason: Optional[str] = None
    primary: Optional[Path] = None
    extras: List[Path] = field(default_factory=list)
    text_file: Optional[Path] = None
    page_source: Optional[str] = None
    pdf_pages: Optional[int] = None
    pdf_page_texts: List[str] = field(default_factory=list)
    pages: List[PageText] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ocr_reliability: Dict[str, str] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All page text of the work (pages joined by blank lines).

        :returns: Text.
        """
        return "\n\n".join(p.text for p in self.pages if not p.not_this_work)


def item_raw_files(item: Dict[str, Any]) -> List[str]:
    """Relative paths of the fetched files of an item (``files.raw`` plus ``files.html`` / ``files.pdf``).

    :param item: Item record.
    :returns: Paths relative to the source dir, in record order, without duplicates.
    """
    files = item.get("files") or {}
    out: List[str] = []
    for key in ("raw", "html", "pdf"):
        vals = files.get(key)
        for v in ([vals] if isinstance(vals, str) else vals or []):
            if isinstance(v, str) and v.startswith("raw/") and v not in out:
                out.append(v)
    return out


def fetch_rows_for(item: Dict[str, Any], rows: Sequence[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
    """Fetch-log rows that belong to an item (same slug, a ``<slug>__NN`` child, or one of its files).

    :param item: Item record.
    :param rows: All ``(line_no, record)`` of the source's ``fetch_log.jsonl``.
    :returns: Matching rows in log order.
    """
    slug = item.get("slug") or ""
    raws = set(item_raw_files(item))
    out = []
    for n, r in rows:
        files = r.get("files") if isinstance(r.get("files"), dict) else {}
        if r.get("slug") == slug or str(r.get("slug", "")).startswith(slug + "__") or r.get("file") in raws \
                or raws & {v for v in files.values() if isinstance(v, str)}:
            out.append((n, r))
    return out


def find_item_pdf(item: Dict[str, Any], source_dir: Path, rows: Sequence[Tuple[int, Dict[str, Any]]]) -> Optional[Path]:
    """The item's PDF in ``raw/``: listed by the fetcher, or dropped in later under a predictable name.

    Names tried: the item's listed raw PDFs; ``<slug>.pdf``; ``<slug>__NN.pdf``; the file name of any PDF URL of
    the item (``children_failed`` / fetch-log ``url``), i.e. what a browser download of a blocked PDF is called.

    :param item: Item record.
    :param source_dir: Source directory.
    :param rows: The item's fetch-log rows.
    :returns: The PDF path or None.
    """
    raw_dir = source_dir / "raw"
    slug = item.get("slug") or ""
    names = [Path(r).name for r in item_raw_files(item) if r.lower().endswith(".pdf")]
    names.append(f"{slug}.pdf")
    names += sorted(p.name for p in raw_dir.glob(f"{glob_escape(slug)}__*.pdf"))
    urls = list(item.get("children_failed") or []) + [r.get("url", "") for _, r in rows]
    names += [Path(urllib.parse.unquote(urllib.parse.urlparse(u).path)).name for u in urls if u.lower().endswith(".pdf")]
    present = {p.name.lower(): p for p in raw_dir.glob("*")} if raw_dir.is_dir() else {}
    for name in names:
        if name and name.lower() in present:
            return present[name.lower()]
    return None


def glob_escape(text: str) -> str:
    """Escape glob metacharacters.

    :param text: Literal text.
    :returns: Escaped pattern fragment.
    """
    return re.sub(r"([*?\[\]])", r"[\1]", text)


def build_pages(plan: WorkPlan, source_dir: Path, pdf: Optional[Path]) -> None:
    """Fill ``plan.pages`` / ``page_source`` / ``text_file`` / PDF fields.

    :param plan: Plan with item set.
    :param source_dir: Source directory.
    :param pdf: The item's PDF, if any.
    """
    item = plan.item
    text_rel = (item.get("files") or {}).get("text")
    text_path = source_dir / text_rel if text_rel and (source_dir / text_rel).exists() else None
    from_pdf_fetch = pdf is not None and item.get("kind") in ELIGIBLE_KINDS
    if pdf is not None:
        texts = pdf_page_texts(pdf)
        plan.pdf_pages = len(texts)
        segs = text_path.read_text(encoding="utf-8").split("\f") if text_path and from_pdf_fetch else []
        if len(segs) == len(texts) and len(segs) > 1:
            segs[0] = strip_fetch_header(segs[0])
            texts, plan.page_source, plan.text_file = segs, "fetch_text_formfeed_pages", text_path
        else:
            owpl = bool(item.get("one_word_per_line")) or is_one_word_per_line(texts)
            plan.page_source = "pdf_text_layer"
            if has_legacy_hebrew_font(pdf):
                texts, plan.page_source = pdf_page_texts(pdf, "legacy_remap"), "pdf_legacy_font_remap"
                plan.flags.append("legacy_font_remapped_by_ingest")
            if owpl:
                texts = [" ".join(t.split()) for t in texts]
                plan.page_source += "_words_joined"
        plan.pdf_page_texts = texts
        for i, t in enumerate(texts, 1):
            if len(t.strip()) < MIN_PAGE_CHARS:
                continue
            flat = " ".join(t.split())
            tag = next((v for k, v in BOILERPLATE_PAGES.items() if k in flat), None) if i == 1 else None
            plan.pages.append(PageText(seq=i, text=t.strip("\n"), pdf_page=i, not_this_work=tag))
        return
    if text_path is None:
        plan.skip_reason = "no_text"
        return
    body = strip_fetch_header(text_path.read_text(encoding="utf-8"))
    plan.text_file, plan.page_source = text_path, "fetch_text_chunks"
    plan.pages = [PageText(seq=i, text=t) for i, t in enumerate(chunk_text(body), 1) if len(t.strip()) >= MIN_PAGE_CHARS]


def item_title(candidates: Sequence[Tuple[Optional[str], Sequence[str]]]) -> Optional[str]:
    """First non-empty title among ``(title, site suffixes)`` candidates, with a matching suffix dropped.

    :param candidates: In priority order, e.g. the landing page's title for a PDF fetched via a landing page, then
        the item title; each with its source-wide site suffixes (see :func:`common_title_suffix`).
    :returns: Title or None.
    """
    for title, suffixes in candidates:
        t = (title or "").strip()
        suffix = next((x for x in suffixes if t.endswith(x)), None)
        if suffix:
            t = t[: -len(suffix)].strip()
        if t:
            return t
    return None


def item_authors(item: Dict[str, Any]) -> Optional[List[str]]:
    """Authors as stored by the fetcher (a list; a string is split on ``;`` only).

    :param item: Item record.
    :returns: Author list or None.
    """
    a = item.get("authors")
    if isinstance(a, list):
        return [str(x).strip() for x in a if str(x).strip()] or None
    if isinstance(a, str) and a.strip():
        return [x.strip() for x in a.split(";") if x.strip()]
    return None


def stable_id(item: Dict[str, Any]) -> Optional[str]:
    """A stable identifier: DOI, archive.org identifier, else the source URL.

    :param item: Item record.
    :returns: Identifier string or None.
    """
    if item.get("doi"):
        return f"doi:{item['doi']}"
    url = item.get("source_url") or ""
    m = re.match(r"https?://archive\.org/details/([^/?#]+)", url)
    if m:
        return f"archive.org:{m.group(1)}"
    return url or None


def build_metadata(plan: WorkPlan, cfg: SourceConfig, files: Dict[str, str], title: Optional[str],
                   created: str) -> Dict[str, Any]:
    """Work metadata in the shape of the existing ``*_metadata.json`` files (unknown fields null).

    :param plan: Plan with pages built.
    :param cfg: Source settings.
    :param files: In-dir file name -> sha256, primary first.
    :param title: Work title.
    :param created: Ingest date (ISO).
    :returns: Metadata dict.
    """
    it = plan.item
    rows = plan.fetch_rows
    primary_name = plan.primary.name if plan.primary else None
    primary_rows = [(n, r) for n, r in rows if primary_name and (Path(str(r.get("file") or "")).name == primary_name
                    or primary_name in {Path(v).name for v in (r.get("files") or {}).values() if isinstance(v, str)})]
    main_row = (primary_rows or rows or [(None, {})])[0][1]
    lang = LANGUAGE_NAMES.get(str(it.get("language") or ""), it.get("language"))
    year = first_year(it.get("year"), it.get("date"), it.get("citation"), it.get("journal"), it.get("venue"))
    if year is None and plan.page_source == "fetch_text_chunks":
        m = WP_PUBLISHED.search(plan.pages[0].text if plan.pages else "")
        year = m.group(1) if m else None
    manual = plan.primary is not None and not primary_rows and it.get("kind") == LANDING_KIND
    review = it.get("review") if isinstance(it.get("review"), dict) else {}
    over = {k: v for k, v in cfg.item_overrides.get(plan.slug, {}).items() if k in ("title", "authors", "year")}
    title = over.get("title", title)
    year = over.get("year", year)
    meta = {
        "title": title,
        "authors": over.get("authors", item_authors(it)),
        "summary": None,
        "doi": it.get("doi"),
        "publication": {
            "type": cfg.publication_type,
            "journal": it.get("journal") or it.get("venue") or cfg.series,
            "volume": it.get("volume"),
            "issue": it.get("issue"),
            "year": year,
            "date": it.get("date") or (f"{it['month']} {it['year']}" if it.get("month") and it.get("year") else None),
            "citation": it.get("citation"),
            "language": lang,
            "publisher": cfg.publisher,
        },
        "subjects": None,
        "source": {
            "provenance": ("user-supplied PDF placed in raw/ for a landing-page-only item of the web fetch"
                           if manual else f"open-access web fetch {it.get('fetched') or ''}".strip()),
            "source_dir": plan.source,
            "url": it.get("source_url"),
            "final_url": main_row.get("final_url") or main_row.get("url"),
            "stable_id": stable_id(it),
            "licence": it.get("licence") or it.get("licence_terms") or cfg.licence_report,
            "licence_terms": it.get("licence_terms"),
            "licence_report": cfg.licence_report,
            "fetched_at": main_row.get("fetched") or it.get("fetched"),
            "fetch_record": {
                "items_jsonl": f"{plan.source}/items.jsonl",
                "items_line": plan.items_line,
                "fetch_log": f"{plan.source}/fetch_log.jsonl",
                "fetch_log_lines": [n for n, _ in rows],
                "url": main_row.get("url") or it.get("source_url"),
                "fetched": main_row.get("fetched") or it.get("fetched"),
                "status": main_row.get("status"),
                "raw_files": [f"{plan.source}/{p.relative_to(plan.primary.parents[1]).as_posix()}"
                              for p in ([plan.primary] if plan.primary else []) + plan.extras] if plan.primary else [],
                "text_file": f"{plan.source}/{plan.text_file.relative_to(plan.text_file.parents[1]).as_posix()}"
                if plan.text_file else None,
                "sha256": files,
                "expected_pdf_urls": it.get("children_failed") if manual else None,
            },
            "fetch_review": {"genre": review.get("genre"), "reviewer": review.get("reviewer")} if review else None,
        },
        "files": list(files),
        "sha256": next(iter(files.values()), None),
        "ingest": {
            "route": ROUTE,
            "version": INGEST_VERSION,
            "page_source": plan.page_source,
            "pages": len(plan.pages),
            "chars": sum(len(p.text) for p in plan.pages),
            "pdf_pages": plan.pdf_pages,
            "ocr_reliability": plan.ocr_reliability,
            "overrides": sorted(over) or None,
            "created": created,
        },
    }
    return meta


def plan_source(source_dir: Path, cfg: SourceConfig, limit: Optional[int] = None) -> List[WorkPlan]:
    """Plan every item of one source directory (no files written).

    :param source_dir: Source directory.
    :param cfg: Its settings.
    :param limit: Plan only the first N works (skipped items do not count).
    :returns: Plans in ``items.jsonl`` order.
    """
    items = read_jsonl(source_dir / "items.jsonl")
    rows = read_jsonl(source_dir / "fetch_log.jsonl")
    suffix = common_title_suffix([it.get("title") or "" for _, it in items])
    landing_titles: Dict[str, Optional[str]] = {}
    for _, it in items:
        htmls = [r for r in item_raw_files(it) if r.lower().endswith((".html", ".htm"))]
        if htmls and (source_dir / htmls[0]).exists():
            landing_titles[str(it.get("slug"))] = html_title(source_dir / htmls[0])
    landing_suffix = common_title_suffix([t or "" for t in landing_titles.values()])
    plans: List[WorkPlan] = []
    works = 0
    for line, it in items:
        slug = str(it.get("slug") or "")
        plan = WorkPlan(source=cfg.name, slug=slug, item=it, items_line=line, fetch_rows=fetch_rows_for(it, rows))
        plans.append(plan)
        kind = it.get("kind")
        pdf = find_item_pdf(it, source_dir, plan.fetch_rows) if kind in ELIGIBLE_KINDS | {LANDING_KIND} else None
        if not SLUG_RE.match(slug):
            plan.skip_reason = "unsafe_slug"
        elif it.get("duplicate_of"):
            plan.skip_reason = f"fetch_duplicate_of:{it['duplicate_of']}"
        elif kind == LANDING_KIND and pdf is None:
            plan.skip_reason = "landing_page_only_awaiting_pdf"
        elif kind not in ELIGIBLE_KINDS | {LANDING_KIND}:
            plan.skip_reason = f"kind:{kind}"
        if plan.skip_reason or (limit is not None and works >= limit):
            plan.skip_reason = plan.skip_reason or "limit"
            continue
        works += 1
        raws = [source_dir / r for r in item_raw_files(it) if (source_dir / r).exists()]
        plan.primary = pdf or next((p for p in raws if p.suffix.lower() in (".html", ".htm")), None) or \
            next(iter(raws), None)
        plan.extras = [p for p in raws if p != plan.primary and p.suffix.lower() != ".pdf"]
        build_pages(plan, source_dir, pdf)
        if not plan.skip_reason and not plan.pages:
            plan.skip_reason = "no_text_pages"
        rel = dict(cfg.ocr_reliability)
        if (plan.page_source or "").endswith("_words_joined"):
            rel["method"] = "pdf_text_layer_words_joined"
        if it.get("legacy_hebrew_font") or (plan.page_source or "").startswith("pdf_legacy_font_remap"):
            rel["hebrew"] = "legacy_font_remapped_line_order_unverified"
        rel.update(cfg.item_overrides.get(slug, {}).get("ocr_reliability", {}))
        plan.ocr_reliability = rel
        if isinstance(it.get("review"), dict) and it["review"].get("genre") == "not_genizah":
            plan.flags.append("fetch_review:not_genizah")
        landing = landing_titles.get(slug) if pdf is not None else None
        plan.metadata = {"title": item_title([(landing, landing_suffix), (it.get("title"), suffix)])}
    return plans


# ============================================================================== dedup
def word_shingles(text: str, n: int = SHINGLE_N, mod: int = SHINGLE_MOD) -> Set[int]:
    """Sampled hashes of word n-grams (letters/digits, case-folded); a hash is kept when ``hash % mod == 0``.

    :param text: Text.
    :param n: Words per shingle.
    :param mod: Sampling modulus.
    :returns: Set of kept hashes.
    """
    words = re.findall(r"[^\W_]+", text.casefold())
    out: Set[int] = set()
    for i in range(len(words) - n + 1):
        h = int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "big")
        if h % mod == 0:
            out.add(h)
    return out


class Deduper:
    """Lookup of already-held works by title, URL, DOI, book_uuid and sampled text shingles."""

    def __init__(self, label: str) -> None:
        """Create an empty lookup.

        :param label: ``local`` or ``index`` (written into match records).
        """
        self.label = label
        self.titles: Dict[str, Set[str]] = collections.defaultdict(set)
        self.urls: Dict[str, Set[str]] = collections.defaultdict(set)
        self.dois: Dict[str, Set[str]] = collections.defaultdict(set)
        self.book_uuids: Dict[str, str] = {}
        self.shingles: Dict[int, Set[str]] = collections.defaultdict(set)
        self.names: Dict[str, str] = {}
        self.refs: Set[str] = set()
        #: ref -> ``pdf_name`` values of its pages (index works: the stem of each ``doc_id``).
        self.pdf_names: Dict[str, Set[str]] = collections.defaultdict(set)

    def add(self, ref: str, title: Optional[str] = None, text: str = "", urls: Iterable[str] = (),
            doi: Optional[str] = None, book_uuid: Optional[str] = None) -> None:
        """Register one work.

        :param ref: Work reference (book dir or book_uuid).
        :param title: Title.
        :param text: Text (shingled).
        :param urls: Source URLs.
        :param doi: DOI.
        :param book_uuid: book_uuid.
        """
        self.refs.add(ref)
        if title and title_key(title):
            self.titles[title_key(title)].add(ref)
            self.names.setdefault(ref, title)
        for u in urls:
            if u:
                self.urls[u.rstrip("/")].add(ref)
        if doi:
            self.dois[doi.lower()].add(ref)
        if book_uuid:
            self.book_uuids[book_uuid] = ref
        for h in word_shingles(text):
            self.shingles[h].add(ref)

    def match(self, title: Optional[str], text: str, urls: Iterable[str] = (), doi: Optional[str] = None,
              book_uuid: Optional[str] = None, exclude: Iterable[str] = ()) -> Dict[str, Any]:
        """Compare one item with the registered works.

        :param title: Item title.
        :param text: Item text.
        :param urls: Item URLs.
        :param doi: Item DOI.
        :param book_uuid: Item's prospective book_uuid.
        :param exclude: Refs to ignore (the item's own work).
        :returns: ``{"book_uuid", "url", "doi", "title", "containment", "best", "n_shingles"}``.
        """
        ex = set(exclude)
        sh = word_shingles(text)
        hits: collections.Counter = collections.Counter()
        for h in sh:
            for ref in self.shingles.get(h, ()):
                if ref not in ex:
                    hits[ref] += 1
        best = hits.most_common(3)
        return {
            "book_uuid": self.book_uuids.get(book_uuid) if book_uuid and self.book_uuids.get(book_uuid) not in ex else None,
            "url": sorted({r for u in urls if u for r in self.urls.get(u.rstrip("/"), ()) if r not in ex}),
            "doi": sorted(r for r in self.dois.get((doi or "").lower(), ()) if r not in ex) if doi else [],
            "title": sorted(r for r in self.titles.get(title_key(title), ()) if r not in ex) if title_key(title) else [],
            "containment": [(ref, round(n / len(sh), 3)) for ref, n in best] if sh else [],
            "n_shingles": len(sh),
        }


def verdict_from(local: Dict[str, Any], index: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Dedup verdict from local and index matches.

    ``already_indexed``: same book_uuid, URL or DOI in the index, or ≥ :data:`CONTAINMENT_DUP` of the item's
    shingles in one indexed work, or a same-title indexed work holding ≥ 0.2 of them. ``local_duplicate``: the same
    tests against local (not necessarily indexed) works. A same-title match alone is only a flag.

    :param local: :meth:`Deduper.match` against local works.
    :param index: :meth:`Deduper.match` against the index, or None when not checked.
    :returns: ``(verdict, reasons)``.
    """
    reasons: List[str] = []
    for label, m in (("index", index), ("local", local)):
        if not m:
            continue
        strong = []
        if m.get("book_uuid"):
            strong.append(f"{label}:book_uuid={m['book_uuid']}")
        strong += [f"{label}:url={r}" for r in m["url"]] + [f"{label}:doi={r}" for r in m["doi"]]
        enough = m["n_shingles"] >= MIN_SHINGLES
        for ref, c in m["containment"]:
            if enough and (c >= CONTAINMENT_DUP or (c >= 0.2 and ref in m["title"])):
                strong.append(f"{label}:text={ref}@{c}")
        if strong:
            return ("already_indexed" if label == "index" else "local_duplicate"), strong
        reasons += [f"{label}:same_title={r}" for r in m["title"]]
        reasons += [f"{label}:partial_text={ref}@{c}" for ref, c in m["containment"] if enough and 0.2 <= c < CONTAINMENT_DUP]
    return "new", reasons


def local_works(root: Path, exclude_dirs: Iterable[Path]) -> Deduper:
    """Register every local work outside the web-fetch sources: title (resolved metadata), DOI and page text
    (largest structured variant, else the OCR-results file).

    :param root: ``academic_literature`` root.
    :param exclude_dirs: Directories to leave out (the web-fetch source dirs).
    :returns: Filled :class:`Deduper`.
    """
    from src.datasets.indexing.bibliography.index_all_bibliography import find_book_metadata

    ex = [d.resolve() for d in exclude_dirs]
    dd = Deduper("local")
    variants: Dict[Path, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in root.rglob("page_*_structured.json"):
        if any(e == p or e in p.parents for e in ex) or p.parent.name.endswith(VARIANT_SUFFIX):
            continue
        sd = p.parent
        book = sd.parent if "_structured" in sd.name else sd.parent.parent
        if "_structured" not in sd.name and "_structured" not in sd.parent.name:
            continue
        variants[book][sd] += 1
    books = set(variants)
    for ocr in root.rglob("*_ocr_results.json"):
        if not any(e in ocr.parents for e in ex) and ocr.parent not in books and ocr.name == f"{ocr.parent.name}_ocr_results.json":
            books.add(ocr.parent)
    for book in sorted(books):
        meta_path = find_book_metadata(book, root)
        meta: Dict[str, Any] = {}
        if meta_path is not None:
            meta = json.loads(meta_path.read_text(encoding="utf-8") or "{}")
        else:
            enh = next(iter(sorted(book.glob("*_enhanced.json"))), None)
            meta = json.loads(enh.read_text(encoding="utf-8") or "{}") if enh else {}
        texts: List[str] = []
        if variants.get(book):
            sd = variants[book].most_common(1)[0][0]
            for p in sorted(sd.glob("page_*_structured.json")):
                data = json.loads(p.read_text(encoding="utf-8") or "{}")
                texts.append(str(data.get("full_main_text") or ""))
        else:
            data = json.loads((book / f"{book.name}_ocr_results.json").read_text(encoding="utf-8") or "{}")
            texts = [str((pg.get("ocr_result") or {}).get("full_text") or "") for pg in data.get("pages", [])
                     if isinstance(pg.get("ocr_result"), dict)]
        src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
        dd.add(book.relative_to(root).as_posix(), title=meta.get("title") if isinstance(meta.get("title"), str) else None,
               text="\n".join(texts), urls=[src.get("url") or "", meta.get("source_url") or ""],
               doi=meta.get("doi") if isinstance(meta.get("doi"), str) else None)
    LOG.info("local works registered: %d", len(books))
    return dd


def index_works(index: str, env_file: Path) -> Deduper:
    """Register every work of an Elasticsearch bibliography index (read-only scroll; credentials from ``env_file``).

    :param index: Index name.
    :param env_file: ``.env`` holding ``ELASTIC_*`` (loaded in-process, never printed).
    :returns: Filled :class:`Deduper` keyed by book_uuid.
    """
    import dotenv
    from elasticsearch import Elasticsearch, helpers

    from src.datasets.indexing.elastic_index_genizah import es_config_from_env

    for name in ("elastic_transport", "elasticsearch", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    dotenv.load_dotenv(env_file)
    es = Elasticsearch(**es_config_from_env(), request_timeout=120)
    books: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {"title": None, "texts": [], "urls": set(), "doi": None, "pdf_names": set()})
    n = 0
    for hit in helpers.scan(es, index=index, query={"query": {"match_all": {}}}, size=500,
                            _source=["title", "book_uuid", "doi", "bibliography", "full_text_content", "doc_id"]):
        s = hit["_source"]
        b = books[s.get("book_uuid") or s.get("doc_id") or hit["_id"]]
        b["title"] = b["title"] or s.get("title")
        b["doi"] = b["doi"] or s.get("doi")
        b["texts"].append(str(s.get("full_text_content") or ""))
        if s.get("doc_id"):
            b["pdf_names"].add(re.sub(r"_p[^_]*$", "", str(s["doc_id"])))
        for bib in s.get("bibliography") or []:
            if isinstance(bib, dict) and bib.get("url"):
                b["urls"].add(bib["url"])
        n += 1
    es.close()
    dd = Deduper("index")
    for uuid, b in books.items():
        dd.add(uuid, title=b["title"], text="\n".join(b["texts"]), urls=b["urls"], doi=b["doi"], book_uuid=uuid)
        dd.pdf_names[uuid] |= b["pdf_names"]
    LOG.info("index works registered: %d books from %d pages", len(books), n)
    return dd


# ============================================================================== writing
def page_json(plan: WorkPlan, page: PageText, resolver: ShelfmarkResolver, created: str) -> Dict[str, Any]:
    """Structured-page JSON in the Pass-1 schema plus an ``ingest`` block.

    :param plan: Work plan.
    :param page: The page.
    :param resolver: Shelfmark resolver.
    :param created: Ingest date (ISO).
    :returns: Page dict.
    """
    marks, detail = resolver.page_marks(page.text, plan.item.get("shelfmarks_mentioned") or [])
    rel = plan.ocr_reliability
    unreliable = [k for k in ("hebrew", "arabic") if str(rel.get(k, "")).startswith("ocr_unreliable")]
    marks_at = [(o, o + len(d["raw"])) for d in detail for o in d["offsets"]]
    spans = garbled_line_spans(page.text, marks_at) if unreliable else []
    meta = plan.metadata
    out: Dict[str, Any] = {
        "shelf_marks_mentioned": marks,
        "footnotes": {},
        "transcriptions": {},
        "extracted_page_number": None,
        "summary": "",
        "classification": "",
        "metadata": {
            "page_number": page.seq,
            "pdf_name": plan.slug,
            "source_url": (meta.get("source") or {}).get("url") or "",
            "language": (meta.get("publication") or {}).get("language") or "Unknown",
            "model_used": "none",
            "provider": ROUTE,
            "processed_at": created,
        },
        "full_main_text": page.text,
        "ingest": {
            "route": ROUTE,
            "version": INGEST_VERSION,
            "page_source": plan.page_source,
            "pdf_page": page.pdf_page,
            "chunk_index": None if page.pdf_page else page.seq,
            "ocr_reliability": rel,
            "unreliable_spans": spans,
            "unreliable_spans_reason": f"OCR of {'/'.join(unreliable)} script read as Latin (heuristic line flag)"
            if spans else None,
            "source_url": (meta.get("source") or {}).get("url"),
            "fetched_at": (meta.get("source") or {}).get("fetched_at"),
            "shelf_marks_detail": detail,
        },
    }
    if page.not_this_work:
        out["not_this_work"] = page.not_this_work
    return out


def _copy(src: Path, dst: Path, dry_run: bool) -> str:
    """Copy one file (never move), keeping bytes; an identical existing copy is left alone.

    :param src: Source file.
    :param dst: Destination.
    :param dry_run: Do not write.
    :returns: sha256 of the source.
    """
    digest = sha256_file(src)
    if not dry_run and not (dst.exists() and sha256_file(dst) == digest):
        shutil.copy2(src, dst)
    return digest


def work_files(plan: WorkPlan) -> List[Tuple[Path, str]]:
    """``(source file, in-dir name)`` for every file copied into the work dir, primary first.

    :param plan: Work plan.
    :returns: Pairs.
    """
    pairs = [(plan.primary, f"{plan.slug}{plan.primary.suffix.lower()}")] if plan.primary else []
    if plan.text_file is not None and not (plan.primary and plan.primary.suffix.lower() == ".txt"):
        pairs.append((plan.text_file, f"{plan.slug}.txt"))
    taken = {name for _, name in pairs}
    for p in plan.extras:
        if p.name not in taken:
            pairs.append((p, p.name))
            taken.add(p.name)
    return pairs


def materialise(plan: WorkPlan, root: Path, cfg: SourceConfig, resolver: ShelfmarkResolver, created: str,
                dry_run: bool, rebuild: bool) -> str:
    """Write one work (files, metadata, OCR-results file for PDFs, structured pages).

    :param plan: Planned work (verdict already ``new`` or allowed).
    :param root: ``academic_literature`` root.
    :param cfg: Source settings.
    :param resolver: Shelfmark resolver.
    :param created: Ingest date (ISO).
    :param dry_run: Report only.
    :param rebuild: Rewrite a work that is already materialised with the same sha256.
    :returns: ``unchanged``, ``would_write``, ``written`` or ``refused:<reason>``.
    """
    work = root / plan.source / plan.slug
    book = work / plan.slug
    variant = book / f"{plan.slug}{VARIANT_SUFFIX}"
    meta_path = work / f"{plan.slug}_metadata.json"
    pairs = work_files(plan)
    files = {name: sha256_file(src) for src, name in pairs}
    plan.metadata = build_metadata(plan, cfg, files, plan.metadata.get("title"), created)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (old.get("ingest") or {}).get("route") != ROUTE:
            return "refused:existing_non_webtext_work"
        same = old.get("sha256") == plan.metadata["sha256"] and old.get("source", {}).get("fetch_record", {}).get(
            "sha256") == files and (old.get("ingest") or {}).get("version") == INGEST_VERSION
        if same and not rebuild:
            return "unchanged"
        stale = {p.name for p in variant.glob("page_*_structured.json")} - {f"page_{pg.seq:03d}_structured.json"
                                                                            for pg in plan.pages}
        if stale:
            return f"refused:stale_pages_{len(stale)}"
    if dry_run:
        return "would_write"
    variant.mkdir(parents=True, exist_ok=True)
    for src, name in pairs:
        _copy(src, work / name, dry_run=False)
    if plan.pdf_pages:
        ocr = {"pdf_path": f"{plan.source}/{plan.slug}/{plan.slug}{plan.primary.suffix.lower()}",
               "total_pages": plan.pdf_pages, "processed_pages": plan.pdf_pages, "images_dir": None,
               "pages": [{"page_number": i, "ocr_result": {"full_text": t}, "error": None}
                         for i, t in enumerate(plan.pdf_page_texts, 1)],
               "processing_info": {"mode": "text_only", "route": ROUTE, "page_source": plan.page_source,
                                   "note": "PDF text layer / fetcher text; no OCR model and no LLM", "created": created}}
        (book / f"{plan.slug}_ocr_results.json").write_text(json.dumps(ocr, ensure_ascii=False, indent=1), encoding="utf-8")
    for page in plan.pages:
        (variant / f"page_{page.seq:03d}_structured.json").write_text(
            json.dumps(page_json(plan, page, resolver, created), ensure_ascii=False, indent=1), encoding="utf-8")
    meta_path.write_text(json.dumps(plan.metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    return "written"


def validate_work(root: Path, plan: WorkPlan) -> Tuple[int, List[str]]:
    """Parse every written page the way the indexer does and build its ES document (no embedding, no network).

    :param root: ``academic_literature`` root.
    :param plan: Written work.
    :returns: ``(pages parsed, errors)``.
    """
    from src.datasets.document_models.bibliography_document import BibliographyDocument

    meta = json.loads((root / plan.source / plan.slug / f"{plan.slug}_metadata.json").read_text(encoding="utf-8"))
    variant = root / plan.source / plan.slug / plan.slug / f"{plan.slug}{VARIANT_SUFFIX}"
    errors, n = [], 0
    for p in sorted(variant.glob("page_*_structured.json")):
        try:
            BibliographyDocument.from_structured_json(p, meta).to_elasticsearch_document()
            n += 1
        except (ValueError, KeyError, TypeError) as e:  # surfaced in the manifest, never fatal
            errors.append(f"{p.name}: {type(e).__name__}: {str(e)[:120]}")
    return n, errors


# ============================================================================== manifest
def plan_record(plan: WorkPlan, verdict: str, reasons: List[str], status: str, resolver: ShelfmarkResolver) -> Dict[str, Any]:
    """Manifest row of one item.

    :param plan: Plan.
    :param verdict: Dedup verdict (or the skip reason).
    :param reasons: Dedup reasons.
    :param status: Materialisation status.
    :param resolver: Shelfmark resolver (for the work-level shelfmark list).
    :returns: Row dict.
    """
    md = plan.metadata
    pub = md.get("publication") or {}
    marks: List[str] = []
    for p in plan.pages:
        if p.not_this_work:
            continue
        for m in resolver.page_marks(p.text, plan.item.get("shelfmarks_mentioned") or [])[0]:
            if m not in marks:
                marks.append(m)
    return {
        "source": plan.source, "slug": plan.slug, "title": md.get("title") or plan.item.get("title") or None,
        "authors": md.get("authors") if "authors" in md else item_authors(plan.item),
        "year": pub.get("year") if pub else first_year(plan.item.get("year"), plan.item.get("date")),
        "language": pub.get("language") if pub else LANGUAGE_NAMES.get(str(plan.item.get("language")),
                                                                         plan.item.get("language")),
        "pages": len(plan.pages), "chars": sum(len(p.text) for p in plan.pages), "page_source": plan.page_source,
        "not_this_work_pages": sum(1 for p in plan.pages if p.not_this_work),
        "shelfmarks_n": len(marks), "shelfmarks_first5": marks[:5],
        "licence": ((md.get("source") or {}).get("licence") if md.get("source") else plan.item.get("licence_terms")),
        "ocr_reliability": plan.ocr_reliability, "verdict": verdict, "dedup_reasons": reasons, "status": status,
        "skip_reason": plan.skip_reason, "flags": plan.flags,
    }


def write_manifest(out_dir: Path, rows: List[Dict[str, Any]], header: Dict[str, Any], dryrun_cmd: str) -> None:
    """Write ``manifest.json`` and ``MANIFEST.md``.

    :param out_dir: Output directory.
    :param rows: Manifest rows.
    :param header: Run facts (args, counts, index checked).
    :param dryrun_cmd: Index dry-run command for the materialised works.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_src: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    totals = {s: {"items": len(rs), **collections.Counter(r["verdict"].split(":")[0] for r in rs),
                  "skip_reasons": dict(collections.Counter(r["verdict"].split(":", 1)[1] for r in rs
                                                           if r["verdict"].startswith("skipped:"))),
                  "pages": sum(r["pages"] for r in rs if r["verdict"] == "new"),
                  "indexable_pages": sum(r["pages"] - r["not_this_work_pages"] for r in rs if r["verdict"] == "new"),
                  "chars": sum(r["chars"] for r in rs if r["verdict"] == "new"),
                  "shelfmarks": sum(r["shelfmarks_n"] for r in rs if r["verdict"] == "new")} for s, rs in by_src.items()}
    (out_dir / "manifest.json").write_text(json.dumps({"header": header, "totals": totals, "index_dryrun_command": dryrun_cmd,
                                                       "items": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    def cell(v: Any, n: int = 60) -> str:
        s = "; ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else str(v))
        s = s.replace("|", "/").replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    md = [f"# Web-items ingest manifest ({header['mode']})", "",
          f"Run {header['created']}; route `{ROUTE}` v{INGEST_VERSION}; index checked: {header.get('es_index') or 'none'}; "
          f"local works compared: {header.get('local_works')}.", "",
          "## Totals per source", "",
          "| source | items | new | already_indexed | local_duplicate | skipped | pages (new) | indexable pages | chars (new) "
          "| shelfmarks (new) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, t in totals.items():
        skipped = t.get("skipped", 0)
        md.append(f"| {s} | {t['items']} | {t.get('new', 0)} | {t.get('already_indexed', 0)} | {t.get('local_duplicate', 0)} | "
                  f"{skipped} | {t['pages']} | {t['indexable_pages']} | {t['chars']} | {t['shelfmarks']} |")
    md += ["", "## Index dry-run command (no `--apply`)", "", "```", dryrun_cmd, "```", ""]
    for s, rs in by_src.items():
        md += [f"## {s}", "", "| slug | title | authors | year | lang | pages | chars | shelfmarks (n: first 5) | licence | verdict |",
               "|---|---|---|---|---|---:|---:|---|---|---|"]
        for r in rs:
            verdict = r["verdict"] + (f" ({cell(r['dedup_reasons'], 80)})" if r["dedup_reasons"] else "")
            verdict += f" [{cell(r['flags'], 60)}]" if r["flags"] else ""
            md.append(f"| {r['slug']} | {cell(r['title'])} | {cell(r['authors'], 40)} | {cell(r['year'])} | {cell(r['language'])} | "
                      f"{r['pages']} | {r['chars']} | {r['shelfmarks_n']}: {cell(r['shelfmarks_first5'], 90)} | "
                      f"{cell(r['licence'], 40)} | {verdict} |")
        md.append("")
    (out_dir / "MANIFEST.md").write_text("\n".join(md), encoding="utf-8")


# ============================================================================== driver
def run(root: Path, sources: Dict[str, SourceConfig], only: Sequence[str] = (), limit: Optional[int] = None,
        dry_run: bool = False, rebuild: bool = False, es_index: Optional[str] = None, env_file: Optional[Path] = None,
        allow: Sequence[str] = (), corpus: Any = "load", manifest_dir: Optional[Path] = None,
        check_local: bool = True) -> List[Dict[str, Any]]:
    """Plan, dedup and (unless ``dry_run``) materialise the web items.

    :param root: ``academic_literature`` root.
    :param sources: Source settings (all are processed unless ``only`` is given).
    :param only: Source names to process.
    :param limit: Works per source.
    :param dry_run: Write no work dirs (the manifest is still written).
    :param rebuild: Rewrite works already materialised with the same sha256.
    :param es_index: Index to dedup against (read-only), or None.
    :param env_file: ``.env`` with the ES credentials.
    :param allow: Slugs materialised despite a duplicate verdict.
    :param corpus: ``"load"`` to load the text27b corpus, None for no canonicalisation, or a loaded corpus.
    :param manifest_dir: Where to write the manifest (None: not written).
    :param check_local: Compare with the local works.
    :returns: Manifest rows.
    """
    created = _dt.date.today().isoformat()
    names = [s for s in sources if not only or s in only]
    missing = sorted(set(only) - set(sources))
    if missing:
        raise SystemExit(f"unknown source(s): {missing}")
    if corpus == "load":
        from src.finetuning.text27b.shelfmarks import load_corpus

        corpus = load_corpus()
    resolver = ShelfmarkResolver(corpus)
    plans = [p for s in names for p in plan_source(root / s, sources[s], limit)]
    local = local_works(root, [root / s for s in sources]) if check_local else Deduper("local")
    index = index_works(es_index, env_file or Path(".env")) if es_index else None
    from src.datasets.document_models import corpus_ids

    taken = collections.Counter(Path(r).name for r in local.refs)
    taken.update(p.slug for p in plans if not p.skip_reason)
    for plan in plans:
        if not plan.skip_reason and taken[plan.slug] > 1:
            plan.skip_reason = "slug_collision"
    batch = Deduper("batch")
    rows: List[Dict[str, Any]] = []
    materialised: List[str] = []
    for plan in plans:
        if plan.skip_reason:
            rows.append(plan_record(plan, f"skipped:{plan.skip_reason}", [], "skipped", resolver))
            continue
        text = plan.text
        md_probe = {"doi": plan.item.get("doi")}
        own_uuid = corpus_ids.book_uuid(corpus_ids.book_key(md_probe, plan.slug))
        own_ref = f"{plan.source}/{plan.slug}/{plan.slug}"
        urls = [plan.item.get("source_url") or ""] + [r.get("final_url") or r.get("url") or "" for _, r in plan.fetch_rows]
        lm = local.match(plan.metadata.get("title"), text, urls, plan.item.get("doi"), exclude=[own_ref])
        # an index book with this work's book_uuid whose pages were all indexed under this slug is the work itself
        # (indexed from its own dir, possibly before a crash lost the dir), not a duplicate
        own_written = (root / plan.source / plan.slug / f"{plan.slug}_metadata.json").exists()
        own = [own_uuid] if own_written or (index and index.pdf_names.get(own_uuid) == {plan.slug}) else []
        im = index.match(plan.metadata.get("title"), text, urls, plan.item.get("doi"), own_uuid, exclude=own) \
            if index else None
        verdict, reasons = verdict_from(lm, im)
        bm = batch.match(None, text)
        reasons += [f"batch:text={r}@{c}" for r, c in bm["containment"] if c >= CONTAINMENT_DUP and bm["n_shingles"] >= MIN_SHINGLES]
        batch.add(f"{plan.source}/{plan.slug}", text=text)
        go = verdict == "new" or plan.slug in allow
        status = materialise(plan, root, sources[plan.source], resolver, created, dry_run, rebuild) if go else "not_materialised"
        if not go:
            plan.metadata = build_metadata(plan, sources[plan.source], {n: sha256_file(s) for s, n in work_files(plan)},
                                           plan.metadata.get("title"), created)
        if status in ("written", "unchanged") and not dry_run:
            _, errs = validate_work(root, plan)
            if errs:
                plan.flags.append(f"validation_errors:{len(errs)}")
                LOG.warning("%s: %s", plan.slug, errs[:3])
        if go and status in ("written", "unchanged", "would_write"):
            materialised.append(plan.slug)
        rows.append(plan_record(plan, verdict, reasons, status, resolver))
        LOG.info("%s/%s: %s %s (%d pages)", plan.source, plan.slug, verdict, status, len(plan.pages))
    if manifest_dir is not None:
        cmd = ("PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.bibliography.web_items_ingest --list-works | "
               "PYTHONPATH=. xargs .venv/bin/python logs/lit_batch_2026-10-20/index_batch_into_bib07.py")
        header = {"mode": "dry-run" if dry_run else "write", "created": created, "es_index": es_index,
                  "local_works": len(local.refs), "sources": names, "limit": limit, "rebuild": rebuild,
                  "materialised": len(materialised)}
        write_manifest(manifest_dir, rows, header, cmd)
        (manifest_dir / "materialised_books.txt").write_text("\n".join(materialised) + "\n", encoding="utf-8")
    return rows


def list_works(root: Path, sources: Iterable[str]) -> List[str]:
    """Book names of the works this ingest has materialised (read from the corpus folder only).

    :param root: ``academic_literature`` root.
    :param sources: Source directory names.
    :returns: Work slugs in source / name order.
    """
    out = []
    for s in sources:
        for meta in sorted((root / s).glob("*/*_metadata.json")):
            if meta.name != f"{meta.parent.name}_metadata.json":
                continue
            if (json.loads(meta.read_text(encoding="utf-8")).get("ingest") or {}).get("route") == ROUTE:
                out.append(meta.parent.name)
    return out


def build_parser() -> argparse.ArgumentParser:
    """CLI parser.

    :returns: Parser.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="academic_literature root")
    p.add_argument("--config", type=Path, help=f"settings file (default <root>/{CONFIG_NAME})")
    p.add_argument("--source", action="append", default=[], help="process only this source dir (repeatable)")
    p.add_argument("--limit", type=int, help="works per source")
    p.add_argument("--dry-run", action="store_true", help="write no work dirs (manifest only)")
    p.add_argument("--rebuild", action="store_true", help="rewrite works already materialised with the same sha256")
    p.add_argument("--es-index", help="bibliography index to dedup against (read-only scan)")
    p.add_argument("--env-file", type=Path, default=Path(".env"), help="file with ELASTIC_* (never printed)")
    p.add_argument("--allow-duplicate", action="append", default=[], help="materialise this slug despite a duplicate verdict")
    p.add_argument("--no-corpus", action="store_true", help="do not canonicalise shelfmarks against the merged corpus")
    p.add_argument("--no-local-dedup", action="store_true", help="skip the comparison with local works")
    p.add_argument("--manifest-dir", type=Path, help="write manifest.json / MANIFEST.md here")
    p.add_argument("--list-works", action="store_true",
                   help="print the materialised work names (one per line) and exit; feeds the index dry run")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Arguments.
    :returns: Exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    for name in ("elastic_transport", "elasticsearch", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    sources = load_sources_config(root, args.config)
    if args.list_works:
        print("\n".join(list_works(root, [s for s in sources if not args.source or s in args.source])))
        return 0
    rows = run(root, sources, only=args.source, limit=args.limit, dry_run=args.dry_run, rebuild=args.rebuild,
               es_index=args.es_index, env_file=args.env_file, allow=args.allow_duplicate,
               corpus=None if args.no_corpus else "load", manifest_dir=args.manifest_dir,
               check_local=not args.no_local_dedup)
    counts = collections.Counter(r["verdict"].split(":")[0] for r in rows)
    status = collections.Counter(r["status"] for r in rows)
    print(f"items {len(rows)}: {dict(counts)}; status {dict(status)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
