# File name: pages.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Academic-literature pages: one structuring per page, joined to its OCR text (text27b).

Layout of ``paths.academic_literature`` (no names are hard-coded here)::

    <work>/[<sub dirs>/]<stem>_structured[_<model>][/<model>]/page_NNN_structured.json
    <work>/[<sub dirs>/]<stem>_ocr_results.json      # {"pages": [{"page_number", "ocr_result": {"full_text"}, ...}]}

``work`` is the top-level directory, ``part`` the path between it and the structuring directory
(nested parts such as ``volume_1/0_200/<stem>`` included; for a structuring directory sitting
directly in the work, the stem). The OCR file is found by its stem (``metadata.pdf_name`` of the
page, else the structuring directory's stem) anywhere under the work, preferring the copy closest
to the page; the OCR page is ``metadata.page_number`` (else the number in the file name). This
join covers nested parts the 2026-10-21 critic's path globbing missed.

``full_main_text`` is the structuring model's rendering of the OCR; :attr:`PageRec.ocr_text` is the
OCR file's text, and :attr:`PageRec.ocr_containment` the share of ``full_main_text``'s 5-word
shingles present in it (plan §9.2: L1 targets come from the OCR text, gated on containment).
Model-written fields (``summary``, ``classification``, ``transcriptions``, ``footnotes``,
``shelf_marks_mentioned``) are never training targets.

Public API (other text27b modules import these names):

Iteration
    :class:`PageRec`, :func:`iter_pages`, :func:`page_key`, :func:`parse_page_key`,
    :func:`page_no_of`, :func:`model_priority`, :func:`ocr_index`, :func:`load_ocr_file`,
    :func:`resolve_ocr`, :func:`shingles`, :func:`containment`, :func:`parse_printed_page`,
    :func:`infer_printed_pages`, :func:`default_repair_root`, :func:`apply_ocr_repair` (derived repairs of
    fragmented text layers, written by :mod:`ocr_repair`, replace ``ocr_text`` when present and current).
Page JSON fields (ported from ``analysis_lib``)
    :func:`sm_keys`, :func:`sm_note`, :func:`tr_keys`, :func:`tr_chars`, :func:`footnote_texts`.
Page-text analysis (ported from ``analysis_lib`` for the literature index / span detector)
    :func:`letter_counts` (``analysis_lib.script_counts``: a ``(he, ar, lat)`` tuple),
    :func:`is_rtl_line`, :func:`ja_or_he`, :func:`lang_of`, :class:`Block`, :func:`line_offsets`,
    :func:`headingish`, :func:`segment_blocks`, :func:`footnote_region_start`, :func:`docness`,
    :func:`rtl_run_chars`, :func:`catalogue_like`, :func:`index_like`, :data:`EN_SW`,
    :data:`JA_WORDS`, :data:`HE_WORDS`.

:class:`PageRec` also answers to the ``analysis_lib.Page`` names ``book``, ``idx``, ``text``
(= ``full_main_text``) and ``data`` so the ported span detector can use it unchanged.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import shelfmarks as _sm

LOG = logging.getLogger(__name__)
_PAGE_RE = re.compile(r"page_(\d+)")
_WORD = re.compile(r"[\w֐-׿؀-ۿ]+")
#: Structuring preference (lower index wins); an empty ``full_main_text`` costs 20.
MODEL_PREFERENCE: Tuple[str, ...] = ("gemini_3.5", "gemini_2.5", "qwen3.6", "qwen3_vl_8b", "qwen_model", "gemma")


# ------------------------------------------------------------------------------------ keys
def page_key(work: str, part: str, page_file: str) -> str:
    """Stable page key ``<work>/<part>/<page_file>`` (the ``(book, part, page_file)`` triple of earlier indexes).

    :param work: Top-level work directory.
    :param part: Part path below the work.
    :param page_file: Structured page file name (``page_038_structured.json``).
    :returns: The key.
    """
    return f"{work}/{part}/{page_file}"


def parse_page_key(key: str) -> Tuple[str, str, str]:
    """Split a page key back into ``(work, part, page_file)``.

    :param key: Output of :func:`page_key`.
    :returns: The triple (``part`` may contain ``/``).
    :raises ValueError: When the key has fewer than three components.
    """
    bits = key.split("/")
    if len(bits) < 3:
        raise ValueError(f"not a page key: {key!r}")
    return bits[0], "/".join(bits[1:-1]), bits[-1]


def page_no_of(page_file: str) -> int:
    """Page index from a page file name (``page_038_structured.json`` → 38).

    :param page_file: File name.
    :returns: The number.
    :raises ValueError: When the name has no ``page_N``.
    """
    m = _PAGE_RE.search(page_file)
    if not m:
        raise ValueError(f"no page number in {page_file!r}")
    return int(m.group(1))


def model_priority(model_dir: str) -> int:
    """Rank a structuring directory (lower = preferred; port of ``analysis_lib._model_priority``).

    :param model_dir: Path fragment naming the structuring model.
    :returns: Priority (9 when no known model is named).
    """
    m = model_dir.lower()
    for i, k in enumerate(MODEL_PREFERENCE):
        if k in m:
            return i
    return 9


# ------------------------------------------------------------------------------------ records
@dataclass
class PageRec:
    """One literature page (the preferred structuring) with its OCR text.

    Fields: ``work`` (top-level directory), ``part`` (path below it), ``page_file``, ``page_no``
    (PDF page index = OCR ``page_number``), ``path`` (structured JSON), ``structuring_dir`` (relative
    to the part), ``structurer_model`` / ``structurer_provider`` (``metadata.model_used`` /
    ``provider``), ``printed_page`` (``extracted_page_number`` as written by the structurer),
    ``printed_page_num`` / ``printed_page_source`` (neighbour-checked printed page; source
    ``extracted`` / ``extracted_unverified`` / ``inferred`` / None, see :func:`infer_printed_pages`),
    ``full_main_text`` (structurer's text), ``page_type`` (``classification``), ``shelf_marks``
    (raw keys of ``shelf_marks_mentioned``; many are fake -- use :meth:`located_shelf_marks`),
    ``ocr_file`` / ``ocr_text`` / ``ocr_source`` (``text_layer`` / ``vision`` / ``ocr``) /
    ``ocr_mode`` (the OCR file's ``processing_info.mode``), ``ocr_containment`` (share of
    ``full_main_text`` 5-word shingles found in ``ocr_text``; None under five words), ``data``
    (the page JSON), ``ocr_text_raw`` / ``ocr_repair`` (set when a derived repair of a fragmented text layer
    replaced ``ocr_text``: the original OCR text and the repair method, see :mod:`ocr_repair`).
    """

    work: str
    part: str
    page_file: str
    page_no: int
    path: str
    structuring_dir: str
    structurer_model: Optional[str]
    structurer_provider: Optional[str]
    printed_page: Optional[str]
    full_main_text: str
    page_type: Optional[str]
    shelf_marks: List[str]
    ocr_file: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_source: Optional[str] = None
    ocr_mode: Optional[str] = None
    ocr_containment: Optional[float] = None
    printed_page_num: Optional[int] = None
    printed_page_source: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict, repr=False)
    ocr_text_raw: Optional[str] = field(default=None, repr=False)
    ocr_repair: Optional[str] = None

    @property
    def page_key(self) -> str:
        """Stable key of this page.

        :returns: ``<work>/<part>/<page_file>``.
        """
        return page_key(self.work, self.part, self.page_file)

    @property
    def book(self) -> str:
        """``analysis_lib.Page.book`` alias of :attr:`work`.

        :returns: Work directory name.
        """
        return self.work

    @property
    def idx(self) -> int:
        """``analysis_lib.Page.idx`` alias of :attr:`page_no`.

        :returns: Page index.
        """
        return self.page_no

    @property
    def text(self) -> str:
        """``analysis_lib.Page.text`` alias of :attr:`full_main_text` (the structurer's text, not the OCR).

        :returns: ``full_main_text``.
        """
        return self.full_main_text

    @property
    def model_dir(self) -> str:
        """``analysis_lib.Page.model_dir`` alias of :attr:`structuring_dir`.

        :returns: Structuring directory (relative to the part).
        """
        return self.structuring_dir

    def located_shelf_marks(self, use_ocr: bool = True) -> List[str]:
        """Shelfmark keys of the page that occur in its text (the located-in-text rule).

        :param use_ocr: Search the OCR text when present (else ``full_main_text``).
        :returns: Located raw keys.
        """
        text = self.ocr_text if (use_ocr and self.ocr_text) else self.full_main_text
        return _sm.located_keys(self.shelf_marks, text or "")


# ------------------------------------------------------------------------------------ page JSON fields
def sm_keys(d: Dict[str, Any]) -> List[str]:
    """Shelfmark keys of a page's ``shelf_marks_mentioned`` (dict keys or list items).

    :param d: Page JSON.
    :returns: Raw shelfmark strings.
    """
    sms = d.get("shelf_marks_mentioned") or {}
    if isinstance(sms, dict):
        return [k for k in sms.keys() if isinstance(k, str)]
    if isinstance(sms, list):
        return [x for x in sms if isinstance(x, str)]
    return []


def sm_note(d: Dict[str, Any], key: str) -> str:
    """The structuring model's note for a shelfmark key.

    :param d: Page JSON.
    :param key: Shelfmark key.
    :returns: Note text or ''.
    """
    sms = d.get("shelf_marks_mentioned") or {}
    if isinstance(sms, dict):
        v = sms.get(key)
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)[:400] if v else ""
    return ""


def tr_keys(d: Dict[str, Any]) -> List[str]:
    """Keys of a page's ``transcriptions`` dict that carry text.

    :param d: Page JSON.
    :returns: Keys.
    """
    tr = d.get("transcriptions") or {}
    if not isinstance(tr, dict):
        return []
    return [k for k, v in tr.items() if isinstance(k, str) and v]


def tr_chars(d: Dict[str, Any], key: str) -> int:
    """Characters of one transcription entry.

    :param d: Page JSON.
    :param key: Transcription key.
    :returns: Characters.
    """
    v = (d.get("transcriptions") or {}).get(key)
    if isinstance(v, dict):
        return sum(len(str(x)) for x in v.values())
    if isinstance(v, list):
        return sum(len(str(x)) for x in v)
    return len(str(v or ""))


def footnote_texts(d: Dict[str, Any]) -> List[str]:
    """The page's footnote strings.

    :param d: Page JSON.
    :returns: Footnote texts.
    """
    fn = d.get("footnotes") or {}
    if isinstance(fn, dict):
        return [str(v) for v in fn.values() if v]
    if isinstance(fn, list):
        return [str(v) for v in fn if v]
    return []


def parse_printed_page(value: Any) -> Optional[str]:
    """Normalise ``extracted_page_number`` (``[118]``, ``118``, ``"[118]"``, ``["xii"]``, ``[]``, None).

    :param value: Raw field value.
    :returns: The first page label as a string, or None.
    """
    if isinstance(value, list):
        value = next((v for v in value if v not in (None, "")), None)
    if value is None or isinstance(value, bool):
        return None
    s = str(value).strip().strip("[]()").strip()
    return s or None


# ------------------------------------------------------------------------------------ OCR
def ocr_index(root: Path) -> Dict[str, List[Path]]:
    """All ``*_ocr_results.json`` files under the literature root, by stem.

    :param root: ``paths.academic_literature``.
    :returns: ``{stem: [paths]}``.
    """
    idx: Dict[str, List[Path]] = collections.defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith("_ocr_results.json"):
                idx[f[: -len("_ocr_results.json")]].append(Path(dirpath) / f)
    return dict(idx)


def load_ocr_file(path: Path) -> Dict[int, Dict[str, Any]]:
    """OCR pages of one ``*_ocr_results.json`` by page number.

    :param path: OCR results file.
    :returns: ``{page_number: {"text", "source", "mode"}}`` (``source`` = per-page ``ocr_source``,
        ``text_layer`` for a ``text_only`` file, else ``ocr``).
    """
    with open(path, encoding="utf-8") as fh:
        o = json.load(fh)
    mode = (o.get("processing_info") or {}).get("mode")
    out: Dict[int, Dict[str, Any]] = {}
    for p in o.get("pages") or []:
        try:
            n = int(p.get("page_number"))
        except (TypeError, ValueError):
            continue
        r = p.get("ocr_result") or {}
        text = r.get("full_text") or r.get("page_text") or ""
        src = p.get("ocr_source") or ("text_layer" if mode == "text_only" else "ocr")
        out[n] = {"text": text if isinstance(text, str) else str(text), "source": src, "mode": mode}
    return out


def _nearest(candidates: Sequence[Path], page_path: Path) -> Path:
    """The candidate whose directory shares the longest prefix with the page's path.

    :param candidates: OCR files with the right stem.
    :param page_path: Structured page file.
    :returns: Best candidate.
    """
    def score(c: Path) -> int:
        """Common directory depth.

        :param c: Candidate.
        :returns: Number of shared leading path components.
        """
        n = 0
        for a, b in zip(c.parent.parts, page_path.parent.parts):
            if a != b:
                break
            n += 1
        return n

    return max(candidates, key=score)


def resolve_ocr(page_path: Path, data: Dict[str, Any], structuring_component: str,
                index: Dict[str, List[Path]], work_root: Path) -> Optional[Path]:
    """Find the OCR file of a structured page.

    Stems tried: ``metadata.pdf_name``, the structuring directory's stem; candidates outside the
    page's work are ignored.

    :param page_path: Structured page file.
    :param data: Page JSON.
    :param structuring_component: Path component containing ``_structured``.
    :param index: :func:`ocr_index` output.
    :param work_root: The work directory of the page.
    :returns: OCR file or None.
    """
    md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    stems = [str(md.get("pdf_name") or ""), re.sub(r"_structured.*$", "", structuring_component)]
    for stem in stems:
        cands = [c for c in index.get(stem, []) if work_root in c.parents]
        if cands:
            return _nearest(cands, page_path)
    return None


def shingles(text: str, n: int = 5) -> Set[str]:
    """Lower-cased word ``n``-gram shingles (Hebrew / Arabic letters count as word characters).

    :param text: Text.
    :param n: Shingle length in words.
    :returns: Set of space-joined shingles.
    """
    t = _WORD.findall((text or "").lower())
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def containment(text: str, reference: str, n: int = 5) -> Optional[float]:
    """Share of ``text``'s shingles present in ``reference``.

    :param text: Text whose provenance is checked (``full_main_text``).
    :param reference: Source text (the OCR page).
    :param n: Shingle length in words.
    :returns: Share in [0, 1], or None when ``text`` has fewer than ``n`` words.
    """
    a = shingles(text, n)
    if not a:
        return None
    return len(a & shingles(reference, n)) / len(a)


# ------------------------------------------------------------------------------------ iteration
def _split_rel(rel: Sequence[str]) -> Tuple[str, str, str, str]:
    """Derive ``(work, part, structuring_dir, structuring_component)`` from a page's relative path parts.

    :param rel: Path components below the literature root (last = page file).
    :returns: The derived fields (``part`` as in ``analysis_lib.iter_pages``).
    """
    work = rel[0]
    si = next(i for i, c in enumerate(rel) if "_structured" in c and i < len(rel) - 1)
    structuring_dir = "/".join(rel[si:-1])
    stem = re.sub(r"_structured.*$", "", rel[si])
    part_bits = list(rel[1:si])
    if not part_bits:
        part = stem
    elif stem != part_bits[-1]:
        part = "/".join(part_bits + [stem])
    else:
        part = "/".join(part_bits)
    return work, part, structuring_dir, rel[si]


#: Printed-page values the structuring prompt's example leaks into pages; trusted only when a
#: neighbouring page confirms the offset.
LEAKED_PAGE_LABELS: Set[int] = {119}


def infer_printed_pages(pages: Sequence[PageRec], max_gap: int = 2) -> None:
    """Fill :attr:`PageRec.printed_page_num` / ``printed_page_source`` from neighbour-consistent offsets.

    Within each part (pages ordered by index), a numeric extracted page is *reliable* when its
    offset (printed minus index) equals that of the previous or next numeric page. Then each page
    gets: ``extracted`` -- its own number, when reliable or matching the offset of the nearest
    reliable page on either side; ``inferred`` -- index + offset when the nearest reliable pages on
    both sides agree, or the nearer one lies within ``max_gap`` pages (this replaces the
    prompt-example label ``119`` and stray numbers); ``extracted_unverified`` -- its own number
    when nothing contradicts it; None otherwise.

    :param pages: Pages (modified in place).
    :param max_gap: Largest index distance to a single reliable neighbour used for inference.
    :returns: None.
    """
    by_part: Dict[Tuple[str, str], List[PageRec]] = collections.defaultdict(list)
    for p in pages:
        by_part[(p.work, p.part)].append(p)
    for group in by_part.values():
        group = sorted(group, key=lambda r: r.page_no)
        nums: List[Optional[int]] = [int(p.printed_page) if p.printed_page and p.printed_page.isdigit() else None
                                     for p in group]
        cand = [i for i, v in enumerate(nums) if v is not None and v not in LEAKED_PAGE_LABELS]
        offs = {i: nums[i] - group[i].page_no for i in cand}
        reliable: Dict[int, int] = {}
        for j, i in enumerate(cand):
            nb = [cand[j - 1]] if j > 0 else []
            nb += [cand[j + 1]] if j + 1 < len(cand) else []
            if any(offs[k] == offs[i] for k in nb):
                reliable[i] = offs[i]
        rel_idx = sorted(reliable)
        for i, p in enumerate(group):
            own = nums[i]
            own_off = own - p.page_no if own is not None else None
            if i in reliable:
                p.printed_page_num, p.printed_page_source = own, "extracted"
                continue
            before = max((k for k in rel_idx if k < i), default=None)
            after = min((k for k in rel_idx if k > i), default=None)
            sides = [k for k in (before, after) if k is not None]
            if own_off is not None and any(reliable[k] == own_off for k in sides):
                p.printed_page_num, p.printed_page_source = own, "extracted"
                continue
            guess = None
            if before is not None and after is not None and reliable[before] == reliable[after]:
                guess = reliable[before]
            elif sides:
                near = min(sides, key=lambda k: abs(group[k].page_no - p.page_no))
                if abs(group[near].page_no - p.page_no) <= max_gap:
                    guess = reliable[near]
            if guess is not None and p.page_no + guess > 0:
                p.printed_page_num, p.printed_page_source = p.page_no + guess, "inferred"
            elif own is not None and own not in LEAKED_PAGE_LABELS:
                p.printed_page_num, p.printed_page_source = own, "extracted_unverified"
            else:
                p.printed_page_num, p.printed_page_source = None, None


def iter_pages(root: Optional[Path] = None, works: Optional[Iterable[str]] = None, with_ocr: bool = True,
               skip: Iterable[str] = (), shingle_n: int = 5, repair_root: Optional[Path] = None,
               use_repairs: bool = True) -> List[PageRec]:
    """Load the literature pages, one structuring per ``(work, part, page number)``.

    Port of ``analysis_lib.iter_pages`` (same ``work`` / ``part`` derivation and model preference,
    empty ``full_main_text`` costs 20) with three changes: pages are deduplicated by page number
    (``metadata.page_number``, else the file name) rather than file name, ties going to the file
    whose name carries that number, then to the smaller path; the OCR text is joined by file stem
    (:func:`resolve_ocr`); printed pages are made consistent per part (:func:`infer_printed_pages`).

    :param root: Literature root (default ``paths.academic_literature``).
    :param works: Restrict to these top-level work directories (for sampling / tests).
    :param with_ocr: Join OCR text and compute containment.
    :param skip: Top-level directories to ignore.
    :param shingle_n: Shingle length for :attr:`PageRec.ocr_containment`.
    :param repair_root: Directory of derived OCR repairs (default ``paths.out_dir/ocr_repair`` when it exists).
    :param use_repairs: Apply the repairs (:func:`apply_ocr_repair`); False keeps the raw OCR text.
    :returns: Pages sorted by work, part, page number.
    """
    root = Path(root) if root is not None else _paths.load_paths().academic_literature
    wanted = set(works) if works is not None else None
    skip_set = set(skip)
    best: Dict[Tuple[str, str, int], Tuple[Tuple[int, int, str], PageRec]] = {}
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for fname in sorted(files):
            if not (fname.startswith("page_") and fname.endswith("_structured.json")):
                continue
            path = Path(dirpath) / fname
            rel = path.relative_to(root).parts
            if len(rel) < 3 or rel[0] in skip_set or (wanted is not None and rel[0] not in wanted):
                continue
            if not any("_structured" in c for c in rel[:-1]):
                continue
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            work, part, sdir, _ = _split_rel(rel)
            md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            try:
                page_no = int(md.get("page_number"))
            except (TypeError, ValueError):
                page_no = page_no_of(fname)
            text = data.get("full_main_text") or ""
            text = text if isinstance(text, str) else str(text)
            pri = (model_priority(sdir) + (20 if not text.strip() else 0),
                   0 if page_no_of(fname) == page_no else 1, str(path))
            rec = PageRec(work=work, part=part, page_file=fname, page_no=page_no, path=str(path),
                          structuring_dir=sdir, structurer_model=md.get("model_used"),
                          structurer_provider=md.get("provider"),
                          printed_page=parse_printed_page(data.get("extracted_page_number")),
                          full_main_text=text, page_type=data.get("classification"), shelf_marks=sm_keys(data),
                          data=data)
            k = (work, part, page_no)
            if k not in best or pri < best[k][0]:
                best[k] = (pri, rec)
    pages = [r for _, r in best.values()]
    pages.sort(key=lambda r: (r.work, r.part, r.page_no))
    if with_ocr:
        _join_ocr(pages, root, shingle_n, (repair_root or default_repair_root()) if use_repairs else None)
    infer_printed_pages(pages)
    return pages


def default_repair_root() -> Optional[Path]:
    """The build's OCR repair directory (``paths.out_dir/ocr_repair``) when it exists.

    :returns: Directory or None (no repairs, or no local paths file).
    """
    from src.finetuning.text27b import ocr_repair as _rep
    try:
        d = _rep.default_repair_root()
    except FileNotFoundError:
        return None
    return d if d.is_dir() else None


def apply_ocr_repair(page: PageRec, repairs: Dict[int, Tuple[str, str, str]], ocr_file_name: Optional[str]) -> bool:
    """Replace a page's OCR text by its derived repair when the repair was made from exactly this OCR text.

    The repair applies only when the repair manifest names the page's OCR file and the sha1 of the page's OCR
    text equals the one recorded when the repair was made; the original text is kept in ``ocr_text_raw``.

    :param page: Page with ``ocr_text`` / ``ocr_file`` set.
    :param repairs: ``{page_number: (text, original sha1, method)}`` of the page's work (:func:`ocr_repair.load_repairs`).
    :param ocr_file_name: OCR file name the repairs were made from.
    :returns: True when the page text was replaced.
    """
    rec = repairs.get(page.page_no)
    if rec is None or page.ocr_text is None or not page.ocr_file or Path(page.ocr_file).name != ocr_file_name:
        return False
    text, sha1, method = rec
    if hashlib.sha1(page.ocr_text.encode("utf-8")).hexdigest() != sha1:
        return False
    page.ocr_text_raw, page.ocr_text, page.ocr_repair = page.ocr_text, text, method
    page.ocr_source = f"{page.ocr_source or 'ocr'}+repair"
    return True


def _join_ocr(pages: Sequence[PageRec], root: Path, shingle_n: int, repair_root: Optional[Path] = None) -> None:
    """Attach OCR text, source and containment to pages (in place), applying derived OCR repairs.

    :param pages: Pages from :func:`iter_pages`.
    :param root: Literature root.
    :param shingle_n: Shingle length.
    :param repair_root: Directory of derived OCR repairs (None: none applied).
    :returns: None.
    """
    from src.finetuning.text27b import ocr_repair as _rep
    index = ocr_index(root)
    cache: Dict[Path, Dict[int, Dict[str, Any]]] = {}
    repairs: Dict[str, Tuple[Optional[str], Dict[int, Tuple[str, str, str]]]] = {}
    repaired = 0
    missing = 0
    for p in pages:
        f = resolve_ocr(Path(p.path), p.data, p.structuring_dir.split("/")[0], index, root / p.work)
        if f is None:
            missing += 1
            continue
        if f not in cache:
            cache[f] = load_ocr_file(f)
        o = cache[f].get(p.page_no)
        p.ocr_file = str(f)
        if o is None:
            missing += 1
            continue
        p.ocr_text, p.ocr_source, p.ocr_mode = o["text"], o["source"], o["mode"]
        if repair_root is not None:
            if p.work not in repairs:
                repairs[p.work] = _rep.load_repairs(repair_root, p.work)
            name, recs = repairs[p.work]
            repaired += apply_ocr_repair(p, recs, name)
        # the join check compares the structurer's text with the OCR file's own text (a repair is verified
        # against that text by its character multiset and sha1 when it is made)
        p.ocr_containment = containment(p.full_main_text, p.ocr_text_raw or p.ocr_text, shingle_n)
    if missing:
        LOG.warning("%d of %d pages without an OCR page", missing, len(pages))
    if repaired:
        LOG.info("%d pages read from derived OCR repairs", repaired)


# ------------------------------------------------------------------------------------ text analysis
RTL_HE = re.compile(r"[֐-׿]")
RTL_AR = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")


def letter_counts(s: str) -> Tuple[int, int, int]:
    """Hebrew, Arabic and Latin letter counts (``analysis_lib.script_counts``).

    :param s: Text.
    :returns: ``(hebrew, arabic, latin)``.
    """
    return len(RTL_HE.findall(s)), len(RTL_AR.findall(s)), len(LATIN.findall(s))


def is_rtl_line(s: str) -> bool:
    """Whether a line is mostly Hebrew / Arabic script.

    :param s: Line.
    :returns: True for ≥ 3 RTL letters outnumbering Latin ones.
    """
    h, a, lat = letter_counts(s)
    return (h + a) >= 3 and (h + a) > lat


JA_WORDS: Set[str] = {"פי", "אן", "עלי", "ען", "מא", "לם", "קד", "הדא", "דלך", "אלדי", "ואנא", "כאן", "אנה", "ולם", "פקד",
                      "וקד", "ואן", "אלי", "מנה", "ענד", "בעד", "גיר", "יא", "אנא", "תם", "אללה", "הו", "הי", "לך", "פאן",
                      "ומא", "אליה"}
HE_WORDS: Set[str] = {"את", "של", "אשר", "הוא", "היא", "כי", "זה", "זאת", "אני", "אתה", "גם", "אבל", "אחרי", "שלא", "אותו",
                      "היה", "הם", "הזה", "אלה", "כמו", "או", "יש", "אין", "שהוא", "לפי", "ראה", "כלומר", "שם", "בין", "עוד",
                      "אך", "ולא"}


def ja_or_he(s: str) -> str:
    """Classify Hebrew-script text as Judaeo-Arabic, Hebrew or unknown.

    :param s: Text.
    :returns: ``ja``, ``he`` or ``?``.
    """
    toks = re.findall(r"[א-ת\"']+", s)
    ja = sum(1 for t in toks if t in JA_WORDS or (len(t) >= 4 and re.match(r"^[ובפל]?אל", t)))
    he = sum(1 for t in toks if t in HE_WORDS or (len(t) >= 4 and t.startswith("וה")))
    if ja == he:
        return "?"
    return "ja" if ja > he else "he"


EN_SW: Set[str] = {"the", "and", "of", "to", "in", "is", "that", "which", "was", "with", "for", "as", "by", "this", "his", "from"}
FR_SW: Set[str] = {"le", "la", "les", "des", "et", "du", "est", "une", "dans", "qui", "que", "pour", "sur", "au", "aux", "par"}
DE_SW: Set[str] = {"der", "die", "und", "das", "den", "ist", "nicht", "mit", "von", "des", "dem", "ein", "eine", "zu", "auf", "im"}


def lang_of(s: str) -> str:
    """Language of a prose passage by script and stop words.

    :param s: Text.
    :returns: ``en``, ``he``, ``fr``, ``de`` or ``other``.
    """
    h, a, lat = letter_counts(s)
    if h + a > lat:
        return "he" if h > a and ja_or_he(s) != "ja" else "other"
    words = re.findall(r"[a-zà-ÿ]+", s.lower())
    if not words:
        return "other"
    sc = {"en": sum(w in EN_SW for w in words), "fr": sum(w in FR_SW for w in words), "de": sum(w in DE_SW for w in words)}
    best_lang = max(sc, key=sc.get)
    return best_lang if sc[best_lang] >= 2 else "other"


@dataclass
class Block:
    """A run of lines between blank lines / heading breaks (character offsets into the page text)."""

    start: int
    end: int
    lines: List[str]

    @property
    def text(self) -> str:
        """Joined text of the block.

        :returns: Lines joined with newlines.
        """
        return "\n".join(self.lines)


def line_offsets(text: str) -> List[int]:
    """Start offset of each line.

    :param text: Text.
    :returns: Offsets.
    """
    offs, o = [], 0
    for ln in text.split("\n"):
        offs.append(o)
        o += len(ln) + 1
    return offs


def headingish(line: str, width: float) -> bool:
    """Whether a line looks like a heading / section break rather than prose.

    :param line: Line.
    :param width: Typical full line width of the page.
    :returns: True for heading-like lines.
    """
    s = line.strip()
    if not s:
        return True
    if len(s) >= 0.62 * width or s.endswith("-"):
        return False
    if re.search(r"[.;:,!?\"”’)\]]$", s) and len(s) > 25:
        return False
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6 and len(letters) > 3:
        return True
    return len(s) < 0.45 * width and not re.search(r"[.,;]$", s)


def segment_blocks(text: str, body_end: Optional[int] = None) -> List[Block]:
    """Split page text into blocks bounded by blank lines / heading-like lines.

    :param text: Page text.
    :param body_end: Offset where the footnote region starts (blocks are cut there).
    :returns: Blocks.
    """
    lines = text.split("\n")
    offs = line_offsets(text)
    lens = sorted(len(x.strip()) for x in lines if len(x.strip()) > 15)
    width = lens[int(len(lens) * 0.8)] if lens else 60
    blocks: List[Block] = []
    cur: List[int] = []

    def flush() -> None:
        """Close the current block.

        :returns: None.
        """
        if cur:
            a = offs[cur[0]]
            b = offs[cur[-1]] + len(lines[cur[-1]])
            blocks.append(Block(a, b, [lines[i] for i in cur]))
        cur.clear()

    for i, ln in enumerate(lines):
        if body_end is not None and offs[i] >= body_end and (not cur or offs[cur[0]] < body_end):
            flush()
        if not ln.strip():
            flush()
            continue
        if headingish(ln, width):
            flush()
            cur.append(i)
            flush()
            continue
        cur.append(i)
    flush()
    return blocks


def footnote_region_start(text: str, d: Dict[str, Any]) -> Optional[int]:
    """Estimate where the footnote area begins in the page text (from the page's footnote strings).

    :param text: Page text.
    :param d: Page JSON.
    :returns: Line-start offset, or None.
    """
    fns = footnote_texts(d)
    if not fns:
        return None
    toks = _sm.tokens_with_pos(text)
    tk = [t for t, _, _ in toks]
    cands = []
    for f in fns:
        ft = [t for t, _, _ in _sm.tokens_with_pos(f)][:6]
        if len(ft) < 4:
            continue
        for i in range(len(tk) - len(ft) + 1):
            if tk[i:i + len(ft)] == ft:
                cands.append(toks[i][1])
                break
    cands = [c for c in cands if c > 0.35 * len(text)]
    if not cands:
        return None
    c = min(cands)
    ls = text.rfind("\n", 0, c)
    return ls + 1 if ls >= 0 else c


FORMULA = re.compile(r"(?i)\b(may god|i am writing|i (have )?(informed|received|sent|wrote)|your (servant|excellency|honou?r|letter)|"
                     r"my (lord|master|brother)|dinars?|dirhams?|witness(es|ed)?|peace be|god willing|in your name|"
                     r"we (testify|the undersigned)|the elder|may (he|his|your)|praise be to god)\b")


def docness(s: str) -> float:
    """How much an English passage reads as translated document text rather than scholarship.

    :param s: Passage.
    :returns: Score (≥ 1.0 reads as a translated letter / deed; 0.0 under 40 words).
    """
    words = re.findall(r"[A-Za-z']+", s)
    if len(words) < 40:
        return 0.0
    fp = sum(w.lower() in {"i", "me", "my", "you", "your", "we", "our", "us", "thee", "thy"} for w in words) / len(words)
    form = len(FORMULA.findall(s))
    linenums = len(re.findall(r"(?:^|\s)\((\d{1,2})\)\s", s)) + len(re.findall(r"(?m)^\s*\d{1,2}\.\s+[A-Z\[]", s))
    schol = len(re.findall(r"(?i)\b(cf\.|see|ibid|p\.|pp\.|goitein|edited|published|scholar|med\. soc|n\.\s?\d)", s))
    return round(fp * 12 + min(form, 6) * 0.25 + min(linenums, 8) * 0.15 - schol * 0.15, 3)


def rtl_run_chars(lines: List[str], need_ja: bool = False) -> int:
    """Characters of the longest run of consecutive RTL-script lines.

    :param lines: Lines.
    :param need_ja: Count only runs classified Judaeo-Arabic / Arabic.
    :returns: Characters of the longest run.
    """
    best = cur = 0
    run: List[str] = []
    for ln in lines + [""]:
        if ln.strip() and is_rtl_line(ln):
            run.append(ln)
            cur += len(ln.strip())
            continue
        if run:
            ok = True
            if need_ja:
                joined = " ".join(run)
                h, a, _ = letter_counts(joined)
                ok = a > h or ja_or_he(joined) == "ja"
            if ok:
                best = max(best, cur)
        run, cur = [], 0
    return best


CATALOG_FIELD = re.compile(r"(?i)^\s*(paper|parchment|vellum|size|date|lines?|script|language|contents|bibl|dimensions|"
                           r"material|condition|recto|verso|\d+(\.\d+)?\s?(x|×)\s?\d+|\d+\s?cm)")


def catalogue_like(lines: List[str]) -> bool:
    """Whether lines have a catalogue-entry layout (three or more field lines).

    :param lines: Lines.
    :returns: True when catalogue-like.
    """
    if len(lines) < 3:
        return False
    return sum(1 for ln in lines if CATALOG_FIELD.search(ln)) >= 3


def index_like(text: str) -> bool:
    """Whether a page is a back-of-book index (mostly number lists).

    :param text: Page text.
    :returns: True for index pages.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 12:
        return False
    numeric = sum(1 for ln in lines if re.fullmatch(r"[\d\s,.\-–n]+", ln) or re.search(r"\d+(,\s*\d+){2,}", ln))
    return numeric / len(lines) > 0.35
