# File name: semantic_text.py
# Date: 10/7/26 (semantic_v4 added 10/9/26)
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Semantic document text: ``semantic_v3`` (frozen, index v9) and ``semantic_v4`` (index v10).

``semantic_v3`` is the production embedding text minus record-identity lines. The fine-tuned text embedder
(``isaacmg/genizah-embed-qwen3-0.6b``) was trained and evaluated only on this text, so every index built with it must
embed exactly this text. It is a port of ``evals/embedding_audit/semantic_text.py`` in the genizah_search repo (branch
``eval-retrieval``); **every regex below is copied verbatim from that reference and must stay identical to it**
(``tests/test_semantic_text.py`` pins them), and :func:`semantic_text`, :func:`content_lines`,
:func:`semantic_ineligible_reason` and :func:`is_semantic_eligible` behave exactly like the reference's
``semantic_text`` / ``content_lines`` / ``ineligible_reason`` / ``is_eligible`` with the frozen labels on any text
(``tests/data/semantic_v3_golden.jsonl`` pins them on real records). Changing one changes the text the model sees and
silently breaks parity with training and the frozen eval: make a new text version instead.

What ``semantic_v3`` does to one record's production text (``GenizahDocument.create_text_representation()``):

* drops the ``Document ID:`` and ``Shelf Mark:`` lines (numbering schemes plus the collection a fragment was shipped
  to, not the subject);
* strips editor credits: ``[Editor: ...]`` tags and the PGP form ``Editor: Surname, Given``;
* drops field lines left without a value (``Transcription:``);
* masks the record's own shelf mark, cited shelf marks of other fragments and PGPIDs in the remaining lines
  (descriptions, join notes) as :data:`MASK`.

A record is *ineligible* under ``semantic_v3`` when that text is empty, shorter than :data:`MIN_CHARS`, or says nothing
beyond a boilerplate catalogue label (:data:`BOILERPLATE_PATH`), metadata lines (:data:`META_PREFIXES`) and an
``Alternate titles:`` line whose every ``;`` part is itself boilerplate. Every other line counts (a ``Subjects:`` line
included), the whole text counts towards :data:`MIN_CHARS`, and alternate-title parts are compared as written.

``semantic_v4`` (index v10: genizah_search ``docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md`` rows B1, B3, B7) is a
pure layout over the frozen ``semantic_v3`` text (:func:`semantic_v4_text`): continuation lines are folded into the
field line before them, the ``Subjects:`` and ``Document Type:`` lines are dropped, and the ``Alternate titles:`` parts
are folded into the ``Description:`` line (a part whose normalised key occurs as whole words in the description's key
or in an earlier kept part's key is left out; whole parts are appended until the next one would take the appended
characters past :data:`MAX_ALT_CHARS`, the first one always). Its eligibility (:func:`semantic_v4_ineligible_reason`)
is decided on the ``semantic_v3`` text without its ``Subjects:`` lines (:func:`v4_measured_text`), so the layout never
removes a record: ``Subjects:`` lines are neither content nor counted towards :data:`MIN_CHARS`, alternate-title parts
are compared with the labels without square brackets, and a short text whose content names a festival or a topic of
the frozen admission patterns (:func:`admit_pattern`) is admitted.

The lines from ``semantic_v4 begin`` to ``semantic_v4 end`` below are a byte-for-byte copy of the genizah_search
reference, ``evals/embedding_audit/semantic_text.py`` (plan B7: the genizah_search copy is the reference and this one
mirrors it), mirrored on 2026-10-10 from the ``eval-retrieval`` working copy (mtime 2026-10-09 16:36:44 EDT). It
replaced the draft code of ``audits/text-pipeline.md`` §2 and §4, which looked a part's key up in the earlier keys
joined by spaces and so left out a part whose words span two earlier keys (14 real records). Only the paths above the
block (:data:`BOILERPLATE_PATH`, :data:`ADMIT_PATH`) are this repo's own. ``tests/test_semantic_v4.py`` pins the
block's sha256, its constants, both data files (:data:`BOILERPLATE_SHA256`, :data:`ADMIT_SHA256`) and its output on
real records (``tests/data/semantic_v4_golden.jsonl``): change the genizah_search copy first, then mirror it here.

``data/boilerplate_descriptions_v3.json`` is an unchanged copy of genizah_search
``docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json``: the 64 labels the reference's
``boilerplate_descriptions(corpus_v1, min_count=50, include_alt_titles=True)`` produced, frozen 2026-10-07 for v9
parity; ``semantic_v4`` uses the same list and never recomputes it. ``data/eligibility_admit_v4.json`` is an unchanged
copy of genizah_search ``docs/handoffs/v10_embedder/eligibility_admit_v4.json`` (mirrored 2026-10-10, mtime 2026-10-09
16:36:35 EDT), written by its ``build_admit_v4.py`` from ``build_subjects.py``: ``FESTIVAL_TERMS`` (case-insensitive)
and ``TOPIC_RULES_V2`` (case-sensitive), the patterns that admit a short record. Its provenance lives here and in its
own ``source`` key only, so that the file keeps the pinned :data:`ADMIT_SHA256`.

Bibliography pages get their own variant (:func:`bibliography_semantic_text`): the shelf marks they mention are the
bridge between scholarship and fragments, so only the ``Source PDF:`` file-name line is dropped.
"""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import AbstractSet, FrozenSet, List, Optional, Pattern, Sequence, Set, Tuple

ID_PREFIXES = ("Document ID:", "Shelf Mark:")
# lines that describe the record's form, not its subject: they never make a record eligible on their own
META_PREFIXES = ("Language:", "Document Type:", "Date:")
ALT_PREFIX = "Alternate titles:"
# "[Editor: KTIV Academic transcription (Frag. 001r)]"; also the truncated form at a cut-off line end
# ("[Editor: KTIV Acad...")
EDITOR_RE = re.compile(r"\[Editors?:[^\]\n]{0,200}(?:\]|$)\s*")
# PGP transcriptions open with unbracketed attributions: "Editor: Gil, Moshe Editor: Goitein, S. D. ",
# "Editors: Goitein, S. D.; Friedman, Mordechai Akiva ", "Editors: [unknown source]". A name is a capitalised surname,
# optionally ", Given names/initials"; the lookahead stops a name from swallowing the next "Editor:".
_EDITOR_NAME = r"(?:\[unknown source\]‎?|[A-Z][\w'’\-]*(?:\.\.\.)?(?:,(?: (?!Editors?:)[A-Z][\w'’\-]*\.?)+)?)"
PGP_EDITOR_RE = re.compile(rf"(?<!\w)Editors?: {_EDITOR_NAME}(?:; {_EDITOR_NAME})*\s*")
EMPTY_FIELD_RE = re.compile(r"^[A-Z][\w /()-]{0,40}:\s*$")  # a field label left with no value, e.g. "Transcription:"
MIN_CHARS = 40

# Shelf-mark detector: the production SHELFMARK_PATTERN of the site's shelfmark_normalizer.py (genizah_search
# src/backend, branch prod-mbp), frozen here as the reference froze it. Do not import the live pattern: if production
# changes it, this text version must not change.
PROD_SHELFMARK_RE = re.compile(
    r'\b('
    r'T-S\s+AS\s+\d+[\w.]*|T-S\s+NS\s+\d+[\w.]*|T-S\s+K\s+\d+[\w.]*|T-S\s+Ar\.?\s*\d+[\w.]*|T-S\s+Misc\.?\s*\d+[\w.]*'
    r'|TS\s+AS\s+\d+[\w.]*|TS\s+NS\s+\d+[\w.]*|TS\s+Misc\.?\s*\d+[\w.]*'
    r'|T-S\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*|T-S\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    r'|TS\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*|TS\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    r'|ENA\s+NS\s+\d+[\w.]*|ENA\s+\d+[\w.]*'
    r'|L-G\s+[A-Za-z0-9]+[\w.]*'
    r'|Mosseri\s+[A-Za-z0-9]+[\w.]*|Moss\.\s+[A-Za-z0-9]+[\w.]*'
    r'|CUL\s+Or\.?\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?[A-Za-z]?\d+(?:\.\d+)?)?|CUL\s+Add\s+\d+[\w.]*'
    r'|Or\.\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?[A-Za-z]?\d+(?:\.\d+)?)?'
    r'|Bodl\.?\s+MS\.?\s+[A-Za-z]+\.?\s*[A-Za-z]?\.?\s*\d+[\w.]*|MS\.?\s+heb\.?\s*[A-Za-z]\.?\s*\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+'
    r'|Manchester\s*[:\-]?\s*Gaster\s+Printed\s+Series\s+\d+[\w.]*|Manchester\s*[:\-]?\s*Gaster\s+\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Glass\s+\d+[\w.]*|Manchester\s*[:\-]?\s*[ABCGLP]\s+\d+[\w.]*'
    r'|Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+|Gaster\s+[A-Z]\s+\d+[\w.]*'
    r'|BL\s+Or\s+\d+[\w.]*|BL\s+Add\s+\d+[\w.]*|Halper\s+\d+|AIU\s+[IVX]+\.[A-Z]\.\d+'
    r')', re.IGNORECASE)
# record-identity strings the production pattern misses, as cited inside PGP / KTIV descriptions: PGP ids, the
# St Petersburg collections, "T-S NS J401.21", "T-S Loan 137", "CUL Add.3339a", "Bodl. Heb. f. 26/1-6",
# "ENA NS I.2" / "ENA L43", "AIU XII.118", "JRL Gaster heb. ms 1636/14", "DK 238.5 alt. XVI" (Kaufmann),
# "Stras. 4038/27", "UPenn E 16510", "Freer F 1908.44H", "Rylands B 8725", "Pococke 400", "Firk. I: 1587",
# "Westminster Frag. Cairens. 25", "NLI#$FL188116337", "NLI 577.4/27", "F 1908.44i", "Sassoon 713", "JRL B 5758",
# "Mosseri VII 155"
EXTRA_ID_RE = re.compile(
    r"\bPGPID\s*:?\s*#?\s*\d+"
    r"|\bAntonin\s+\d+[\w.]*"
    r"|\bYevr\.?(?:[-\s]Arab\.?)?\s+[IVX]+(?:\s*[A-Z]\b)?(?:\s*\d+[\w./]*)?"
    r"|\bT-?S\s+NS\s+J\.?\s?\d+[\w.]*"
    r"|\bT-?S\s+(?:Loan|Arabic|Misc\.?|[A-Z]{1,2}\.?)\s?\d+[\w.]*"
    r"|\bCUL\s+(?:Add|Or)\.\s*\d+[\w.]*(?:[–-][a-z]\b)?"
    r"|\bBodl\.?\s+(?:MS\.?\s+)?(?:Heb\.?\s*)?[a-z]\.?\s*\d+[\w./-]*"
    r"|\bJTS:?\s+(?:MS\s+|Geniza\s+Misc\.?\s+|ENA\s+)?\d+[\w.]*"
    r"|\bENA\s+(?:NS\s+|Misc\.?\s+)?(?:[IVX]+\.\d+|[A-Z]\d+|\d+)[\w.]*"
    r"|\bAIU\s+(?:[IVX]+\.?\s*)?[A-Z]?\.?\s*\d+[\w.]*"
    r"|\b(?:JRL\s+)?Gaster\s+(?:hebr?|ar)\.?\s*(?:ms\.?\s*)?\d+[\w./]*"
    r"|\bDK\s*\d+[\w./]*(?:\s+alt\.?\s*[IVXLC]+)?"
    r"|\bStras(?:bourg)?\.?\s*\d+[\w./]*"
    r"|\bU?Penn\s+E\s*\d+[\w.]*"
    r"|\bFreer:?\s+F\s*\d{4}\.\d+[\w.]*"
    r"|\bRylands\s+[A-Z]\s*\d+[\w./]*"
    r"|\bPococke\s+\d+[\w.]*"
    r"|\bFirk\.?\s*[IV]+\s*:?\s*\d+[\w.]*"
    r"|\bWestminster\s+Frag\.?\s+Cairens\.?\s*\d+[\w.]*"
    r"|\bNLI\s*#?\$?FL\d+"
    r"|\bNLI\s+\d+(?:\.\d+)*(?:/\d+)?"
    r"|\bF\s+1908\.\d+[\w.]*"
    r"|\bSassoon\s+\d+[\w.]*"
    r"|\bJRL\s+[A-Z]\s*\d+[\w./]*",
    re.IGNORECASE)
# applied BEFORE the production pattern, which would stop "Mosseri VII 155" after "Mosseri VII"
PRE_ID_RE = re.compile(r"\bMosseri\s+[IVX]+[a-z]?\s*[.,]?\s*\d+[\w.]*")
MASK = "[shelfmark]"

# Frozen inputs of the semantic_v4 text function (the v4 block below pins both by sha256, BOILERPLATE_SHA256 and
# ADMIT_SHA256): byte-identical copies of genizah_search docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json and
# docs/handoffs/v10_embedder/eligibility_admit_v4.json (the latter mirrored 2026-10-10 from the eval-retrieval working
# copy, mtime 2026-10-09 16:36:35 EDT). These two paths are the only repo-specific names the v4 block reads.
BOILERPLATE_PATH = Path(__file__).resolve().parent / "data" / "boilerplate_descriptions_v3.json"
ADMIT_PATH = Path(__file__).resolve().parent / "data" / "eligibility_admit_v4.json"
# The one line of a bibliography page representation that is dropped (``BibliographyDocument``'s first line).
BIBLIOGRAPHY_DROP_PREFIX = "Source PDF:"


def mask_shelfmarks(text: str, own_mark: str = "") -> Tuple[str, int]:
    """Replace the record's own shelf mark, cited shelf marks and PGPIDs with :data:`MASK`.

    :param text: Text (a stripped record text or one of its lines).
    :param own_mark: The record's ``Shelf Mark:`` value (masked literally when longer than 4 characters).
    :returns: (masked text, number of spans masked).
    :rtype: Tuple[str, int]
    """
    n = 0
    if len(own_mark) > 4 and own_mark in text:
        # whole-token match only: "T-S 12.34" must not turn "T-S 12.345" into "[shelfmark]5"
        text, n = re.subn(rf"(?<!\w){re.escape(own_mark)}(?!\w)", MASK, text)
    text, k0 = PRE_ID_RE.subn(MASK, text)
    text, k1 = PROD_SHELFMARK_RE.subn(MASK, text)
    text, k2 = EXTRA_ID_RE.subn(MASK, text)
    return text, n + k0 + k1 + k2


def shelf_mark_from_text(text: str) -> str:
    """The ``Shelf Mark:`` value of a production text (empty when the line is absent).

    The production text writes the line only when the shelf mark differs from the document id.

    :param text: Production embedding text.
    :returns: Shelf mark, stripped.
    :rtype: str
    """
    return next((ln[len("Shelf Mark:"):].strip() for ln in text.split("\n") if ln.startswith("Shelf Mark:")), "")


def semantic_text(text: str, own_shelfmark: Optional[str] = None) -> str:
    """Drop ``Document ID:`` / ``Shelf Mark:`` lines and editor credits; mask record identifiers.

    Removes ``[Editor: ...]`` tags and the PGP form ``Editor: Surname, Given`` (who edited the transcription, not what
    the record is about), drops field lines left without a value, and replaces the record's own shelf mark, cited
    shelf marks of other fragments and PGPIDs anywhere in the remaining lines with :data:`MASK`.

    :param text: Production embedding text of one record (``GenizahDocument.create_text_representation()``).
    :param own_shelfmark: The record's own shelf mark. ``None`` reads it from the text's ``Shelf Mark:`` line, which
        is exactly what training did; pass a value only when it equals that line's value.
    :returns: Semantic text (may be empty).
    :rtype: str
    """
    lines = [PGP_EDITOR_RE.sub("", EDITOR_RE.sub("", ln)).rstrip() for ln in text.split("\n")
             if not ln.startswith(ID_PREFIXES)]
    out = "\n".join(ln for ln in lines if ln.strip() and not EMPTY_FIELD_RE.match(ln)).strip()
    own = shelf_mark_from_text(text) if own_shelfmark is None else own_shelfmark.strip()
    return mask_shelfmarks(out, own)[0]


def alt_title_parts(text: str) -> Set[str]:
    """The ``;``-separated parts of a record's ``Alternate titles:`` line(s).

    :param text: Record text (raw or semantic).
    :returns: Set of non-empty parts.
    :rtype: Set[str]
    """
    parts: Set[str] = set()
    for ln in text.split("\n"):
        if ln.startswith(ALT_PREFIX):
            parts.update(p.strip() for p in ln[len(ALT_PREFIX):].split(";") if p.strip())
    return parts


def content_lines(stripped: str, boiler: AbstractSet[str]) -> List[str]:
    """Lines of a semantic text that say something beyond boilerplate labels and metadata (frozen ``semantic_v3``).

    Exactly the reference's ``content_lines``: an ``Alternate titles:`` line counts only when one of its parts (as
    written, brackets included) is not boilerplate; every line that is not metadata, a boilerplate ``Description:`` or
    such an ``Alternate titles:`` line counts, a ``Subjects:`` line included. :func:`content_lines_v4` is the v4 rule.

    :param stripped: Output of :func:`semantic_text`.
    :param boiler: Boilerplate labels.
    :returns: Remaining content lines.
    :rtype: List[str]
    """
    out = []
    for ln in stripped.split("\n"):
        if not ln.strip() or ln.startswith(META_PREFIXES):
            continue
        if ln.startswith("Description:") and ln[len("Description:"):].strip() in boiler:
            continue
        if ln.startswith(ALT_PREFIX) and alt_title_parts(ln) <= boiler:
            continue
        out.append(ln)
    return out


@lru_cache(maxsize=None)
def load_boilerplate_labels(path: Path = BOILERPLATE_PATH) -> FrozenSet[str]:
    """Load the frozen boilerplate labels once.

    :param path: JSON file with a ``boilerplate`` list and the ``min_chars`` it was frozen with.
    :returns: The labels.
    :rtype: FrozenSet[str]
    :raises ValueError: When the file's ``min_chars`` disagrees with :data:`MIN_CHARS`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("min_chars") != MIN_CHARS:
        raise ValueError(f"{path}: min_chars={data.get('min_chars')} but this text version uses {MIN_CHARS}")
    return frozenset(data["boilerplate"])


def _label_set(labels: Optional[Sequence[str]]) -> AbstractSet[str]:
    """Return ``labels`` as a set, defaulting to the frozen v3 list.

    :param labels: Boilerplate labels, or ``None`` for :func:`load_boilerplate_labels`.
    :returns: Set of labels.
    :rtype: AbstractSet[str]
    """
    if labels is None:
        return load_boilerplate_labels()
    return labels if isinstance(labels, (set, frozenset)) else frozenset(labels)


def semantic_ineligible_reason(text: str, labels: Optional[Sequence[str]] = None) -> str:
    """Name why a ``semantic_v3`` text gets no vector (the reference's ``ineligible_reason``).

    :param text: Output of :func:`semantic_text`.
    :param labels: Boilerplate labels (``None``: the frozen v3 list).
    :returns: ``empty``, ``short`` (the whole text is shorter than :data:`MIN_CHARS`), ``boilerplate_only`` or
        ``eligible``.
    :rtype: str
    """
    if not text:
        return "empty"
    if len(text) < MIN_CHARS:
        return "short"
    return "eligible" if content_lines(text, _label_set(labels)) else "boilerplate_only"


def is_semantic_eligible(text: str, labels: Optional[Sequence[str]] = None) -> bool:
    """Whether a record's ``semantic_v3`` text may be embedded (the reference's ``is_eligible``).

    :param text: Output of :func:`semantic_text`.
    :param labels: Boilerplate labels (``None``: the frozen v3 list, :func:`load_boilerplate_labels`).
    :returns: False when the text is empty, shorter than :data:`MIN_CHARS`, or only a boilerplate label plus
        ``Language`` / ``Document Type`` / ``Date`` lines.
    :rtype: bool
    """
    if len(text) < MIN_CHARS:
        return False
    return bool(content_lines(text, _label_set(labels)))


# Mirrored 2026-10-10 from genizah_search evals/embedding_audit/semantic_text.py (branch eval-retrieval, uncommitted
# working copy, mtime 2026-10-09 16:36:44 EDT, file sha256 92caf87e6546338f...): the lines from "semantic_v4 begin" to
# "semantic_v4 end" are byte-identical to that copy's (block sha256 721e3db0c5a0612b..., tests/test_semantic_v4.py).
# =====================================================================================================================
# semantic_v4 begin. Everything from this line to the "semantic_v4 end" line is copied byte for byte into
# historical-document-analysis src/embeddings/semantic_text.py; parity_v4.py compares the two blocks and every record's
# output. The block reads only names both copies define above it (ALT_PREFIX, META_PREFIXES, MIN_CHARS,
# alt_title_parts, load_boilerplate_labels, semantic_text) plus the repo-specific BOILERPLATE_PATH / ADMIT_PATH.
# Changing it changes the embedded text: that is a new text version and a full re-embed, never an edit of one copy.
# =====================================================================================================================
TEXT_VERSION_SEMANTIC_V4 = "semantic_v4"
# the same layout over the production text rendered without the KTIV scholarly frames (the H0 "no frames" arm)
TEXT_VERSION_SEMANTIC_V4_NF = "semantic_v4_nf"
DESC_PREFIX = "Description:"
SUBJECTS_PREFIX = "Subjects:"
# Every field label GenizahDocument.create_text_representation() writes; a line starting with none of them continues
# the value of the line before it (a description that contained a newline).
FIELD_PREFIXES = ("Document ID:", "Shelf Mark:", DESC_PREFIX, ALT_PREFIX, SUBJECTS_PREFIX, "Title:", "Summary:",
                  "Language:", "Document Type:", "Date:", "Transcription:", "Translation:", "Related People:",
                  "Related Places:", "Physical:", "Additional Info:")
# never embedded by v4: catalogue keywords (the training labels' names) and the English keyword guess at the type
V4_DROP_PREFIXES = (SUBJECTS_PREFIX, "Document Type:")
TITLE_SEP = "; "
MAX_ALT_CHARS = 1000
TITLE_KEY_RE = re.compile(r"[\W_]+")
BOILERPLATE_SHA256 = "d57204f7278cc38bc713d54c05533fe87a40c79e3110cbbd51e42cb10e372677"
ADMIT_SHA256 = "2922b7747033c4d1d9b9845fee01b7c8fc0228b532d45c0e359db5786637ec01"


def text_sha256(text: str) -> str:
    """sha256 hex digest of the exact UTF-8 bytes handed to the embedder (index field ``embedded_text_sha256``).

    :param text: Embedded text.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file's bytes.

    :param path: File path.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str) -> Path:
    """Refuse a frozen data file whose bytes are not the pinned ones.

    :param path: Data file.
    :param expected_sha256: Pinned sha256 hex digest.
    :returns: ``path`` as a Path.
    :rtype: Path
    :raises ValueError: When the file's sha256 differs from the pin.
    """
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: sha256 {actual} is not the pinned {expected_sha256} (frozen text-function input)")
    return Path(path)


def v4_text_version(scholarly_frames: bool) -> str:
    """Text version name of a v4 variant.

    :param scholarly_frames: Whether the production text was rendered with the KTIV scholarly frames.
    :returns: :data:`TEXT_VERSION_SEMANTIC_V4` or :data:`TEXT_VERSION_SEMANTIC_V4_NF`.
    :rtype: str
    """
    return TEXT_VERSION_SEMANTIC_V4 if scholarly_frames else TEXT_VERSION_SEMANTIC_V4_NF


def fold_field_lines(text: str) -> str:
    """The text with every line that starts no field label appended (one space) to the line before it.

    Blank continuation lines are dropped; a first line that starts no label is kept as it is.

    :param text: A production or semantic text.
    :returns: The text with one line per field.
    :rtype: str
    """
    out: List[str] = []
    for line in text.split("\n"):
        if out and not line.startswith(FIELD_PREFIXES):
            if line.strip():
                out[-1] = f"{out[-1].rstrip()} {line.strip()}"
        else:
            out.append(line)
    return "\n".join(out)


def title_key(text: str) -> str:
    """Comparison key of a title: case-folded, every run of punctuation / brackets / underscores one space.

    :param text: A title, an alternate-title part or a description.
    :returns: Key (words separated by single spaces).
    :rtype: str
    """
    return " ".join(TITLE_KEY_RE.sub(" ", text.casefold()).split())


def title_parts(lines: Sequence[str]) -> List[str]:
    """The non-empty ``;`` parts of every ``Alternate titles:`` line, in text order.

    :param lines: Lines of a folded text (:func:`fold_field_lines`).
    :returns: Parts, stripped, duplicates kept.
    :rtype: List[str]
    """
    parts: List[str] = []
    for line in lines:
        if line.startswith(ALT_PREFIX):
            parts.extend(p.strip() for p in line[len(ALT_PREFIX):].split(";") if p.strip())
    return parts


def merge_title_parts(description: str, parts: Sequence[str]) -> Tuple[List[str], int, int]:
    """Choose the alternate-title parts appended to a description.

    A part is left out when its :func:`title_key` occurs, as whole words, in the description's key or in an earlier
    kept part's key (or is empty); appending stops before the first part that would take the appended characters
    past :data:`MAX_ALT_CHARS` (the first kept part is always kept whole).

    :param description: The ``Description:`` value (may be empty).
    :param parts: Candidate parts in order (:func:`title_parts`).
    :returns: (kept parts in order, parts left out as duplicates, parts left out by the cap).
    :rtype: Tuple[List[str], int, int]
    """
    keys = [f" {title_key(description)} "] if description else []
    kept: List[str] = []
    added = duplicates = 0
    for i, part in enumerate(parts):
        key = title_key(part)
        if not key or any(f" {key} " in seen for seen in keys):
            duplicates += 1
            continue
        if kept and added + len(part) > MAX_ALT_CHARS:
            return kept, duplicates, len(parts) - i
        kept.append(part)
        added += len(part) + len(TITLE_SEP)
        keys.append(f" {key} ")
    return kept, duplicates, 0


def merge_title_lines(lines: List[str]) -> List[str]:
    """Fold the ``Alternate titles:`` parts into the ``Description:`` line (one catalogue line under one label).

    The merged line takes the description's place, or the first ``Alternate titles:`` line's place when the record
    has no description; every other line keeps its order.

    :param lines: Lines of a folded text.
    :returns: Lines with no ``Alternate titles:`` line.
    :rtype: List[str]
    """
    alt_at = {i for i, line in enumerate(lines) if line.startswith(ALT_PREFIX)}
    if not alt_at:
        return list(lines)
    desc_i = next((i for i, line in enumerate(lines) if line.startswith(DESC_PREFIX)), None)
    description = lines[desc_i][len(DESC_PREFIX):].strip() if desc_i is not None else ""
    kept = merge_title_parts(description, title_parts(lines))[0]
    value = TITLE_SEP.join(([description] if description else []) + kept)
    merged_at = desc_i if desc_i is not None else min(alt_at)
    out: List[str] = []
    for i, line in enumerate(lines):
        if i == merged_at:
            if value:
                out.append(f"{DESC_PREFIX} {value}")
        elif i not in alt_at:
            out.append(line)
    return out


def semantic_v4_text(v3_text: str) -> str:
    """``semantic_v4`` text of a record from its ``semantic_v3`` text.

    Folds continuation lines, drops the :data:`V4_DROP_PREFIXES` lines, then folds the alternate titles into the
    description (:func:`merge_title_lines`).

    :param v3_text: Output of ``semantic_text()``.
    :returns: Embedded text (may be empty).
    :rtype: str
    """
    lines = [line for line in fold_field_lines(v3_text).split("\n") if not line.startswith(V4_DROP_PREFIXES)]
    return "\n".join(merge_title_lines(lines)).strip()


@lru_cache(maxsize=None)
def admit_pattern(path: Path = ADMIT_PATH) -> Pattern[str]:
    """Frozen festival / topic patterns that admit a short record (``eligibility_admit_v4.json``).

    :param path: Admission file: ``{"festival_terms": {topic: regex}, "topic_rules": {topic: regex},
        "flags": {"festival_terms": "i", "topic_rules": ""}, ...}``.
    :returns: One compiled alternation (each block with its own case flag).
    :rtype: Pattern[str]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: List[str] = []
    for block in ("festival_terms", "topic_rules"):
        opener = "(?i:" if "i" in data["flags"][block] else "(?:"
        groups.extend(f"{opener}{pattern})" for _, pattern in sorted(data[block].items()))
    return re.compile("|".join(groups))


@lru_cache(maxsize=None)
def frozen_v4_labels() -> FrozenSet[str]:
    """The frozen boilerplate labels, after checking the file against :data:`BOILERPLATE_SHA256`.

    :returns: The 64 labels.
    :rtype: FrozenSet[str]
    """
    return load_boilerplate_labels(verify_frozen_file(BOILERPLATE_PATH, BOILERPLATE_SHA256))


@lru_cache(maxsize=None)
def frozen_admit_pattern() -> Pattern[str]:
    """The frozen admission pattern, after checking the file against :data:`ADMIT_SHA256`.

    :returns: Compiled pattern (:func:`admit_pattern`).
    :rtype: Pattern[str]
    """
    return admit_pattern(verify_frozen_file(ADMIT_PATH, ADMIT_SHA256))


def v4_measured_text(v3_text: str) -> str:
    """The part of a ``semantic_v3`` text that decides v4 eligibility: every line but the ``Subjects:`` lines (and
    their continuation lines), otherwise unchanged.

    :param v3_text: Output of ``semantic_text()``.
    :returns: The measured text, stripped (equal to ``v3_text`` when it has no ``Subjects:`` line).
    :rtype: str
    """
    out: List[str] = []
    skip = False
    for line in v3_text.split("\n"):
        if line.startswith(FIELD_PREFIXES):
            skip = line.startswith(SUBJECTS_PREFIX)
        if not skip:
            out.append(line)
    return "\n".join(out).strip()


def unbracketed(part: str) -> str:
    """An alternate-title part without square brackets, for the boilerplate comparison (a KTIV frame brackets its
    work, ``[טוביה בן אליעזר, לקח טוב]: ...``; the frozen list holds the same identification without them).

    :param part: One ``;`` part.
    :returns: The part with ``[`` and ``]`` removed, stripped.
    :rtype: str
    """
    return part.replace("[", "").replace("]", "").strip()


def content_lines_v4(measured: str, boiler: AbstractSet[str]) -> List[str]:
    """Lines that say something beyond boilerplate labels and metadata: the v3 rule, ``Subjects:`` never counts and
    alternate-title parts are compared without square brackets.

    :param measured: Output of :func:`v4_measured_text`.
    :param boiler: Frozen boilerplate labels.
    :returns: Content lines.
    :rtype: List[str]
    """
    out: List[str] = []
    for line in measured.split("\n"):
        if not line.strip() or line.startswith(META_PREFIXES + (SUBJECTS_PREFIX,)):
            continue
        if line.startswith(DESC_PREFIX) and line[len(DESC_PREFIX):].strip() in boiler:
            continue
        if line.startswith(ALT_PREFIX) and {unbracketed(p) for p in alt_title_parts(line)} <= boiler:
            continue
        out.append(line)
    return out


def admission_value(content: Sequence[str]) -> str:
    """The values of content lines (labels removed), joined, for the admission test.

    :param content: Output of :func:`content_lines_v4`.
    :returns: ``" ; "``-joined values.
    :rtype: str
    """
    return " ; ".join(line.split(":", 1)[1].strip() if line.startswith(FIELD_PREFIXES) else line.strip()
                      for line in content)


def semantic_v4_ineligible_reason(v3_text: str, labels: Optional[AbstractSet[str]] = None,
                                  admit: Optional[Pattern[str]] = None) -> str:
    """``empty`` / ``short`` / ``boilerplate_only`` / ``eligible`` for a record's ``semantic_v3`` text.

    The v3 rule on :func:`v4_measured_text` (:data:`MIN_CHARS`, then at least one content line), plus one admission:
    a short text whose content names a festival or a topic (:func:`admit_pattern`) is eligible.

    :param v3_text: Output of ``semantic_text()``.
    :param labels: Boilerplate labels (``None``: the frozen list, :func:`frozen_v4_labels`).
    :param admit: Admission pattern (``None``: the frozen one, :func:`frozen_admit_pattern`).
    :returns: Reason.
    :rtype: str
    """
    boiler = frozen_v4_labels() if labels is None else labels
    pattern = frozen_admit_pattern() if admit is None else admit
    measured = v4_measured_text(v3_text)
    if not measured:
        return "empty"
    content = content_lines_v4(measured, boiler)
    if len(measured) < MIN_CHARS:
        value = admission_value(content)
        return "eligible" if value and pattern.search(value) else "short"
    return "eligible" if content else "boilerplate_only"


def semantic_v4_record(v8_text: str) -> Tuple[str, str, str]:
    """The v4 text of one record from its production (``v8``) text, its reason, and its v3 text.

    :param v8_text: ``GenizahDocument.create_text_representation()`` output (with or without scholarly frames).
    :returns: (v4 text, :func:`semantic_v4_ineligible_reason` with the frozen inputs, v3 text).
    :rtype: Tuple[str, str, str]
    """
    v3_text = semantic_text(v8_text)
    return semantic_v4_text(v3_text), semantic_v4_ineligible_reason(v3_text), v3_text
# semantic_v4 end
# =====================================================================================================================


def bibliography_semantic_text(text: str) -> str:
    """``semantic_v3`` text of a bibliography page: the production text without its ``Source PDF:`` line.

    Keeps the ``Shelf marks mentioned:`` line and every inline shelf mark (the bridge from scholarship to fragments)
    and changes nothing else; only the leading file-name line that ``BibliographyDocument`` writes is dropped.

    :param text: Production text of one page (``BibliographyDocument.create_text_representation()``, which is also
        the stored ``full_text_content``).
    :returns: The page text without the file-name line (empty when nothing else is left).
    :rtype: str
    """
    first, _, rest = text.partition("\n")
    if first.startswith(BIBLIOGRAPHY_DROP_PREFIX):
        return rest
    return text
