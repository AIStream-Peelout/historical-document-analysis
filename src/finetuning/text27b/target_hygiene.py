# File name: target_hygiene.py
# Date: 10/26/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Target hygiene of the t27b prep pass (2026-10-26): analysis targets that cite evidence the input lacks.

An ``analyse_document`` target is an editor's introduction or commentary. Its author held the manuscript (or a
photograph), the rest of the book and its bibliography; the row's input is only the document text (+ the catalogue
context, sometimes the book's translation). Sentences that rest on that unavailable evidence teach the model to
invent it, so :func:`strip_unavailable_evidence` removes, sentence by sentence (``textnorm.sentence_spans``):

``physical``
    physical description: dimensions (``25 x 8 cm``, ``1 x 33 inches``), writing material (paper, vellum,
    parchment), ink, the hand / handwriting / script style (``cursive Hebrew characters``, ``the same hand``),
    layout (columns, margins, written in several directions or upside down, a blank side, numbered lines on a
    side), the object's form and condition (strip, leaf, sheet, folio, torn / cut / tapering / mutilated).
``cross_ref``
    cross-references to other numbered documents, plates or figures of the book that are not in the input
    (``No. XXIII``, ``Nos. XLI and XLVIII``, ``see doc. 12``, ``pl. IV``, ``fig. 3``, ``Document 7``, ``above, p. 45``,
    Hebrew ``תעודה 12`` / ``מס' 12`` / ``לוח 3``).
``biblio``
    bibliographic apparatus left after the B4 citation strip: page / volume references, ``ibid.``, ``op. cit.``,
    ``s.v.``, journal abbreviations, ``ed.`` with a year, author-year citations, Hebrew ``עמ' 12``.

Uncertainty and alternative readings stay: a sentence that discusses a reading (``perhaps``, ``uncertain``,
``may be read``, ``(?)``, ``or rather``, Hebrew ``אולי`` / ``ספק``) is kept when its only trigger is a *soft* one
(layout / form words such as margin, blank, leaf, torn); hard triggers (dimensions, material, ink, hand, numbered
cross-references, bibliographic references) remove it. A target left shorter than :data:`MIN_CHARS_AFTER` (120)
characters is dropped by the callers (:func:`builder.analysis_evidence_strip`).

The audit helper (:func:`audit_family_files`, CLI ``audit``) compares the ``analyse_document`` rows of a family file
before and after a rebuild and writes before/after counts plus a sample of before/after pairs.

Public API: :data:`MIN_CHARS_AFTER`, :data:`OP_PREFIX`, :data:`REASONS`, :func:`sentence_reasons`, :func:`decide`, :func:`units`,
:func:`is_uncertainty`, :func:`strip_unavailable_evidence`, :func:`analyse_rows_by_id`, :func:`audit_family_files`,
:func:`main`.

CLI::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.text27b.target_hygiene audit --before DIR --after DIR
        [--files book_pairs analyse_document_v2] [--n 10] [--out audit/analysis_strip_sample.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows
from src.finetuning.text27b import textnorm as tn

LOG = logging.getLogger(__name__)

#: A stripped analysis target shorter than this is dropped (prep pass 2026-10-26).
MIN_CHARS_AFTER = 120
#: Prefix of the strip ops recorded in ``meta.strip_ops``.
OP_PREFIX = "E"
#: Reasons in the order they are reported.
REASONS: Tuple[str, ...] = ("physical", "cross_ref", "biblio")
ANALYSE_FAMILY = "analyse_document"

_I = re.I
_NUM = r"(?:\d+(?:[.,]\d+)?|[IVX]{1,4})"
_ROMAN_OR_NUM = r"(?:[IVXLC]{1,7}|\d{1,4}[a-z]?)\b"

# ------------------------------------------------------------------------------------------ physical (hard)
_PHYS_HARD: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("dimensions", re.compile(r"\b\d+(?:[.,]\d+)?\s*[½¼¾⅛⅜⅝⅞]?\s*(?:cm|mm|centimet(?:er|re)s?|millimet(?:er|re)s?|"
                              r"inch(?:es)?|in\.)\s*(?:[x×X]|by)\s*\d|"
                              r"\b(?:about|ca\.|c\.)?\s*[\dIl]+(?:[.,]\d+)?\s*[½¼¾⅛⅜⅝⅞]?\s*[x×X]\s*\d+(?:[.,]\d+)?\s*"
                              r"[½¼¾⅛⅜⅝⅞]?\s*(?:cm|mm|inch(?:es)?|in\.)|"
                              r"\b\d+(?:[.,]\d+)?\s*(?:cm|mm|centimet(?:er|re)s?|millimet(?:er|re)s?|inch(?:es)?)\b|"
                              r"\b(?:measures|measuring|measurements?|dimensions)\b|"
                              r"\d+(?:[.,]\d+)?\s*(?:ס[\"״׳']?מ|מ[\"״׳']?מ)", _I)),
    ("material", re.compile(r"(?:^|(?<=[.;:(\]]\s))(?:Paper|Parchment|Vellum)\b|(?<=\s)(?:Paper|Parchment|Vellum)\s*[.;]|"
                            r"\b(?:on|in)\s+(?:\w+\s+)?(?:paper|vellum|parchment)\b(?!\s+(?:trade|money|merchants?|"
                            r"makers?|making|mills?|industry|prices?))|"
                            r"\b(?:strip|piece|sheet|leaf|side|bottom|top|edge|back|half)s?\s+of\s+(?:the\s+)?(?:\w+\s+)?"
                            r"(?:paper|vellum|parchment)\b|"
                            r"\b(?:the|this|its)\s+(?:paper|vellum|parchment)\s+(?:is|was|has|had|which|of)\b|"
                            r"\b(?:paper|vellum|parchment)\s+(?:folded|cut|strip|leaf|sheet|fragment|document|deed)\b|"
                            r"(?:על\s+|ב)?(?:נייר|קלף)(?=[\s,.;]|$)", _I)),
    ("ink", re.compile(r"\binks?\b(?=\s+(?:is|was|has|had|are|were|transfer|border\S*|blots?|stains?|traces?|"
                       r"colou?r|lines?)\b)|"
                       r"\b(?:the|its|in|with|black|brown|red|pale|faded|dark|gummy|thick|thin|different|same|another)\s+"
                       r"(?:\w+\s+)?inks?\b(?!\s*,)|\bבדיו\b|\bהדיו\b", _I)),
    ("hand", re.compile(r"\bhandwritings?\b|"
                        r"\b(?:same|similar|different|another|second|third|later|two\s+other|other\s+two|scribe'?s|scribal|his own|her own|own|"
                        r"neat|clear|beautiful|good|fine|elegant|cursive|careless|hasty|unpractised|unpracticed|"
                        r"practised|practiced|large|small|bold|square|semi-?cursive|calligraphic|formal|informal|"
                        r"distinctive|characteristic|typical|clumsy|crude|regular|irregular|careful|skilled|unskilled|"
                        r"rapid|flowing|legible|crabbed|cramped|book|documentary|chancery|Maghrib[iī]|Oriental|"
                        r"Sephardi(?:c)?|Spanish|Byzantine|Yemenite|Iraqi|Palestinian|Egyptian|Persian)\s+"
                        r"(?:\w+\s+)?hands?\b(?!\s+(?:over|of\s+(?:god|the\s+lord)))|"
                        r"\b(?:written|penned|copied|signed)\s+(?:\w+\s+){0,3}(?:in|by)\s+(?:the|a|an|his|her)\s+"
                        r"(?:\w+\s+){0,2}hand\b|"
                        r"\bthe\s+hand\s+(?:is|was|seems|appears|resembles|looks)\b|"
                        r"\b(?:Hebrew|Arabic|Judaeo-Arabic|Coptic|Syriac)\s+hand\b|"
                        r"\b(?:cursive|semi-?cursive|square|rabbinic|calligraphic|Maghrib[iī]|Oriental|Sephardi(?:c)?|"
                        r"Ashkenazi(?:c)?|Byzantine|Yemenite|Persian|Iraqi|Palestinian|Egyptian|Spanish|"
                        r"North[- ]African)\s+(?:\w+\s+)?(?:script|characters|letters|writing|lettering)\b|"
                        r"\b(?:script|characters|letters|writing)\s+(?:is|are|was|were)\s+(?:small|large|neat|cursive|"
                        r"distinct|clear|regular|irregular|careless|beautiful)\b|"
                        r"\bכתב[-־\s](?:ידו|ידה|ידם|יד\s+(?:אחר|אחד|שונה|זהה))\b|\bבכתב[-־\s](?:ידו|ידה|יד)\b|"
                        r"\bכתיבה\s+(?:מרובעת|רהוטה|בינונית|מזרחית|ספרדית|מערבית)\b|"
                        r"\bכתב\s+(?:מרובע|רהוט|בינוני|מזרחי|ספרדי|מערבי|מגרבי)\b", _I)),
)
# ------------------------------------------------------------------------------------------ physical (soft)
_PHYS_SOFT: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("layout", re.compile(r"\bmargins?\b|\bmarginal(?:ia)?\b|"
                          r"\b(?:in|into)\s+(?:two|three|four|\d)\s+columns\b|\bcolumns?\s+(?:of|on)\s+the\s+"
                          r"(?:page|leaf|sheet|recto|verso)\b|\b(?:right|left|first|second|third|outer|inner)\s+column\b|"
                          r"\bwritten\s+(?:in\s+(?:several|different|two|three|various|all)\s+directions|"
                          r"upside[\s-]down|at\s+right\s+angles|across|diagonally|vertically|perpendicular\w*|sideways)|"
                          r"\bupside[\s-]down\b|\bat\s+right\s+angles\b|\bdiagonally\b|"
                          r"\bin\s+(?:several|different|various)\s+directions\b|"
                          r"\b(?:recto|verso|reverse|obverse|back|other\s+side|second\s+side)\s+(?:is|are|was|being|"
                          r"has\s+been|remains?)\s+(?:\w+\s+)?(?:blank|empty|left\s+blank|unused|uninscribed)\b|"
                          r"\bblank\b|\buninscribed\b|"
                          r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|"
                          r"thirty|forty|fifty)(?:\s+\w+)?\s+lines\s+(?:of\s+(?:writing|text|script)|on\s+the\s+"
                          r"(?:recto|verso|front|back)|are\s+(?:preserved|extant|written)|remain|survive)\b|"
                          r"\(\s*\d+\s+lines\b|"
                          r"\bשוליים\b|\bבשול(?:י|יים)\b|\bעמודות\b|\b(?:הצד|הגב|העמוד)\s+(?:\S+\s+)?ריק\b", _I)),
    ("form", re.compile(r"\b(?:a|an|the|this|that|one|two|three|four|both|single|double|second|present|main)\s+"
                        r"(?:(?:narrow|long|small|large|conjoined|torn|damaged|irregularly\s+shaped|paper|parchment|"
                        r"vellum|single|double|first|second|last|whole|entire)\s+)?(?:strip|leaf|leaves|sheet|sheets|"
                        r"folio|folios|bifolium|bifolio|quire)\b(?!\s+(?:the|for|on|at|in|with|to|from|his|her|their|"
                        r"its|after|before|when|and\s+(?:goes|comes|travels)))|"
                        r"\b(?:tapering|torn|trimmed|mutilated|worm-?eaten|stained|faded|rubbed|crumpled|creased|"
                        r"folded|water-?damaged|effaced)\b|"
                        r"\bcut\s+(?:from|out\s+of)\s+(?:a|an|the)\s+(?:\w+\s+)?(?:letter|document|leaf|sheet|page|"
                        r"book|scroll|codex)\b|\b(?:has\s+been|have\s+been|was|were)\s+(?:cut|trimmed|torn)\b|"
                        r"\bwatermark\w*\b|\bruled\b|\bruling\b|\bcatchwords?\b|\bstitch\w*\b|\bconservation\w*\b|"
                        r"\bקרוע(?:ה|ים|ות)?\b|\bפגום(?:ה|ים|ות)?\b|\bמטושטש(?:ת|ים|ות)?\b|\bמחוק(?:ה|ים|ות)?\b", _I)),
)
#: A sentence longer than this many words is not stripped on a soft trigger alone (an argument that mentions a margin
#: or a torn letter in passing is not a physical description).
SOFT_MAX_WORDS = 30
# ------------------------------------------------------------------------------------------ cross references
_CROSS: "re.Pattern[str]" = re.compile(
    r"(?<!\bMS\s)(?<!\bMS)\b[Nn]os?\.\s*" + _ROMAN_OR_NUM + r"|"
    r"\b(?:letters?|documents?|docs?|texts?)\s+[A-Z]\.\d+(?:\.[a-z])?\b|"
    r"\b(?:docs?|documents?|texts?|deeds?|letters?|plates?|pls?|figs?|figures?|appendix|chap|chapter|tables?)"
    r"\.?\s*(?:no\.?\s*)?(?:[IVXLC]{1,7}(?:\.\d+(?:\.[a-z])?)?\b|\d{1,4}[a-z]?\b)(?!\s*(?:dinars?|dirhams?|"
    r"qirats?|years?|days?|months?|lines?|words?))|"
    r"\bsee\s+(?:further\s+|also\s+)?pages?\s+\d|"
    r"\b(?:above|below|supra|infra)\s*,?\s*(?:pp?\.|n\.|nos?\.|notes?|chap)\s*\d|"
    r"\bour\s+(?:No|Document|Text|Plate|Fig)\b|"
    r"(?:תעוד(?:ה|ות)|מסמכ(?:ים)?|מסמך|מס['׳]|סימן|לוח(?:ות)?|איור|נספח|פרק)\s*(?:\d{1,4}|[א-ת]['׳\"״])|"
    r"(?:לעיל|להלן)\s*,?\s*(?:עמ['׳]|מס['׳]|הערה|תעודה|סימן|פרק)",
    re.UNICODE)
#: Capitalised words that make ``<Word> 12`` a document reference only with an explicit number marker.
_CROSS_CASE_SENSITIVE: "re.Pattern[str]" = re.compile(r"\b(?:Document|Text|Deed|Letter|Plate|Figure|Fig|Appendix)\s+"
                                                      r"(?:No\.?\s*)?(?:[IVXLC]{1,7}(?:\.\d+(?:\.[a-z])?)?|\d{1,4}[a-z]?)\b")
# ------------------------------------------------------------------------------------------ bibliography
_JOURNALS = (r"(?i:Mediterranean\s+Society)|JQR|REJ|ZDMG|JAOS|BSOAS|HUCA|PAAJR|JNES|JSS|JJS|JSAI|AJS\s+Review|Tarbiz|Tarbits|Zion|Sefunot|"
             r"Shalem|Pe['’]amim|Kiryat\s+Sefer|Qiryat\s+Sefer|Med\.?\s*Soc|EI2|EJ2|Encyclop(?:a|ae|e)dia|"
             r"Jewish\s+Quarterly\s+Review|Revue\s+des\s+[ÉE]tudes\s+Juives|Bulletin\s+of\s+the\s+School|Speculum|"
             r"Islamic\s+Law\s+and\s+Society|Studia\s+Islamica")
_BIBLIO: "re.Pattern[str]" = re.compile(
    r"\bpp?\.\s*\d|\bvols?\.\s*[\dIVX]|\bibid\b|\bop\.?\s*cit\b|\bloc\.?\s*cit\b|\bs\.\s?v\.|"
    r"\b(?:eds?|edd)\.\s+(?:[A-Z]|by\b)|\(\s*(?:ed|eds)\.\s*\)|"
    r"\b(?:" + _JOURNALS + r")\b|"
    r"\b[A-Z][A-Za-zÀ-ž'’-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-zÀ-ž'’-]+)?\s*,?\s*\(?(?:1[6-9]|20)\d{2}[a-z]?\)?\s*[,:]\s*"
    r"(?:pp?\.\s*)?\d|"
    r"\((?:[^()]{0,60}[,:]\s*)?(?:1[6-9]|20)\d{2}[a-z]?\)\s*[,:.]?\s*(?:pp?\.\s*)?\d|"
    r"\bעמ['׳]\s*\d|\bשם\s*,\s*עמ['׳]|\bכרך\s+[\dא-ת]")
# ------------------------------------------------------------------------------------------ uncertainty
_UNCERTAIN: "re.Pattern[str]" = re.compile(
    r"\bperhaps\b|\bpossibl[ey]\b|\bprobabl[ey]\b|\buncertain\b|\bdoubtful\b|\bunclear\b|\bobscure\b|"
    r"\bmay\s+(?:also\s+)?(?:be|have\s+been)\s+read\b|\bcould\s+(?:also\s+)?be\s+read\b|\bto\s+be\s+read\b|"
    r"\bread(?:s|ing|ings)?\b|\balternative(?:ly)?\b|\bor\s+rather\b|\bconjectur\w*|\bemend\w*|\bseems?\s+to\b|"
    r"\b(?:un)?decipher\w*|\billegible\b|\b(?:not\s+)?visible\b|\bunable\s+to\s+(?:read|decipher)\b|"
    r"\bcannot\s+be\s+(?:read|deciphered)\b|\(\?\)|אולי|ספק|נראה|קריא|לקרוא|מסופק|משובש|השערה", _I)


#: Idioms that contain "hand" without describing handwriting (masked before the hand rule runs).
_HAND_IDIOMS: "re.Pattern[str]" = re.compile(r"\bon\s+the\s+(?:other|one)\s+hand\b|\bat\s+hand\b|\bhand\s+in\s+hand\b|"
                                             r"\b(?:right|left)[\s-]hand\s+(?:side|corner|margin)\b", _I)
#: ``in / from the hand of <Name | someone>`` names the writer's hand -- unless the sentence is about sending something
#: "by the hand of" a messenger (the Hebrew idiom be-yad).
_HAND_OF: "re.Pattern[str]" = re.compile(r"\b(?:in|from)\s+the\s+hand\s+of\s+(?:someone|somebody|an?\s|one\s|"
                                         r"(?-i:[A-ZĀĒĪŌŪḤĤŠṢṬẒʿ‘]))", _I)
_SENT_BY: "re.Pattern[str]" = re.compile(r"\b(?:sent|send|sends|sending|dispatched|delivered|forwarded|carried|brought|"
                                         r"entrusted|given)\b", _I)
#: The sentence points into the document text itself (line references; in a Latin-script sentence, quoted Hebrew /
#: Arabic words): a layout or form word in it describes the edition the input carries, so soft triggers do not fire.
_TEXT_POINTER: "re.Pattern[str]" = re.compile(r"\bll?\.\s*\d|\blines?\s+\d|\bline\s+\d|\bשורו?ת\s+\d")
_RTL_WORD: "re.Pattern[str]" = re.compile("[\u0590-\u05FF\u0600-\u06FF]{2,}")
_RTL_CHAR: "re.Pattern[str]" = re.compile("[\u0590-\u05FF\u0600-\u06FF]")
_LAT_CHAR: "re.Pattern[str]" = re.compile("[A-Za-z\u00c0-\u017e]")
#: A title followed by its date in brackets, then the next sentence (``Petition to ... (Sixth century A.H./...) The
#: leaf ...``): the bracket closes a heading and starts a new unit.
_HEADING_BREAK: "re.Pattern[str]" = re.compile(r"\((?:[^()]{0,80}(?:A\.\s?H\.|A\.\s?D\.|century|\bCE\b|\bAH\b)[^()]{0,40})\)"
                                               r"[ \t]+(?=[A-Z])")


def _match_names(text: str, table: Sequence[Tuple[str, "re.Pattern[str]"]]) -> List[str]:
    """Names of the patterns in ``table`` that match ``text``.

    :param text: Sentence.
    :param table: ``(name, pattern)`` pairs.
    :returns: Matching names in table order.
    """
    return [name for name, pat in table if pat.search(text)]


def is_uncertainty(sentence: str) -> bool:
    """Whether a sentence discusses an uncertain or alternative reading (kept against soft triggers).

    :param sentence: Sentence text.
    :returns: True when it carries uncertainty / alternative-reading language.
    """
    return bool(_UNCERTAIN.search(sentence))


def sentence_reasons(sentence: str) -> Dict[str, List[str]]:
    """Strip triggers of one sentence, by reason.

    :param sentence: Sentence text.
    :returns: ``{"physical": [...], "cross_ref": [...], "biblio": [...], "soft": [...]}`` -- ``physical`` holds the
        hard physical triggers (dimensions / material / ink / hand), ``soft`` the layout / form triggers; empty lists
        mean the sentence is clean.
    """
    masked = _HAND_IDIOMS.sub(lambda m: " " * len(m.group(0)), sentence)
    out: Dict[str, List[str]] = {"physical": _match_names(masked, _PHYS_HARD), "cross_ref": [], "biblio": [],
                                 "soft": _match_names(masked, _PHYS_SOFT)}
    if "hand" not in out["physical"] and _HAND_OF.search(masked) and not _SENT_BY.search(masked):
        out["physical"].append("hand")
    if _CROSS.search(sentence) or _CROSS_CASE_SENSITIVE.search(sentence):
        out["cross_ref"].append("numbered_reference")
    if _BIBLIO.search(sentence):
        out["biblio"].append("bibliographic")
    return out


def decide(sentence: str) -> Optional[str]:
    """Strip reason of one sentence or clause (None = keep).

    Hard triggers (numbered cross-reference, dimensions / material / ink / hand, bibliographic reference) always
    remove the unit. A soft trigger (layout, form) removes it only when the unit has no uncertainty / alternative-
    reading language, does not point into the document text (:func:`points_into_text`) and is at most
    :data:`SOFT_MAX_WORDS` words long.

    :param sentence: Sentence or clause text.
    :returns: ``physical`` / ``cross_ref`` / ``biblio`` or None.
    """
    r = sentence_reasons(sentence)
    if r["cross_ref"]:
        return "cross_ref"
    if r["physical"]:
        return "physical"
    if r["biblio"]:
        return "biblio"
    if r["soft"] and not is_uncertainty(sentence) and not points_into_text(sentence) \
            and len(sentence.split()) <= SOFT_MAX_WORDS:
        return "physical"
    return None


def points_into_text(sentence: str) -> bool:
    """Whether a sentence points into the document text: a line reference, or (in a Latin-script sentence) a quoted
    Hebrew / Arabic word. A Hebrew-language sentence is not a quotation of the text just because it is in Hebrew.

    :param sentence: Sentence text.
    :returns: True for pointers into the edition.
    """
    if _TEXT_POINTER.search(sentence):
        return True
    latin = len(_LAT_CHAR.findall(sentence)) > len(_RTL_CHAR.findall(sentence))
    return latin and bool(_RTL_WORD.search(sentence))


def units(text: str) -> List[Tuple[int, int]]:
    """Strip units of a text: sentences (:func:`textnorm.sentence_spans`), split after a dated heading in brackets
    (``Petition to ... (Sixth century A.H./...) The leaf ...`` is two units). The strip is sentence-level: a sentence
    goes whole, so no clause is left dangling.

    :param text: Text.
    :returns: ``(start, end)`` spans in text order.
    """
    spans: List[Tuple[int, int]] = []
    for a, b in tn.sentence_spans(text):
        cuts = [a] + [a + m.end() for m in _HEADING_BREAK.finditer(text[a:b])] + [b]
        for x, y in zip(cuts, cuts[1:]):
            seg = text[x:y]
            if not seg.strip():
                continue
            spans.append((x + (len(seg) - len(seg.lstrip())), x + len(seg.rstrip())))
    return [(x, y) for x, y in spans if text[x:y].strip()]


def strip_unavailable_evidence(text: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Remove the sentences of an analysis target that rest on evidence the row's input lacks.

    :param text: Analysis target (after the B3 / B4 strips).
    :returns: ``(text, ops, removed)``: the stripped text, the ops (``E:physical`` / ``E:cross_ref`` / ``E:biblio``,
        each once, in :data:`REASONS` order) and the removed ``(reason, unit)`` pairs in text order.
    """
    if not text:
        return text, [], []
    cut: List[Tuple[int, int]] = []
    removed: List[Tuple[str, str]] = []
    for a, b in units(text):
        why = decide(text[a:b])
        if why:
            cut.append((a, b))
            removed.append((why, text[a:b]))
    if not cut:
        return text, [], []
    out = tn._delete_spans(text, cut)
    seen = {w for w, _ in removed}
    ops = [f"{OP_PREFIX}:{r}" for r in REASONS if r in seen]
    return out, ops, removed


# ============================================================================================ audit
def analyse_rows_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    """``analyse_document`` rows of a family file by row id.

    :param path: Family JSONL file (missing → empty).
    :returns: ``{row_id: row}``.
    """
    if not Path(path).exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in _rows.read_jsonl(path):
        meta = row.get("meta") or {}
        if (meta.get("family") or meta.get("task")) == ANALYSE_FAMILY:
            out[str(meta.get("row_id"))] = row
    return out


def _assistant(row: Mapping[str, Any]) -> str:
    """Assistant content of a row.

    :param row: Row.
    :returns: Content ('' when absent).
    """
    return next((str(m.get("content") or "") for m in row.get("messages") or [] if m.get("role") == "assistant"), "")


def _block(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Row / character / token totals of a set of analysis rows, overall and for train.

    :param rows: Rows.
    :returns: ``{rows, chars, target_tokens, train_rows, train_chars, train_target_tokens}``.
    """
    c: collections.Counter = collections.Counter()
    for r in rows:
        m = r.get("meta") or {}
        n = len(_assistant(r))
        tt = int(m.get("target_tokens") or 0)
        c["rows"] += 1
        c["chars"] += n
        c["target_tokens"] += tt
        if m.get("split") == "train":
            c["train_rows"] += 1
            c["train_chars"] += n
            c["train_target_tokens"] += tt
    return dict(c)


def audit_family_files(before_dir: Path, after_dir: Path, stems: Sequence[str], n: int = 10,
                       seed: int = 3407, include: Sequence[str] = ()) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compare the analysis rows of family files before and after the evidence strip.

    :param before_dir: Families directory of the previous build.
    :param after_dir: Families directory of the new build.
    :param stems: Family file stems holding ``analyse_document`` rows.
    :param n: Before/after examples to sample (rows whose target changed, seeded, one per document first).
    :param seed: Sampling seed.
    :param include: Row ids always shown first (e.g. the rows a reviewer named), when they changed.
    :returns: ``(report, examples)``; the report has per-file before / after totals, dropped / changed / unchanged / new
        row counts, removed sentences by reason (from ``meta.evidence_strip``) and ops counts; each example carries the
        before / after targets and the units the strip removes from the before text.
    """
    report: Dict[str, Any] = {"files": {}, "min_chars_after": MIN_CHARS_AFTER}
    changed_pool: List[Dict[str, Any]] = []
    for stem in stems:
        before = analyse_rows_by_id(Path(before_dir) / f"{stem}.jsonl")
        after = analyse_rows_by_id(Path(after_dir) / f"{stem}.jsonl")
        removed_by_reason: collections.Counter = collections.Counter()
        ops: collections.Counter = collections.Counter()
        status: collections.Counter = collections.Counter()
        for rid, row in after.items():
            ev = (row.get("meta") or {}).get("evidence_strip") or {}
            for reason, k in (ev.get("removed_by_reason") or {}).items():
                removed_by_reason[reason] += int(k)
            ops.update(o for o in (row.get("meta") or {}).get("strip_ops") or [] if o.startswith(OP_PREFIX + ":"))
        for rid, row in before.items():
            if rid not in after:
                status["dropped"] += 1
                continue
            a, b = _assistant(row), _assistant(after[rid])
            if a == b:
                status["unchanged"] += 1
                continue
            status["changed"] += 1
            changed_pool.append({"file": stem, "row_id": rid, "split": after[rid]["meta"].get("split"),
                                 "doc": after[rid]["meta"].get("doc_id") or after[rid]["meta"].get("pgpid"),
                                 "source": after[rid]["meta"].get("source"), "before": a, "after": b,
                                 "removed": [{"reason": w, "text": t} for w, t in strip_unavailable_evidence(a)[2]],
                                 "ops": [o for o in after[rid]["meta"].get("strip_ops") or []
                                         if o.startswith(OP_PREFIX + ":")]})
        status["new"] = sum(1 for rid in after if rid not in before)
        report["files"][stem] = {"before": _block(before.values()), "after": _block(after.values()),
                                 "rows": dict(status), "removed_sentences_by_reason": dict(removed_by_reason),
                                 "ops": dict(ops)}
    key = lambda e: hashlib.sha1(f"{seed}|{e['file']}|{e['row_id']}".encode()).hexdigest()  # noqa: E731
    pool = sorted(changed_pool, key=key)
    picked: List[Dict[str, Any]] = [e for rid in include for e in pool if e["row_id"] == rid][:n]
    docs: set = {e["doc"] for e in picked}
    for e in pool:
        if len(picked) >= n:
            break
        if e["doc"] in docs or e in picked:
            continue
        docs.add(e["doc"])
        picked.append(e)
    for e in pool:
        if len(picked) >= n:
            break
        if e not in picked:
            picked.append(e)
    report["examples"] = len(picked)
    return report, picked


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``audit`` -- before/after report of the analysis evidence strip.

    :param argv: Arguments (None = ``sys.argv``).
    :returns: Exit code.
    """
    ap = argparse.ArgumentParser(prog="target_hygiene", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--before", type=Path, required=True, help="families directory of the previous build")
    a.add_argument("--after", type=Path, default=None, help="families directory of the new build (default out_dir)")
    a.add_argument("--files", nargs="*", default=["book_pairs", "analyse_document_v2"])
    a.add_argument("--n", type=int, default=10)
    a.add_argument("--include", nargs="*", default=[], help="row ids always shown first")
    a.add_argument("--out", type=Path, default=None, help="examples JSONL (default out_dir/audit/analysis_strip_sample.jsonl)")
    a.add_argument("--report", type=Path, default=None, help="report JSON (default next to --out)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = _paths.load_paths()
    after = args.after or p.out("families")
    out = args.out or p.out("audit", "analysis_strip_sample.jsonl")
    report, examples = audit_family_files(args.before, after, args.files, args.n, include=args.include)
    out.parent.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(out, examples)
    rep = args.report or out.with_name("analysis_strip_report.json")
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
