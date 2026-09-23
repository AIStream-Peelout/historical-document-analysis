# File name: build_pgp_qa.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``pgp_qa_v1``: extractive question answering on side-verified PGP edition pages.

HELD OUT: these rows stay out of the training mixture until the user has reviewed
``qa_review_sample.md`` (design: ``docs/v22_dataset.md`` §3.4).

Every answer is a verbatim span of ONE line of the page transcription that
``build_pgp_editions`` produced for the same image (its ``manifest.jsonl``), returned with that
line's one-based index as ``{"line": N, "text": "..."}``; :func:`quote` asserts the span is in
the line. PGP metadata only VALIDATES a line, it never becomes an answer:

* ``qa_person`` — sender, recipient, witness, party, validating judge (never scribe, never
  "mentioned": a page mentions many people): the PGP relation's romanized name is inverted to
  Hebrew script
  (:mod:`src.datasets.qa.invert_names`) and must be located EXACTLY (token-in-order match) with
  a tier-1 (given name or kunya + connector + father) or kunya+given query; the answer is the
  matched span as written. Uncertain relations, roles PGP lists for more than one person, and
  spans found on more than one line are skipped.
* ``qa_date`` — the page has a date-formula line (a Hebrew month name with a date word such as
  שנת/לשטרות/ליצירה/בחדש) and PGP ``doc_date_original`` names exactly one month, whose Hebrew
  spelling is on that line and on no other date line; the line must also carry the year (a
  שנת-word followed by a token, or ליצירה/לשטרות/לבריאת/למניין/למנין); the answer is the whole
  line.
* ``qa_ketubah_parties`` — marriage documents: a formula line (איך/אך ... אמר לה ... לאנתו) or the
  line holding both names, on which the groom or bride from the PGP description is located
  exactly; the answer is the whole line.
* ``qa_party`` — legal documents: an acknowledgment line (אנא ... בן/בר/בת, מודה/מודים/אשהד/נשהד)
  on which a PGP Party or Witness is located exactly; the answer is the whole line.
* ``qa_abstain`` — the date question answered ``{"answer": "not stated"}`` on pages with no date
  indication anywhere in the edition and no PGP date of any kind (at most 10% of QA rows and
  20% of pages). The same abstention clause is part of every ``qa_date`` question.

At most 3 QA rows per page image; answer lines containing ``[...]`` are skipped. The split is
the page's document split from ``build_pgp_editions``. Outputs (NAS): the DatasetDict (one
split per family + ``val``), ``stats.json``, ``manifest.jsonl`` and a 200-row stratified
human-review sample (``qa_review_sample.jsonl`` / ``.md``).

Usage (repo root, after ``build_pgp_editions``)::

    nice -n 10 .venv/bin/python -m src.finetuning.qwen_hebrew.build_pgp_qa
"""
import argparse
import collections
import csv
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.datasets.qa import invert_names as inv
from src.finetuning.qwen_hebrew import build_pgp_editions as eds_builder
from src.finetuning.qwen_hebrew.build_pgp_editions import DOCUMENTS_CSV, load_csv_by, save_dataset, stable_hash
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

NAS_DATASETS = eds_builder.NAS_DATASETS
DEFAULT_EDITIONS = NAS_DATASETS / "pgp_editions_v1"
DEFAULT_OUT = NAS_DATASETS / "pgp_qa_v1"
RELATIONS_CSV = inv.RELATIONS_CSV
LABEL_SOURCE = "pgp_edition_qa"
MAX_ROWS_PER_PAGE = 3
ABSTAIN_MAX_ROW_SHARE = 0.10
ABSTAIN_MAX_PAGE_SHARE = 0.20
REVIEW_SAMPLE = 200
SEED = eds_builder.SPLIT_SEED

FAMILIES = ("qa_person", "qa_date", "qa_ketubah_parties", "qa_party", "qa_abstain")
# "Mentioned" is not a role here: a page mentions many people, so "who is the person mentioned?"
# has no single answer (the one sample hit was the document's signatory)
PERSON_ROLES = {"Sender": "sender", "Recipient": "recipient", "Witness": "witness", "Party": "party",
                "Validating judge": "validating judge"}
ROLE_PRIORITY = ("Sender", "Recipient", "Party", "Validating judge", "Witness")

PERSON_PROMPT = ('Who is the {role}? Quote the name exactly as written on the page, as JSON '
                 '{{"line": N, "text": "<name as written>"}}.')
DATE_PROMPT = ('Quote the line that gives the date of this document, as JSON {"line": N, "text": "<line>"}, '
               'or answer {"answer": "not stated"} if the page carries no date.')
KETUBAH_PROMPT = ('Who are the groom and the bride? Quote the line naming them, as JSON '
                  '{"line": N, "text": "<line>"}.')
PARTY_PROMPT = ('Quote the line in which a party to the document identifies themself, as JSON '
                '{"line": N, "text": "<line>"}.')
ABSTAIN_ANSWER = json.dumps({"answer": "not stated"})

# ----------------------------------------------------------------------------- answers


def quote(lines: Sequence[str], n: int, text: str) -> str:
    """The JSON answer ``{"line": n, "text": text}``, asserting the extractive contract.

    :param lines: Page transcription lines.
    :param n: One-based line index.
    :param text: Quoted span.
    :returns: JSON answer.
    :raises AssertionError: When ``text`` is empty or not a substring of line ``n``.
    """
    assert 1 <= n <= len(lines), f"line {n} out of range"
    assert text and text in lines[n - 1], f"answer {text!r} is not a span of line {n}: {lines[n - 1]!r}"
    return json.dumps({"line": n, "text": text}, ensure_ascii=False)


# ----------------------------------------------------------------------------- months / dates

MONTHS: Dict[str, List[str]] = {
    "tishrei": ["תשרי"],
    "marheshvan": ["מרחשון", "מרחשוון", "חשון", "חשוון"],
    "kislev": ["כסלו", "כסליו"],
    "tevet": ["טבת", "טבית"],
    "shevat": ["שבט"],
    "adar": ["אדר"],
    "nisan": ["ניסן", "נסן"],
    "iyyar": ["אייר", "איר", "אייאר"],
    "sivan": ["סיון", "סיוון"],
    "tammuz": ["תמוז"],
    "av": ["אב"],
    "elul": ["אלול"],
}
ROMAN_MONTHS: Dict[str, str] = {
    "tishrei": "tishrei", "tishri": "tishrei", "tishre": "tishrei",
    "marheshvan": "marheshvan", "marheshwan": "marheshvan", "marcheshvan": "marheshvan",
    "heshvan": "marheshvan", "heshwan": "marheshvan", "cheshvan": "marheshvan", "hesvan": "marheshvan",
    "kislev": "kislev", "kislew": "kislev", "kisliv": "kislev",
    "tevet": "tevet", "tebet": "tevet", "teveth": "tevet",
    "shevat": "shevat", "shebat": "shevat", "shvat": "shevat",
    "adar": "adar", "nisan": "nisan", "nissan": "nisan", "iyyar": "iyyar", "iyar": "iyyar",
    "sivan": "sivan", "siwan": "sivan", "tammuz": "tammuz", "tamuz": "tammuz",
    "av": "av", "ab": "av", "elul": "elul",
}
_MONTH_PREFIXES = ("", "ב", "ל", "ד", "ו", "מ", "וב", "ול", "וד")
_ALL_MONTH_SPELLINGS = {s for v in MONTHS.values() for s in v}
DATE_WORDS = {"שנת", "בשנת", "לשנת", "משנת", "דשנת", "שנה", "בשנה", "לשטרות", "לשטרי", "ליצירה", "ליצירת",
              "לבריאת", "לבריאה", "לחשבון", "למנין", "למנינא", "לפרט", "בחדש", "לחדש", "בחודש", "לחודש",
              "בירח", "לירח", "ירח", "יומין", "יום", "ביום", "ימים", "בשבה", "בשבא", "בשבת", "בשבוע", "סנה",
              "סנת"}
# the answer line of qa_date must carry the year: an era word, or a year word followed by at least
# one token on the same line -- Hebrew שנת (שנת, בשנת, דשנת, משנתינו, ...) or Judaeo-Arabic סנה/סנת/
# אלסנה with a ב/ו prefix ("פי סנה" is the token סנה after the separate word פי)
YEAR_ERA_WORDS = {"ליצירה", "לשטרות", "לבריאת", "למניין", "למנין"}
_YEAR_WORD_RE = re.compile(r"(?:ו?[במלד])?שנת(?:ינו|נו)?")
_JA_YEAR_WORD_RE = re.compile(r"(?:ו?ב|ו)?(?:אל)?סנ[הת]")
# broad detector for the abstention branch: every date word incl. day words (יום, יומין) and
# Islamic month names; a false alarm only costs an abstain row, a miss would teach "not stated"
# for a dated page
ISLAMIC_MONTHS = {"מחרם", "צפר", "רביע", "גמאדי", "גמאדא", "רגב", "שעבאן", "רמצאן", "שואל", "קעדה", "אלקעדה",
                  "חגה", "אלחגה"}
BROAD_DATE_WORDS = DATE_WORDS | {"תאריך", "אלתאריך", "בתאריך", "שהר"}
_DESC_DATE_RE = re.compile(r"\b(?:dated?|dating|dates)\b|\b\d{3,4}\b|\bC\.?E\.?\b|\bA\.?M\.?\b|\bA\.?H\.?\b|"
                           r"Seleucid|Hijr|Anno Mundi|centur", re.I)
_PGP_DATE_FIELDS = ("doc_date_original", "doc_date_calendar", "doc_date_standard", "inferred_date_display",
                    "inferred_date_standard")


def fold(text: str) -> str:
    """Lower-case, strip diacritics and ʿʾ' marks (romanized month matching).

    :param text: Romanized text.
    :returns: Folded text.
    """
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[ʿʾ'’‘`]", "", s).lower()


def pgp_month(doc_date_original: str) -> Optional[str]:
    """The single month named by a PGP ``doc_date_original`` (None if none, several, or restored).

    ``"19 Adar 1427"`` -> ``"adar"``; ``"Tishrei–Ṭevet 1495"`` (two months), ``"[Tammuz] 4831"``
    (month restored by the editor) and ``"Ḥannuka 1548"`` (no month) -> None.

    :param doc_date_original: PGP field.
    :returns: Month key of :data:`MONTHS`, or None.
    """
    found = []
    folded = fold(doc_date_original)
    for m in re.finditer(r"[^\W\d_]+", folded):
        key = ROMAN_MONTHS.get(m.group(0))
        if key:
            left = folded[:m.start()]           # folding changes lengths: index the folded text
            if left.count("[") > left.count("]"):
                return None
            found.append(key)
    return found[0] if len(set(found)) == 1 else None


def tokens(line: str) -> List[str]:
    """Hebrew-letter tokens of a line.

    :param line: Text.
    :returns: Tokens (punctuation and geresh removed).
    """
    return re.findall(r"[א-ת]+", line)


def month_tokens(line: str, spellings: Optional[Set[str]] = None) -> Set[str]:
    """Month spellings written as a token of the line (optionally with a one/two-letter prefix).

    :param line: Text.
    :param spellings: Spellings to look for (default: every Hebrew month).
    :returns: The spellings found.
    """
    spellings = spellings or _ALL_MONTH_SPELLINGS
    out = set()
    for t in tokens(line):
        for p in _MONTH_PREFIXES:
            if t.startswith(p) and t[len(p):] in spellings:
                out.add(t[len(p):])
    return out


def is_date_line(line: str) -> bool:
    """A date-formula line: a Hebrew month name AND a date word (שנת, לשטרות, בחדש, יום, ...).

    :param line: Text.
    :returns: Whether the line gives a date.
    """
    return bool(month_tokens(line)) and bool(set(tokens(line)) & DATE_WORDS)


def has_year(line: str) -> bool:
    """The line carries the year, not only the month.

    Either an era word (:data:`YEAR_ERA_WORDS`) or a year word followed by at least one token on
    the line: a שנת-word (שנת/בשנת with a ב/מ/ל/ד/ו prefix or the ינו suffix: דשנת, משנתינו) or
    a Judaeo-Arabic סנה/סנת/אלסנה (also בסנה, וסנה; "פי סנה"). "...לחדש תשרי שנת" (the year
    continues on the next line) does not.

    :param line: Text.
    :returns: Whether the line has a year expression.
    """
    toks = tokens(line)
    if set(toks) & YEAR_ERA_WORDS:
        return True
    return any((_YEAR_WORD_RE.fullmatch(t) or _JA_YEAR_WORD_RE.fullmatch(t)) and k + 1 < len(toks)
               for k, t in enumerate(toks))


def has_date_indication(line: str) -> bool:
    """Broad date detector for the abstention branch (any month name or dating word).

    :param line: Text.
    :returns: Whether the line might carry a date.
    """
    toks = set(tokens(line))
    return bool(month_tokens(line)) or bool(toks & BROAD_DATE_WORDS) or bool(toks & ISLAMIC_MONTHS)


def pgp_has_date(meta: Dict[str, str]) -> bool:
    """Any PGP date: the date fields, inferred dates, or a date/year/century in the description.

    :param meta: ``documents.csv`` row.
    :returns: Whether PGP records any date for the document.
    """
    return any((meta.get(f) or "").strip() for f in _PGP_DATE_FIELDS) or bool(
        _DESC_DATE_RE.search(meta.get("description") or ""))


# ----------------------------------------------------------------------------- page index / persons


@dataclass
class PageIndex:
    """Name-matching index over one page transcription.

    :param doc: :class:`invert_names.DocIndex` of the page lines.
    :param line_of: Index line -> page line (0-based).
    """

    doc: inv.DocIndex
    line_of: List[int]


def page_index(lines: Sequence[str]) -> PageIndex:
    """Index a page transcription with :func:`invert_names.build_doc_index`.

    :param lines: Page lines (no newlines inside).
    :returns: The index and the mapping back to page lines.
    """
    doc = inv.build_doc_index("\n".join(lines))
    line_of = [i for i, ln in enumerate(lines)
               if len(inv.HEB_LETTER_RE.findall(ln)) >= 3 and not inv.is_editorial(ln)]
    assert len(line_of) == len(doc.lines_b) and all(
        doc.lines_b[k].original == lines[line_of[k]] for k in range(len(line_of)))
    return PageIndex(doc, line_of)


@dataclass
class Located:
    """A name located exactly on a page line.

    :param line: One-based page line.
    :param span: The matched text as written.
    :param query: The Hebrew-script query that matched.
    :param tier: Query tier.
    :param kind: Query kind.
    :param source: ``romanized`` / ``latin_variant``.
    """

    line: int
    span: str
    query: str
    tier: int
    kind: str
    source: str


def allowed_query(q: inv.Query) -> bool:
    """Only given name/kunya + father (tier 1) or kunya + given name queries may locate a person.

    :param q: Query.
    :returns: Whether the query kind is strict enough.
    """
    return q.tier == 1 or q.kind == "kunya+given"


def locate_exact(latin_forms: Sequence[str], script_variants: Sequence[str], relation: str,
                 index: PageIndex, lines: Sequence[str]) -> Optional[Located]:
    """Locate a person on the page under the strict rule, or None.

    Queries come from :func:`invert_names.queries_from_romanized` on each Latin form, filtered
    to :func:`allowed_query`; the hit must be ``status == "located"`` AND an exact token-in-order
    match (never fuzzy, never kunya-only or family-only).

    :param latin_forms: PGP name first, then Latin variants.
    :param script_variants: Hebrew/Arabic-script variants (contradiction checks only).
    :param relation: PGP relation label (contradiction rules).
    :param index: Page index.
    :param lines: Page lines.
    :returns: The located span, or None.
    """
    queries: List[inv.Query] = []
    for k, form in enumerate(latin_forms):
        queries += [q for q in inv.queries_from_romanized(form, "romanized" if k == 0 else "latin_variant")
                    if allowed_query(q)]
    uniq = list({q.text: q for q in queries[::-1]}.values())
    if not uniq:
        return None
    res = inv.locate(uniq, index.doc, inv.person_info(list(latin_forms), list(script_variants)), relation)
    if not (res["status"] == "located" and res.get("exact") and res.get("exact_hit")
            and (res.get("tier") == 1 or res.get("kind") == "kunya+given") and res.get("span")):
        return None
    n = index.line_of[res["line_no"]] + 1
    assert res["span"] in lines[n - 1]
    return Located(n, res["span"], res["query"], res["tier"], res["kind"], res["source"])


def single_line_of(span: str, lines: Sequence[str]) -> bool:
    """The span occurs on exactly one page line (so "line N" is unambiguous).

    :param span: Text.
    :param lines: Page lines.
    :returns: Whether exactly one line contains it.
    """
    return sum(span in ln for ln in lines) == 1


# ----------------------------------------------------------------------------- ketubba names

_KETUBAH_RE = re.compile(r"^\s*(?:Ketubba|Ketubah|Marriage contract|Marriage document|Betrothal)", re.I)
_CONNECTORS = {"b.", "bt.", "b", "bt", "ben", "bat", "bint", "bnt."}
_NAME_HEADS = {"Abū", "Abu", "Umm", "Sitt", "Ibn"}
_NAME_TOKEN_RE = re.compile(r"^\(?(?:[A-ZÀ-ÞĀ-ŽḀ-Ỿ]|[ʿʾ'][A-ZÀ-ÞĀ-ŽḀ-Ỿ]|(?:al|ha|he|l|el)-[ʿʾ']?[A-ZÀ-ÞĀ-ŽḀ-Ỿ])")


def parse_name(tokens_: Sequence[str]) -> Optional[str]:
    """Read one romanized name from the start of a token list.

    Grammar: a name starts with a capital (or Abū/Umm/Sitt/Ibn), continues with capitalised or
    article-prefixed tokens (``al-``, ``ha-``) and ``b.``/``bt.`` chains, and ends at the first
    other token or at sentence punctuation. Uncertain names (``?``, ``[``, ``...``) are rejected.

    :param tokens_: Whitespace tokens following a cue ("Groom:", "between").
    :returns: The name, or None.
    """
    out: List[str] = []
    for tok in tokens_:
        if any(c in tok for c in "?[]…") or "..." in tok:
            return None
        core = tok.rstrip(".,;:")
        if tok in _CONNECTORS or core in _CONNECTORS:
            if not out:
                return None
            out.append(tok if tok.endswith(".") or tok in _CONNECTORS else core)
            continue
        if not (_NAME_TOKEN_RE.match(core) or core in _NAME_HEADS):
            break
        out.append(core)
        if tok != core:                     # sentence punctuation ends the name
            break
    while out and out[-1].rstrip(".") in {c.rstrip(".") for c in _CONNECTORS}:
        out.pop()
    if not out or not (_NAME_TOKEN_RE.match(out[0]) or out[0] in _NAME_HEADS):
        return None
    return " ".join(out)


def parse_couple(description: str) -> Dict[str, str]:
    """Groom and bride names from a PGP marriage-document description.

    Cues: ``Groom:``/``Bride:`` (``Fiancé:``/``Fiancée:`` in betrothal deeds), "the (bride)groom('s
    name) is ...", "the bride('s name) is ...", and "of|between X and Y" (X = groom, Y = bride).

    :param description: PGP description.
    :returns: ``{"groom": ..., "bride": ...}`` for the names found.
    """
    out: Dict[str, str] = {}
    for role, rx in (("groom", r"\b(?:Groom|Bridegroom|Fianc[eé]):\s*"), ("bride", r"\b(?:Bride|Fianc[eé]e):\s*"),
                     ("groom", r"\b[Tt]he (?:bride)?groom(?:'s name)? is\s+"),
                     ("bride", r"\b[Tt]he bride(?:'s name)? is\s+")):
        if role in out:
            continue
        m = re.search(rx, description)
        if m:
            name = parse_name(description[m.end():].split())
            if name:
                out[role] = name
    if not out:
        m = re.search(r"\b(?:of|between)\s+", description[:300])
        if m:
            toks = description[m.end():].split()
            x = parse_name(toks)
            if x:
                rest = toks[len(x.split()):]
                if rest and rest[0] == "and":
                    y = parse_name(rest[1:])
                    if y:
                        out = {"groom": x, "bride": y}
    return out


_FORMULA_RE = re.compile(r"(?:^|\s)ו?א[י]?ך\s|אמר\s+(?:לה|להדא)|לאנת[ו]|לאינתו")
_PARTY_LINE_RE = re.compile(r"(?:^|\s)ו?(?:אנא|אנן|אנחנו|נחנא)\s(?:.*\s)?(?:בן|בר|בת)(?:\s|$)|מודה|מודים|אשהד|נשהד")


# ----------------------------------------------------------------------------- page rows


@dataclass
class QARow:
    """One candidate QA row with the evidence that validated it.

    :param family: Row family.
    :param section: Role / kind label.
    :param question: Prompt.
    :param answer: JSON answer.
    :param line: One-based answer line (0 for abstain).
    :param evidence: What validated the row (for review).
    :param priority: Lower wins the per-page cap.
    """

    family: str
    section: str
    question: str
    answer: str
    line: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: int = 9


@dataclass
class QAContext:
    """Per-run inputs.

    :param pgp_docs: ``documents.csv`` by pgpid.
    :param relations: PGP relation rows by pgpid.
    :param by_name: people.csv by name.
    :param by_slug: people.csv by slug.
    :param edition_lines: pgpid -> every raw + rendered edition line (all sides).
    """

    pgp_docs: Dict[str, Dict[str, str]]
    relations: Dict[str, List[Dict[str, str]]]
    by_name: Dict[str, Dict]
    by_slug: Dict[str, Dict]
    edition_lines: Dict[str, List[str]]


def person_rows(page: Dict[str, Any], ctx: QAContext, index: PageIndex,
                stats: collections.Counter) -> Tuple[List[QARow], List[Tuple[Dict[str, str], Located]]]:
    """``qa_person`` rows of a page and every relation located on it.

    :param page: Editions manifest record.
    :param ctx: Context.
    :param index: Page index.
    :param stats: Counters (updated).
    :returns: ``(rows, located relation rows)``; the second list also serves ``qa_party``.
    """
    lines = page["lines"]
    rels = ctx.relations.get(page["pgpid"], [])
    per_role = collections.Counter(inv.base_relation(r["relation"]) for r in rels)
    rows, located = [], []
    for r in rels:
        base = inv.base_relation(r["relation"])
        if base not in PERSON_ROLES:
            continue
        stats[f"relations_{base}"] += 1
        if "uncertain" in r["relation"]:
            stats["person_skip_uncertain"] += 1
            continue
        script, latin = inv.person_forms(r, ctx.by_name, ctx.by_slug)
        hit = locate_exact(latin, script, r["relation"], index, lines)
        if hit is None:
            stats["person_not_located_exact"] += 1
            continue
        located.append((r, hit))
        stats["person_located_exact"] += 1
        if per_role[base] > 1:
            stats["person_skip_role_not_unique"] += 1
            continue
        if not single_line_of(hit.span, lines):
            stats["person_skip_span_on_several_lines"] += 1
            continue
        if GAP_TOKEN in lines[hit.line - 1]:
            stats["person_skip_gap_in_line"] += 1
            continue
        rows.append(QARow("qa_person", base.lower().replace(" ", "_"), PERSON_PROMPT.format(role=PERSON_ROLES[base]),
                          quote(lines, hit.line, hit.span), hit.line,
                          {"relation": r["relation"], "person_name": r["person_name"], "person_slug": r["person_slug"],
                           "query": hit.query, "tier": hit.tier, "kind": hit.kind, "source": hit.source},
                          priority=3 + ROLE_PRIORITY.index(base)))
    return rows, located


def date_row(page: Dict[str, Any], meta: Dict[str, str], stats: collections.Counter) -> Optional[QARow]:
    """``qa_date`` row: the single date line carrying the PGP month and the year.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    lines = page["lines"]
    date_lines = [i for i, ln in enumerate(lines) if is_date_line(ln)]
    if not date_lines:
        return None
    stats["date_pages_with_date_line"] += 1
    month = pgp_month(meta.get("doc_date_original") or "")
    if month is None:
        stats["date_skip_no_single_pgp_month"] += 1
        return None
    hits = [i for i in date_lines if month_tokens(lines[i], set(MONTHS[month]))]
    if len(hits) != 1:
        stats["date_skip_month_not_on_one_date_line" if not hits else "date_skip_month_on_several_lines"] += 1
        return None
    i = hits[0]
    if GAP_TOKEN in lines[i]:
        stats["date_skip_gap_in_line"] += 1
        return None
    if not has_year(lines[i]):
        stats["date_skip_no_year_on_line"] += 1
        return None
    return QARow("qa_date", "date", DATE_PROMPT, quote(lines, i + 1, lines[i]), i + 1,
                 {"doc_date_original": meta.get("doc_date_original"), "pgp_month": month,
                  "hebrew_spellings": MONTHS[month]}, priority=2)


def ketubah_row(page: Dict[str, Any], meta: Dict[str, str], index: PageIndex,
                stats: collections.Counter) -> Optional[QARow]:
    """``qa_ketubah_parties`` row: the formula line naming groom and/or bride.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param index: Page index.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    desc = meta.get("description") or ""
    if not _KETUBAH_RE.match(desc):
        return None
    stats["ketubah_pages"] += 1
    couple = parse_couple(desc)
    if not couple:
        stats["ketubah_skip_names_not_parsed"] += 1
        return None
    lines = page["lines"]
    hits = {role: locate_exact([name], [], "Party", index, lines) for role, name in couple.items()}
    hits = {k: v for k, v in hits.items() if v}
    if not hits:
        stats["ketubah_skip_names_not_located"] += 1
        return None
    by_line = collections.defaultdict(list)
    for role, h in hits.items():
        by_line[h.line].append(role)
    cands = [n for n in sorted(by_line) if len(by_line[n]) == 2 or _FORMULA_RE.search(lines[n - 1])]
    cands.sort(key=lambda n: (-len(by_line[n]), not re.search(r"(?:^|\s)ו?א[י]?ך\s", lines[n - 1]), n))
    cands = [n for n in cands if GAP_TOKEN not in lines[n - 1]]
    if not cands:
        stats["ketubah_skip_no_clean_formula_line"] += 1
        return None
    n = cands[0]
    return QARow("qa_ketubah_parties", "groom_bride", KETUBAH_PROMPT, quote(lines, n, lines[n - 1]), n,
                 {"description": desc[:300], "parsed": couple, "located_on_line": by_line[n],
                  "spans": {k: v.span for k, v in hits.items()}}, priority=0)


def party_row(page: Dict[str, Any], meta: Dict[str, str], located: Sequence[Tuple[Dict[str, str], Located]],
              stats: collections.Counter) -> Optional[QARow]:
    """``qa_party`` row: an acknowledgment line on which a Party/Witness is located.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param located: Relations located exactly on the page.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    if "Legal" not in (meta.get("type") or ""):
        return None
    lines = page["lines"]
    for r, h in sorted(located, key=lambda x: x[1].line):
        if inv.base_relation(r["relation"]) not in ("Party", "Witness"):
            continue
        ln = lines[h.line - 1]
        if _PARTY_LINE_RE.search(ln) and GAP_TOKEN not in ln:
            return QARow("qa_party", inv.base_relation(r["relation"]).lower(), PARTY_PROMPT,
                         quote(lines, h.line, ln), h.line,
                         {"relation": r["relation"], "person_name": r["person_name"], "span": h.span,
                          "query": h.query, "tier": h.tier, "kind": h.kind}, priority=1)
    stats["party_skip_no_located_acknowledgment_line"] += 1
    return None


def abstain_eligible(page: Dict[str, Any], meta: Dict[str, str], ctx: QAContext) -> bool:
    """No date on the page, none anywhere in the edition (raw text included), none in PGP.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param ctx: Context.
    :returns: Whether the "not stated" answer is safe.
    """
    if pgp_has_date(meta):
        return False
    return not any(has_date_indication(ln) for ln in page["lines"] + ctx.edition_lines.get(page["pgpid"], []))


def page_candidates(page: Dict[str, Any], ctx: QAContext,
                    stats: collections.Counter) -> Tuple[List[QARow], bool]:
    """All validated QA rows of one page (before caps) and its abstention eligibility.

    :param page: Editions manifest record.
    :param ctx: Context.
    :param stats: Counters (updated).
    :returns: ``(rows sorted by priority, abstain eligible)``.
    """
    meta = ctx.pgp_docs.get(page["pgpid"], {})
    index = page_index(page["lines"])
    rows, located = person_rows(page, ctx, index, stats)
    for extra in (date_row(page, meta, stats), ketubah_row(page, meta, index, stats),
                  party_row(page, meta, located, stats)):
        if extra:
            rows.append(extra)
    rows.sort(key=lambda r: (r.priority, r.line))
    return rows, abstain_eligible(page, meta, ctx)


def apply_caps(pages: Sequence[Dict[str, Any]], cands: Dict[str, List[QARow]], eligible: Dict[str, bool],
               max_rows: int = MAX_ROWS_PER_PAGE, abstain_row_share: float = ABSTAIN_MAX_ROW_SHARE,
               abstain_page_share: float = ABSTAIN_MAX_PAGE_SHARE) -> Dict[str, List[QARow]]:
    """Per-page cap, then abstention rows within the row and page shares.

    :param pages: Manifest records (``key`` field set).
    :param cands: ``page key -> candidate rows`` sorted by priority.
    :param eligible: ``page key -> abstain eligible``.
    :param max_rows: QA rows per page image.
    :param abstain_row_share: Maximum share of abstain rows among all QA rows.
    :param abstain_page_share: Maximum share of pages carrying an abstain row.
    :returns: ``page key -> kept rows``.
    """
    kept = {p["key"]: list(cands.get(p["key"], []))[:max_rows] for p in pages}
    n_other = sum(len(v) for v in kept.values())
    budget = min(int(abstain_row_share * n_other / (1 - abstain_row_share)), int(abstain_page_share * len(pages)))
    order = sorted((p["key"] for p in pages if eligible.get(p["key"]) and len(kept[p["key"]]) < max_rows),
                   key=lambda k: stable_hash(f"{SEED}:abstain:{k}"))
    for k in order[:max(0, budget)]:
        kept[k].append(QARow("qa_abstain", "no_date", DATE_PROMPT, ABSTAIN_ANSWER, 0,
                             {"pgp_date_fields": "none", "date_indications_in_edition": "none"}, priority=8))
    return kept


# ----------------------------------------------------------------------------- review sample


def review_sample(rows: Sequence[Dict[str, Any]], n: int = REVIEW_SAMPLE) -> List[Dict[str, Any]]:
    """Stratified sample across families (all rows if fewer than ``n``).

    :param rows: Manifest rows (``family`` set).
    :param n: Sample size.
    :returns: Sampled rows, grouped by family.
    """
    if len(rows) <= n:
        return sorted(rows, key=lambda r: (FAMILIES.index(r["family"]), r["stem"]))
    by_fam: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for v in by_fam.values():
        v.sort(key=lambda r: stable_hash(f"{SEED}:review:{r['stem']}"))
    quota = {f: 0 for f in by_fam}
    left = n
    while left:
        open_f = [f for f in by_fam if quota[f] < len(by_fam[f])]
        if not open_f:
            break
        for f in sorted(open_f, key=lambda f: FAMILIES.index(f)):
            if left and quota[f] < len(by_fam[f]):
                quota[f] += 1
                left -= 1
    return [r for f in sorted(by_fam, key=FAMILIES.index) for r in by_fam[f][:quota[f]]]


def review_markdown(sample: Sequence[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    """Render the review sample as Markdown.

    :param sample: Sampled rows.
    :param stats: Build stats.
    :returns: Markdown text.
    """
    out = ["# pgp_qa_v1 — human review sample", "",
           "**These QA rows are NOT in the training mixture.** They stay out until this sample has been "
           "reviewed and the families accepted (`docs/v22_dataset.md` §3.4).", "",
           f"Built {stats['generated_at']} from `{stats['editions_manifest']}`. Rows per family: "
           f"{json.dumps(stats['rows_by_family'])}. Sample: {len(sample)} rows, stratified by family.", "",
           "Each row shows the page image link, the question, the target JSON, the full page line the "
           "answer quotes, and the PGP evidence that validated it. Mark rows that are wrong.", ""]
    fam = None
    for i, r in enumerate(sample, 1):
        if r["family"] != fam:
            fam = r["family"]
            out += [f"## {fam}", ""]
        out += [f"### {i}. `{r['stem']}` — {r['canonical_id']} (image {r['image_index']})", "",
                f"- image: {r['image_url']}", f"- question: {r['question']}", f"- answer: `{r['answer']}`",
                f"- answer line {r['line'] or '-'}: {r['answer_line'] or '(abstention: no date line on the page)'}",
                f"- evidence: `{json.dumps(r['evidence'], ensure_ascii=False)}`", ""]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- build


def load_relations(path: Path, pgpids: Set[str]) -> Dict[str, List[Dict[str, str]]]:
    """PGP person-document relations of the given documents.

    :param path: Relations CSV.
    :param pgpids: Documents of interest.
    :returns: ``pgpid -> relation rows``.
    """
    out: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["pgpid"] in pgpids:
                out[r["pgpid"]].append(r)
    return dict(out)


def edition_lines_of(pgpids: Set[str]) -> Dict[str, List[str]]:
    """Every raw and rendered line of each document's edition (all sides, all blocks).

    :param pgpids: Documents of interest.
    :returns: ``pgpid -> lines``.
    """
    eds = eds_builder.load_editions()
    out = {}
    for p in pgpids:
        if p in eds:
            content = eds[p]["content"]
            parsed = eds_builder.parse_edition(content)
            out[p] = content.splitlines() + [ln for b in parsed.blocks for ln in b.lines]
    return out


def build(editions_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Build ``pgp_qa_v1`` from the editions manifest.

    :param editions_dir: ``pgp_editions_v1`` output (``manifest.jsonl``).
    :param out_dir: Destination (NAS).
    :returns: The stats dict.
    """
    manifest_path = editions_dir / "manifest.jsonl"
    pages = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for p in pages:
        p["key"] = f"{p['canonical_id']}__{p['image_index']}"
    pgpids = {p["pgpid"] for p in pages}
    by_name, by_slug = inv.load_people()
    inv.NAME_LEXICON.update(inv.build_name_lexicon(list(by_name.values())))
    ctx = QAContext(load_csv_by(DOCUMENTS_CSV, "pgpid"), load_relations(RELATIONS_CSV, pgpids), by_name, by_slug,
                    edition_lines_of(pgpids))
    stats_c: collections.Counter = collections.Counter()
    cands, eligible = {}, {}
    for p in pages:
        cands[p["key"]], eligible[p["key"]] = page_candidates(p, ctx, stats_c)
    kept = apply_caps(pages, cands, eligible)

    split_rows: Dict[str, List[Dict[str, Any]]] = {f"train_{f}": [] for f in FAMILIES}
    split_rows["val"] = []
    manifest: List[Dict[str, Any]] = []
    for p in pages:
        for j, q in enumerate(kept[p["key"]]):
            stem = f"pgpqa_{p['pgpid']}_{p['image_index']}_{q.family}_{j}"
            if q.family != "qa_abstain":
                obj = json.loads(q.answer)
                assert obj["text"] in p["lines"][obj["line"] - 1]
            r = {"image": p["image_path"], "question": q.question, "answer": q.answer, "task": q.family,
                 "section": q.section, "stem": stem, "label_source": LABEL_SOURCE, "target_chars": len(q.answer),
                 "target_tokens": 0, "image_width": p["image_width"], "image_height": p["image_height"]}
            split_rows["val" if p["split"] == "val" else f"train_{q.family}"].append(r)
            manifest.append({"stem": stem, "family": q.family, "section": q.section, "pgpid": p["pgpid"],
                             "canonical_id": p["canonical_id"], "image_index": p["image_index"],
                             "image_url": p["image_url"], "split": p["split"], "question": q.question,
                             "answer": q.answer, "line": q.line,
                             "answer_line": p["lines"][q.line - 1] if q.line else "",
                             "line_countable_from_top": bool(q.line and p["countable"][q.line - 1]),
                             "line_region": p["regions"][q.line - 1] if q.line else "", "evidence": q.evidence})
    fam_counts = collections.Counter(m["family"] for m in manifest)
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "HELD OUT of the training mixture until qa_review_sample.md is reviewed",
        "editions_manifest": str(manifest_path), "pages": len(pages),
        "pages_with_qa_rows": sum(1 for v in kept.values() if v),
        "rows_by_family": dict(fam_counts), "rows_by_split": {k: len(v) for k, v in split_rows.items()},
        "rows_by_family_and_section": dict(collections.Counter(f"{m['family']}:{m['section']}" for m in manifest)),
        "candidates_before_page_cap": dict(collections.Counter(q.family for v in cands.values() for q in v)),
        "abstain_eligible_pages": sum(eligible.values()),
        "answer_lines_countable_from_top": sum(m["line_countable_from_top"] for m in manifest if m["line"]),
        "validation_counters": dict(stats_c),
        "rules": {"max_rows_per_page": MAX_ROWS_PER_PAGE, "abstain_max_row_share": ABSTAIN_MAX_ROW_SHARE,
                  "abstain_max_page_share": ABSTAIN_MAX_PAGE_SHARE,
                  "person_rule": "status located AND exact token match AND tier 1 or kunya+given; "
                                 "not uncertain; role unique in PGP; span on one line; no [...] in the line"},
    }
    sample = review_sample(manifest)
    save_dataset(split_rows, out_dir, {
        "stats.json": json.dumps(stats, ensure_ascii=False, indent=1),
        "manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
        "qa_review_sample.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in sample),
        "qa_review_sample.md": review_markdown(sample, stats)})
    logger.info("stats: %s", json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editions-dir", type=Path, default=DEFAULT_EDITIONS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    build(a.editions_dir, a.output_dir)


if __name__ == "__main__":
    main()
