# File name: build_pgp_qa.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``pgp_qa``: extractive question answering on side-verified PGP edition pages.

Status: included in the v22 mixture; review sample available at ``qa_review_sample.md`` (design:
``docs/v22_dataset.md`` §3.4). The ``qa_party_formula`` rule was admitted after a manual spot check
(``qa_party_formula_spotcheck.md``). ``pgp_qa_v1`` fed the v22a pilot, which learned the answer
format but not retrieval; v2 (2026-09-28, default output ``pgp_qa_v2``) answers text first, caps the
share of any one answer, balances abstention against ``qa_date`` and paraphrases the questions (see
"Balance" below).

Every answer is a verbatim span of ONE line of the page transcription that
``build_pgp_editions`` produced for the same image (its ``manifest.jsonl``), returned text first
with that line's one-based index as ``{"text": "...", "line": N}`` (the model writes the quote
before it commits to a line number). The span is always WHOLE token(s) exactly
as written, never part of a word: a name or place written with an attached prefix letter is quoted
with it ("בדמשק", not "דמשק"); :func:`quote` raises :class:`AnswerSpanError` otherwise, and the
build re-checks every emitted answer. PGP metadata only VALIDATES a line, it never becomes an
answer:

* ``qa_person`` — sender, recipient, witness, party, validating judge (never scribe, never
  "mentioned": a page mentions many people): the PGP relation's romanized name is inverted to
  Hebrew script
  (:mod:`src.datasets.qa.invert_names`) and must be located EXACTLY (token-in-order match) with
  a tier-1 (given name or kunya + connector + father) or kunya+given query; the answer is the
  matched name as whole token(s), with an attached prefix particle (ו/ל/ב ...) when there is one.
  Uncertain relations, roles PGP lists for more than one person, and spans found on more than one
  line are skipped.
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
  indication anywhere in the edition and no PGP date of any kind (about ``--abstain-ratio`` x the
  ``qa_date`` rows, on at most 20% of pages). The same abstention clause is part of every
  ``qa_date`` question, and both families draw their wordings from the same table.

Set-valued and span rows (added 2026-09-24):

* ``qa_witnesses_list`` / ``qa_parties_list`` — roles PGP lists for two or more people: the
  answer is a JSON list of ``{"text": name, "line": N}`` in line order, emitted only when EVERY
  holder is located exactly (certain relations, one line each, no ``[...]`` in the line) and a
  completeness check passes, per name: witnesses -- every ``X בן|בר|ביר|ברבי Y`` in the signature
  region (last 8 main-text lines plus margins/address) is a located witness span, and every
  located witness sits in that region; parties -- every ``X בן|בר|ביר|ברבי|בת Y`` on the page is
  a located span of some PGP relation of the document.
* ``qa_witness_line`` / ``qa_party_line`` (sections ``witness_any`` / ``party_any``) — the
  fallback when at least one holder is located but the list is not emitted: the first located
  holder's whole line (for witnesses, a line in the signature region).
* ``qa_date_month`` / ``qa_date_year`` — spans of the date: the month token as written on the
  PGP-month date line, prefix included (בניסן; ``[...]`` elsewhere on the line is allowed), and the
  year expression after
  the year word (שנת/בשנת/משנתינו/דשנת/סנה/סנת ...), through the era word when there is one on the
  line, else 1-3 numeral tokens (marked letter numerals or number words) that do not run on;
  the year may sit on the line after the month line. The year value is not checked against PGP
  (no calendar conversion) but PGP must record a date.
* ``qa_ketubah_groom`` / ``qa_ketubah_bride`` — the groom or bride of the PGP description located
  exactly on any line of a marriage document.
* ``qa_party_formula`` — legal documents: the page's single self-identification line recognised
  by its wording alone (singular אנא in apposition to a name, or מודה/מודים + name; no court or
  witness formula), on pages without a ``qa_party`` row.
* ``qa_place`` (sections ``written`` / ``sent``) — PGP origin/location or destination, spelled in
  Hebrew script (a table of common places plus places.csv variants), matched as exact tokens on
  exactly one line; the answer is the whole token(s) as written, prefix included (בדמשק).

Balance (v2). A validated QA pair is a *fact*; at most ``--max-rows-per-page`` (3) facts per page
image (a list counts as one) and answer lines containing ``[...]`` are skipped. Before that per-page
cut, within each family no single answer -- compared after stripping nikud, folding final letters
and collapsing whitespace (:func:`answer_key`; a list by its sorted items) -- may exceed
``--max-answer-share`` (0.20) of the family's train facts, though one fact per answer is always
allowed: the excess candidates are dropped in ``stable_hash`` order and the pages' next candidates
take the freed slots (v1: 47% of the ``qa_place`` answers were בפסטאט and the model learned the
prior). Val pages keep their natural answer distribution. Abstention facts number
``--abstain-ratio`` (0.8) x the ``qa_date`` facts, bounded by the eligible pages with a free slot and
by 20% of all pages (v1 capped them at 10% of all QA rows: 100 of 384 eligible pages). Each train
fact is emitted ``--paraphrases`` (2) times with distinct question wordings drawn by ``stable_hash``
from :data:`PARAPHRASES`; the paraphrase twins share the answer, task, section and stem and differ
only in the question. Val keeps one row per fact with the canonical wording (index 0), so held-out
scoring stays one row per fact. The split is the page's document split from ``build_pgp_editions``.
Outputs (NAS): the DatasetDict (one split per family + ``val``), ``stats.json``, ``manifest.jsonl``
(one record per fact, with the wordings it was emitted with) and a 250-fact stratified human-review
sample (``qa_review_sample.jsonl`` / ``.md``).

Usage (repo root, after ``build_pgp_editions``)::

    nice -n 10 .venv/bin/python -m src.finetuning.qwen_hebrew.build_pgp_qa \\
        --editions-dir /Volumes/home/studio_offload/datasets/pgp_editions_v1 \\
        --output-dir /Volumes/home/studio_offload/datasets/pgp_qa_v2 \\
        --max-rows-per-page 3 --max-answer-share 0.20 --abstain-ratio 0.8 --paraphrases 2
"""
import argparse
import collections
import csv
import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.datasets.evaluations.metrics import normalize_whitespace, strip_nikud
from src.datasets.qa import invert_names as inv
from src.finetuning.qwen_hebrew import build_pgp_editions as eds_builder
from src.finetuning.qwen_hebrew.build_pgp_editions import DOCUMENTS_CSV, load_csv_by, save_dataset, stable_hash
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

NAS_DATASETS = eds_builder.NAS_DATASETS
DEFAULT_EDITIONS = NAS_DATASETS / "pgp_editions_v1"
DEFAULT_OUT = NAS_DATASETS / "pgp_qa_v2"
V1_DIR = NAS_DATASETS / "pgp_qa_v1"           # its spot-check file is carried into new builds
RELATIONS_CSV = inv.RELATIONS_CSV
PLACES_CSV = eds_builder.CG / "pgp_raw/data/places.csv"
SPOTCHECK_FILE = "qa_party_formula_spotcheck.md"
LABEL_SOURCE = "pgp_edition_qa"
MAX_ROWS_PER_PAGE = 3          # facts per page image (a list answer counts as one)
MAX_ANSWER_SHARE = 0.20        # no answer above this share of its family's train facts
ABSTAIN_RATIO = 0.8            # abstention facts per qa_date fact
ABSTAIN_MAX_PAGE_SHARE = 0.20
PARAPHRASES_PER_FACT = 2       # train rows per fact, each with its own question wording (val: 1)
REVIEW_SAMPLE = 250
SEED = eds_builder.SPLIT_SEED

FAMILIES = ("qa_person", "qa_witnesses_list", "qa_witness_line", "qa_parties_list", "qa_party_line", "qa_date",
            "qa_date_month", "qa_date_year", "qa_ketubah_parties", "qa_ketubah_groom", "qa_ketubah_bride",
            "qa_party", "qa_party_formula", "qa_place", "qa_abstain")
# lower wins the per-page cap; the families that existed first keep their relative order
PRIORITY = {"qa_ketubah_parties": 0, "qa_party": 1, "qa_date": 2, "qa_witnesses_list": 3, "qa_parties_list": 3,
            "qa_ketubah_groom": 4, "qa_ketubah_bride": 4, "qa_place": 4, "qa_person": 5, "qa_witness_line": 10,
            "qa_party_line": 10, "qa_party_formula": 10, "qa_date_year": 11, "qa_date_month": 12, "qa_abstain": 20}
SIGNATURE_WINDOW = 8           # the last main-text lines searched for witness signatures
# "Mentioned" is not a role here: a page mentions many people, so "who is the person mentioned?"
# has no single answer (the one sample hit was the document's signatory)
PERSON_ROLES = {"Sender": "sender", "Recipient": "recipient", "Witness": "witness", "Party": "party",
                "Validating judge": "validating judge"}
ROLE_PRIORITY = ("Sender", "Recipient", "Party", "Validating judge", "Witness")

PERSON_PROMPT = ('Who is the {role}? Quote the name exactly as written on the page, including any attached '
                 'prefix letter, as JSON {{"text": "<name as written>", "line": N}}.')
DATE_PROMPT = ('Quote the line that gives the date of this document, as JSON {"text": "<line>", "line": N}, '
               'or answer {"answer": "not stated"} if the page carries no date.')
KETUBAH_PROMPT = ('Who are the groom and the bride? Quote the line naming them, as JSON '
                  '{"text": "<line>", "line": N}.')
PARTY_PROMPT = ('Quote the line in which a party to the document identifies themself, as JSON '
                '{"text": "<line>", "line": N}.')
ABSTAIN_ANSWER = json.dumps({"answer": "not stated"})
WITNESS_LIST_PROMPT = ('Quote the names of the witnesses who signed this document exactly as written, including '
                       'any attached prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.')
WITNESS_LINE_PROMPT = 'Quote one line in which a witness signs, as JSON {"text": "<line>", "line": N}.'
PARTY_LIST_PROMPT = ('Quote the names of the parties to this document exactly as written, including any attached '
                     'prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.')
PARTY_LINE_PROMPT = 'Quote one line naming a party to this document, as JSON {"text": "<line>", "line": N}.'
MONTH_PROMPT = ('In which month was this document written? Quote the month exactly as written, including any '
                'attached prefix letter, as JSON {"text": "<month token>", "line": N}.')
YEAR_PROMPT = ('In which year was this document written? Quote the year exactly as written, as JSON '
               '{"text": "<year expression>", "line": N}.')
KETUBAH_NAME_PROMPT = ('Who is the {role}? Quote the name exactly as written, including any attached prefix '
                       'letter, as JSON {{"text": "<name>", "line": N}}.')
PLACE_WRITTEN_PROMPT = ('Where was this document written? Quote the word(s) naming the place exactly as written on '
                        'the page, including any attached prefix letter, as JSON {"text": "<as written>", "line": N}.')
PLACE_SENT_PROMPT = ('To where was this document sent? Quote the word(s) naming the place exactly as written on the '
                     'page, including any attached prefix letter, as JSON {"text": "<as written>", "line": N}.')

# Question wordings per family (``qa_place`` per section). Index 0 is the canonical prompt above,
# the one QARow.question carries and the val split uses. Every wording asks the same question with
# the same answer instruction: the text-first JSON example, "including any attached prefix letter"
# wherever the canonical prompt has it, and the abstention clause in every date wording. The person
# and ketubba-name wordings are str.format templates ({role}; literal braces doubled).
_DATE_WORDINGS = (
    DATE_PROMPT,
    'Which line of the page gives the date of this document? Quote that whole line as JSON '
    '{"text": "<line>", "line": N}; if the page carries no date, answer {"answer": "not stated"}.',
    'Copy the line that records the date of this document exactly as written, as JSON '
    '{"text": "<line>", "line": N}, or answer {"answer": "not stated"} if no date appears on the page.',
    'Find the dating line of this document and quote it as JSON {"text": "<line>", "line": N}. If the page '
    'carries no date, answer {"answer": "not stated"}.')
_PARTY_WORDINGS = (
    PARTY_PROMPT,
    'Which line contains the self-identification of a party to this document? Quote it as JSON '
    '{"text": "<line>", "line": N}.',
    'Copy the line where a party to the document introduces themself, exactly as written, as JSON '
    '{"text": "<line>", "line": N}.',
    'Find the line in which one of the parties to the document states who they are, and quote it as JSON '
    '{"text": "<line>", "line": N}.')
_KETUBAH_NAME_WORDINGS = (
    KETUBAH_NAME_PROMPT,
    'Name the {role} of this marriage document, quoting the name as written, including any attached prefix '
    'letter, as JSON {{"text": "<name>", "line": N}}.',
    'What is the name of the {role}? Copy it exactly as written on the page, including any attached prefix '
    'letter, as JSON {{"text": "<name>", "line": N}}.',
    "Find the {role}'s name on the page and quote it exactly as written, including any attached prefix letter, "
    'as JSON {{"text": "<name>", "line": N}}.')
PARAPHRASES: Dict[str, Tuple[str, ...]] = {
    "qa_person": (
        PERSON_PROMPT,
        'Name the {role} of this document. Copy the name exactly as it is written on the page, including any '
        'attached prefix letter, as JSON {{"text": "<name as written>", "line": N}}.',
        'Which person is the {role}? Give the name as written on the page, including any attached prefix '
        'letter, as JSON {{"text": "<name as written>", "line": N}}.',
        'Find the name of the {role} on the page and quote it exactly as written, including any attached '
        'prefix letter, as JSON {{"text": "<name as written>", "line": N}}.'),
    "qa_date": _DATE_WORDINGS,
    "qa_abstain": _DATE_WORDINGS,          # the date question: the wording never tells the two branches apart
    "qa_ketubah_parties": (
        KETUBAH_PROMPT,
        'Quote the line of this marriage document that names the groom and the bride, as JSON '
        '{"text": "<line>", "line": N}.',
        'Which line names the couple, the groom and the bride? Answer with the whole line as JSON '
        '{"text": "<line>", "line": N}.',
        'Copy the line in which the groom and the bride are named, exactly as written, as JSON '
        '{"text": "<line>", "line": N}.'),
    "qa_party": _PARTY_WORDINGS,
    "qa_party_formula": _PARTY_WORDINGS,   # same question as qa_party
    "qa_witnesses_list": (
        WITNESS_LIST_PROMPT,
        'Who signed this document as witnesses? List every witness name exactly as written, including any '
        'attached prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.',
        'List all the witnesses whose signatures appear on this document, quoting each name exactly as written, '
        'including any attached prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.',
        'Give the names of the witnesses who signed this document as written on the page, including any '
        'attached prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.'),
    "qa_witness_line": (
        WITNESS_LINE_PROMPT,
        'Copy one of the lines that bear a witness signature, exactly as written, as JSON '
        '{"text": "<line>", "line": N}.',
        'Which line holds the signature of a witness? Quote one such line as JSON {"text": "<line>", "line": N}.',
        'Find a line in which one of the witnesses signs and quote it as JSON {"text": "<line>", "line": N}.'),
    "qa_parties_list": (
        PARTY_LIST_PROMPT,
        'Who are the parties to this document? List each name exactly as written, including any attached '
        'prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.',
        'List all the parties to this document, quoting each name as written on the page, including any '
        'attached prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.',
        'Give the names of the parties to this document as written on the page, including any attached '
        'prefix letter, as a JSON list of {"text": "<name as written>", "line": N}.'),
    "qa_party_line": (
        PARTY_LINE_PROMPT,
        'Copy one line of the page that names a party to this document, exactly as written, as JSON '
        '{"text": "<line>", "line": N}.',
        'Which line names one of the parties to this document? Quote one such line as JSON '
        '{"text": "<line>", "line": N}.',
        'Find a line in which a party to this document is named and quote it as JSON '
        '{"text": "<line>", "line": N}.'),
    "qa_date_month": (
        MONTH_PROMPT,
        'What month does the date of this document give? Quote the month name as written, including any '
        'attached prefix letter, as JSON {"text": "<month token>", "line": N}.',
        'Name the month in which this document was written, quoting it exactly as written, including any '
        'attached prefix letter, as JSON {"text": "<month token>", "line": N}.',
        "Quote the month of this document's date exactly as written on the page, including any attached "
        'prefix letter, as JSON {"text": "<month token>", "line": N}.'),
    "qa_date_year": (
        YEAR_PROMPT,
        'What year does the date of this document give? Quote the year expression as written, as JSON '
        '{"text": "<year expression>", "line": N}.',
        'Name the year in which this document was written, quoting it exactly as written, as JSON '
        '{"text": "<year expression>", "line": N}.',
        "Quote the year of this document's date exactly as written on the page, as JSON "
        '{"text": "<year expression>", "line": N}.'),
    "qa_ketubah_groom": _KETUBAH_NAME_WORDINGS,
    "qa_ketubah_bride": _KETUBAH_NAME_WORDINGS,
    "qa_place:written": (
        PLACE_WRITTEN_PROMPT,
        'In which place was this document written? Quote the place name exactly as written, including any '
        'attached prefix letter, as JSON {"text": "<as written>", "line": N}.',
        'Name the place where this document was written, quoting the word(s) as written on the page, '
        'including any attached prefix letter, as JSON {"text": "<as written>", "line": N}.',
        'At which place was this document written? Copy the word(s) naming it exactly as written, including '
        'any attached prefix letter, as JSON {"text": "<as written>", "line": N}.'),
    "qa_place:sent": (
        PLACE_SENT_PROMPT,
        'What is the destination of this document? Quote the place name exactly as written, including any '
        'attached prefix letter, as JSON {"text": "<as written>", "line": N}.',
        'Name the place this document was sent to, quoting the word(s) as written on the page, including any '
        'attached prefix letter, as JSON {"text": "<as written>", "line": N}.',
        'To which place was this document addressed? Copy the word(s) naming it exactly as written, including '
        'any attached prefix letter, as JSON {"text": "<as written>", "line": N}.'),
}

# ----------------------------------------------------------------------------- answers


class AnswerSpanError(ValueError):
    """An answer text that is not whole whitespace token(s) of its line."""


def check_whole_tokens(text: str, line: str) -> None:
    """Invariant: the answer is whole token(s) of the line exactly as written, never part of a word.

    :param text: Answer text.
    :param line: The line it quotes.
    :raises AnswerSpanError: When ``text`` is empty or does not sit between whitespace/line ends.
    """
    if not text or not re.search(r"(?:^|\s)" + re.escape(text) + r"(?:\s|$)", line):
        raise AnswerSpanError(f"answer {text!r} is not whole token(s) of the line {line!r}")


def name_as_written(line: str, span: str) -> Optional[str]:
    """A located name as whole token(s) of the line: the span widened to the surrounding whitespace,
    which takes in an attached prefix particle ("יוסף בן יעקב" in "ליוסף בן יעקב" -> "ליוסף בן יעקב")
    and glued punctuation or brackets, never other letters.

    The located span leaves the prefix out (``invert_names`` indexes tokens without it). An
    occurrence whose widening would take in letters other than one of
    :data:`invert_names.ALLOWED_PREFIXES` (a word joined by a hyphen, a longer word) is not used.

    :param line: Page line.
    :param span: Located name span.
    :returns: The first clean occurrence as written, or None.
    """
    i = line.find(span)
    while i >= 0:
        start = line.rfind(" ", 0, i) + 1
        end = line.find(" ", i + len(span))
        end = len(line) if end < 0 else end
        left = "".join(tokens(line[start:i]))
        if (not left or left in inv.ALLOWED_PREFIXES) and not tokens(line[i + len(span):end]):
            return line[start:end]
        i = line.find(span, i + 1)
    return None


def quote(lines: Sequence[str], n: int, text: str) -> str:
    """The JSON answer ``{"text": text, "line": n}`` (text first), enforcing the extractive contract.

    :param lines: Page transcription lines.
    :param n: One-based line index.
    :param text: Quoted span (whole token(s) of line ``n``).
    :returns: JSON answer.
    :raises AssertionError: When ``n`` is out of range.
    :raises AnswerSpanError: When ``text`` is not whole token(s) of line ``n``.
    """
    assert 1 <= n <= len(lines), f"line {n} out of range"
    check_whole_tokens(text, lines[n - 1])
    return json.dumps({"text": text, "line": n}, ensure_ascii=False)


def quote_list(lines: Sequence[str], items: Sequence[Tuple[int, str]]) -> str:
    """The JSON list answer ``[{"text": text, "line": n}, ...]``, each item asserted like :func:`quote`.

    :param lines: Page transcription lines.
    :param items: ``(one-based line, span)`` in answer order.
    :returns: JSON answer.
    :raises AssertionError: When the list is empty.
    :raises AnswerSpanError: When an item is not whole token(s) of its line.
    """
    assert items, "empty list answer"
    for n, text in items:
        quote(lines, n, text)
    return json.dumps([{"text": t, "line": n} for n, t in items], ensure_ascii=False)


def answer_items(answer: str) -> List[Dict[str, Any]]:
    """The ``{"text", "line"}`` items of a JSON answer (one for a span answer, several for a list).

    :param answer: JSON answer.
    :returns: Items (empty for the abstention answer).
    """
    obj = json.loads(answer)
    if isinstance(obj, list):
        return obj
    return [obj] if "line" in obj else []


_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_QUOTES = str.maketrans({"׳": "'", "’": "'", "‘": "'", "״": '"', "“": '"', "”": '"'})


def normalise_answer(text: str) -> str:
    """Fold an answer text for the per-answer cap: nikud stripped, typographic quotes mapped to ASCII,
    final letters folded, whitespace collapsed.

    The same folding as ``normalise_hebrew`` of the v22 task evaluation (a test keeps them equal),
    so "one answer" here is one answer to the scorer too.

    :param text: Answer text.
    :returns: Folded text.
    """
    return normalize_whitespace(strip_nikud(text or "").translate(_QUOTES).translate(_FINALS))


def answer_key(answer: str) -> str:
    """The identity of a JSON answer under the per-answer cap.

    :param answer: JSON answer.
    :returns: The folded item texts, sorted (a list answer is a set of names) and joined with " | ".
    """
    return " | ".join(sorted(normalise_answer(str(it["text"])) for it in answer_items(answer)))


def check_answer(family: str, answer: str, lines: Sequence[str]) -> None:
    """The emission invariant: every span answer is whole token(s) of its line exactly as written.

    :param family: Row family (only ``qa_abstain`` may carry no span).
    :param answer: JSON answer.
    :param lines: Page transcription lines.
    :raises AssertionError: When a non-abstention answer has no span or names a line off the page.
    :raises AnswerSpanError: When a span is not whole token(s) of its line.
    """
    items = answer_items(answer)
    assert items or family == "qa_abstain", f"{family} answer without a span"
    for it in items:
        assert 1 <= it["line"] <= len(lines), f"line {it['line']} out of range"
        check_whole_tokens(it["text"], lines[it["line"] - 1])


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


# ----------------------------------------------------------------------------- month / year spans

# number words (Hebrew, Aramaic, Judaeo-Arabic) that may form a written year; a leading ו is allowed
NUMBER_WORDS = set("""
אחד אחת שנים שתים שתי שני שלש שלוש שלשה שלושה שלשת שלושת ארבע ארבעה ארבעת חמש חמשה חמשת שש ששה ששת שבע
שבעה שבעת שמונה שמנה שמונת שמנת תשע תשעה תשעת עשר עשרה עשרת עשרים שלשים שלושים ארבעים חמשים חמישים ששים שישים
שבעים שמונים שמנים תשעים מאה מאת מאות מאתים אלף אלפים אלפי
חד חדא תרין תרתין תרי תלת תלתא תלתין ארבעא ארבעין חמשא חמשין שית שיתא שתין שבעא שבעין תמני תמניא תמנין תשעא
תשעין עסר עשרא עשרין מאא מאתן מאתין מאוון אלפא אלפין
ואחד אתנין אתנתין תלאת תלאתה כמס כמסה סת סתה סבע סבעה תמאן תמאניה תסע תסעה עשרה תלאתין כמסין סתין סבעין
תמאנין תסעין מאיה מאתי אלאף
""".split())
ADAR_QUALIFIERS = {"א", "ב", "ראשון", "שני", "הראשון", "השני", "הא", "הב", "קמא", "תניינא", "בתרא", "אלאול",
                   "אלתאני"}
_NUMERAL_LETTERS_RE = re.compile(r"[א-ט]?(?:ת{0,2}[קרש]?)[יכלמנסעפצךםןףץ]?[א-ט]?")
_MARKED_TOKEN_RE = re.compile(r"[א-ת]*[׳״'\"][א-ת׳״'\"]*")
# abbreviations that happen to read as letter numerals ("הנז׳" = the aforementioned)
_NUMERAL_STOP = {"הנז", "הנזל", "הנל", "הזה", "זו", "זאת", "הזאת"}
ERA_SPAN_MAX = 8               # tokens allowed between the year word and the era word
# a number-word year is taken only when a word like "years" or an era abbreviation closes it
# (otherwise a number word missing from NUMBER_WORDS, or a glued typo, would cut the year short)
YEAR_TERMINATORS = {"שנין", "שנים", "שנה", "שנא", "שנן", "סנין", "למנינא", "למניינא", "לבריתיה", "לבריתה", "לבריאה",
                    "לבריה", "לברית", "לשטרי", "ליצי", "לפק", "לפרט", "לחשבון"}
_DOUBLE_MARK_RE = re.compile(r"״|׳׳|''|\"")


def is_year_word(tok: str) -> bool:
    """A year word token: שנת-family (שנת, בשנת, דשנת, משנתינו, ...) or Judaeo-Arabic סנה/סנת/אלסנה.

    :param tok: Whitespace token.
    :returns: Whether it introduces a year.
    """
    lets = "".join(tokens(tok))
    return bool(_YEAR_WORD_RE.fullmatch(lets) or _JA_YEAR_WORD_RE.fullmatch(lets))


def numeral_letters(lets: str) -> bool:
    """The letters read as a Hebrew letter numeral (thousands, hundreds, tens, units in order).

    :param lets: Hebrew letters only.
    :returns: Whether they form a numeral of at least two letters.
    """
    return len(lets) >= 2 and lets not in _NUMERAL_STOP and bool(_NUMERAL_LETTERS_RE.fullmatch(lets))


def is_numeral(tok: str, unmarked_ok: bool = False) -> bool:
    """A numeral token: a letter numeral marked with geresh/gershayim (התקע״ח, ד׳תתי״א), a number
    word (אלף, ושלש, מאות, אלפא ...), or with ``unmarked_ok`` an unmarked letter numeral (אתתצט).

    :param tok: Whitespace token.
    :param unmarked_ok: Accept unmarked letter numerals (only inside an era-word span).
    :returns: Whether the token is a numeral.
    """
    core = tok.strip(".,:;")
    lets = "".join(tokens(core))
    if not lets:
        return False
    if _MARKED_TOKEN_RE.fullmatch(core):
        return numeral_letters(lets)
    word = lets[1:] if lets.startswith("ו") and lets[1:] in NUMBER_WORDS else lets
    return word in NUMBER_WORDS or (unmarked_ok and numeral_letters(lets))


def year_after(lines: Sequence[str], j: int, start: int) -> Optional[Tuple[int, str]]:
    """The year expression starting at token ``start`` of line ``j``.

    Through the era word (ליצירה/לשטרות/לבריאת/למניין/למנין) when it follows within
    :data:`ERA_SPAN_MAX` tokens and the span holds a numeral; otherwise 1-3 numeral tokens, which
    must end in a numeral closed by gershayim (התקס״ה) or be followed by a
    :data:`YEAR_TERMINATORS` word ("שנין", "לפ״ק" ...) on the line or at the start of the next line
    -- a run that simply stops could be a year cut short.

    :param lines: Page lines.
    :param j: Line (0-based).
    :param start: First token of the year.
    :returns: ``(line, span)`` or None.
    """
    rest = lines[j].split(" ")[start:]
    era = next((e for e, t in enumerate(rest[:ERA_SPAN_MAX + 1]) if "".join(tokens(t)) in YEAR_ERA_WORDS), None)
    if era == 0:
        return None
    if era is not None:
        if not any(is_numeral(t, unmarked_ok=True) for t in rest[:era]):
            return None
        span_toks = rest[:era + 1]
    else:
        run = []
        for t in rest:
            if not is_numeral(t):
                break
            run.append(t)
        if not 1 <= len(run) <= 3:
            return None
        if not _DOUBLE_MARK_RE.search(run[-1]):      # a numeral closed by gershayim ends the year by itself
            if len(run) < len(rest):
                if "".join(tokens(rest[len(run)])) not in YEAR_TERMINATORS:
                    return None
            elif j + 1 >= len(lines) or "".join(tokens(lines[j + 1].split(" ")[0])) not in (
                    YEAR_TERMINATORS | YEAR_ERA_WORDS):
                return None
        span_toks = run
    span = " ".join(span_toks)
    return None if GAP_TOKEN in span else (j, span)


def year_span(lines: Sequence[str], i: int) -> Optional[Tuple[int, str]]:
    """Find the year expression of the date on line ``i`` (the month line) or on the next line.

    A year word after the month token on line ``i`` is followed by its year on the same line, or,
    when the year word ends the line ("... תשרי שנת"), at the start of line ``i + 1``; without a year
    word on line ``i``, a year word on line ``i + 1`` is used.

    :param lines: Page lines.
    :param i: Month line (0-based).
    :returns: ``(line, span)`` or None.
    """
    toks = lines[i].split(" ")
    m_idx = next((k for k, t in enumerate(toks) if month_tokens(t)), -1)
    for k in range(m_idx + 1, len(toks)):
        if is_year_word(toks[k]):
            if k + 1 < len(toks):
                return year_after(lines, i, k + 1)
            return year_after(lines, i + 1, 0) if i + 1 < len(lines) and lines[i + 1] else None
    if i + 1 < len(lines):
        nxt = lines[i + 1].split(" ")
        for k, t in enumerate(nxt[:-1]):
            if is_year_word(t):
                return year_after(lines, i + 1, k + 1)
    return None


def pgp_month_line(lines: Sequence[str], meta: Dict[str, str]) -> Tuple[Optional[int], Optional[str]]:
    """The single date line carrying PGP's month (the ``qa_date`` rule without its gap/year checks).

    :param lines: Page lines.
    :param meta: PGP document row.
    :returns: ``(line index or None, PGP month or None)``.
    """
    month = pgp_month(meta.get("doc_date_original") or "")
    if month is None:
        return None, None
    hits = [i for i, ln in enumerate(lines) if is_date_line(ln) and month_tokens(ln, set(MONTHS[month]))]
    return (hits[0] if len(hits) == 1 else None), month


def month_row(page: Dict[str, Any], meta: Dict[str, str], stats: collections.Counter) -> Optional["QARow"]:
    """``qa_date_month``: the month token as written on the PGP-month date line.

    The answer is the whole token as written, with an attached prefix letter when there is one
    (לחדש אדר -> "אדר", ... בניסן -> "בניסן"); an Adar qualifier that follows it (אדר שני, אדר א׳) is
    part of the span. A gap elsewhere on the line is allowed.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    lines = page["lines"]
    i, month = pgp_month_line(lines, meta)
    if i is None:
        return None
    toks = lines[i].split(" ")
    written = {p + s for p in _MONTH_PREFIXES for s in MONTHS[month]}
    for k, t in enumerate(toks):
        core = t.strip(".,:;")                 # letters only: a bracketed/abbreviated token is not quoted
        if core not in written:
            continue
        span = t
        if month == "adar" and t == core and k + 1 < len(toks) and "".join(tokens(toks[k + 1])) in ADAR_QUALIFIERS:
            span = f"{t} {toks[k + 1]}"
        return QARow("qa_date_month", "month", MONTH_PROMPT, quote(lines, i + 1, span), i + 1,
                     {"doc_date_original": meta.get("doc_date_original"), "pgp_month": month,
                      "date_line": lines[i]}, priority=PRIORITY["qa_date_month"])
    stats["month_skip_no_standalone_month_token"] += 1
    return None


def year_row(page: Dict[str, Any], meta: Dict[str, str], stats: collections.Counter) -> Optional["QARow"]:
    """``qa_date_year``: the year expression of the document's date line (or the line after it).

    The date line is the PGP-month date line, or the page's only date line when PGP's date has no
    month; PGP must record a date (the value is not compared: no calendar conversion).

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    if not ((meta.get("doc_date_original") or "").strip() or (meta.get("doc_date_standard") or "").strip()):
        return None
    lines = page["lines"]
    i, month = pgp_month_line(lines, meta)
    if i is None and month is None:
        date_lines = [k for k, ln in enumerate(lines) if is_date_line(ln)]
        i = date_lines[0] if len(date_lines) == 1 else None
    if i is None:
        return None
    found = year_span(lines, i)
    if found is None:
        stats["year_skip_no_year_expression"] += 1
        return None
    j, span = found
    return QARow("qa_date_year", "year", YEAR_PROMPT, quote(lines, j + 1, span), j + 1,
                 {"doc_date_original": meta.get("doc_date_original"),
                  "doc_date_standard": meta.get("doc_date_standard"), "month_line": i + 1,
                  "year_on_next_line": j != i}, priority=PRIORITY["qa_date_year"])


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


_FORMULA_RE = re.compile(r"(?:^|\s)ו?א[י]?ך\s|אמר\s+(?:לה|להדא)|לאנת[ו]|לאינתו|^\s*ו?(?:אלחתן|אלכלה)(?:\s|$)")
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
    :param prompt_args: Fills the family's :data:`PARAPHRASES` templates (``{"role": ...}`` for
        persons and ketubba names; empty when the wordings are plain strings).
    """

    family: str
    section: str
    question: str
    answer: str
    line: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: int = 9
    prompt_args: Dict[str, str] = field(default_factory=dict)


@dataclass
class QAContext:
    """Per-run inputs.

    :param pgp_docs: ``documents.csv`` by pgpid.
    :param relations: PGP relation rows by pgpid.
    :param by_name: people.csv by name.
    :param by_slug: people.csv by slug.
    :param edition_lines: pgpid -> every raw + rendered edition line (all sides).
    :param places: places.csv rows by place name.
    """

    pgp_docs: Dict[str, Dict[str, str]]
    relations: Dict[str, List[Dict[str, str]]]
    by_name: Dict[str, Dict]
    by_slug: Dict[str, Dict]
    edition_lines: Dict[str, List[str]]
    places: Dict[str, Dict[str, str]] = field(default_factory=dict)


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
        text = name_as_written(lines[hit.line - 1], hit.span)
        if text is None:
            stats["person_skip_not_whole_token"] += 1
            continue
        rows.append(QARow("qa_person", base.lower().replace(" ", "_"), PERSON_PROMPT.format(role=PERSON_ROLES[base]),
                          quote(lines, hit.line, text), hit.line,
                          {"relation": r["relation"], "person_name": r["person_name"], "person_slug": r["person_slug"],
                           "query": hit.query, "tier": hit.tier, "kind": hit.kind, "source": hit.source},
                          priority=PRIORITY["qa_person"] + ROLE_PRIORITY.index(base),
                          prompt_args={"role": PERSON_ROLES[base]}))
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
    couple, hits = ketubah_hits(page, meta, index)
    if not couple:
        stats["ketubah_skip_names_not_parsed"] += 1
        return None
    lines = page["lines"]
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


_NAME_CONNECTORS = {"בן", "בר", "ביר", "ברבי"}


def has_name_pattern(line: str, connectors: Set[str] = frozenset(_NAME_CONNECTORS)) -> bool:
    """The line holds ``<token> (בן|בר|ביר|ברבי) <token>`` (a signature-like name; any token, a gap too).

    :param line: Text.
    :param connectors: Connector words (letters only).
    :returns: Whether the pattern occurs.
    """
    toks = line.split(" ")
    return any("".join(tokens(toks[k])) in connectors and toks[k - 1] and toks[k + 1]
               for k in range(1, len(toks) - 1))


def signature_region(page: Dict[str, Any], window: int = SIGNATURE_WINDOW) -> List[int]:
    """Lines where witnesses sign: the last ``window`` main-text lines plus every non-main line.

    :param page: Editions manifest record (``regions`` per line).
    :param window: Main-text lines counted from the end.
    :returns: 0-based line indices.
    """
    regions = page["regions"]
    main = [i for i, r in enumerate(regions) if r == "main"]
    return sorted(set(main[-window:]) | {i for i, r in enumerate(regions) if r != "main"})


def name_occurrences(line: str, connectors: Set[str] = frozenset(_NAME_CONNECTORS)) -> List[Tuple[int, int]]:
    """Character ranges of every ``<token> <connector> <token>`` in a line.

    :param line: Text.
    :param connectors: Connector words (letters only).
    :returns: ``(start, end)`` ranges.
    """
    toks = line.split(" ")
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t) + 1
    return [(starts[k - 1], starts[k + 1] + len(toks[k + 1])) for k in range(1, len(toks) - 1)
            if "".join(tokens(toks[k])) in connectors and toks[k - 1] and toks[k + 1]]


def covered(line: str, rng: Tuple[int, int], spans: Sequence[str]) -> bool:
    """Some located span overlaps the character range (a name chain ``X בר Y בר Z`` counts as covered).

    :param line: Text.
    :param rng: ``(start, end)``.
    :param spans: Located spans.
    :returns: Whether the occurrence is accounted for.
    """
    for sp in spans:
        i = line.find(sp)
        while i >= 0:
            if i < rng[1] and i + len(sp) > rng[0]:
                return True
            i = line.find(sp, i + 1)
    return False


def witnesses_complete(page: Dict[str, Any], spans: Sequence[str]) -> bool:
    """Every signature-like name in the signature region is one of the located witness spans.

    Checked per name, not per line, so a line with two signatures of which PGP lists one fails.

    :param page: Editions manifest record.
    :param spans: Located witness spans.
    :returns: Whether PGP's witness list accounts for every signature on the page.
    """
    lines = page["lines"]
    return all(covered(lines[i], rng, spans) for i in signature_region(page) for rng in name_occurrences(lines[i]))


def parties_complete(page: Dict[str, Any], spans: Sequence[str]) -> bool:
    """Every ``X בן|בר|ביר|ברבי|בת Y`` name on the page is a located span of some PGP relation.

    A document naming someone PGP does not account for may have a party PGP missed (e.g. the
    declarant of "יקול X בן Y ... ואקפת Z"), so the party list is not emitted.

    :param page: Editions manifest record.
    :param spans: Located spans of every located relation of the document.
    :returns: Whether every name on the page is accounted for.
    """
    return all(covered(ln, rng, spans) for ln in page["lines"]
               for rng in name_occurrences(ln, _NAME_CONNECTORS | {"בת"}))


SET_ROLES = {"Witness": ("qa_witnesses_list", "qa_witness_line", WITNESS_LIST_PROMPT, WITNESS_LINE_PROMPT, "witness"),
             "Party": ("qa_parties_list", "qa_party_line", PARTY_LIST_PROMPT, PARTY_LINE_PROMPT, "party")}


def set_valued_rows(page: Dict[str, Any], ctx: QAContext, located: Sequence[Tuple[Dict[str, str], Located]],
                    role: str, stats: collections.Counter) -> Tuple[List[QARow], Dict[str, bool]]:
    """List row (every holder, complete) or single-line fallback for a role PGP gives several holders.

    :param page: Editions manifest record.
    :param ctx: Context.
    :param located: Relations located exactly on the page (certain relations only).
    :param role: ``"Witness"`` or ``"Party"``.
    :param stats: Counters (updated).
    :returns: ``(rows, funnel flags has_role / all_located / complete)``.
    """
    lines = page["lines"]
    list_family, line_family, list_prompt, line_prompt, section = SET_ROLES[role]
    holders = [r for r in ctx.relations.get(page["pgpid"], []) if inv.base_relation(r["relation"]) == role]
    flags = {"has_role": len(holders) >= 2, "all_located": False, "complete": False}
    if len(holders) < 2:
        return [], flags
    hit_of = {id(r): h for r, h in located}
    hits = [hit_of.get(id(r)) for r in holders]          # uncertain relations were never located
    region = set(signature_region(page)) if role == "Witness" else None
    usable = [h for h in hits if h and single_line_of(h.span, lines) and GAP_TOKEN not in lines[h.line - 1]
              and (region is None or h.line - 1 in region)]
    usable.sort(key=lambda h: (h.line, lines[h.line - 1].find(h.span)))
    flags["all_located"] = all(hits)
    if flags["all_located"]:
        spans = [h.span for h in hits]
        if role == "Witness":
            flags["complete"] = witnesses_complete(page, spans)
        else:
            flags["complete"] = parties_complete(page, [h.span for _, h in located])
    distinct = len({(h.line, h.span) for h in usable}) == len(usable)
    reason = "not_all_located" if not flags["all_located"] else "incomplete" if not flags["complete"] else "line_rule"
    if flags["complete"] and len(usable) == len(holders) and distinct:
        items = [(h.line, name_as_written(lines[h.line - 1], h.span)) for h in usable]
        if all(text for _, text in items):
            return [QARow(list_family, f"{section}_all", list_prompt, quote_list(lines, items), items[0][0],
                          {"relation": role, "holders": [r["person_name"] for r in holders],
                           "spans": [h.span for h in usable], "queries": [h.query for h in usable]},
                          priority=PRIORITY[list_family])], flags
        reason = "not_whole_token"
    stats[f"{section}_list_not_emitted:{reason}"] += 1
    if not usable:
        return [], flags
    h = usable[0]
    return [QARow(line_family, f"{section}_any", line_prompt, quote(lines, h.line, lines[h.line - 1]), h.line,
                  {"relation": role, "holders": [r["person_name"] for r in holders], "span": h.span,
                   "query": h.query}, priority=PRIORITY[line_family])], flags


def ketubah_hits(page: Dict[str, Any], meta: Dict[str, str],
                 index: PageIndex) -> Tuple[Dict[str, str], Dict[str, Located]]:
    """Groom and bride parsed from a marriage-document description, and those located exactly.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param index: Page index.
    :returns: ``(parsed couple, role -> located name)``; both empty for other documents.
    """
    if not _KETUBAH_RE.match(meta.get("description") or ""):
        return {}, {}
    couple = parse_couple(meta.get("description") or "")
    hits = {role: locate_exact([name], [], "Party", index, page["lines"]) for role, name in couple.items()}
    return couple, {k: v for k, v in hits.items() if v}


def ketubah_name_rows(page: Dict[str, Any], meta: Dict[str, str], index: PageIndex,
                      stats: collections.Counter) -> List[QARow]:
    """``qa_ketubah_groom`` / ``qa_ketubah_bride``: the name located exactly on any line.

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param index: Page index.
    :param stats: Counters (updated).
    :returns: Rows (at most one per role).
    """
    lines = page["lines"]
    couple, hits = ketubah_hits(page, meta, index)
    rows = []
    for role, h in hits.items():
        if not single_line_of(h.span, lines) or GAP_TOKEN in lines[h.line - 1]:
            stats[f"ketubah_{role}_skip_span_or_gap"] += 1
            continue
        text = name_as_written(lines[h.line - 1], h.span)
        if text is None:
            stats[f"ketubah_{role}_skip_not_whole_token"] += 1
            continue
        fam = f"qa_ketubah_{role}"
        rows.append(QARow(fam, role, KETUBAH_NAME_PROMPT.format(role=role), quote(lines, h.line, text), h.line,
                          {"description": (meta.get("description") or "")[:300], "parsed": couple.get(role),
                           "query": h.query, "tier": h.tier, "kind": h.kind}, priority=PRIORITY[fam],
                          prompt_args={"role": role}))
    return rows


# qa_party_formula (round 2 of the spot check, docs: qa_party_formula_spotcheck.md): the plural
# "we" formulas (חצר אלינא אנן חתומי מטה X, שהדותא ... אנן שהדי) introduce a party in the third
# person and were the main false hits of round 1 (precision 10/22), so only the singular אנא with the
# name in apposition, or מודה/מודים + name, in legal documents, without court/witness markers
FORMULA_PRONOUNS = {"אנא"}
FORMULA_VERBS = {"מודה", "מודים"}
FORMULA_CONNECTORS = {"בן", "בר", "ביר", "בת", "ברת"}
FORMULA_MIN_LETTERS = 12
FORMULA_APPOSITION = 3         # the connector sits at most this many tokens after the pronoun / verb
_COURT_MARKER_RE = re.compile(r"חתומי|חתמי|שהדי|שהד|דינא|^חצר|^אחצר")


def party_formula_line(line: str) -> bool:
    """A party's self-identification line, recognised by its wording alone.

    ``אנא`` within the first 4 tokens, or ``מודה``/``מודים`` anywhere, followed within
    :data:`FORMULA_APPOSITION` tokens by a name ``<token> (בן|בר|ביר|בת|ברת) <token>`` whose first
    token does not open with the conjunction ו ("אנא ועמאר בר ..." = "I and ʿAmmār ..."); no token
    marks a court or witness formula (חתומי, שהדי, דינא, חצר ...); no ``[...]``; at least
    :data:`FORMULA_MIN_LETTERS` Hebrew letters.

    :param line: Page line.
    :returns: Whether the line matches.
    """
    if GAP_TOKEN in line or len("".join(tokens(line))) < FORMULA_MIN_LETTERS:
        return False
    toks = line.split(" ")
    lets = ["".join(tokens(t)) for t in toks]
    if any(_COURT_MARKER_RE.search(t) for t in lets):
        return False
    starts = [k for k in range(min(4, len(toks))) if lets[k] in FORMULA_PRONOUNS]
    starts += [k for k in range(len(toks)) if lets[k] in FORMULA_VERBS]
    for k in starts:
        first = k + 2 if k + 1 < len(toks) and lets[k + 1] in {"אני", "אנא"} else k + 1
        if first >= len(toks) or not lets[first] or lets[first].startswith("ו"):
            continue
        if any(lets[c] in FORMULA_CONNECTORS and toks[c + 1] for c in range(first + 1, min(first + FORMULA_APPOSITION,
                                                                                              len(toks) - 1))):
            return True
    return False


def party_formula_row(page: Dict[str, Any], meta: Dict[str, str], stats: collections.Counter) -> Optional[QARow]:
    """``qa_party_formula``: the single self-identification line of a legal document page.

    Emitted only when exactly one line of the page matches :func:`party_formula_line` (so "the
    line" is unambiguous) and the page has no ``qa_party`` row (same question).

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param stats: Counters (updated).
    :returns: The row, or None.
    """
    if "Legal" not in (meta.get("type") or ""):
        return None
    lines = page["lines"]
    hits = [i for i, ln in enumerate(lines) if party_formula_line(ln)]
    if len(hits) != 1:
        if hits:
            stats["party_formula_skip_several_lines"] += 1
        return None
    i = hits[0]
    return QARow("qa_party_formula", "party_formula", PARTY_PROMPT, quote(lines, i + 1, lines[i]), i + 1,
                 {"rule": "formula wording only (no PGP person located)", "type": meta.get("type")},
                 priority=PRIORITY["qa_party_formula"])


# ----------------------------------------------------------------------------- places

# Hebrew-script names of common places (keys: folded PGP names); places.csv adds its own variants
PLACE_TABLE: Dict[str, List[str]] = {
    "fustat": ["פסטאט", "פוסטאט", "פסטט", "מצרים"], "cairo": ["אלקאהרה", "קאהרה"],
    "alexandria": ["אלאסכנדריה", "אסכנדריה", "נא אמון"], "jerusalem": ["ירושלים", "אלקדס", "בית המקדש"],
    "damascus": ["דמשק"], "tyre": ["צור"], "al-ramla": ["רמלה", "אלרמלה"], "ramla": ["רמלה", "אלרמלה"],
    "aden": ["עדן"], "qayrawan": ["קירואן", "אלקירואן"], "al-mahdiyya": ["אלמהדיה"], "mahdiyya": ["אלמהדיה"],
    "sicily": ["צקליה", "סקליה"], "palermo": ["צקליה", "סקליה", "פלרמו"], "ascalon": ["אשקלון", "עסקלאן"],
    "tiberias": ["טבריה"], "baghdad": ["בגדאד"], "byzantium": ["ארץ אדום", "רום", "קסטנטיניה"],
    "spain": ["ספרד", "אלאנדלס"], "al-andalus": ["ספרד", "אלאנדלס"], "almeria": ["אלמריה"], "denia": ["דאניה"],
    "tinnis": ["תניס"], "damietta": ["דמיאט"], "minyat zifta": ["מנית זפתא", "מניה זפתי", "מניה זפתא"],
    "bilbays": ["בלביס"], "sunbat": ["סנבאט"],
}
# names that are also ordinary words (rock, height, milk, judge, desire, lady): never matched alone
PLACE_STOP = {"צור", "רום", "חלב", "דן", "חמדה", "בעלת"}
# a place token after these words is not the place (נוחו עדן = "rest in Eden", יציאת מצרים = the Exodus)
PLACE_CONTEXT_STOP = {"עדן": {"נוחו", "גן", "בגן", "וגן"}, "מצרים": {"ארץ", "מארץ", "יציאת", "ביציאת", "ליציאת"}}
_PLACE_PREFIXES = ("", "ב", "ל", "מ", "ו", "ד", "וב", "ול", "ומ", "וד")


def place_spellings(name: str, places: Dict[str, Dict[str, str]]) -> List[str]:
    """Hebrew-script spellings of a PGP place: the table plus places.csv Hebrew/Arabic-script variants
    (and the article written apart: אלקאהרה -> אל קאהרה).

    :param name: PGP place name (e.g. "Fustat", "al-Mahdiyya").
    :param places: places.csv rows by name.
    :returns: Distinct spellings (Arabic script mapped to Hebrew letters), stop words removed.
    """
    out = list(PLACE_TABLE.get(fold(name).strip(), []))
    for v in re.split(r"[;,|]", (places.get(name) or {}).get("name_variants") or ""):
        v = v.strip()
        if inv.ARABIC_RE.search(v):
            v = inv.arabic_to_hebrew_script(v)
        if v and re.fullmatch(r"[א-ת ]+", v):
            out.append(v)
    out += ["אל " + sp[2:] for sp in out if sp.startswith("אל") and len(sp) > 4 and " " not in sp]  # "אל קאהרה"
    return [sp for sp in dict.fromkeys(out) if sp not in PLACE_STOP]


def place_hits(lines: Sequence[str], spellings: Sequence[str]) -> List[Tuple[int, str]]:
    """Exact token matches of place spellings (a ב/ל/מ/ו/ד prefix on the first token is allowed).

    :param lines: Page lines.
    :param spellings: Place spellings.
    :returns: ``(0-based line, whole token(s) as written, prefix included)`` in page order.
    """
    out = []
    for i, ln in enumerate(lines):
        toks = ln.split(" ")
        lets = ["".join(tokens(t)) for t in toks]
        for k in range(len(toks)):
            for sp in spellings:
                words = sp.split(" ")
                if k + len(words) > len(toks) or lets[k + 1:k + len(words)] != words[1:]:
                    continue
                pre = next((p for p in _PLACE_PREFIXES if lets[k] == p + words[0]), None)
                if pre is None or (k and lets[k - 1] in PLACE_CONTEXT_STOP.get(words[0], set())):
                    continue
                out.append((i, " ".join(toks[k:k + len(words)])))    # whole tokens: letters == prefix + place
    return out


def place_rows(page: Dict[str, Any], meta: Dict[str, str], places: Dict[str, Dict[str, str]],
               stats: collections.Counter, funnel: Optional[Dict[str, Set[str]]] = None) -> List[QARow]:
    """``qa_place``: where the document was written (PGP origin/location) or sent (destination).

    Each question needs a single PGP place, a match on exactly one line, no ``[...]`` in that line,
    and no line holding both a written-place and a sent-place match (then both are skipped).

    :param page: Editions manifest record.
    :param meta: PGP document row.
    :param places: places.csv rows by name.
    :param stats: Counters (updated).
    :param funnel: ``stage -> pgpids`` (updated when given).
    :returns: Rows (at most one per role).
    """
    lines = page["lines"]
    roles = {"written": {v.strip() for f in ("origin", "location") for v in re.split(r",\s*", meta.get(f) or "")
                         if v.strip()},
             "sent": {v.strip() for v in re.split(r",\s*", meta.get("destination") or "") if v.strip()}}
    if not any(roles.values()):
        return []
    if funnel is not None:
        funnel["with_place"].add(page["pgpid"])
    found: Dict[str, List[Tuple[int, str]]] = {}
    for role, names in roles.items():
        if len(names) == 1:
            found[role] = place_hits(lines, place_spellings(next(iter(names)), places))
        elif names:
            stats[f"place_{role}_skip_several_pgp_places"] += 1
    if funnel is not None and any(found.values()):
        funnel["located"].add(page["pgpid"])
    shared = {i for i, _ in found.get("written", [])} & {i for i, _ in found.get("sent", [])}
    rows = []
    for role, hits in found.items():
        if not hits:
            continue
        line_set = {i for i, _ in hits}
        if shared:
            stats["place_skip_written_and_sent_on_one_line"] += 1
        elif len(line_set) > 1:
            stats[f"place_{role}_skip_on_several_lines"] += 1
        elif GAP_TOKEN in lines[hits[0][0]]:
            stats[f"place_{role}_skip_gap_in_line"] += 1
        else:
            i, span = hits[0]
            rows.append(QARow("qa_place", role, PLACE_WRITTEN_PROMPT if role == "written" else PLACE_SENT_PROMPT,
                              quote(lines, i + 1, span), i + 1,
                              {"pgp_place": next(iter(roles[role])), "fields": "origin/location" if role == "written"
                               else "destination"}, priority=PRIORITY["qa_place"]))
    return rows


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


def page_candidates(page: Dict[str, Any], ctx: QAContext, stats: collections.Counter,
                    funnel: Optional[Dict[str, Dict[str, Set[str]]]] = None) -> Tuple[List[QARow], bool]:
    """All validated QA rows of one page (before caps) and its abstention eligibility.

    :param page: Editions manifest record.
    :param ctx: Context.
    :param stats: Counters (updated).
    :param funnel: ``role -> stage -> pgpids`` for the witness/party funnel (updated when given).
    :returns: ``(rows sorted by priority, abstain eligible)``.
    """
    meta = ctx.pgp_docs.get(page["pgpid"], {})
    index = page_index(page["lines"])
    rows, located = person_rows(page, ctx, index, stats)
    party = party_row(page, meta, located, stats)
    for extra in (date_row(page, meta, stats), ketubah_row(page, meta, index, stats), party,
                  month_row(page, meta, stats), year_row(page, meta, stats),
                  None if party else party_formula_row(page, meta, stats)):
        if extra:
            rows.append(extra)
    rows += ketubah_name_rows(page, meta, index, stats)
    rows += place_rows(page, meta, ctx.places, stats, funnel.get("Place") if funnel is not None else None)
    for role in SET_ROLES:
        set_rows, flags = set_valued_rows(page, ctx, located, role, stats)
        rows += set_rows
        if funnel is not None and role in funnel:
            for stage, ok in flags.items():
                if ok:
                    funnel[role][stage].add(page["pgpid"])
    rows.sort(key=lambda r: (r.priority, r.line))
    return rows, abstain_eligible(page, meta, ctx)


def answer_limit(n_family: int, share: float) -> int:
    """Most facts one answer may have in a family of ``n_family`` train facts (never below one).

    :param n_family: The family's train facts.
    :param share: Maximum answer share.
    :returns: ``max(1, floor(share * n_family))``.
    """
    return max(1, math.floor(share * n_family + 1e-9))


def cap_answer_shares(pages: Sequence[Dict[str, Any]], cands: Dict[str, List[QARow]], max_rows: int,
                      share: float, dropped: Optional[collections.Counter] = None) -> Dict[str, List[QARow]]:
    """Drop train candidates whose answer exceeds ``share`` of its family's train facts.

    The count is taken on what the per-page cut would keep (the first ``max_rows`` remaining
    candidates of every train page). An answer (:func:`answer_key`) above :func:`answer_limit`
    keeps the candidates with the lowest ``stable_hash`` and the rest are removed from the candidate
    lists, so each page's next candidate moves up into the freed slot. Dropping shrinks the family
    and a refill can add rows, so the count repeats until no answer is above its limit; candidates
    only ever leave, so this ends. Val pages and ``qa_abstain`` are never touched.

    :param pages: Manifest records (``key`` and ``split`` set).
    :param cands: ``page key -> candidate rows`` sorted by priority.
    :param max_rows: Facts per page image.
    :param share: Maximum answer share within a family.
    :param dropped: ``family -> candidates dropped`` (updated when given).
    :returns: ``page key -> remaining candidates`` (same order).
    """
    train = [p["key"] for p in pages if p.get("split") != "val"]
    gone: Set[Tuple[str, int]] = set()
    while True:
        groups: Dict[Tuple[str, str], List[Tuple[int, str, int]]] = collections.defaultdict(list)
        n_family: collections.Counter = collections.Counter()
        for k in train:
            live = [(i, r) for i, r in enumerate(cands.get(k, [])) if (k, i) not in gone][:max_rows]
            for i, r in live:
                if r.family == "qa_abstain":
                    continue
                a = answer_key(r.answer)
                groups[(r.family, a)].append((stable_hash(f"{SEED}:answer_cap:{k}:{r.family}:{r.section}:{a}"), k, i))
                n_family[r.family] += 1
        excess = [m for (fam, _), members in groups.items()
                  for m in sorted(members)[answer_limit(n_family[fam], share):]]
        if not excess:
            break
        gone.update((k, i) for _, k, i in excess)
    if dropped is not None:
        dropped.update(cands[k][i].family for k, i in gone)
    return {k: [r for i, r in enumerate(v) if (k, i) not in gone] for k, v in cands.items()}


def abstain_budget(n_date: int, ratio: float, n_pages: int, page_share: float, n_open: int) -> int:
    """How many abstention facts to add.

    :param n_date: ``qa_date`` facts kept (all splits).
    :param ratio: Abstention facts per ``qa_date`` fact.
    :param n_pages: Pages in the build.
    :param page_share: Maximum share of pages carrying an abstention fact.
    :param n_open: Eligible pages that still have a free slot.
    :returns: ``min(floor(ratio * n_date), floor(page_share * n_pages), n_open)``.
    """
    return max(0, min(math.floor(ratio * n_date + 1e-9), math.floor(page_share * n_pages + 1e-9), n_open))


def apply_caps(pages: Sequence[Dict[str, Any]], cands: Dict[str, List[QARow]], eligible: Dict[str, bool],
               max_rows: int = MAX_ROWS_PER_PAGE, abstain_ratio: float = ABSTAIN_RATIO,
               abstain_page_share: float = ABSTAIN_MAX_PAGE_SHARE, max_answer_share: float = MAX_ANSWER_SHARE,
               info: Optional[Dict[str, Any]] = None) -> Dict[str, List[QARow]]:
    """Per-answer cap on the train candidates, the per-page cut, then the abstention facts.

    :param pages: Manifest records (``key`` and ``split`` set).
    :param cands: ``page key -> candidate rows`` sorted by priority.
    :param eligible: ``page key -> abstain eligible``.
    :param max_rows: Facts per page image.
    :param abstain_ratio: Abstention facts per kept ``qa_date`` fact.
    :param abstain_page_share: Maximum share of pages carrying an abstention fact.
    :param max_answer_share: Maximum share of one answer within a family's train facts.
    :param info: Filled with ``answer_cap_dropped`` (family -> candidates) and ``abstain`` (budget
        arithmetic) when given.
    :returns: ``page key -> kept facts``.
    """
    dropped: collections.Counter = collections.Counter()
    capped = cap_answer_shares(pages, cands, max_rows, max_answer_share, dropped)
    kept = {p["key"]: list(capped.get(p["key"], []))[:max_rows] for p in pages}
    n_date = sum(r.family == "qa_date" for v in kept.values() for r in v)
    order = sorted((p["key"] for p in pages if eligible.get(p["key"]) and len(kept[p["key"]]) < max_rows),
                   key=lambda k: stable_hash(f"{SEED}:abstain:{k}"))
    budget = abstain_budget(n_date, abstain_ratio, len(pages), abstain_page_share, len(order))
    for k in order[:budget]:
        kept[k].append(QARow("qa_abstain", "no_date", DATE_PROMPT, ABSTAIN_ANSWER, 0,
                             {"pgp_date_fields": "none", "date_indications_in_edition": "none"}, priority=8))
    if info is not None:
        info["answer_cap_dropped"] = dict(dropped)
        info["abstain"] = {"ratio": abstain_ratio, "qa_date_facts": n_date,
                           "ratio_budget": math.floor(abstain_ratio * n_date + 1e-9),
                           "page_share_budget": math.floor(abstain_page_share * len(pages) + 1e-9),
                           "eligible_pages": sum(bool(eligible.get(p["key"])) for p in pages),
                           "eligible_pages_with_free_slot": len(order), "facts": budget}
    return kept


def row_wordings(row: QARow) -> List[str]:
    """Every wording of a row's question from :data:`PARAPHRASES`, the canonical one first.

    :param row: QA row (family and, for ``qa_place``, section pick the table; ``prompt_args`` fill it).
    :returns: The wordings.
    :raises AssertionError: When the canonical wording is not the row's question.
    """
    table = PARAPHRASES.get(f"{row.family}:{row.section}") or PARAPHRASES[row.family]
    out = [t.format(**row.prompt_args) if row.prompt_args else t for t in table]
    assert out[0] == row.question, f"{row.family}/{row.section}: the question is not the canonical wording"
    return out


def draw_wordings(n: int, k: int, fact_key: str) -> List[int]:
    """``k`` distinct wording indices out of ``n``, drawn deterministically per fact.

    :param n: Wordings available.
    :param k: Wordings wanted.
    :param fact_key: The fact's identity (its stem).
    :returns: Indices in draw order.
    :raises AssertionError: Unless ``1 <= k <= n``.
    """
    assert 1 <= k <= n, f"{k} wordings requested, {n} available"
    return sorted(range(n), key=lambda v: stable_hash(f"{SEED}:paraphrase:{fact_key}:{v}"))[:k]


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
    facts = stats.get("facts_by_family", stats["rows_by_family"])
    out = [f"# {stats.get('dataset', 'pgp_qa')} — human review sample", "",
           f"**Status: {stats['status']}.**", "",
           f"Built {stats['generated_at']} from `{stats['editions_manifest']}`. Facts per family: "
           f"{json.dumps(facts)}. Sample: {len(sample)} facts, stratified by family (a train fact is "
           f"emitted once per question wording; the other wordings are listed under it).", "",
           "Each row shows the page image link, the question, the target JSON, the full page line the "
           "answer quotes, and the PGP evidence that validated it. Mark rows that are wrong.", ""]
    fam = None
    for i, r in enumerate(sample, 1):
        if r["family"] != fam:
            fam = r["family"]
            out += [f"## {fam}", ""]
        out += [f"### {i}. `{r['stem']}` — {r['canonical_id']} (image {r['image_index']})", "",
                f"- image: {r['image_url']}", f"- question: {r['question']}"]
        out += [f"- also asked as: {q}" for q in r.get("questions", [])[1:]]
        out += [f"- answer: `{r['answer']}`",
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


def emit_rows(pages: Sequence[Dict[str, Any]], kept: Dict[str, List[QARow]],
              paraphrases: int = PARAPHRASES_PER_FACT) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Dataset rows and the per-fact manifest of the kept facts.

    A train fact becomes ``paraphrases`` rows whose questions are distinct wordings drawn by
    :func:`draw_wordings`; the twins share image, answer, task, section and stem. A val fact becomes
    one row with the canonical wording. Every answer is re-checked (:func:`check_answer`) first.

    :param pages: Manifest records (``key`` set).
    :param kept: ``page key -> kept facts``.
    :param paraphrases: Wordings per train fact.
    :returns: ``(split name -> rows, one manifest record per fact)``.
    :raises AnswerSpanError: When an answer is not whole token(s) of its line (a bug, never a skip).
    """
    split_rows: Dict[str, List[Dict[str, Any]]] = {f"train_{f}": [] for f in FAMILIES}
    split_rows["val"] = []
    manifest: List[Dict[str, Any]] = []
    for p in pages:
        is_val = p["split"] == "val"
        for j, q in enumerate(kept[p["key"]]):
            stem = f"pgpqa_{p['pgpid']}_{p['image_index']}_{q.family}_{j}"
            check_answer(q.family, q.answer, p["lines"])     # raises: a violation is a bug, never a skip
            wordings = row_wordings(q)
            variants = [0] if is_val else draw_wordings(len(wordings), paraphrases, stem)
            for v in variants:
                split_rows["val" if is_val else f"train_{q.family}"].append(
                    {"image": p["image_path"], "question": wordings[v], "answer": q.answer, "task": q.family,
                     "section": q.section, "stem": stem, "label_source": LABEL_SOURCE,
                     "target_chars": len(q.answer), "target_tokens": 0, "image_width": p["image_width"],
                     "image_height": p["image_height"]})
            items = answer_items(q.answer)
            manifest.append({"stem": stem, "family": q.family, "section": q.section, "pgpid": p["pgpid"],
                             "canonical_id": p["canonical_id"], "image_index": p["image_index"],
                             "image_url": p["image_url"], "split": p["split"], "question": wordings[variants[0]],
                             "questions": [wordings[v] for v in variants], "prompt_variants": variants,
                             "answer": q.answer, "line": q.line,
                             "answer_line": " || ".join(f"[{n}] {p['lines'][n - 1]}" for n in
                                                        sorted({it["line"] for it in items})) if len(items) > 1
                             else (p["lines"][q.line - 1] if q.line else ""),
                             "line_countable_from_top": bool(q.line and p["countable"][q.line - 1]),
                             "line_region": p["regions"][q.line - 1] if q.line else "", "evidence": q.evidence})
    return split_rows, manifest


def top_answers(manifest: Sequence[Dict[str, Any]], n: int = 5) -> Dict[str, List[Tuple[str, int, float]]]:
    """The most frequent answers of each family's train facts (the prior a model could learn).

    :param manifest: Per-fact manifest records.
    :param n: Answers per family.
    :returns: ``family -> [(answer key, facts, share of the family's train facts), ...]``.
    """
    by_fam: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for m in manifest:
        if m["split"] != "val" and m["family"] != "qa_abstain":
            by_fam[m["family"]][answer_key(m["answer"])] += 1
    return {f: [(a, c, round(c / sum(cnt.values()), 3)) for a, c in cnt.most_common(n)]
            for f, cnt in sorted(by_fam.items(), key=lambda kv: FAMILIES.index(kv[0]))}


def build(editions_dir: Path, out_dir: Path, max_rows: int = MAX_ROWS_PER_PAGE,
          max_answer_share: float = MAX_ANSWER_SHARE, abstain_ratio: float = ABSTAIN_RATIO,
          paraphrases: int = PARAPHRASES_PER_FACT) -> Dict[str, Any]:
    """Build the QA dataset from the editions manifest.

    :param editions_dir: ``pgp_editions_v1`` output (``manifest.jsonl``).
    :param out_dir: Destination (NAS).
    :param max_rows: Facts per page image.
    :param max_answer_share: Maximum share of one answer within a family's train facts.
    :param abstain_ratio: Abstention facts per ``qa_date`` fact.
    :param paraphrases: Question wordings (rows) per train fact.
    :returns: The stats dict.
    :raises ValueError: When ``paraphrases`` is not between 1 and the smallest wording table.
    """
    min_wordings = min(len(t) for t in PARAPHRASES.values())
    if not 1 <= paraphrases <= min_wordings:
        raise ValueError(f"--paraphrases must be between 1 and {min_wordings}, got {paraphrases}")
    manifest_path = editions_dir / "manifest.jsonl"
    pages = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for p in pages:
        p["key"] = f"{p['canonical_id']}__{p['image_index']}"
    pgpids = {p["pgpid"] for p in pages}
    by_name, by_slug = inv.load_people()
    inv.NAME_LEXICON.update(inv.build_name_lexicon(list(by_name.values())))
    ctx = QAContext(load_csv_by(DOCUMENTS_CSV, "pgpid"), load_relations(RELATIONS_CSV, pgpids), by_name, by_slug,
                    edition_lines_of(pgpids), load_csv_by(PLACES_CSV, "name"))
    stats_c: collections.Counter = collections.Counter()
    funnel: Dict[str, Dict[str, Set[str]]] = {r: collections.defaultdict(set) for r in list(SET_ROLES) + ["Place"]}
    cands, eligible = {}, {}
    for p in pages:
        cands[p["key"]], eligible[p["key"]] = page_candidates(p, ctx, stats_c, funnel)
    cap_info: Dict[str, Any] = {}
    kept = apply_caps(pages, cands, eligible, max_rows=max_rows, abstain_ratio=abstain_ratio,
                      max_answer_share=max_answer_share, info=cap_info)
    split_rows, manifest = emit_rows(pages, kept, paraphrases)
    all_rows = [r for rows in split_rows.values() for r in rows]
    assert len({(r["stem"], r["question"]) for r in all_rows}) == len(all_rows), "rows are distinct by question"

    fam_counts = collections.Counter(m["family"] for m in manifest)
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "dataset": out_dir.name,
        "status": "included in the v22 mixture; review sample available at qa_review_sample.md",
        "editions_manifest": str(manifest_path), "pages": len(pages),
        "pages_with_qa_rows": sum(1 for v in kept.values() if v),
        "facts_by_family": dict(fam_counts),
        "facts_by_family_and_split": {f: dict(collections.Counter(m["split"] for m in manifest if m["family"] == f))
                                      for f in FAMILIES if fam_counts.get(f)},
        "rows_by_family": dict(collections.Counter(r["task"] for r in all_rows)),
        "rows_by_split": {k: len(v) for k, v in split_rows.items()},
        "rows_total": len(all_rows),
        "rows_by_family_and_section": dict(collections.Counter(f"{r['task']}:{r['section']}" for r in all_rows)),
        "candidates_before_page_cap": dict(collections.Counter(q.family for v in cands.values() for q in v)),
        "answer_cap": {"max_answer_share": max_answer_share, "dropped_candidates": cap_info["answer_cap_dropped"],
                       "top_train_answers": top_answers(manifest)},
        "abstain": cap_info["abstain"],
        "abstain_eligible_pages": sum(eligible.values()),
        "paraphrases": {"train_rows_per_fact": paraphrases, "val_rows_per_fact": 1,
                        "val_wording": "canonical (index 0)",
                        "wordings_used": dict(collections.Counter(
                            f"{m['family']}:{v}" for m in manifest for v in m["prompt_variants"]))},
        "answer_lines_countable_from_top": sum(m["line_countable_from_top"] for m in manifest if m["line"]),
        "validation_counters": dict(stats_c),
        "set_valued_funnel": {role: {
            "documents_with_2plus_holders": len(funnel[role]["has_role"]),
            "documents_all_holders_located": len(funnel[role]["all_located"]),
            "documents_complete": len(funnel[role]["complete"]),
            "list_facts": fam_counts.get(SET_ROLES[role][0], 0),
            "fallback_facts": fam_counts.get(SET_ROLES[role][1], 0)} for role in SET_ROLES},
        "place_funnel": {"documents_with_origin_destination_or_location": len(funnel["Place"]["with_place"]),
                         "documents_with_a_candidate_located": len(funnel["Place"]["located"]),
                         "facts": fam_counts.get("qa_place", 0),
                         "facts_by_section": dict(collections.Counter(m["section"] for m in manifest
                                                                      if m["family"] == "qa_place"))},
        "answer_invariant": "every answer text is whole token(s) of its line exactly as written "
                            "(prefix letters kept); checked on every emitted fact",
        "rules": {"max_rows_per_page": max_rows, "max_answer_share": max_answer_share,
                  "abstain_ratio": abstain_ratio, "abstain_max_page_share": ABSTAIN_MAX_PAGE_SHARE,
                  "paraphrases": paraphrases, "answer_format": 'text first: {"text": "...", "line": N}',
                  "person_rule": "status located AND exact token match AND tier 1 or kunya+given; "
                                 "not uncertain; role unique in PGP; span on one line; no [...] in the line",
                  "set_rule": "roles with >= 2 PGP holders: list only when every holder is located, on one "
                              "line, gap-free, and the completeness check passes; else the first located "
                              "holder's line (witness_any / party_any)",
                  "priority": PRIORITY},
    }
    sample = review_sample(manifest)
    sidecars = {
        "stats.json": json.dumps(stats, ensure_ascii=False, indent=1),
        "manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
        "qa_review_sample.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in sample),
        "qa_review_sample.md": review_markdown(sample, stats)}
    # the manual spot check survives rebuilds; a new version directory takes v1's
    spotcheck = next((d / SPOTCHECK_FILE for d in (out_dir, V1_DIR) if (d / SPOTCHECK_FILE).exists()), None)
    if spotcheck is not None:
        sidecars[SPOTCHECK_FILE] = spotcheck.read_text(encoding="utf-8")
    save_dataset(split_rows, out_dir, sidecars)
    logger.info("stats: %s", json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editions-dir", type=Path, default=DEFAULT_EDITIONS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-rows-per-page", type=int, default=MAX_ROWS_PER_PAGE,
                    help="facts per page image; a list answer counts as one (default %(default)s)")
    ap.add_argument("--max-answer-share", type=float, default=MAX_ANSWER_SHARE,
                    help="maximum share of one normalised answer within a family's train facts "
                         "(default %(default)s)")
    ap.add_argument("--abstain-ratio", type=float, default=ABSTAIN_RATIO,
                    help="abstention facts per qa_date fact, bounded by the eligible pages and by "
                         f"{ABSTAIN_MAX_PAGE_SHARE * 100:.0f}%% of all pages (default %(default)s)")
    ap.add_argument("--paraphrases", type=int, default=PARAPHRASES_PER_FACT,
                    help="question wordings (rows) per train fact; val keeps one (default %(default)s)")
    a = ap.parse_args()
    build(a.editions_dir, a.output_dir, max_rows=a.max_rows_per_page, max_answer_share=a.max_answer_share,
          abstain_ratio=a.abstain_ratio, paraphrases=a.paraphrases)


if __name__ == "__main__":
    main()
