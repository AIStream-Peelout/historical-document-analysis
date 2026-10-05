# File name: build_vqa_parse.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``pgp_vqa_parse_v1``: parse a page to JSON, then answer from the image plus that parse.

Why (v22b diagnosis, 2026-10-04): asked "In which month was this document written?", the fine-tuned
reader answers with the most common training month instead of reading the page, on questions it
trained on as on held-out ones. With the page's line-by-line reading pasted into the prompt as JSON
(image still attached) the same checkpoint answers month questions 8 times out of 12 instead of 1,
without retraining (``logs/v22b_diagnosis/qa_parse_in_prompt.py``). This pilot teaches both halves
inside the model: (1) parse a page to JSON; (2) fill fields and answer questions from the image plus a
JSON parse, citing the line.

User rule: no text written by our own model is ever a TARGET. Every target text (parse lines, answers,
field values) is a human edition line of ``pgp_editions_v1`` (side-verified pages) or whole tokens of
one; the cached v21b reading of a page appears only inside a PROMPT, as context; Kraken's text is only
evidence for WHERE an edition line is (its boxes), never a target.

Families (one train split each, ``train_<family>``, plus ``val`` = the edition manifest's val pages,
which are registered held-out documents):

1. ``parse_lines`` -- image -> ``[{"n": 1, "text": "<edition line>"}, ...]`` (one element per edition
   line, in edition order, gaps as ``[...]``), every page.
2. ``parse_lines_boxes`` -- the same with ``"bbox_2d"`` (0-1000 ints) per line, from Kraken geometry:
   the page's Kraken fragments that carry a Hebrew letter are clustered into rows
   (:func:`kraken_row_boxes`), every edition line is assigned a distinct row in order (monotonic,
   :func:`align_lines`) at letters-only similarity >= :data:`BOX_MIN_SIM`, the box is the union of the
   row's fragments, and the page is kept only when every line gets a row, every box passes the
   documentary-grounding geometry check (``box_rejection``: height <= 3 median fragment heights, >= 60
   px wide), the boxes run top to bottom in line order without overlapping by more than
   :data:`MAX_BOX_OVERLAP` of the smaller height, and every row reads its line from edge to edge
   (:func:`edge_coverage` >= :data:`BOX_MIN_EDGE_COVERAGE`; added to the specified rule after a visual
   check: where Kraken did not segment a line's first or last words the box stopped short of the line).
3. ``fields_from_parse`` -- prompt = :data:`PARSE_LEAD` + the parse + a request for named fields, each
   defined in one line taken from the QA builder's canonical question wording; target = a JSON object
   with exactly those keys, each ``{"text": ..., "line": n}`` (a list of them for the list fields) or
   null. A page's fields are its ``pgp_qa`` facts (:data:`FIELD_OF`); an abstention fact gives a null
   ``date`` and, when no token of the page is a month or a year word, null ``month`` and ``year``.
4. ``question_from_parse`` -- prompt = lead + parse + one QA question in the QA builder's wording
   (train: the fact's two paraphrases, val: the canonical wording) with the answer instruction replaced
   by :data:`ANSWER_INSTRUCTION` (:data:`LIST_ANSWER_INSTRUCTION` for list facts); one row per (fact,
   wording).
5. ``lookup_from_parse`` -- two generic look-ups per page: "Which line contains «phrase»?" ->
   ``{"line": n, "text": <the whole line>}`` and "What is written after «phrase» on the same line?" ->
   ``{"answer": <the rest of the line>, "line": n}``; phrases of 2-3 whole words that occur once on the
   page, from gap-free lines (``build_pgp_editions.unique_phrases``), drawn by ``stable_hash``.

The parse in a prompt (families 3-5): a page is a model-parse page with probability
:data:`MODEL_PARSE_SHARE` (seeded by page); its rows show the cached model reading numbered the same way
instead of the edition lines, but only when every answer line of the row maps to exactly one reading
line at similarity >= :data:`MODEL_LINE_MIN_SIM` that maps back to it (:func:`map_reading`), the cited
number is that reading line's, and the answer text stays the edition's. Otherwise the row falls back to
the gold parse (reason recorded). A row that asserts the page carries no date never shows a reading
with a date word in it.

Invariants, checked on every row (:func:`check_row`, a violation raises): targets parse as JSON; parse
targets are the edition lines exactly; every non-null answer text is whole token(s) of the EDITION line
behind its cited number (``build_pgp_qa.check_whole_tokens``); every cited number exists in the parse
the prompt shows and stands for that edition line. Split hygiene (:func:`check_splits`): no val page
image in a train split, no registered benchmark document in train. Prompts over
:data:`MAX_PROMPT_TOKENS` (v21b tokenizer) are skipped (a model-parse row first falls back to gold).

Outputs: the DatasetDict (KTIV ``FEATURES``; ``task`` = family, ``section`` = ``<detail>|<parse>``),
``manifest.jsonl`` (one record per row), ``stats.json``, ``review_sample.md`` and the images-once export.

Usage (repo root)::

    nice -n 10 .venv/bin/python -m src.finetuning.qwen_hebrew.build_vqa_parse
"""
import argparse
import collections
import functools
import json
import logging
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import Levenshtein

from src.datasets.consensus.line_rule import letters, similarity
from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents
from src.finetuning.qwen_hebrew import build_pgp_editions as eds
from src.finetuning.qwen_hebrew import build_pgp_qa as qa
from src.finetuning.qwen_hebrew.build_documentary_grounding import box_rejection, median_height, union_box
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
NAS_DATASETS = eds.NAS_DATASETS
DEFAULT_EDITIONS = NAS_DATASETS / "pgp_editions_v1"
DEFAULT_QA = NAS_DATASETS / "pgp_qa_v2"
DEFAULT_OUT = NAS_DATASETS / "pgp_vqa_parse_v1"
DEFAULT_IMAGES_ONCE = NAS_DATASETS / "pgp_vqa_parse_v1_images_once"
DEFAULT_TOKENIZER = _REPO / "models/qwen3-vl-8b-heb-v21b-step1200/tokenizer.json"
READER_MODEL = eds.MODEL                 # the cached reading shown as context (v21b step 1200)
LABEL_SOURCE = "pgp_edition_vqa"
SEED = 20261005
MODEL_PARSE_SHARE = 0.30                 # pages whose context rows show the model reading
MODEL_LINE_MIN_SIM = 0.8                 # an answer line must be read at least this well to cite the reading
BOX_MIN_SIM = 0.5                        # an edition line needs a Kraken row at least this similar
BOX_MIN_EDGE_COVERAGE = 0.75             # ... that reads it from (near) its first to (near) its last letter
MAX_BOX_OVERLAP = 0.5                    # consecutive line boxes may share this much of the smaller height
MAX_PROMPT_TOKENS = 4500
MAX_TARGET_TOKENS = 4500                 # image ~6.4-7.4k + prompt + target stays under the 12,288 cap
LOOKUP_MIN_REST_LETTERS = 2
REVIEW_ROWS_PER_FAMILY = 8
FAMILIES = ("parse_lines", "parse_lines_boxes", "fields_from_parse", "question_from_parse", "lookup_from_parse")
CONTEXT_FAMILIES = FAMILIES[2:]

Box = Tuple[int, int, int, int]
Cite = Tuple[int, int, str]              # (number shown in the prompt's parse, edition line 0-based, text)
TokenCounter = Callable[[str], int]

# ----------------------------------------------------------------------------- prompts

_INTRO = ("This image is a manuscript fragment from the Cairo Genizah — handwritten Hebrew script (the language may "
          "be Hebrew, Judeo-Arabic, or Aramaic).")
PARSE_PROMPT = (f"{_INTRO}\n\n"
                'Parse the page into JSON: one element per line of text, in reading order, {"n": <line number, from '
                '1>, "text": "<the line exactly as written>"}. Where text is lost or illegible due to damage, write '
                "[...]. Do NOT correct, restore, or complete from memory.\n\n"
                "Return ONLY the JSON array.")
PARSE_BOXES_PROMPT = (f"{_INTRO}\n\n"
                      'Parse the page into JSON with line boxes: one element per line of text, in reading order, '
                      '{"n": <line number, from 1>, "text": "<the line exactly as written>", "bbox_2d": [x1, y1, x2, '
                      "y2]} giving the line's bounding box, coordinates normalized to 0-1000. Where text is lost or "
                      "illegible due to damage, write [...]. Do NOT correct, restore, or complete from memory.\n\n"
                      "Return ONLY the JSON array.")
# the lead of the zero-shot test that motivated this set (qa_parse_in_prompt.py), byte for byte
PARSE_LEAD = "Line-by-line reading of this page (JSON, in reading order):\n"
ANSWER_INSTRUCTION = ('Answer with ONLY JSON {"answer": "<words exactly as written on the page>", "line": <n>}, where '
                      'n is the number of that line in the reading above, or {"answer": null} when the page does '
                      "not state it.")
LIST_ANSWER_INSTRUCTION = ('Answer with ONLY JSON {"answer": [{"text": "<name exactly as written on the page>", '
                           '"line": <n>}, ...]}, one item per name, where n is the number of its line in the reading '
                           'above, or {"answer": null} when the page does not state it.')
LOOKUP_LINE_PROMPT = ('Which line contains «{phrase}»? Answer with ONLY JSON {{"line": <n>, "text": "<that whole line '
                      'exactly as written on the page>"}}, where n is the number of that line in the reading above.')
LOOKUP_AFTER_PROMPT = ('What is written after «{phrase}» on the same line? Answer with ONLY JSON {{"answer": "<the '
                       'rest of that line exactly as written on the page>", "line": <n>}}, where n is the number of '
                       "that line in the reading above.")
FIELDS_HEAD = "Fill in these fields from this page:"
FIELDS_TAIL = ('Answer with ONLY a JSON object with exactly these keys, in this order: {keys}. Each value is '
               '{{"text": "<words exactly as written on the page>", "line": <n>}}, where n is the number of that line '
               "in the reading above, or null when the page does not state it.")
FIELDS_LIST_TAIL = " The value of {keys} is a JSON list of such objects, one per name, or null."

# ----------------------------------------------------------------------------- fields

# (QA family, section) -> field name; "*" = any section of the family. Sections are build_pgp_qa's.
FIELD_OF: Dict[Tuple[str, str], str] = {
    ("qa_date", "*"): "date", ("qa_abstain", "*"): "date", ("qa_date_month", "*"): "month",
    ("qa_date_year", "*"): "year", ("qa_place", "written"): "place", ("qa_place", "sent"): "destination",
    ("qa_person", "sender"): "sender", ("qa_person", "recipient"): "recipient", ("qa_person", "party"): "party",
    ("qa_person", "witness"): "witness", ("qa_person", "validating_judge"): "judge",
    ("qa_parties_list", "*"): "parties", ("qa_witnesses_list", "*"): "witnesses",
    ("qa_party", "*"): "party_statement", ("qa_party_formula", "*"): "party_statement",
    ("qa_party_line", "*"): "party_line", ("qa_witness_line", "*"): "witness_line",
    ("qa_ketubah_parties", "*"): "couple_line", ("qa_ketubah_groom", "*"): "groom",
    ("qa_ketubah_bride", "*"): "bride"}
FIELD_ORDER = ("date", "month", "year", "place", "destination", "sender", "recipient", "party", "witness", "judge",
               "parties", "witnesses", "party_statement", "party_line", "witness_line", "couple_line", "groom",
               "bride")
LIST_FIELDS = frozenset({"parties", "witnesses"})
# the QA wording each field's one-line definition is cut from: (PARAPHRASES key, template arguments)
FIELD_WORDING: Dict[str, Tuple[str, Dict[str, str]]] = {
    "date": ("qa_date", {}), "month": ("qa_date_month", {}), "year": ("qa_date_year", {}),
    "place": ("qa_place:written", {}), "destination": ("qa_place:sent", {}),
    "sender": ("qa_person", {"role": "sender"}), "recipient": ("qa_person", {"role": "recipient"}),
    "party": ("qa_person", {"role": "party"}), "witness": ("qa_person", {"role": "witness"}),
    "judge": ("qa_person", {"role": "validating judge"}), "parties": ("qa_parties_list", {}),
    "witnesses": ("qa_witnesses_list", {}), "party_statement": ("qa_party", {}),
    "party_line": ("qa_party_line", {}), "witness_line": ("qa_witness_line", {}),
    "couple_line": ("qa_ketubah_parties", {}), "groom": ("qa_ketubah_groom", {"role": "groom"}),
    "bride": ("qa_ketubah_bride", {"role": "bride"})}
_ANSWER_FORMAT_MARKERS = (", as a JSON", " as a JSON", ", as JSON", " as JSON")


def question_stem(wording: str) -> str:
    """A QA builder question without its answer-format clause (the JSON example and, for the date
    question, the "not stated" clause that follows it).

    :param wording: A :data:`build_pgp_qa.PARAPHRASES` wording (templates already formatted).
    :returns: The question part, ending in "?" or ".".
    :raises ValueError: When the wording has no recognised answer-format clause, or the cut leaves any
        JSON or abstention wording behind.
    """
    cut = min((i for i in (wording.find(m) for m in _ANSWER_FORMAT_MARKERS) if i >= 0), default=-1)
    if cut < 0:
        raise ValueError(f"no answer-format clause in {wording!r}")
    stem = wording[:cut].rstrip(" ,;")
    if not stem or "{" in stem or "JSON" in stem or "not stated" in stem:
        raise ValueError(f"answer-format clause not cut cleanly from {wording!r}")
    return stem if stem.endswith(("?", ".")) else stem + "."


def field_definition(name: str) -> str:
    """One-line definition of a field: the canonical QA question of its family, answer format cut.

    :param name: Field name (:data:`FIELD_ORDER`).
    :returns: The definition.
    """
    key, args = FIELD_WORDING[name]
    wording = qa.PARAPHRASES[key][0]
    return question_stem(wording.format(**args) if args else wording)


def field_name(fact: Dict[str, Any]) -> str:
    """The field a QA fact fills.

    :param fact: ``pgp_qa`` manifest record (``family``, ``section``).
    :returns: Field name.
    :raises KeyError: For a family/section without a mapping (a new QA family must be mapped first).
    """
    return FIELD_OF.get((fact["family"], fact["section"])) or FIELD_OF[(fact["family"], "*")]


@dataclass
class FieldValue:
    """One requested field of a page.

    :param name: Field name.
    :param items: ``(text, edition line 0-based)`` per value; None = the page does not state it.
    :param is_list: The value is a list of names.
    :param source: The fact stem it comes from (``derived:<stem>`` for a null implied by an abstention).
    """

    name: str
    items: Optional[List[Tuple[str, int]]]
    is_list: bool
    source: str


def fact_items(fact: Dict[str, Any]) -> List[Tuple[str, int]]:
    """``(text, edition line 0-based)`` of a QA fact's answer.

    :param fact: ``pgp_qa`` manifest record.
    :returns: Items (empty for an abstention fact).
    """
    return [(str(it["text"]), int(it["line"]) - 1) for it in qa.answer_items(fact["answer"])]


def fact_is_list(fact: Dict[str, Any]) -> bool:
    """Whether a QA fact answers with a list of names.

    :param fact: ``pgp_qa`` manifest record.
    :returns: True for list answers.
    """
    return isinstance(json.loads(fact["answer"]), list)


def has_year_word(line: str) -> bool:
    """A token of the line introduces or closes a year (שנת-family, סנה/סנת, or an era word).

    :param line: Edition line.
    :returns: Whether a year expression could be on the line.
    """
    return any(qa.is_year_word(t) or "".join(qa.tokens(t)) in qa.YEAR_ERA_WORDS for t in line.split(" "))


def page_fields(lines: Sequence[str], facts: Sequence[Dict[str, Any]]) -> List[FieldValue]:
    """The fields requested for one page, in :data:`FIELD_ORDER`.

    Each fact fills its field (:func:`field_name`); an abstention fact gives a null ``date`` and, when no
    token of the page is a month name or a year word, null ``month`` and ``year`` (the abstention rule
    already excludes any month name or dating word on the page and any PGP date).

    :param lines: Edition lines of the page.
    :param facts: The page's ``pgp_qa`` facts.
    :returns: Field values.
    :raises ValueError: When two facts fill the same field, or a derived null meets a fact.
    """
    out: Dict[str, FieldValue] = {}
    for f in facts:
        name = field_name(f)
        if name in out:
            raise ValueError(f"facts {out[name].source} and {f['stem']} both fill {name!r}")
        out[name] = FieldValue(name, fact_items(f) or None, fact_is_list(f), f["stem"])
        if (out[name].items is None) != (f["family"] == "qa_abstain"):
            raise ValueError(f"fact {f['stem']}: only an abstention fact may be null")
    abstain = next((f for f in facts if f["family"] == "qa_abstain"), None)
    if abstain is not None:
        for name, absent in (("month", not any(qa.month_tokens(ln) for ln in lines)),
                             ("year", not any(has_year_word(ln) for ln in lines))):
            if name in out:
                raise ValueError(f"page with an abstention fact has a {name} fact")
            if absent:
                out[name] = FieldValue(name, None, False, f"derived:{abstain['stem']}")
    return [out[n] for n in FIELD_ORDER if n in out]


def fields_request(fields: Sequence[FieldValue]) -> str:
    """The request part of a ``fields_from_parse`` prompt.

    :param fields: Requested fields in order.
    :returns: Head, one definition line per field, and the answer instruction.
    """
    keys = ", ".join(f'"{f.name}"' for f in fields)
    out = [FIELDS_HEAD] + [f"- {f.name}: {field_definition(f.name)}" for f in fields]
    tail = FIELDS_TAIL.format(keys=keys)
    lists = [f'"{f.name}"' for f in fields if f.is_list]
    if lists:
        tail += FIELDS_LIST_TAIL.format(keys=" and ".join(lists))
    return "\n".join(out + [tail])


def render_fields(fields: Sequence[FieldValue], to_shown: Dict[int, int]) -> str:
    """The ``fields_from_parse`` target.

    :param fields: Requested fields in order.
    :param to_shown: Edition line (0-based) -> number in the prompt's parse.
    :returns: JSON object with exactly the requested keys.
    """
    obj: Dict[str, Any] = {}
    for f in fields:
        if f.items is None:
            obj[f.name] = None
        elif f.is_list:
            obj[f.name] = [{"text": t, "line": to_shown[e]} for t, e in f.items]
        else:
            (t, e), = f.items
            obj[f.name] = {"text": t, "line": to_shown[e]}
    return json.dumps(obj, ensure_ascii=False)


# ----------------------------------------------------------------------------- questions


def question_request(wording: str, is_list: bool) -> str:
    """A QA question with its answer instruction replaced (``question_from_parse``).

    :param wording: The QA builder's wording.
    :param is_list: The fact answers with a list of names.
    :returns: Question stem + :data:`ANSWER_INSTRUCTION` (or :data:`LIST_ANSWER_INSTRUCTION`).
    """
    return f"{question_stem(wording)} {LIST_ANSWER_INSTRUCTION if is_list else ANSWER_INSTRUCTION}"


def render_question(items: Sequence[Tuple[str, int]], is_list: bool, to_shown: Dict[int, int]) -> str:
    """The ``question_from_parse`` target.

    :param items: ``(text, edition line 0-based)``; empty = not stated.
    :param is_list: The answer is a list of names.
    :param to_shown: Edition line (0-based) -> number in the prompt's parse.
    :returns: ``{"answer": null}``, ``{"answer": text, "line": n}`` or ``{"answer": [{"text", "line"}, ...]}``.
    """
    if not items:
        return json.dumps({"answer": None})
    if is_list:
        return json.dumps({"answer": [{"text": t, "line": to_shown[e]} for t, e in items]}, ensure_ascii=False)
    (t, e), = items
    return json.dumps({"answer": t, "line": to_shown[e]}, ensure_ascii=False)


# ----------------------------------------------------------------------------- look-ups


@dataclass(frozen=True)
class Lookup:
    """One generic look-up of a page.

    :param kind: ``line_of_phrase`` (answer = the whole line) or ``after_phrase`` (answer = the rest of
        the line after the phrase).
    :param phrase: The quoted phrase (whole words of the edition line).
    :param line: Edition line (0-based).
    :param text: The answer text.
    """

    kind: str
    phrase: str
    line: int
    text: str


def lookup_candidates(lines: Sequence[str]) -> List[Tuple[int, str]]:
    """Phrases of 2-3 whole words that occur once on the page, from gap-free lines.

    Host lines carry no ``[...]`` and >= ``LINE_MIN_LETTERS`` letters; the phrases are the editions
    builder's (:func:`build_pgp_editions.unique_phrases`: clean words, >= 8 letters, unique on the page
    both letters-only and as a string).

    :param lines: Edition lines.
    :returns: ``(line 0-based, phrase)`` candidates.
    """
    n = len(lines)
    ans = eds.Answer(list(lines), ["main"] * n, [True] * n, [frozenset()] * n, [])
    return [(i, ph) for i, ln in enumerate(lines)
            if GAP_TOKEN not in ln and eds.n_hebrew(ln) >= eds.LINE_MIN_LETTERS for ph in eds.unique_phrases(ans, i)]


def rest_after(line: str, phrase: str) -> str:
    """The whole tokens of a line after a phrase of whole tokens.

    :param line: Edition line.
    :param phrase: Whole-token phrase of the line.
    :returns: The rest ("" when the phrase ends the line).
    :raises ValueError: When the phrase is not whole tokens of the line.
    """
    toks, words = line.split(" "), phrase.split(" ")
    for s in range(len(toks) - len(words) + 1):
        if toks[s:s + len(words)] == words:
            return " ".join(toks[s + len(words):])
    raise ValueError(f"{phrase!r} is not whole tokens of {line!r}")


def pick_lookups(lines: Sequence[str], key: str, seed: int = SEED) -> Tuple[Optional[Lookup], Optional[Lookup]]:
    """The two look-ups of a page, drawn by ``stable_hash`` (stable across rebuilds).

    The line look-up takes the first candidate in hash order. The after-phrase look-up takes the first
    candidate with a different phrase whose rest of line has >= :data:`LOOKUP_MIN_REST_LETTERS` letters,
    preferring a different line.

    :param lines: Edition lines.
    :param key: Page key.
    :param seed: Seed.
    :returns: ``(line look-up or None, after-phrase look-up or None)``.
    """
    cands = sorted(lookup_candidates(lines), key=lambda c: eds.stable_hash(f"{seed}:lookup:{key}:{c[0]}:{c[1]}"))
    if not cands:
        return None, None
    i, phrase = cands[0]
    first = Lookup("line_of_phrase", phrase, i, lines[i])
    afters = [(j, p, rest_after(lines[j], p)) for j, p in cands if p != phrase]
    afters = [a for a in afters if eds.n_hebrew(a[2]) >= LOOKUP_MIN_REST_LETTERS]
    afters.sort(key=lambda a: a[0] == i)          # stable: other lines first, hash order kept
    second = Lookup("after_phrase", afters[0][1], afters[0][0], afters[0][2]) if afters else None
    return first, second


def lookup_request(lk: Lookup) -> str:
    """The request part of a ``lookup_from_parse`` prompt.

    :param lk: Look-up.
    :returns: The question with its answer instruction.
    """
    tpl = LOOKUP_LINE_PROMPT if lk.kind == "line_of_phrase" else LOOKUP_AFTER_PROMPT
    return tpl.format(phrase=lk.phrase)


def render_lookup(lk: Lookup, to_shown: Dict[int, int]) -> str:
    """The ``lookup_from_parse`` target.

    :param lk: Look-up.
    :param to_shown: Edition line (0-based) -> number in the prompt's parse.
    :returns: ``{"line": n, "text": line}`` or ``{"answer": rest, "line": n}``.
    """
    if lk.kind == "line_of_phrase":
        return json.dumps({"line": to_shown[lk.line], "text": lk.text}, ensure_ascii=False)
    return json.dumps({"answer": lk.text, "line": to_shown[lk.line]}, ensure_ascii=False)


# ----------------------------------------------------------------------------- parses


def parse_json(lines: Sequence[str]) -> str:
    """A page parse: a JSON array, one ``{"n", "text"}`` element per line, one element per text line.

    :param lines: Lines in order.
    :returns: JSON text.
    """
    return "[\n" + ",\n".join(json.dumps({"n": i + 1, "text": t}, ensure_ascii=False)
                              for i, t in enumerate(lines)) + "\n]"


def parse_boxes_json(lines: Sequence[str], boxes: Sequence[Box]) -> str:
    """A page parse with line boxes: ``{"n", "text", "bbox_2d"}`` per line.

    :param lines: Lines in order.
    :param boxes: One 0-1000 box per line.
    :returns: JSON text.
    """
    assert len(lines) == len(boxes)
    return "[\n" + ",\n".join(json.dumps({"n": i + 1, "text": t, "bbox_2d": list(b)}, ensure_ascii=False)
                              for i, (t, b) in enumerate(zip(lines, boxes))) + "\n]"


def context_prompt(shown: Sequence[str], request: str) -> str:
    """A context-family prompt: lead, the parse shown, a blank line, the request.

    :param shown: Lines of the parse the prompt shows.
    :param request: Question / field request with its answer instruction.
    :returns: Prompt text.
    """
    return f"{PARSE_LEAD}{parse_json(shown)}\n\n{request}"


def model_reading(raw: Dict[str, Any]) -> List[str]:
    """The cached model reading of a page as parse lines (prompt context only, never a target).

    :param raw: ``ai_reads/raw`` record.
    :returns: VLM lines (``reader_from_raw``: parsed lines, else recovered from the raw reply), NFC,
        whitespace collapsed, empty lines dropped.
    """
    return [unicodedata.normalize("NFC", " ".join(t.split())) for t in eds.reader_from_raw(raw).vlm if t.split()]


@dataclass
class ReadingMap:
    """Edition lines that a model reading line stands for.

    :param to_reading: Edition line (0-based) -> reading line (0-based), for the lines that map.
    :param reasons: Edition line -> why it does not map.
    :param best_sim: Edition line -> best similarity to any reading line.
    """

    to_reading: Dict[int, int]
    reasons: Dict[int, str]
    best_sim: Dict[int, float]


def map_reading(edition: Sequence[str], reading: Sequence[str], min_sim: float = MODEL_LINE_MIN_SIM) -> ReadingMap:
    """Map edition lines to model reading lines.

    Edition line ``e`` maps to reading line ``m`` when ``m`` is the ONLY reading line at similarity >=
    ``min_sim`` to ``e`` and ``e`` is the most similar edition line of ``m`` (mutual; first on ties).

    :param edition: Edition lines.
    :param reading: Model reading lines.
    :param min_sim: Minimum letters-only similarity.
    :returns: The map (injective).
    """
    if not reading:
        return ReadingMap({}, {e: "no_model_reading" for e in range(len(edition))}, {})
    sims = [[similarity(a, b) for b in reading] for a in edition]
    owner = [max(range(len(edition)), key=lambda e: sims[e][m]) for m in range(len(reading))]
    out = ReadingMap({}, {}, {})
    for e, row in enumerate(sims):
        out.best_sim[e] = max(row)
        above = [m for m, s in enumerate(row) if s >= min_sim]
        if not above:
            out.reasons[e] = "answer_line_below_min_similarity"
        elif len(above) > 1:
            out.reasons[e] = "answer_line_matches_several_reading_lines"
        elif owner[above[0]] != e:
            out.reasons[e] = "reading_line_closer_to_another_line"
        else:
            out.to_reading[e] = above[0]
    assert len(set(out.to_reading.values())) == len(out.to_reading)
    return out


def wants_model_parse(key: str, share: float = MODEL_PARSE_SHARE, seed: int = SEED) -> bool:
    """Whether a page's context rows should show the model reading (seeded per page).

    :param key: Page key.
    :param share: Target share of pages.
    :param seed: Seed.
    :returns: True for about ``share`` of pages.
    """
    return eds.stable_hash(f"{seed}:model_parse:{key}") % 10000 < round(share * 10000)


# ----------------------------------------------------------------------------- Kraken rows and boxes


@dataclass(frozen=True)
class KrakenRow:
    """One row of Kraken fragments.

    :param text: Fragment texts right to left (by horizontal centre), concatenated.
    :param box: Union of the fragment boxes, 0-1000 ints.
    :param n_frags: Fragments in the row.
    """

    text: str
    box: Box
    n_frags: int


def _y_centre(frag: Dict[str, Any]) -> float:
    """Vertical centre of a fragment box.

    :param frag: Fragment with ``box = [x1, y1, x2, y2]``.
    :returns: ``(y1 + y2) / 2``.
    """
    return (frag["box"][1] + frag["box"][3]) / 2


def kraken_row_boxes(frags: Sequence[Dict[str, Any]]) -> List[KrakenRow]:
    """Cluster a page's Kraken fragments into rows, top to bottom, with their boxes.

    Only fragments with a 4-number box and at least one Hebrew letter take part (bare brackets, dots
    and misread marks carry no text evidence and would only widen a box). Fragments are visited by
    vertical centre; each joins the most recently opened row whose CORE band -- the median top to the
    median bottom of the row's fragments -- holds its centre, else it opens a new row. A band that grows
    to the union of its members (``build_pgp_editions.kraken_rows``) stretches over a fragment spanning
    two lines and chains the lines into one row; the core band does not (measured on the 2,353 edition
    pages: 235 pages with every line matched vs 205). Rows are ordered by the median centre of their
    fragments; a row's text is its fragments right to left, its box the union of their boxes.

    :param frags: Kraken fragments ``{"text", "box"}`` (0-1000 units of the oriented image).
    :returns: Rows.
    """
    usable = [f for f in frags if len(f.get("box") or []) == 4 and letters(f.get("text") or "")]
    rows: List[List[Dict[str, Any]]] = []
    for f in sorted(usable, key=_y_centre):
        cy = _y_centre(f)
        home = None
        for r in reversed(rows):
            if statistics.median(g["box"][1] for g in r) <= cy <= statistics.median(g["box"][3] for g in r):
                home = r
                break
        if home is None:
            rows.append([f])
        else:
            home.append(f)
    rows.sort(key=lambda r: statistics.median(_y_centre(g) for g in r))
    return [KrakenRow("".join(g["text"] for g in sorted(r, key=lambda g: -(g["box"][0] + g["box"][2]))),
                      union_box([g["box"] for g in r]), len(r)) for r in rows]


def similarity_matrix(lines: Sequence[str], texts: Sequence[str]) -> List[List[float]]:
    """Letters-only similarity of every line against every text.

    :param lines: Edition lines.
    :param texts: Reader texts.
    :returns: ``sims[i][j]``.
    """
    return [[similarity(a, b) for b in texts] for a in lines]


def align_lines(sims: Sequence[Sequence[float]], min_sim: float) -> Optional[List[int]]:
    """Assign every line a distinct row in order (line i -> row a_i, a_1 < a_2 < ...), each at
    similarity >= ``min_sim``, maximising the total similarity (rows may be skipped).

    :param sims: ``sims[line][row]``.
    :param min_sim: Minimum similarity of an assigned pair.
    :returns: The row of each line, or None when no such assignment exists.
    """
    n = len(sims)
    m = len(sims[0]) if n else 0
    if n > m:
        return None
    neg = float("-inf")
    best = [[0.0] * (m + 1)] + [[neg] * (m + 1) for _ in range(n)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = best[i][j - 1]
            s = sims[i - 1][j - 1]
            if s >= min_sim and best[i - 1][j - 1] > neg and best[i - 1][j - 1] + s > best[i][j]:
                best[i][j] = best[i - 1][j - 1] + s
                take[i][j] = True
    if best[n][m] == neg:
        return None
    out = [0] * n
    i, j = n, m
    while i > 0:
        if take[i][j]:
            out[i - 1] = j - 1
            i -= 1
        j -= 1
    return out


def edge_coverage(line: str, row_text: str) -> float:
    """Share of a line's letters that lie between the first and the last letter a row reads.

    The letters-only Levenshtein alignment of the line against the row is taken; the span from the
    first to the last line letter in an ``equal`` block, over the line's letter count, is the part of
    the line the row's box can cover. A row that misses the line's first or last words (Kraken did not
    segment them) scores low even when the words it read are right (T-S 10J15.23 line 14: 0.61, its
    box ends at x=597 of a line that runs to ~990); a complete but noisy read scores near 1.

    :param line: Edition line.
    :param row_text: Kraken row text.
    :returns: Coverage in [0, 1] (0 when nothing matches).
    """
    a, b = letters(line), letters(row_text)
    eq = [(i1, i2) for tag, i1, i2, _, _ in Levenshtein.opcodes(a, b) if tag == "equal"]
    return (eq[-1][1] - eq[0][0]) / len(a) if eq else 0.0


def vertical_overlap(a: Box, b: Box) -> float:
    """Height shared by two boxes.

    :param a: Box.
    :param b: Box.
    :returns: Overlap of the ``[y1, y2]`` extents (0 when disjoint).
    """
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


@dataclass
class BoxResult:
    """Outcome of the line-box rule on one page.

    :param boxes: One box per edition line (None when the page does not qualify).
    :param reason: Why it does not qualify ("" when it does).
    :param sims: Similarity of each line to its row (when every line got one).
    :param rows: Kraken rows found.
    :param coverage: Edge coverage of each line by its row (when the geometry checks passed).
    """

    boxes: Optional[List[Box]]
    reason: str
    sims: List[float] = field(default_factory=list)
    rows: int = 0
    coverage: List[float] = field(default_factory=list)


def line_boxes(lines: Sequence[str], frags: Sequence[Dict[str, Any]], image_width: int,
               min_sim: float = BOX_MIN_SIM, max_overlap: float = MAX_BOX_OVERLAP,
               min_coverage: float = BOX_MIN_EDGE_COVERAGE) -> BoxResult:
    """One Kraken box per edition line, or the reason the page does not qualify.

    Checks in order: rows exist; a monotonic one-to-one line -> row assignment at ``min_sim``; the
    geometry of every box (``box_rejection``); centres top to bottom and consecutive overlap; finally
    every row reads its line from edge to edge (:func:`edge_coverage` >= ``min_coverage``), so no box
    stops short of its line.

    :param lines: Edition lines.
    :param frags: The page's Kraken fragments (raw cache ``frags``).
    :param image_width: Oriented image width in pixels (the 60 px width check).
    :param min_sim: Minimum similarity of a line to its row.
    :param max_overlap: Largest vertical overlap of consecutive boxes, as a share of the smaller height.
    :param min_coverage: Minimum edge coverage of every line by its row.
    :returns: The result.
    """
    rows = kraken_row_boxes(frags)
    if not rows:
        return BoxResult(None, "no_kraken_rows")
    sims = similarity_matrix(lines, [r.text for r in rows])
    assign = align_lines(sims, min_sim)
    if assign is None:
        weak = any(max(row) < min_sim for row in sims)
        return BoxResult(None, "line_without_row_at_min_similarity" if weak else "no_monotonic_assignment",
                         rows=len(rows))
    boxes = [rows[j].box for j in assign]
    line_sims = [sims[i][j] for i, j in enumerate(assign)]
    med = median_height(f["box"] for f in frags if len(f.get("box") or []) == 4 and letters(f.get("text") or ""))
    for b in boxes:
        why = box_rejection(b, med, image_width)
        if why:
            return BoxResult(None, why, line_sims, len(rows))
    for a, b in zip(boxes, boxes[1:]):
        if not (a[1] + a[3]) / 2 < (b[1] + b[3]) / 2:
            return BoxResult(None, "boxes_not_top_to_bottom", line_sims, len(rows))
        if vertical_overlap(a, b) > max_overlap * min(a[3] - a[1], b[3] - b[1]):
            return BoxResult(None, "boxes_overlap", line_sims, len(rows))
    cover = [edge_coverage(lines[i], rows[j].text) for i, j in enumerate(assign)]
    if min(cover) < min_coverage:
        return BoxResult(None, "row_misses_part_of_a_line", line_sims, len(rows), cover)
    return BoxResult(boxes, "", line_sims, len(rows), cover)


# ----------------------------------------------------------------------------- rows


@dataclass
class PageContext:
    """What the context families need about one page.

    :param key: Page key.
    :param lines: Edition lines.
    :param want_model: The page was drawn as a model-parse page.
    :param reading: The cached model reading ([] when absent or not loaded).
    :param reading_map: Edition -> reading line map (None when not loaded).
    :param reading_has_date: A reading line carries a month name or a dating word.
    """

    key: str
    lines: List[str]
    want_model: bool
    reading: List[str] = field(default_factory=list)
    reading_map: Optional[ReadingMap] = None
    reading_has_date: bool = False

    def shown(self, source: str) -> Tuple[List[str], Dict[int, int]]:
        """The parse a prompt shows and how its numbers stand for edition lines.

        :param source: ``gold`` or ``model``.
        :returns: ``(lines shown, edition line 0-based -> number shown)``.
        """
        if source == "gold":
            return self.lines, {e: e + 1 for e in range(len(self.lines))}
        assert source == "model" and self.reading_map is not None
        return self.reading, {e: m + 1 for e, m in self.reading_map.to_reading.items()}


@dataclass
class ContextSpec:
    """A context-family row before its parse is chosen.

    :param family: Row family.
    :param detail: First part of ``section`` (QA family, ``fields``, look-up kind).
    :param stem: Row id.
    :param request: Question / field request with its answer instruction.
    :param items: ``(text, edition line 0-based)`` the answer cites.
    :param asserts_no_date: The answer says the page carries no date (or month / year).
    :param render: Target from an ``edition line -> number shown`` map.
    :param meta: Manifest extras.
    """

    family: str
    detail: str
    stem: str
    request: str
    items: List[Tuple[str, int]]
    asserts_no_date: bool
    render: Callable[[Dict[int, int]], str]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Draft:
    """A finished row with what the invariant check needs.

    :param family: Row family.
    :param section: ``section`` column.
    :param stem: Row id.
    :param question: Prompt text.
    :param answer: Target text.
    :param parse: ``none`` (parse families), ``gold`` or ``model``.
    :param cites: ``(number shown, edition line 0-based, text)`` per cited value.
    :param shown_lines: Lines of the parse the prompt shows (context families).
    :param prompt_tokens: Prompt tokens.
    :param target_tokens: Target tokens.
    :param meta: Manifest extras.
    """

    family: str
    section: str
    stem: str
    question: str
    answer: str
    parse: str
    cites: List[Cite]
    shown_lines: List[str]
    prompt_tokens: int
    target_tokens: int
    meta: Dict[str, Any] = field(default_factory=dict)


def choose_parse(ctx: PageContext, spec: ContextSpec) -> Tuple[str, str]:
    """Pick the parse a context row shows.

    :param ctx: Page context.
    :param spec: Row spec.
    :returns: ``(source, fallback reason)``; the reason is "" unless a model-parse page fell back to gold.
    """
    if not ctx.want_model:
        return "gold", ""
    if not ctx.reading or ctx.reading_map is None:
        return "gold", "no_model_reading"
    for _, e in spec.items:
        if e not in ctx.reading_map.to_reading:
            return "gold", ctx.reading_map.reasons[e]
    if spec.asserts_no_date and ctx.reading_has_date:
        return "gold", "model_reading_has_a_date_word"
    return "model", ""


def make_context_row(ctx: PageContext, spec: ContextSpec, count: TokenCounter, max_prompt: int,
                     max_target: int) -> Tuple[Optional[Draft], str, str]:
    """Finish a context row: parse, prompt, target, cites, token budget.

    A model-parse prompt over ``max_prompt`` tokens falls back to the gold parse; a gold prompt over it
    is skipped.

    :param ctx: Page context.
    :param spec: Row spec.
    :param count: Token counter.
    :param max_prompt: Prompt token cap.
    :param max_target: Target token cap.
    :returns: ``(draft or None, fallback reason, skip reason)``.
    """
    source, fallback = choose_parse(ctx, spec)
    while True:
        shown, to_shown = ctx.shown(source)
        question = context_prompt(shown, spec.request)
        n_prompt = count(question)
        if n_prompt <= max_prompt or source == "gold":
            break
        source, fallback = "gold", "model_parse_prompt_over_max_tokens"
    if n_prompt > max_prompt:
        return None, fallback, "prompt_over_max_tokens"
    answer = spec.render(to_shown)
    n_target = count(answer)
    if n_target > max_target:
        return None, fallback, "target_over_max_tokens"
    cites = [(to_shown[e], e, t) for t, e in spec.items]
    meta = dict(spec.meta, fallback=fallback, model_parse_page=ctx.want_model)
    return Draft(spec.family, f"{spec.detail}|{source}", spec.stem, question, answer, source, cites, list(shown),
                 n_prompt, n_target, meta), fallback, ""


def fields_spec(page: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> Optional[ContextSpec]:
    """The ``fields_from_parse`` row of a page (None without facts).

    :param page: Editions manifest record (``key`` set).
    :param facts: The page's QA facts.
    :returns: Spec.
    """
    if not facts:
        return None
    fields = page_fields(page["lines"], facts)
    items = [it for f in fields if f.items for it in f.items]
    return ContextSpec("fields_from_parse", "fields", f"vqa_{page['pgpid']}_{page['image_index']}_fields",
                       fields_request(fields), items, any(f.items is None for f in fields),
                       functools.partial(render_fields, fields),
                       {"fields": [f.name for f in fields], "field_sources": {f.name: f.source for f in fields},
                        "null_fields": [f.name for f in fields if f.items is None]})


def question_specs(facts: Sequence[Dict[str, Any]]) -> List[ContextSpec]:
    """The ``question_from_parse`` rows of a page: one per (fact, wording the QA set used).

    The QA manifest's ``questions`` are the wordings each fact was emitted with (train: two paraphrases,
    val: the canonical one).

    :param facts: The page's QA facts.
    :returns: Specs.
    """
    out = []
    for f in facts:
        items, is_list = fact_items(f), fact_is_list(f)
        for v, wording in zip(f["prompt_variants"], f["questions"]):
            out.append(ContextSpec(
                "question_from_parse", f["family"], f"vqa_{f['stem']}_w{v}", question_request(wording, is_list), items,
                not items, functools.partial(render_question, items, is_list),
                {"qa_stem": f["stem"], "qa_family": f["family"], "qa_section": f["section"], "wording": v,
                 "qa_question": wording}))
    return out


def lookup_specs(page: Dict[str, Any]) -> Tuple[List[ContextSpec], List[str]]:
    """The ``lookup_from_parse`` rows of a page.

    :param page: Editions manifest record (``key`` set).
    :returns: ``(specs, skip reasons)``.
    """
    first, second = pick_lookups(page["lines"], page["key"])
    out, skips = [], []
    for lk, name in ((first, "line_of_phrase"), (second, "after_phrase")):
        if lk is None:
            skips.append(f"lookup_{name}_no_candidate")
            continue
        out.append(ContextSpec(
            "lookup_from_parse", lk.kind, f"vqa_{page['pgpid']}_{page['image_index']}_{lk.kind}", lookup_request(lk),
            [(lk.text, lk.line)], False, functools.partial(render_lookup, lk), {"phrase": lk.phrase}))
    return out, skips


def parse_draft(page: Dict[str, Any], count: TokenCounter, boxes: Optional[Sequence[Box]] = None) -> Draft:
    """The ``parse_lines`` row, or the ``parse_lines_boxes`` row when boxes are given.

    :param page: Editions manifest record.
    :param count: Token counter.
    :param boxes: One box per line.
    :returns: Draft.
    """
    lines = page["lines"]
    if boxes is None:
        family, question, answer, suffix = "parse_lines", PARSE_PROMPT, parse_json(lines), "parse"
    else:
        family, question, answer, suffix = ("parse_lines_boxes", PARSE_BOXES_PROMPT, parse_boxes_json(lines, boxes),
                                            "parse_boxes")
    stem = f"vqa_{page['pgpid']}_{page['image_index']}_{suffix}"
    return Draft(family, "page" if boxes is None else "page_boxes", stem, question, answer, "none",
                 [(i + 1, i, t) for i, t in enumerate(lines)], [], count(question), count(answer))


# ----------------------------------------------------------------------------- invariants


def target_cites(family: str, answer: str) -> List[Tuple[str, int]]:
    """The ``(text, cited number)`` values of a context-family target, read back from its JSON.

    :param family: Row family.
    :param answer: Target text.
    :returns: Values (none for a null answer).
    :raises ValueError: When the target does not have the family's shape.
    """
    obj = json.loads(answer)
    out: List[Tuple[str, int]] = []
    if family == "fields_from_parse":
        if not isinstance(obj, dict):
            raise ValueError("fields target is not an object")
        for v in obj.values():
            for it in ([] if v is None else v if isinstance(v, list) else [v]):
                if set(it) != {"text", "line"}:
                    raise ValueError(f"field value {it!r} is not {{text, line}}")
                out.append((it["text"], it["line"]))
    elif family == "question_from_parse":
        if obj.get("answer") is None:
            if obj != {"answer": None}:
                raise ValueError("a null answer carries other keys")
        elif isinstance(obj["answer"], list):
            if set(obj) != {"answer"} or any(set(it) != {"text", "line"} for it in obj["answer"]):
                raise ValueError("list answer items are not {text, line}")
            out = [(it["text"], it["line"]) for it in obj["answer"]]
        else:
            if set(obj) != {"answer", "line"}:
                raise ValueError("answer is not {answer, line}")
            out = [(obj["answer"], obj["line"])]
    elif family == "lookup_from_parse":
        if set(obj) == {"line", "text"}:
            out = [(obj["text"], obj["line"])]
        elif set(obj) == {"answer", "line"}:
            out = [(obj["answer"], obj["line"])]
        else:
            raise ValueError("look-up target has neither {line, text} nor {answer, line}")
    else:
        raise ValueError(f"not a context family: {family}")
    return out


def check_row(d: Draft, edition: Sequence[str], to_shown: Optional[Dict[int, int]] = None) -> None:
    """The row invariants (a violation is a bug, never a skip).

    Parse families: the target is a JSON array of exactly the edition lines with ``n`` = 1..N (boxes:
    four ints in 0-1000, positive, centres top to bottom). Context families: the target parses as JSON
    of the family's shape; its values are exactly the row's cites; each cited number exists in the
    parse the prompt shows and stands for the cited edition line; each text is whole token(s) of that
    EDITION line.

    :param d: Draft.
    :param edition: Edition lines of the page.
    :param to_shown: Edition line -> number shown (context families; from :meth:`PageContext.shown`).
    :raises AssertionError: On a structural violation.
    :raises build_pgp_qa.AnswerSpanError: When a text is not whole token(s) of its edition line.
    :raises ValueError: When a target does not parse or has the wrong shape.
    """
    if d.family in ("parse_lines", "parse_lines_boxes"):
        arr = json.loads(d.answer)
        assert [el["n"] for el in arr] == list(range(1, len(edition) + 1)), "parse numbers are not 1..N"
        assert [el["text"] for el in arr] == list(edition), "parse lines are not the edition lines"
        if d.family == "parse_lines_boxes":
            boxes = [el["bbox_2d"] for el in arr]
            assert all(len(b) == 4 and all(isinstance(v, int) and 0 <= v <= 1000 for v in b) and b[0] < b[2]
                       and b[1] < b[3] for b in boxes), "a box is not four ascending ints in 0-1000"
            assert all((a[1] + a[3]) < (b[1] + b[3]) for a, b in zip(boxes, boxes[1:])), "boxes not top to bottom"
        else:
            assert all(set(el) == {"n", "text"} for el in arr)
        return
    assert d.parse in ("gold", "model") and to_shown is not None
    values = target_cites(d.family, d.answer)
    assert sorted(values) == sorted((t, n) for n, _, t in d.cites), "target values are not the row's cites"
    shown_of = {n: e for e, n in to_shown.items()}
    for n, e, text in d.cites:
        assert isinstance(n, int) and 1 <= n <= len(d.shown_lines), f"line {n} is not in the parse shown"
        assert shown_of.get(n) == e, f"line {n} of the parse shown does not stand for edition line {e + 1}"
        qa.check_whole_tokens(text, edition[e])
    if d.family == "lookup_from_parse" and json.loads(d.answer).keys() == {"line", "text"}:
        (n, e, text), = d.cites
        assert text == edition[e], "line look-up text is not the whole edition line"


def check_splits(rows: Dict[str, List[Dict[str, Any]]], page_of_stem: Dict[str, Dict[str, Any]],
                 registered: Tuple[Set[str], Set[str]]) -> Dict[str, int]:
    """Split hygiene across the finished rows.

    :param rows: ``split -> feature rows``.
    :param page_of_stem: Row stem -> editions manifest record.
    :param registered: ``(canonical ids, pgpids)`` of registered benchmark documents.
    :returns: Counts that were checked.
    :raises AssertionError: When a val page image (path or sha256) is in a train split, a registered
        document is in train, or a val row comes from a train page.
    """
    val_paths, val_shas, train_paths, train_shas = set(), set(), set(), set()
    ids, pgpids = registered
    for split, rs in rows.items():
        for r in rs:
            p = page_of_stem[r["stem"]]
            if split == "val":
                assert p["split"] == "val", f"{r['stem']}: train page in val"
                val_paths.add(r["image"])
                val_shas.add(p["image_sha256"])
            else:
                assert p["split"] != "val", f"{r['stem']}: val page in {split}"
                assert p["canonical_id"] not in ids and p["pgpid"] not in pgpids, f"{r['stem']}: registered in train"
                train_paths.add(r["image"])
                train_shas.add(p["image_sha256"])
    assert not val_paths & train_paths and not val_shas & train_shas, "a val page image is in a train split"
    return {"val_images": len(val_paths), "train_images": len(train_paths)}


# ----------------------------------------------------------------------------- build


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File.
    :returns: Records.
    """
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def count_tokens(tokenizer: Any, text: str) -> int:
    """Tokens of a text, without special tokens.

    :param tokenizer: A ``tokenizers.Tokenizer``.
    :param text: Text.
    :returns: Token count.
    """
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def tokenizer_counter(path: Path) -> TokenCounter:
    """Token counter of a ``tokenizers`` file (the v21b tokenizer by default).

    :param path: ``tokenizer.json``.
    :returns: ``text -> tokens`` without special tokens.
    """
    from tokenizers import Tokenizer  # local import: the module stays importable without the package

    return functools.partial(count_tokens, Tokenizer.from_file(str(path)))


def percentiles(values: Sequence[int]) -> Dict[str, int]:
    """p50 / p90 / p95 / p99 / max of a list (nearest rank).

    :param values: Numbers.
    :returns: Percentiles (empty for no values).
    """
    if not values:
        return {}
    s = sorted(values)
    return {f"p{q}": s[min(len(s) - 1, int(q / 100 * len(s)))] for q in (50, 90, 95, 99)} | {"max": s[-1], "n": len(s)}


def feature_row(page: Dict[str, Any], d: Draft) -> Dict[str, Any]:
    """A dataset row in the KTIV ``FEATURES`` schema.

    :param page: Editions manifest record.
    :param d: Draft.
    :returns: Row (the image as its NAS path; ``save_to_disk`` embeds the bytes).
    """
    return {"image": page["image_path"], "question": d.question, "answer": d.answer, "task": d.family,
            "section": d.section, "stem": d.stem, "label_source": LABEL_SOURCE, "target_chars": len(d.answer),
            "target_tokens": d.target_tokens, "image_width": page["image_width"], "image_height": page["image_height"]}


def manifest_record(page: Dict[str, Any], d: Draft, split: str) -> Dict[str, Any]:
    """One manifest record per row (what the review and later audits need).

    :param page: Editions manifest record.
    :param d: Draft.
    :param split: Split name.
    :returns: Record.
    """
    rec = {"stem": d.stem, "family": d.family, "section": d.section, "split": split, "parse": d.parse,
           "pgpid": page["pgpid"], "canonical_id": page["canonical_id"], "image_index": page["image_index"],
           "image_url": page["image_url"], "prompt_tokens": d.prompt_tokens, "target_tokens": d.target_tokens}
    if d.family in CONTEXT_FAMILIES:
        rec["cites"] = [{"shown_line": n, "edition_line": e + 1, "text": t,
                         "shown_text": d.shown_lines[n - 1]} for n, e, t in d.cites]
    rec.update(d.meta)
    return rec


@dataclass
class BuildResult:
    """Everything one build produced.

    :param rows: ``split -> feature rows``.
    :param manifest: One record per row.
    :param stats: Build statistics.
    :param review: ``review_sample.md`` text.
    """

    rows: Dict[str, List[Dict[str, Any]]]
    manifest: List[Dict[str, Any]]
    stats: Dict[str, Any]
    review: str


def load_page_reads(raw_dir: Path, page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The page's raw reader record, when it describes the page's image.

    :param raw_dir: ``ai_reads/raw``.
    :param page: Editions manifest record.
    :returns: The record, or None when missing or for another image (URL or sha256 differ).
    """
    path = eds.evidence_path(raw_dir, page["canonical_id"], page["image_index"])
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("image_url") != page["image_url"] or raw.get("sha256") != page["image_sha256"]:
        return None
    return raw


def build(editions_dir: Path = DEFAULT_EDITIONS, qa_dir: Path = DEFAULT_QA, raw_dir: Path = eds.RAW_DIR,
          count: Optional[TokenCounter] = None, model_parse_share: float = MODEL_PARSE_SHARE,
          max_prompt_tokens: int = MAX_PROMPT_TOKENS, max_target_tokens: int = MAX_TARGET_TOKENS,
          registered: Optional[Tuple[Set[str], Set[str]]] = None, limit_pages: int = 0,
          dataset_name: str = DEFAULT_OUT.name) -> BuildResult:
    """Build every row in memory (no writes).

    :param editions_dir: ``pgp_editions_v1`` (``manifest.jsonl``).
    :param qa_dir: ``pgp_qa_v2`` (``manifest.jsonl``).
    :param raw_dir: ``ai_reads/raw`` (read only).
    :param count: Token counter (default: the v21b tokenizer).
    :param model_parse_share: Share of pages whose context rows show the model reading.
    :param max_prompt_tokens: Prompt token cap.
    :param max_target_tokens: Target token cap.
    :param registered: Registered benchmark ``(ids, pgpids)`` (default: the registry).
    :param limit_pages: Use only the first N pages (smoke runs; 0 = all).
    :param dataset_name: Name recorded in the stats.
    :returns: The result.
    :raises AssertionError: On an invariant or split-hygiene violation.
    """
    t0 = time.time()
    count = count or tokenizer_counter(DEFAULT_TOKENIZER)
    registered = registered if registered is not None else registered_benchmark_documents()
    pages = read_jsonl(editions_dir / "manifest.jsonl")
    if limit_pages:
        pages = pages[:limit_pages]
    facts_by_page: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for f in read_jsonl(qa_dir / "manifest.jsonl"):
        facts_by_page[f"{f['canonical_id']}__{f['image_index']}"].append(f)
    tallies = Tallies()
    rows: Dict[str, List[Dict[str, Any]]] = {f"train_{f}": [] for f in FAMILIES}
    rows["val"] = []
    manifest: List[Dict[str, Any]] = []
    page_of_stem: Dict[str, Dict[str, Any]] = {}
    ids, pgpids = registered
    for page in pages:
        page["key"] = eds.page_key(page["canonical_id"], page["image_index"])
        split = "val" if page["split"] == "val" else "train"
        registered_doc = page["canonical_id"] in ids or page["pgpid"] in pgpids
        tallies.pages[f"pages_{split}"] += 1
        if split == "train" and registered_doc:
            tallies.skips["page_registered_benchmark_document_in_train"] += 1
            continue
        tallies.pages["val_pages_registered"] += split == "val" and registered_doc
        facts = facts_by_page.get(page["key"], [])
        for f in facts:
            if f["split"] != page["split"]:
                raise AssertionError(f"fact {f['stem']} is {f['split']} on a {page['split']} page")
        raw = load_page_reads(raw_dir, page)
        for d, to_shown in page_drafts(page, facts, raw, count, tallies, model_parse_share, max_prompt_tokens,
                                       max_target_tokens):
            check_row(d, page["lines"], to_shown)
            name = "val" if split == "val" else f"train_{d.family}"
            assert d.stem not in page_of_stem, f"duplicate stem {d.stem}"
            page_of_stem[d.stem] = page
            rows[name].append(feature_row(page, d))
            manifest.append(manifest_record(page, d, name))
    hygiene = check_splits(rows, page_of_stem, registered)
    stats = build_stats(pages, rows, manifest, tallies, hygiene,
                        dict(editions=str(editions_dir / "manifest.jsonl"), qa=str(qa_dir / "manifest.jsonl"),
                             raw_dir=str(raw_dir), reader_model=READER_MODEL, dataset=dataset_name,
                             model_parse_share=model_parse_share, max_prompt_tokens=max_prompt_tokens,
                             max_target_tokens=max_target_tokens, seconds=round(time.time() - t0, 1)))
    return BuildResult(rows, manifest, stats, review_markdown(review_sample(manifest), stats, rows))


def fallback_counters() -> Dict[str, collections.Counter]:
    """One empty fallback counter per context family.

    :returns: ``family -> Counter``.
    """
    return {f: collections.Counter() for f in CONTEXT_FAMILIES}


@dataclass
class Tallies:
    """Counters a build accumulates.

    :param skips: Skip reasons (rows and pages).
    :param fallbacks: ``family -> model-parse fallback reasons``.
    :param boxes: Line-box rule outcome per page.
    :param pages: Page counters.
    """

    skips: collections.Counter = field(default_factory=collections.Counter)
    fallbacks: Dict[str, collections.Counter] = field(default_factory=fallback_counters)
    boxes: collections.Counter = field(default_factory=collections.Counter)
    pages: collections.Counter = field(default_factory=collections.Counter)


def page_drafts(page: Dict[str, Any], facts: Sequence[Dict[str, Any]], raw: Optional[Dict[str, Any]],
                count: TokenCounter, tallies: Tallies, model_parse_share: float = MODEL_PARSE_SHARE,
                max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                max_target_tokens: int = MAX_TARGET_TOKENS) -> List[Tuple[Draft, Optional[Dict[int, int]]]]:
    """Every row of one page, before the invariant check.

    :param page: Editions manifest record (``key`` set).
    :param facts: The page's QA facts.
    :param raw: The page's reader record (None when missing).
    :param count: Token counter.
    :param tallies: Counters (updated).
    :param model_parse_share: Share of pages whose context rows show the model reading.
    :param max_prompt_tokens: Prompt token cap.
    :param max_target_tokens: Target token cap.
    :returns: ``(draft, edition line -> number shown or None for the parse families)``.
    """
    split = "val" if page["split"] == "val" else "train"
    if raw is None:
        tallies.skips["page_without_reader_record"] += 1
    drafts: List[Tuple[Draft, Optional[Dict[int, int]]]] = []
    d = parse_draft(page, count)
    if d.target_tokens > max_target_tokens:
        tallies.skips["parse_lines_target_over_max_tokens"] += 1
    else:
        drafts.append((d, None))
    res = line_boxes(page["lines"], (raw or {}).get("frags") or [], page["image_width"])
    tallies.boxes[res.reason or "qualified"] += 1
    if res.boxes is not None:
        tallies.boxes[f"qualified_{split}"] += 1
        d = parse_draft(page, count, res.boxes)
        d.meta = {"row_similarity": [round(s, 3) for s in res.sims],
                  "row_edge_coverage": [round(c, 3) for c in res.coverage]}
        if d.target_tokens > max_target_tokens:
            tallies.skips["parse_lines_boxes_target_over_max_tokens"] += 1
        else:
            drafts.append((d, None))

    ctx = PageContext(page["key"], page["lines"], wants_model_parse(page["key"], model_parse_share))
    if ctx.want_model:
        tallies.pages[f"model_parse_pages_{split}"] += 1
        ctx.reading = model_reading(raw) if raw else []
        ctx.reading_map = map_reading(page["lines"], ctx.reading)
        ctx.reading_has_date = any(qa.has_date_indication(t) for t in ctx.reading)
    specs: List[ContextSpec] = []
    spec = fields_spec(page, facts)
    if spec is not None:
        specs.append(spec)
        tallies.pages[f"pages_with_facts_{split}"] += 1
    specs += question_specs(facts)
    look, look_skips = lookup_specs(page)
    specs += look
    tallies.skips.update(look_skips)
    for spec in specs:
        d, fallback, skip = make_context_row(ctx, spec, count, max_prompt_tokens, max_target_tokens)
        if fallback:
            tallies.fallbacks[spec.family][fallback] += 1
        if d is None:
            tallies.skips[f"{spec.family}_{skip}"] += 1
            continue
        drafts.append((d, ctx.shown(d.parse)[1]))
    return drafts


def build_stats(pages: Sequence[Dict[str, Any]], rows: Dict[str, List[Dict[str, Any]]],
                manifest: Sequence[Dict[str, Any]], tallies: Tallies, hygiene: Dict[str, int],
                run: Dict[str, Any]) -> Dict[str, Any]:
    """The ``stats.json`` content.

    :param pages: Pages considered.
    :param rows: ``split -> feature rows``.
    :param manifest: One record per row.
    :param tallies: Build counters.
    :param hygiene: Split-hygiene counts.
    :param run: Run parameters.
    :returns: Stats.
    """
    skips, fallbacks, box_funnel, page_counts = tallies.skips, tallies.fallbacks, tallies.boxes, tallies.pages
    by_fs = collections.Counter((m["family"], "val" if m["split"] == "val" else "train") for m in manifest)
    parse_use = collections.Counter((m["family"], m["parse"]) for m in manifest)
    ctx_rows = [m for m in manifest if m["family"] in CONTEXT_FAMILIES]
    tokens = {f: {"prompt": percentiles([m["prompt_tokens"] for m in manifest if m["family"] == f]),
                  "target": percentiles([m["target_tokens"] for m in manifest if m["family"] == f])}
              for f in FAMILIES}
    tokens["context_families_prompt_by_parse"] = {
        p: percentiles([m["prompt_tokens"] for m in ctx_rows if m["parse"] == p]) for p in ("gold", "model")}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), **run,
        "status": "pilot, not yet reviewed",
        "target_rule": "every target text is a human edition line (or whole tokens of one); the model reading "
                       "appears only in prompts; Kraken text is only box evidence",
        "pages": {"total": len(pages), **dict(page_counts)},
        "rows_by_split": {k: len(v) for k, v in rows.items()},
        "rows_total": sum(len(v) for v in rows.values()),
        "rows_by_family_and_split": {f: {s: by_fs[(f, s)] for s in ("train", "val")} for f in FAMILIES},
        "rows_by_section": dict(collections.Counter(m["section"] for m in manifest)),
        "question_rows_by_qa_family": dict(collections.Counter(
            f"{m['qa_family']}:{'val' if m['split'] == 'val' else 'train'}" for m in manifest
            if m["family"] == "question_from_parse")),
        "parse_context": {
            "rows_by_family_and_parse": {f: {p: parse_use[(f, p)] for p in ("gold", "model")}
                                         for f in CONTEXT_FAMILIES},
            "rows_with_model_reading": sum(m["parse"] == "model" for m in ctx_rows),
            "context_rows": len(ctx_rows),
            "share_with_model_reading": round(sum(m["parse"] == "model" for m in ctx_rows) / max(1, len(ctx_rows)), 4),
            "rows_on_model_parse_pages": sum(bool(m.get("model_parse_page")) for m in ctx_rows),
            "fallbacks_to_gold": {f: dict(c) for f, c in fallbacks.items()},
            "fallbacks_total": sum(sum(c.values()) for c in fallbacks.values())},
        "boxes": {"rule": f"Kraken fragments with a Hebrew letter -> core-band rows; monotonic one-to-one line->row "
                          f"assignment at similarity >= {BOX_MIN_SIM} for EVERY line (max total similarity); box = "
                          f"union of the row's fragments; documentary-grounding geometry check (height <= 3 median "
                          f"fragment heights, >= 60 px wide); centres top to bottom; consecutive overlap <= "
                          f"{MAX_BOX_OVERLAP} of the smaller height; every row reads its line edge to edge (edge "
                          f"coverage >= {BOX_MIN_EDGE_COVERAGE}, added after a visual check found boxes stopping "
                          f"short where Kraken did not segment a line's first words)",
                  "page_outcomes": dict(box_funnel),
                  "qualified_without_edge_coverage_guard": box_funnel["qualified"]
                  + box_funnel["row_misses_part_of_a_line"]},
        "fields": {"mapping": {f"{fam}:{sec}": name for (fam, sec), name in FIELD_OF.items()},
                   "definitions": {name: field_definition(name) for name in FIELD_ORDER},
                   "rows_by_field": dict(collections.Counter(n for m in manifest if m["family"] == "fields_from_parse"
                                                             for n in m["fields"])),
                   "null_values_by_field": dict(collections.Counter(n for m in manifest
                                                                    if m["family"] == "fields_from_parse"
                                                                    for n in m["null_fields"]))},
        "tokens": tokens,
        "skipped": dict(skips),
        "split_hygiene": hygiene,
        "invariants": "checked on every row: targets parse as JSON; parse targets equal the edition lines; every "
                      "answer text is whole token(s) of the edition line behind its cited number; every cited "
                      "number exists in the parse shown; no val page image in train; no registered benchmark "
                      "document in train; prompt tokens <= cap",
    }


# ----------------------------------------------------------------------------- review sample


def review_kind(m: Dict[str, Any]) -> str:
    """What the review sample spreads a family's rows over.

    :param m: Manifest record.
    :returns: QA family (questions), requested fields (fields), look-up kind (look-ups), split otherwise.
    """
    if m["family"] == "question_from_parse":
        return m["qa_family"]
    if m["family"] == "fields_from_parse":
        return ",".join(m["fields"])
    if m["family"] == "lookup_from_parse":
        return m["section"].split("|")[0]
    return m["split"]


def spread(recs: Sequence[Dict[str, Any]], n: int, taken: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Up to ``n`` records in the given order, one per :func:`review_kind` first, then the rest.

    :param recs: Records in draw order.
    :param n: How many.
    :param taken: Kinds already represented (counted as seen).
    :returns: The picks.
    """
    seen = set(taken or ())
    first, rest = [], []
    for m in recs:
        (rest if review_kind(m) in seen else first).append(m)
        seen.add(review_kind(m))
    return (first + rest)[:n]


def review_sample(manifest: Sequence[Dict[str, Any]], k: int = REVIEW_ROWS_PER_FAMILY,
                  seed: int = SEED) -> List[Dict[str, Any]]:
    """``k`` rows per family drawn by ``stable_hash``, spread over :func:`review_kind`; context families
    take half their rows from model-reading parses when available (each half spread on its own; the
    question family also spreads its QA families across the two halves).

    :param manifest: One record per row.
    :param k: Rows per family.
    :param seed: Seed.
    :returns: Sampled manifest records, grouped by family.
    """
    out: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        recs = sorted((m for m in manifest if m["family"] == fam),
                      key=lambda m: eds.stable_hash(f"{seed}:review:{m['stem']}"))
        if fam in CONTEXT_FAMILIES:
            model = spread([m for m in recs if m["parse"] == "model"], k // 2)
            taken = {review_kind(m) for m in model} if fam == "question_from_parse" else None
            gold = spread([m for m in recs if m["parse"] == "gold"], k - len(model), taken)
            out += model + gold
        else:
            out += spread(recs, k)
    return out


def review_markdown(sample: Sequence[Dict[str, Any]], stats: Dict[str, Any],
                    rows: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render the review sample.

    :param sample: Sampled manifest records.
    :param stats: Build stats.
    :param rows: ``split -> feature rows`` (prompt and target text).
    :returns: Markdown.
    """
    text_of = {r["stem"]: r for rs in rows.values() for r in rs}
    out = [f"# {stats['dataset']} — review sample", "",
           f"Built {stats['generated_at']}. Rows: {json.dumps(stats['rows_by_split'])}. "
           f"{REVIEW_ROWS_PER_FAMILY} rows per family; context families show half model-reading parses when "
           "available. Every target text must be the human edition (whole tokens of the cited edition line); the "
           "model reading may appear only inside a prompt. Mark rows that are wrong.", ""]
    fam = None
    for i, m in enumerate(sample, 1):
        if m["family"] != fam:
            fam = m["family"]
            out += [f"## {fam}", ""]
        r = text_of[m["stem"]]
        out += [f"### {i}. `{m['stem']}` — {m['canonical_id']} (image {m['image_index']}), {m['split']}", "",
                f"- image: {m['image_url']}", f"- section: `{m['section']}` — parse in prompt: **{m['parse']}**"
                + (f" (model-parse page, fell back: {m['fallback']})" if m.get("fallback") else ""),
                f"- tokens: prompt {m['prompt_tokens']}, target {m['target_tokens']}"]
        for c in m.get("cites", []):
            out.append(f"- cites line {c['shown_line']} of the parse shown = edition line {c['edition_line']}; "
                       f"shown text: {c['shown_text']}")
        out += ["", "Prompt:", "", "```text", r["question"], "```", "", "Target:", "", "```json", r["answer"], "```",
                ""]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- outputs


def write_outputs(result: BuildResult, out_dir: Path, images_once_dir: Optional[Path], dry_run: bool) -> None:
    """Write the DatasetDict and sidecars, then the images-once export.

    :param result: Build result.
    :param out_dir: Dataset directory (NAS); in a dry run only the sidecars are written there.
    :param images_once_dir: Images-once export directory (None = no export).
    :param dry_run: Write the sidecars only.
    """
    sidecars = {"stats.json": json.dumps(result.stats, ensure_ascii=False, indent=1),
                "manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in result.manifest),
                "review_sample.md": result.review}
    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in sidecars.items():
            (out_dir / name).write_text(text, encoding="utf-8")
        return
    eds.save_dataset(result.rows, out_dir, sidecars)
    if images_once_dir is not None:
        from src.finetuning.qwen_hebrew.images_once import export_images_once  # torch import only when exporting

        export_images_once(out_dir, images_once_dir)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editions-dir", type=Path, default=DEFAULT_EDITIONS)
    ap.add_argument("--qa-dir", type=Path, default=DEFAULT_QA)
    ap.add_argument("--raw-dir", type=Path, default=eds.RAW_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--images-once-dir", type=Path, default=DEFAULT_IMAGES_ONCE)
    ap.add_argument("--no-export", action="store_true", help="skip the images-once export")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--model-parse-share", type=float, default=MODEL_PARSE_SHARE)
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    ap.add_argument("--max-target-tokens", type=int, default=MAX_TARGET_TOKENS)
    ap.add_argument("--limit-pages", type=int, default=0, help="first N pages only (smoke runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build in memory and write only stats.json, manifest.jsonl and review_sample.md")
    a = ap.parse_args()
    result = build(a.editions_dir, a.qa_dir, a.raw_dir, tokenizer_counter(a.tokenizer), a.model_parse_share,
                   a.max_prompt_tokens, a.max_target_tokens, limit_pages=a.limit_pages, dataset_name=a.output_dir.name)
    logger.info("stats: %s", json.dumps(result.stats, ensure_ascii=False, indent=1))
    write_outputs(result, a.output_dir, None if a.no_export else a.images_once_dir, a.dry_run)


if __name__ == "__main__":
    main()
