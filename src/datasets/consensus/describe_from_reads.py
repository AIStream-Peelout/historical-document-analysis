# File name: describe_from_reads.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Machine-generated catalogue descriptions from consensus page reads (text only, the image is never shown).

About 14.5k records of ``genizah_merged_v9`` carry no vector because their catalogue text is empty or boilerplate.
The consensus pipeline (:mod:`src.datasets.consensus.two_reader_lines`) reads their pages with two machine readers;
this module turns one document's reads into a short catalogue description in the register of PGP descriptions,
which the fine-tuned text embedder can embed (:mod:`src.datasets.indexing.update_generated_descriptions` writes it
into the index).

Principles (non-negotiable):

* the generator is a text-only model and never sees an image;
* every people / places / dates / commodities / institutions span it returns must occur verbatim in the
  transcription it was shown (:func:`grounding_check`); violators are dropped and recorded, and the confidence is
  capped at :data:`CONFIDENCE_CAP`;
* outputs carry provenance (pages, readers, files, model, prompt version, time) and are labelled machine-derived
  (:data:`MACHINE_TAG` in the text, ``description_source`` in the row and in the index);
* generated text is never a training target for any reader;
* an identified literary work stays ``unverified`` unless a symbolic check confirms it (``verify_work`` hook of
  :func:`grounding_check`; none is wired in yet).

The per-page record (``two_reader_lines``) is ``{doc_id, source_index, image_index, image_url, image_sha256,
ai_read: {vlm_model, vlm_revision, htr_model, htr_cache_key?, rule_version, decoded_at, parsed, n_lines, n_agreed,
lines: [{index, text, bbox, agreement, status, htr_text, htr_fragments}]}}``. Design choices it forces:

* ``doc_id`` / ``image_index`` belong to the index the record was stamped against (``source_index`` v5 to v9):
  ids were renamed and image lists reordered between versions, and two records can list the same images. A page
  is therefore identified by its bytes (``image_sha256``, else the URL); with an image map (the vector-less pool
  file: v9 ``doc_id``, ``image_index``, ``image_url`` per image) each read is re-keyed by its exact image URL to
  every v9 document that lists it, and reads of URLs outside the map are ignored. Without a map the record's own
  ``doc_id`` / ``image_index`` are used.
* A page may hold several reads (checkpoints, Kraken keys, rule versions). One is used (:func:`page_rank`): an
  optional reader preference first, then a parsed reply, most ``[agreed]`` lines, highest agreed share, newest decode.
* A line is ``[agreed]`` when its ``status`` is ``agreed`` (:mod:`~src.datasets.consensus.line_rule`: agreement >=
  0.8 against Kraken's text); every other line is shown as ``[unverified]``.
* Records carry no read-health block today: ``parsed: false`` and empty pages are flagged, and a ``read_health``
  block is passed through when a record has one.

Grounding normalisation (:func:`normalise_for_grounding`): NFKD, nonspacing marks dropped (niqqud, cantillation,
dagesh, Arabic harakat), geresh / gershayim and curly quotes made ASCII, maqaf, hyphen and ``. , ; : ! ? ׀ ׃`` (and
their Arabic forms) turned into spaces, tatweel removed, Hebrew final letters folded to the medial forms (on by
default; ``fold_finals=False`` keeps them), casefolded, whitespace collapsed. A span is grounded when its normalised
form has at least :data:`MIN_SPAN_LETTERS` letters and is a substring of the normalised transcription, so it may sit
inside a longer word (Hebrew proclitics: ``ובפסטאט`` grounds ``פסטאט``) but never spans a ``[...]`` gap.

LM Studio: OpenAI-compatible chat completions (text only, the conventions of
:mod:`src.models.ocr.lms_transcriber`: one request at a time, low temperature, LM Studio's error body surfaced,
retries with back-off), ``response_format`` ``json_schema`` built from :data:`OUTPUT_SCHEMA` (when the server
refuses it the run falls back to instructions plus parsing), one repair retry on invalid JSON. Structural validation
uses ``jsonschema`` (installed with ``kraken``) against the same schema.

Reasoning (``--reasoning on|off``, default ``off``): ``qwen/qwen3.8-27b`` is a hybrid-thinking model whose chat
template thinks by default at reasoning effort ``xhigh`` and switches thinking off only through the template
variable ``enable_thinking=false``. The site's LM Studio client notes that this LM Studio build honours the
top-level ``reasoning_effort`` field (``low`` / ``medium`` / ``xhigh``) and ignores ``chat_template_kwargs``, so
``off`` sends ``reasoning_effort: low`` plus ``chat_template_kwargs: {enable_thinking: false}`` (no thinking where
honoured, brief thinking otherwise) with a 2,048-token budget, and ``on`` sends ``reasoning_effort: xhigh`` with an
8,192-token budget. The reasoning channel is never stored or used, except that under a JSON schema a reply whose
entire reasoning channel is one JSON object is taken as the answer (LM Studio can put constrained output there).

Usage (repo root)::

    .venv/bin/python -m src.datasets.consensus.describe_from_reads \\
        --reads src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl \\
        --pool logs/next_round/consensus_pool/jobs_vectorless_pool.jsonl --pool-read-class already_read \\
        --catalogue src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl \\
        --out logs/next_round/consensus_pool/generated_descriptions.jsonl [--dry-run] [--limit 10]
"""
import argparse
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import requests
from jsonschema import Draft202012Validator

from src.datasets.consensus.line_rule import line_status
from src.datasets.document_models.genizah_document import _ktiv_alt_titles, _ktiv_scholarly_frames
from src.embeddings.semantic_text import semantic_ineligible_reason
from src.embeddings.text_representation import DEFAULT_TEXT_VERSION, embed_text_for_version
from src.models.ocr import lms_yield_gate
from src.models.ocr.lms_transcriber import DEFAULT_CONFIG

PROMPT_VERSION = "gen-desc-v1-20261008"   # bump whenever SYSTEM_PROMPT, OUTPUT_INSTRUCTIONS or the user-prompt layout change
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DESCRIPTION_SOURCE = "generated_from_consensus"
MACHINE_TAG = "[machine-generated from automatic transcription]"
DEFAULT_MAX_PAGES = 4
DEFAULT_MAX_CHARS = 8000        # transcription characters per prompt (~4-6k tokens of Hebrew): load the model with >= 8k context
TEMPERATURE = 0.1               # lms_transcriber's default: low temperature, little invention
REQUEST_TIMEOUT_S = 600.0       # a 27B model on Apple Silicon; lms_transcriber uses the same budget
MAX_RETRIES = 2                 # retries of a failed request (HTTP 5xx, connection); a 400 is not retried as is
REASONING_MODES = ("off", "on")
REASONING_FIELDS: Dict[str, Dict[str, Any]] = {
    "off": {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}},
    "on": {"reasoning_effort": "xhigh"},
}
MAX_TOKENS = {"off": 2048, "on": 8192}   # the JSON is ~400-700 tokens; thinking needs the rest
MAX_TOKENS_CEILING = 16384      # a reply cut by the budget (finish_reason=length) is asked again with the budget doubled up to here
CONFIDENCE_CAP = 0.5            # a reply with any ungrounded span or misquote is capped here
MIN_EMBED_CONFIDENCE = 0.25     # below this the description is kept but not embedded
MIN_SUMMARY_CHARS = 40          # semantic_text.MIN_CHARS, applied to the summary alone: metadata lines never carry it
MIN_SPAN_LETTERS = 2            # a grounded span needs at least this many letters after normalisation
MAX_VALUE_CHARS = 300           # catalogue values are cut here in the prompt
REPAIR_MAX_ERRORS = 8           # validation errors quoted back to the model in the repair turn
SCHEMA_NAME = "catalogue_description"

GENRES = ("documentary", "liturgical", "biblical", "rabbinic", "literary", "paraliterary", "unclear")
LANGUAGES = ("Hebrew", "Aramaic", "Judaeo-Arabic", "Arabic", "Judaeo-Persian", "Judaeo-Spanish", "Judaeo-Greek",
             "Greek", "Syriac", "Coptic", "Latin", "Other", "Unclear")
SCRIPTS = ("Hebrew", "Arabic", "Hebrew and Arabic", "Latin", "Other", "Unclear")
ENTITY_KINDS = ("people", "places", "dates", "commodities", "institutions")
DOCUMENT_TYPES = ("Letter", "Legal document", "Legal query or responsum", "State document", "List or table", "Account",
                  "Bible", "Targum", "Mishnah", "Talmud", "Midrash", "Halakhic work", "Commentary", "Prayer book",
                  "Piyyut", "Poetry", "Philosophy", "Medicine", "Magic", "Writing exercise", "Unidentified")

_SPAN_LIST: Dict[str, Any] = {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 120},
                              "maxItems": 25}
OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["genre", "document_type", "language", "script", "summary", "identified_work", "entities", "themes",
                 "confidence", "notes"],
    "properties": {
        "genre": {"type": "string", "enum": list(GENRES)},
        "document_type": {"type": "string", "minLength": 1, "maxLength": 80},
        "language": {"type": "array", "items": {"type": "string", "enum": list(LANGUAGES)}, "minItems": 1,
                     "maxItems": 4},
        "script": {"type": "string", "enum": list(SCRIPTS)},
        "summary": {"type": "string", "minLength": 20, "maxLength": 1200},
        "identified_work": {"anyOf": [
            {"type": "null"},
            {"type": "object", "additionalProperties": False, "required": ["title", "section", "confidence"],
             "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 200},
                            "section": {"anyOf": [{"type": "string", "maxLength": 200}, {"type": "null"}]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}]},
        "entities": {"type": "object", "additionalProperties": False, "required": list(ENTITY_KINDS),
                     "properties": {kind: _SPAN_LIST for kind in ENTITY_KINDS}},
        "themes": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 60}, "maxItems": 8},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string", "maxLength": 600},
    },
}
_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)

SYSTEM_PROMPT = """You are a cataloguer of Cairo Genizah fragments. You write catalogue descriptions in the register of the Princeton Geniza Project's document descriptions: factual, concise, third person, present tense, in English, with Hebrew, Aramaic and Arabic names and terms transliterated (ketubba, piyyut, nagid, dinar).

You never see the manuscript. You work only from catalogue metadata and an automatic transcription made by two independent machine readers (a vision-language model and a handwriting-recognition model). Each transcription line is prefixed:
[agreed]      both readers produced essentially the same text: reliable;
[unverified]  only the vision-language model's reading: it may contain misreadings or invented words.
"[...]" marks a gap or an illegible passage. Base every statement on the transcription and give [agreed] lines more weight than [unverified] ones. Catalogue labels can be generic or wrong: use them only as hints.

Rules:
1. Describe only what the text supports. Never invent people, places, dates, sums, goods or events. When the text is too short or damaged to say what it is, say so plainly and give a low confidence.
2. The summary has 2 to 4 sentences. Start with what the fragment is (for example "Letter ...", "Legal deed ...", "Fragment of a piyyut ..."), then its content. Do not mention the shelf mark, the library, the machine readers or the transcription.
3. Entity lists hold spans copied character for character from the transcription, in its original script: people (personal names), places, dates (dates and years as written in the text), commodities (goods, materials, coins and sums), institutions (courts, synagogues, academies, offices). Do not translate, transliterate, correct or complete a span. Leave a list empty rather than guess.
4. If you quote the transcription in the summary, copy the words exactly, in the original script, inside double quotes.
5. identified_work: only when the text is clearly a known work (a biblical passage, Mishnah, Talmud, a known prayer or piyyut, a known book), give its title and section; otherwise null. Identifications are checked separately: rate your confidence honestly.
6. Reply with one JSON object that follows the requested keys and nothing else."""

OUTPUT_INSTRUCTIONS = (
    "Return one JSON object with exactly these keys:\n"
    f"- \"genre\": one of {', '.join(json.dumps(g) for g in GENRES)} (documentary = letters, legal and state "
    "documents, accounts, lists; liturgical = prayers and piyyutim; biblical = Bible text, targum, masora; rabbinic = "
    "Mishnah, Talmud, midrash, halakha, responsa, commentaries; literary = poetry, philosophy, science, grammar and "
    "other books; paraliterary = amulets, magic, medical recipes, writing exercises, calendars, jottings).\n"
    f"- \"document_type\": a short catalogue type, such as {', '.join(DOCUMENT_TYPES)}.\n"
    f"- \"language\": a list of one or more of {', '.join(json.dumps(lang) for lang in LANGUAGES)}.\n"
    f"- \"script\": one of {', '.join(json.dumps(s) for s in SCRIPTS)}; normally the computed script above.\n"
    "- \"summary\": 2 to 4 sentences in English, catalogue register.\n"
    "- \"identified_work\": null, or {\"title\": ..., \"section\": ... or null, \"confidence\": 0 to 1}.\n"
    "- \"entities\": {\"people\": [...], \"places\": [...], \"dates\": [...], \"commodities\": [...], "
    "\"institutions\": [...]}, each a list of spans copied exactly from the transcription (empty lists are fine).\n"
    "- \"themes\": up to 8 short English topic keywords.\n"
    "- \"confidence\": 0 to 1, how certain the summary is.\n"
    "- \"notes\": one short sentence on doubts (damage, unverified lines), or \"\".\n"
    "Reply with the JSON object only."
)

REPAIR_PROMPT = ("Your reply did not pass validation:\n{errors}\n"
                 "Reply again with only the corrected JSON object, with the same keys and rules.")

_RECORD_HEAD = re.compile(r'^\{"doc_id":\s*"((?:[^"\\]|\\.)*)",\s*"source_index":\s*(?:"(?:[^"\\]|\\.)*"|null),'
                          r'\s*"image_index":\s*-?\d+,\s*"image_url":\s*"((?:[^"\\]|\\.)*)"')
_ID_KEY = re.compile(r'"(?:canonical_id|doc_id|_id)":\s*"((?:[^"\\]|\\.)*)"')
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(?:unknown|unknown/לא ידוע|לא ידוע|none|null|n/?a|\[\]|\{\})\.?$", re.IGNORECASE)


def _char_class(*ranges: Tuple[int, int]) -> str:
    """Regex character-class body for code-point ranges (written in hex, so no invisible character sits in the source).

    :param ranges: Inclusive ``(first, last)`` code points.
    :return: Class body, e.g. ``א-ת`` for ``(0x05D0, 0x05EA)``.
    :rtype: str
    """
    return "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in ranges)


_HEB_LETTER = re.compile("[א-ת]")     # the 27 Hebrew letters, final forms included
_ARA_LETTER = re.compile(f"[{_char_class((0x0620, 0x064A), (0x0671, 0x06D3))}]")
_LAT_LETTER = re.compile(f"[A-Za-z{_char_class((0x00C0, 0x024F))}]")
# Hebrew and Arabic blocks plus their presentation forms
_SCRIPT_CHARS = _char_class((0x0590, 0x05FF), (0x0600, 0x06FF), (0xFB1D, 0xFB4F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_SCRIPT_RUN = re.compile(rf"[{_SCRIPT_CHARS}](?:[{_SCRIPT_CHARS}'\"\s]*[{_SCRIPT_CHARS}'\"])?")
# An ASCII quote opens only after start / space / bracket and closes only before end / space / punctuation, so the
# gershayim of a Hebrew abbreviation typed as '"' (רמב"ם) never pairs up with a real quotation mark.
_QUOTED = re.compile(r'(?:^|(?<=[\s(\[]))"([^"\n]{1,300})"(?=$|[\s.,;:!?)\]])'
                     r"|\N{LEFT DOUBLE QUOTATION MARK}([^\N{RIGHT DOUBLE QUOTATION MARK}\n]{1,300})"
                     r"\N{RIGHT DOUBLE QUOTATION MARK}"
                     r"|\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}"
                     r"([^\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}\n]{1,300})"
                     r"\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_QUOTE_MAP = str.maketrans({
    "\N{HEBREW PUNCTUATION GERESH}": "'", "\N{HEBREW PUNCTUATION GERSHAYIM}": '"',
    "\N{LEFT SINGLE QUOTATION MARK}": "'", "\N{RIGHT SINGLE QUOTATION MARK}": "'",
    "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}": "'", "\N{LEFT DOUBLE QUOTATION MARK}": '"',
    "\N{RIGHT DOUBLE QUOTATION MARK}": '"', "\N{DOUBLE LOW-9 QUOTATION MARK}": '"', "\N{ACUTE ACCENT}": "'",
    "`": "'", "\N{ARABIC TATWEEL}": None})
_SPACE_MAP = str.maketrans({ch: " " for ch in (
    "-.,;:!?\N{HEBREW PUNCTUATION MAQAF}\N{HEBREW PUNCTUATION PASEQ}\N{HEBREW PUNCTUATION SOF PASUQ}"
    "\N{ARABIC COMMA}\N{ARABIC SEMICOLON}\N{ARABIC QUESTION MARK}\N{ARABIC FULL STOP}")})
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
NORMALISATION = "nfkd-drop-marks, ascii-quotes, punctuation-to-space, finals-folded, casefold, collapse-ws"


# ----------------------------------------------------------------------------- reads


@dataclass(frozen=True)
class ReadLine:
    """One transcription line of a page read.

    :param text: The VLM reader's text (whitespace collapsed).
    :param agreed: The pipeline's agreement flag: Kraken read the same line (``status == "agreed"``).
    :param agreement: Letters-only similarity of the two readings (None without Kraken text).
    :param reader: The VLM checkpoint that wrote the text.
    """

    text: str
    agreed: bool
    agreement: Optional[float]
    reader: str


@dataclass
class PageReads:
    """The read of one page (image) that a document's description is built from.

    :param image_index: Position in the target document's image list (the image map's, else the record's own).
    :param image_url: Exact image URL the readers saw.
    :param image_sha256: Hash of the image bytes (page identity across index versions), when recorded.
    :param record_doc_id: ``doc_id`` the read was recorded under.
    :param record_image_index: ``image_index`` the read was recorded under.
    :param source_index: Index the record was stamped against.
    :param source_file: File name of the reads JSONL.
    :param vlm_model: VLM checkpoint.
    :param vlm_revision: Its revision.
    :param htr_model: Kraken cache key (``htr_cache_key``), else the HTR model name.
    :param rule_version: Line rule the two readings were matched under.
    :param decoded_at: When the read was made.
    :param parsed: Whether the VLM reply parsed as a JSON array.
    :param flags: Failure / health markers (``unparsed``, ``no_lines``, ``health:<label>``).
    :param health: A ``read_health`` block passed through from the record, when present.
    :param lines: Non-empty lines in reading order.
    :param alternatives: Other reads of the same page that were not used.
    """

    image_index: int
    image_url: str
    image_sha256: Optional[str]
    record_doc_id: str
    record_image_index: int
    source_index: Optional[str]
    source_file: str
    vlm_model: str
    vlm_revision: Optional[str]
    htr_model: str
    rule_version: Optional[str]
    decoded_at: Optional[str]
    parsed: bool
    flags: List[str]
    health: Optional[Dict[str, Any]]
    lines: List[ReadLine]
    alternatives: int = 0

    @property
    def n_agreed(self) -> int:
        """Agreed lines of the page.

        :return: Count.
        :rtype: int
        """
        return sum(line.agreed for line in self.lines)

    def provenance(self) -> Dict[str, Any]:
        """JSON-ready provenance of the page (no line texts).

        :return: The page's fields plus ``n_lines`` and ``n_agreed``.
        :rtype: Dict[str, Any]
        """
        out = {k: v for k, v in asdict(self).items() if k not in ("lines", "health")}
        out.update(n_lines=len(self.lines), n_agreed=self.n_agreed)
        if self.health is not None:
            out["health"] = self.health
        return out


@dataclass
class DocumentReads:
    """The page reads one description is generated from.

    :param doc_id: Target document id (the v9 ``_id`` when an image map is used).
    :param pages: Used pages, first ``max_pages`` by image position.
    :param n_pages_read: Pages with a read before the ``max_pages`` cut.
    """

    doc_id: str
    pages: List[PageReads]
    n_pages_read: int

    @property
    def lines(self) -> List[ReadLine]:
        """All lines of the used pages, in page order.

        :return: Lines.
        :rtype: List[ReadLine]
        """
        return [line for page in self.pages for line in page.lines]

    @property
    def n_lines(self) -> int:
        """Lines of the used pages.

        :return: Count.
        :rtype: int
        """
        return len(self.lines)

    @property
    def n_agreed(self) -> int:
        """Agreed lines of the used pages.

        :return: Count.
        :rtype: int
        """
        return sum(page.n_agreed for page in self.pages)

    @property
    def agreed_lines_pct(self) -> float:
        """Share of agreed lines in percent (0 without lines).

        :return: Percentage, one decimal.
        :rtype: float
        """
        return round(100.0 * self.n_agreed / self.n_lines, 1) if self.n_lines else 0.0

    @property
    def readers(self) -> Dict[str, List[str]]:
        """VLM checkpoints and Kraken keys behind the used pages.

        :return: ``{"vlm": [...], "htr": [...]}``.
        :rtype: Dict[str, List[str]]
        """
        return {"vlm": sorted({p.vlm_model for p in self.pages}), "htr": sorted({p.htr_model for p in self.pages})}

    @property
    def source_files(self) -> List[str]:
        """Reads files behind the used pages.

        :return: File names.
        :rtype: List[str]
        """
        return sorted({p.source_file for p in self.pages})

    def transcription_text(self) -> str:
        """All line texts of the used pages joined by newlines.

        :return: Text.
        :rtype: str
        """
        return "\n".join(line.text for line in self.lines)


def complete_lines(path: Path) -> Iterator[str]:
    """Non-blank, newline-terminated lines of a JSONL file (a record still being appended is skipped).

    :param path: File (may be appended to by a running pipeline).
    :type path: Path
    :return: Lines without their newline.
    :rtype: Iterator[str]
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.endswith("\n") and line.strip():
                yield line.rstrip("\n")


def record_targets(record_doc_id: str, image_url: str, image_index: int, doc_ids: Optional[Set[str]],
                   image_map: Optional[Mapping[str, Sequence[Tuple[str, int]]]]) -> List[Tuple[str, int]]:
    """Target ``(doc_id, image_index)`` pairs a read belongs to.

    :param record_doc_id: ``doc_id`` the read was recorded under.
    :param image_url: Its image URL.
    :param image_index: ``image_index`` it was recorded under.
    :param doc_ids: Wanted documents (None = all).
    :param image_map: URL to target ``(doc_id, image_index)`` pairs; None = the record's own pair.
    :return: Pairs, filtered by ``doc_ids``.
    :rtype: List[Tuple[str, int]]
    """
    targets = list(image_map.get(image_url, ())) if image_map is not None else [(record_doc_id, image_index)]
    return [(d, int(i)) for d, i in targets if doc_ids is None or d in doc_ids]


def record_head(line: str) -> Optional[Tuple[str, str]]:
    """``(doc_id, image_url)`` of a record line read from its prefix, without parsing the whole record.

    :param line: One line of a reads JSONL in the pipeline's key order.
    :return: The pair, or None when the line does not start in that order.
    :rtype: Optional[Tuple[str, str]]
    """
    m = _RECORD_HEAD.match(line)
    if m is None:
        return None
    return json.loads(f'"{m.group(1)}"'), json.loads(f'"{m.group(2)}"')


def page_from_record(record: Mapping[str, Any], source_file: str, image_index: int) -> PageReads:
    """Build a :class:`PageReads` from one ``two_reader_lines`` record.

    :param record: The record.
    :param source_file: File name it came from.
    :param image_index: Target position of the page.
    :return: The page read.
    :rtype: PageReads
    """
    ai = record.get("ai_read") or {}
    reader = str(ai.get("vlm_model") or "unknown")
    lines = []
    for ln in ai.get("lines") or []:
        text = " ".join(str(ln.get("text") or "").split())
        if not text:
            continue
        if "status" in ln:
            agreed = ln["status"] == "agreed"
        else:
            agreed = line_status(ln.get("agreement"), ln.get("htr_text")) == "agreed"
        lines.append(ReadLine(text=text, agreed=agreed, agreement=ln.get("agreement"), reader=reader))
    health = record.get("read_health") or ai.get("read_health")
    flags = []
    if not ai.get("parsed", True):
        flags.append("unparsed")
    if not lines:
        flags.append("no_lines")
    if isinstance(health, Mapping) and health.get("label") not in (None, "ok"):
        flags.append(f"health:{health['label']}")
    return PageReads(
        image_index=image_index, image_url=str(record.get("image_url") or ""),
        image_sha256=record.get("image_sha256"), record_doc_id=str(record.get("doc_id")),
        record_image_index=int(record.get("image_index", -1)), source_index=record.get("source_index"),
        source_file=source_file, vlm_model=reader, vlm_revision=ai.get("vlm_revision"),
        htr_model=str(ai.get("htr_cache_key") or ai.get("htr_model") or "unknown"),
        rule_version=ai.get("rule_version"), decoded_at=ai.get("decoded_at"), parsed=bool(ai.get("parsed", True)),
        flags=flags, health=dict(health) if isinstance(health, Mapping) else None, lines=lines)


def page_rank(page: PageReads, prefer: Sequence[str] = ()) -> Tuple[int, bool, int, float, str]:
    """Sort key of the reads of one page: the highest one is used.

    :param page: One candidate read.
    :param prefer: VLM checkpoints in order of preference (an explicit choice outranks the data).
    :return: ``(preference, parsed, agreed lines, agreed share, decoded_at)``.
    :rtype: Tuple[int, bool, int, float, str]
    """
    preference = len(prefer) - list(prefer).index(page.vlm_model) if page.vlm_model in prefer else 0
    share = page.n_agreed / len(page.lines) if page.lines else 0.0
    return preference, page.parsed, page.n_agreed, share, page.decoded_at or ""


def load_document_reads(paths: Iterable[Any], doc_ids: Optional[Iterable[str]] = None,
                        max_pages: int = DEFAULT_MAX_PAGES,
                        image_map: Optional[Mapping[str, Sequence[Tuple[str, int]]]] = None,
                        prefer: Sequence[str] = ()) -> Dict[str, DocumentReads]:
    """Group per-page consensus records into documents: one read per page, the first ``max_pages`` pages.

    Pages are identified by image bytes (``image_sha256``, else the URL), so a renamed image read under two index
    versions counts once; when a page has several reads :func:`page_rank` picks one. Pages are ordered by their
    target image position.

    :param paths: Reads JSONL files (``two_reader_lines`` output; files still being appended are fine).
    :param doc_ids: Wanted target documents (None = every document in the files).
    :param max_pages: Pages kept per document.
    :param image_map: Exact image URL to target ``(doc_id, image_index)`` pairs (see :func:`load_image_map`); reads
        of URLs outside the map are ignored. None keys reads by their own ``doc_id`` / ``image_index``.
    :param prefer: VLM checkpoints to prefer, most preferred first.
    :return: ``{doc_id: DocumentReads}`` for every wanted document with at least one read.
    :rtype: Dict[str, DocumentReads]
    """
    wanted = set(doc_ids) if doc_ids is not None else None
    groups: Dict[Tuple[str, str], List[PageReads]] = {}
    for path in map(Path, paths):
        for line in complete_lines(path):
            head = record_head(line)
            if head is not None and not record_targets(head[0], head[1], 0, wanted, image_map):
                continue
            record = json.loads(line)
            key = record.get("image_sha256") or record.get("image_url")
            for doc_id, index in record_targets(record["doc_id"], record.get("image_url", ""),
                                                int(record.get("image_index", -1)), wanted, image_map):
                groups.setdefault((doc_id, key), []).append(page_from_record(record, path.name, index))
    by_doc: Dict[str, List[PageReads]] = {}
    for (doc_id, _), candidates in groups.items():
        best = max(candidates, key=lambda p: page_rank(p, prefer))
        best.image_index = min(p.image_index for p in candidates)
        best.alternatives = len(candidates) - 1
        by_doc.setdefault(doc_id, []).append(best)
    out = {}
    for doc_id, pages in by_doc.items():
        pages.sort(key=lambda p: (p.image_index, p.image_url))
        out[doc_id] = DocumentReads(doc_id=doc_id, pages=pages[:max_pages], n_pages_read=len(pages))
    return out


def load_image_map(pool_path: Any) -> Dict[str, List[Tuple[str, int]]]:
    """Image URL to target ``(doc_id, image_index)`` pairs from a job / pool JSONL.

    :param pool_path: JSONL with ``doc_id``, ``image_index``, ``image_url`` per row (the vector-less pool file).
    :return: The map (an image listed by several documents maps to each of them).
    :rtype: Dict[str, List[Tuple[str, int]]]
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for line in complete_lines(Path(pool_path)):
        row = json.loads(line)
        if row.get("image_url") and "image_index" in row:
            out.setdefault(row["image_url"], []).append((row["doc_id"], int(row["image_index"])))
    return out


def pool_doc_ids(pool_path: Any, read_class: Optional[str] = None) -> Set[str]:
    """Documents of a pool file, optionally only those whose every image row has one ``reread_class``.

    :param pool_path: Pool JSONL (``doc_id``, ``reread_class`` per image row).
    :param read_class: E.g. ``already_read``; None keeps every document.
    :return: Document ids.
    :rtype: Set[str]
    """
    classes: Dict[str, Set[str]] = {}
    for line in complete_lines(Path(pool_path)):
        row = json.loads(line)
        classes.setdefault(row["doc_id"], set()).add(row.get("reread_class"))
    return {d for d, seen in classes.items() if read_class is None or seen == {read_class}}


def read_doc_ids(path: Any) -> Set[str]:
    """Document ids from a text file (one per line) or a JSONL with a ``doc_id`` key.

    :param path: File.
    :return: Ids.
    :rtype: Set[str]
    """
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.add(json.loads(line)["doc_id"] if line.startswith("{") else line)
    return out


# ----------------------------------------------------------------------------- catalogue


def _clean_value(value: Any) -> str:
    """One catalogue value as prompt text: lists joined, whitespace collapsed, links and placeholders dropped.

    :param value: Raw value (string, number, list or None).
    :return: Clean text, or ``""`` when nothing usable is left.
    :rtype: str
    """
    if value is None or isinstance(value, (dict, bool)):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [_clean_value(v) for v in value]
        return "; ".join(dict.fromkeys(p for p in parts if p))
    text = " ".join(str(value).split())
    if not text or _URL.search(text) or _PLACEHOLDER.match(text):
        return ""
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS].rstrip() + " ..."


def _merged_fields(record: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """Whitelisted catalogue fields of a ``merged_shelfmarks.jsonl`` record (PGP, KTIV and Bodleian blocks only).

    :param record: Merged record.
    :return: ``(label, raw value)`` pairs in prompt order.
    :rtype: List[Tuple[str, Any]]
    """
    sources = record.get("sources") or {}
    pgp_docs = (sources.get("pgp") or {}).get("documents") or []
    ktiv = sources.get("ktiv") or {}
    basic = ktiv.get("basic_catalog") if isinstance(ktiv.get("basic_catalog"), Mapping) else {}
    full = ktiv.get("full_catalog") if isinstance(ktiv.get("full_catalog"), Mapping) else {}
    tei = (sources.get("bodleian") or {}).get("tei") or {}
    tei = tei if isinstance(tei, Mapping) else {}
    items = tei.get("items") if isinstance(tei.get("items"), list) else []
    return [
        ("Shelf mark", record.get("shelfmark_display") or record.get("canonical_id")),
        ("Institution", record.get("institution")),
        ("Collection", [record.get("collection"), record.get("subcollection")]),
        ("Existing catalogue label", record.get("description")),
        ("Catalogue date", record.get("date")),
        ("PGP document type", [d.get("type") for d in pgp_docs]),
        ("PGP languages", [d.get(k) for d in pgp_docs for k in ("languages_primary", "languages_secondary")]),
        ("PGP description", [d.get("description") for d in pgp_docs]),
        ("PGP date", [d.get("doc_date_standard") or d.get("inferred_date_display") for d in pgp_docs]),
        ("KTIV title", basic.get("title")),
        ("KTIV alternate titles", _ktiv_alt_titles(ktiv)),
        ("KTIV scholarly identifications", _ktiv_scholarly_frames(ktiv)),
        ("KTIV language", basic.get("language")),
        ("KTIV script", basic.get("script")),
        ("KTIV date", basic.get("date")),
        ("KTIV subjects", full.get("subjects")),
        ("KTIV notes", full.get("notes")),
        ("Bodleian titles", tei.get("titles")),
        ("Bodleian languages", tei.get("languages")),
        ("Bodleian notes", [n for it in items if isinstance(it, Mapping) for n in (it.get("notes") or [])]),
        ("Bodleian physical description", tei.get("phys_desc")),
    ]


def _index_fields(source: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """Whitelisted catalogue fields of an index document ``_source``.

    :param source: Index document.
    :return: ``(label, raw value)`` pairs in prompt order.
    :rtype: List[Tuple[str, Any]]
    """
    return [
        ("Shelf mark", source.get("shelf_mark")),
        ("Institution", source.get("institution")),
        ("Collection", [source.get("collection"), source.get("sub_collection")]),
        ("Existing catalogue label", source.get("description")),
        ("Alternate titles", source.get("alt_titles")),
        ("Catalogue date", source.get("date_display")),
        ("Language", [source.get("main_language")] + list(source.get("other_languages") or [])),
        ("Document type", source.get("document_type")),
    ]


def _catalogue_source(record: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Unwrap an index hit (``_source``) or return the record itself.

    :param record: Merged record, index hit, index ``_source``, or None.
    :return: The record to read fields from (empty for None).
    :rtype: Mapping[str, Any]
    """
    if not record:
        return {}
    return record.get("_source") or record


def catalogue_metadata(record: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Known catalogue metadata of one document as prompt lines (label to value).

    Accepts a ``merged_shelfmarks.jsonl`` record, an index hit / ``_source``, or a mapping that is already
    label-to-value (returned cleaned). Only whitelisted fields are read; values holding a link are dropped.

    :param record: The catalogue record, or None.
    :return: Ordered ``{label: value}``; empty when nothing is known.
    :rtype: Dict[str, str]
    """
    source = _catalogue_source(record)
    if not source:
        return {}
    if "sources" in source or "canonical_id" in source:
        fields = _merged_fields(source)
    elif "doc_id" in source or "shelf_mark" in source:
        fields = _index_fields(source)
    else:
        fields = list(source.items())
    out: Dict[str, str] = {}
    for label, value in fields:
        text = _clean_value(value)
        if text:
            out[label] = text
    return out


def catalogue_date(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The catalogue's own date of a document (the merged scalar date, or the index ``date_display``).

    :param record: Merged record or index hit / ``_source`` (None allowed).
    :return: Clean date text, or None.
    :rtype: Optional[str]
    """
    source = _catalogue_source(record)
    raw = source.get("date") if ("sources" in source or "canonical_id" in source) else source.get("date_display")
    return _clean_value(raw) or None


def catalogue_id(record: Mapping[str, Any]) -> Optional[str]:
    """Document id of a catalogue line (merged ``canonical_id``, index ``_id`` / ``doc_id``).

    :param record: Parsed catalogue line.
    :return: The id, or None.
    :rtype: Optional[str]
    """
    source = record.get("_source") or {}
    return record.get("canonical_id") or record.get("_id") or record.get("doc_id") or source.get("doc_id")


def load_catalogue(path: Any, doc_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Catalogue records of the wanted documents from a merged JSONL or an index export (streamed).

    :param path: ``merged_shelfmarks.jsonl``, or a JSONL of index hits / ``_source`` documents.
    :param doc_ids: Wanted ids.
    :return: ``{doc_id: record}``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    out: Dict[str, Dict[str, Any]] = {}
    for line in complete_lines(Path(path)):
        m = _ID_KEY.search(line)
        if m is None or json.loads(f'"{m.group(1)}"') not in doc_ids:
            continue
        record = json.loads(line)
        doc_id = catalogue_id(record)
        if doc_id in doc_ids:
            out[doc_id] = record
    return out


# ----------------------------------------------------------------------------- prompt


def computed_script(text: str) -> str:
    """Script of a transcription counted from its letters.

    :param text: Transcription.
    :return: One of :data:`SCRIPTS` (``Unclear`` without letters; ``Hebrew and Arabic`` when each holds >= 10 %).
    :rtype: str
    """
    counts = {"Hebrew": len(_HEB_LETTER.findall(text or "")), "Arabic": len(_ARA_LETTER.findall(text or "")),
              "Latin": len(_LAT_LETTER.findall(text or ""))}
    total = sum(counts.values())
    if not total:
        return "Unclear"
    if counts["Hebrew"] >= 0.1 * total and counts["Arabic"] >= 0.1 * total:
        return "Hebrew and Arabic"
    name = max(counts, key=counts.get)
    return name if counts[name] >= 0.5 * total else "Other"


def page_budgets(sizes: Sequence[int], budget: int) -> List[int]:
    """Share a character budget among pages, short pages whole first (water-filling).

    :param sizes: Characters per page.
    :param budget: Total characters.
    :return: Characters allowed per page.
    :rtype: List[int]
    """
    if sum(sizes) <= budget:
        return list(sizes)
    alloc = [0] * len(sizes)
    remaining = budget
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    for k, i in enumerate(order):
        alloc[i] = min(sizes[i], remaining // (len(order) - k))
        remaining -= alloc[i]
    return alloc


def select_prompt_lines(doc: DocumentReads, max_chars: int = DEFAULT_MAX_CHARS
                        ) -> List[Tuple[PageReads, List[ReadLine]]]:
    """The lines shown to the model: whole lines in reading order, each page cut at its share of the budget.

    :param doc: Document reads.
    :param max_chars: Transcription characters allowed in the prompt.
    :return: ``(page, shown lines)`` per page.
    :rtype: List[Tuple[PageReads, List[ReadLine]]]
    """
    budgets = page_budgets([sum(len(line.text) for line in page.lines) for page in doc.pages], max_chars)
    selected = []
    for page, budget in zip(doc.pages, budgets):
        shown, used = [], 0
        for line in page.lines:
            if used + len(line.text) > budget:
                break
            shown.append(line)
            used += len(line.text)
        selected.append((page, shown))
    return selected


def prompt_transcription(doc: DocumentReads, max_chars: int = DEFAULT_MAX_CHARS) -> Tuple[str, str]:
    """Texts the grounding check compares against: every shown line, and the shown agreed lines.

    :param doc: Document reads.
    :param max_chars: The prompt's character budget.
    :return: ``(all shown lines, agreed shown lines)`` joined by newlines.
    :rtype: Tuple[str, str]
    """
    shown = [line for _, lines in select_prompt_lines(doc, max_chars) for line in lines]
    return "\n".join(line.text for line in shown), "\n".join(line.text for line in shown if line.agreed)


def build_prompt(doc: DocumentReads, catalogue: Optional[Mapping[str, Any]],
                 max_chars: int = DEFAULT_MAX_CHARS) -> List[Dict[str, str]]:
    """Chat messages for one document: cataloguer instructions, then metadata, transcription and output rules.

    :param doc: Document reads.
    :param catalogue: Its catalogue record (merged, index, or label-to-value; None when unknown); never fetched here.
    :param max_chars: Transcription characters allowed (see :func:`select_prompt_lines`).
    :return: ``[{"role": "system", ...}, {"role": "user", ...}]``.
    :rtype: List[Dict[str, str]]
    """
    meta = catalogue_metadata(catalogue)
    selected = select_prompt_lines(doc, max_chars)
    shown = [line for _, lines in selected for line in lines]
    script = computed_script("\n".join(line.text for line in shown))
    parts = ["Catalogue metadata (existing records; labels may be generic or wrong):"]
    parts += [f"- {label}: {value}" for label, value in meta.items()] or ["- (none known)"]
    parts += ["", f"Script of the transcription (counted from its characters): {script}", ""]
    header = (f"Transcription: {len(selected)} page(s), {len(shown)} lines, "
              f"{sum(line.agreed for line in shown)} of them [agreed].")
    if doc.n_pages_read > len(doc.pages):
        header += f" The document has {doc.n_pages_read} read pages; the first {len(doc.pages)} are shown."
    if len(shown) < doc.n_lines:
        header += (f" Only {len(shown)} of the {doc.n_lines} lines fit the character budget; the ends of long "
                   f"pages are cut.")
    parts.append(header)
    for n, (page, lines) in enumerate(selected, 1):
        note = f" (read flagged: {', '.join(page.flags)})" if page.flags else ""
        parts.append(f"=== Page {n} of {len(selected)} (image {page.image_index}){note} ===")
        parts += [f"[{'agreed' if line.agreed else 'unverified'}] {line.text}" for line in lines] or ["(no lines)"]
    parts += ["", OUTPUT_INSTRUCTIONS]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]


# ----------------------------------------------------------------------------- output parsing and validation


def parse_json_object(text: Optional[str], whole: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse a model reply as one JSON object.

    :param text: Reply text (``<think>`` blocks and code fences are removed first).
    :param whole: Require the whole reply to be the object (no prose around it).
    :return: ``(object, None)`` or ``(None, error)``.
    :rtype: Tuple[Optional[Dict[str, Any]], Optional[str]]
    """
    body = _FENCE.sub("", _THINK.sub("", text or "").strip()).strip()
    if not whole:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return None, "no JSON object in the reply"
        body = body[start:end + 1]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:     # the model's reply is data: report, then repair once
        return None, f"invalid JSON: {exc.msg} at character {exc.pos}"
    if not isinstance(obj, dict):
        return None, "the reply is JSON but not an object"
    return obj, None


def validate_output(obj: Any) -> List[str]:
    """Check a parsed reply against :data:`OUTPUT_SCHEMA`.

    :param obj: Parsed reply.
    :return: Error messages (``path: message``), empty when valid.
    :rtype: List[str]
    """
    errors = sorted(_VALIDATOR.iter_errors(obj), key=lambda e: [str(p) for p in e.absolute_path])
    messages = []
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "(root)"
        message = err.message if len(err.message) <= 200 else err.message[:200] + " ..."
        messages.append(f"{path}: {message}")
    return messages


def parse_and_validate(text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse a reply and validate it.

    :param text: Reply text.
    :return: ``(object, [])`` when valid, else ``(object or None, errors)``.
    :rtype: Tuple[Optional[Dict[str, Any]], List[str]]
    """
    obj, error = parse_json_object(text)
    if obj is None:
        return None, [error]
    return obj, validate_output(obj)


# ----------------------------------------------------------------------------- grounding


def normalise_for_grounding(text: Optional[str], fold_finals: bool = True) -> str:
    """Light normalisation under which an entity span must occur in the transcription (see the module docstring).

    :param text: Span or transcription.
    :param fold_finals: Fold Hebrew final letters (ך ם ן ף ץ) into their medial forms.
    :return: Normalised text.
    :rtype: str
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    out = stripped.translate(_QUOTE_MAP).translate(_SPACE_MAP)
    if fold_finals:
        out = out.translate(_FINALS)
    return " ".join(out.casefold().split())


def summary_spans(summary: str) -> List[str]:
    """Original-script spans of a summary: quoted segments holding Hebrew or Arabic letters, and unquoted runs.

    :param summary: Summary text.
    :return: Spans in order, without duplicates.
    :rtype: List[str]
    """
    spans = []
    for m in _QUOTED.finditer(summary):
        segment = next(g for g in m.groups() if g is not None).strip()
        if _HEB_LETTER.search(segment) or _ARA_LETTER.search(segment):
            spans.append(segment)
    for m in _SCRIPT_RUN.finditer(_QUOTED.sub(" ", summary)):
        run = m.group(0).strip(" '\"")
        if len(_HEB_LETTER.findall(run)) + len(_ARA_LETTER.findall(run)) >= MIN_SPAN_LETTERS:
            spans.append(run)
    return list(dict.fromkeys(spans))


def count_sentences(text: str) -> int:
    """Approximate sentence count (splits after ``. ! ?`` followed by whitespace).

    :param text: Text.
    :return: Number of sentences (0 for blank text).
    :rtype: int
    """
    return len([s for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()])


@dataclass
class GroundingReport:
    """What :func:`grounding_check` kept, dropped, corrected and flagged.

    :param kept: Grounded spans per entity kind.
    :param dropped: Dropped spans as ``{"kind", "span", "reason"}`` (``absent`` / ``too_short``).
    :param support: Kept span to ``agreed`` / ``unverified`` (whether it occurs in an agreed line); empty when no
        agreed text was given.
    :param summary_spans: Original-script spans found in the summary.
    :param summary_spans_absent: Those not in the transcription (flagged, not removed).
    :param summary_sentences: Approximate sentence count of the summary (2-4 asked; not enforced).
    :param script: ``{"model", "computed", "used"}``.
    :param language_corrections: Deterministic language fixes applied.
    :param identified_work: ``none``, ``unverified`` or ``verified``.
    :param confidence_model: Confidence the model gave.
    :param confidence: Confidence after the cap.
    :param confidence_capped: Whether the cap lowered it.
    :param normalisation: Name of the normalisation applied.
    :param grounded_json: The reply after grounding (entities filtered, script / language corrected, work marked).
    """

    kept: Dict[str, List[str]]
    dropped: List[Dict[str, str]]
    support: Dict[str, str]
    summary_spans: List[str]
    summary_spans_absent: List[str]
    summary_sentences: int
    script: Dict[str, Optional[str]]
    language_corrections: List[str]
    identified_work: str
    confidence_model: Optional[float]
    confidence: Optional[float]
    confidence_capped: bool
    normalisation: str
    grounded_json: Dict[str, Any] = field(repr=False)

    @property
    def violations(self) -> int:
        """Dropped spans plus summary spans absent from the transcription.

        :return: Count.
        :rtype: int
        """
        return len(self.dropped) + len(self.summary_spans_absent)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready report (without the grounded reply).

        :return: Report fields.
        :rtype: Dict[str, Any]
        """
        return {k: v for k, v in asdict(self).items() if k != "grounded_json"}

    def summary(self) -> Dict[str, Any]:
        """Compact form for the index (``generated_description_meta.grounding``).

        :return: Counts and flags.
        :rtype: Dict[str, Any]
        """
        return summarise_grounding(self.to_dict())


def summarise_grounding(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact grounding summary from a stored report (a row's ``grounding``).

    :param report: :meth:`GroundingReport.to_dict` output.
    :return: ``entities_kept``, ``entities_dropped``, ``summary_spans_absent``, ``confidence_capped``,
        ``identified_work``, ``normalisation``.
    :rtype: Dict[str, Any]
    """
    return {
        "entities_kept": sum(len(v) for v in (report.get("kept") or {}).values()),
        "entities_dropped": len(report.get("dropped") or []),
        "summary_spans_absent": len(report.get("summary_spans_absent") or []),
        "confidence_capped": bool(report.get("confidence_capped")),
        "identified_work": report.get("identified_work", "none"),
        "normalisation": report.get("normalisation"),
    }


def grounding_check(json_out: Mapping[str, Any], transcription_text: str, agreed_text: Optional[str] = None,
                    fold_finals: bool = True,
                    verify_work: Optional[Callable[[Mapping[str, Any], str], bool]] = None) -> GroundingReport:
    """Keep only entity spans that occur in the transcription; flag misquotes; cap confidence; mark the work.

    * every entity span must occur in the transcription under :func:`normalise_for_grounding` (and hold at least
      :data:`MIN_SPAN_LETTERS` letters): violators are dropped and recorded;
    * original-script spans in the summary (quoted, or unquoted Hebrew / Arabic runs) that are absent are flagged;
    * any violation caps the confidence at :data:`CONFIDENCE_CAP`;
    * the script is set to the one counted from the transcription; Arabic in Hebrew script is Judaeo-Arabic;
    * ``identified_work`` gets ``verified`` / ``verification``: unverified unless ``verify_work`` confirms it.

    :param json_out: Validated model reply.
    :param transcription_text: The transcription the model was shown.
    :param agreed_text: Its agreed lines only (adds ``support`` per kept span), or None.
    :param fold_finals: Fold Hebrew final letters when matching.
    :param verify_work: Symbolic check ``(identified_work, transcription) -> bool``; None = no check.
    :return: The report; ``grounded_json`` is the reply to describe from.
    :rtype: GroundingReport
    """
    grounded: Dict[str, Any] = json.loads(json.dumps(json_out))
    norm_text = normalise_for_grounding(transcription_text, fold_finals)
    norm_agreed = normalise_for_grounding(agreed_text, fold_finals) if agreed_text is not None else None
    kept: Dict[str, List[str]] = {kind: [] for kind in ENTITY_KINDS}
    dropped: List[Dict[str, str]] = []
    support: Dict[str, str] = {}
    entities = grounded.get("entities") or {}
    for kind in ENTITY_KINDS:
        seen: Set[str] = set()
        for raw in entities.get(kind) or []:
            span = " ".join(str(raw).split())
            norm = normalise_for_grounding(span, fold_finals)
            if norm in seen:
                continue
            seen.add(norm)
            if sum(ch.isalpha() for ch in norm) < MIN_SPAN_LETTERS:
                dropped.append({"kind": kind, "span": span, "reason": "too_short"})
            elif norm not in norm_text:
                dropped.append({"kind": kind, "span": span, "reason": "absent"})
            else:
                kept[kind].append(span)
                if norm_agreed is not None:
                    support[span] = "agreed" if norm in norm_agreed else "unverified"
    grounded["entities"] = kept

    spans = summary_spans(str(grounded.get("summary") or ""))
    absent = [s for s in spans if normalise_for_grounding(s, fold_finals) not in norm_text]

    computed = computed_script(transcription_text)
    model_script = grounded.get("script")
    if computed != "Unclear":
        grounded["script"] = computed
    corrections = []
    languages = list(grounded.get("language") or [])
    if grounded.get("script") == "Hebrew" and "Arabic" in languages:
        languages = list(dict.fromkeys("Judaeo-Arabic" if lang == "Arabic" else lang for lang in languages))
        corrections.append("Arabic -> Judaeo-Arabic (Hebrew script)")
    grounded["language"] = languages

    work = grounded.get("identified_work")
    work_status = "none"
    if isinstance(work, Mapping) and work.get("title"):
        verified = bool(verify_work(work, transcription_text)) if verify_work is not None else False
        work_status = "verified" if verified else "unverified"
        grounded["identified_work"] = {**work, "verified": verified,
                                       "verification": "symbolic" if verified else "unverified"}
    else:
        grounded["identified_work"] = None

    raw_conf = grounded.get("confidence")
    confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) else None
    capped = confidence is not None and bool(dropped or absent) and confidence > CONFIDENCE_CAP
    if capped:
        confidence = CONFIDENCE_CAP
    grounded["confidence"] = confidence
    return GroundingReport(
        kept=kept, dropped=dropped, support=support, summary_spans=spans, summary_spans_absent=absent,
        summary_sentences=count_sentences(str(grounded.get("summary") or "")),
        script={"model": model_script, "computed": computed, "used": grounded.get("script")},
        language_corrections=corrections, identified_work=work_status,
        confidence_model=float(raw_conf) if confidence is not None else None, confidence=confidence,
        confidence_capped=capped, normalisation=NORMALISATION + ("" if fold_finals else " (finals kept)"),
        grounded_json=grounded)


# ----------------------------------------------------------------------------- embedding text


def description_text(json_out: Mapping[str, Any], catalogue_date: Optional[str] = None,
                     include_tag: bool = True) -> str:
    """Catalogue-style text to embed, in the line shape of a record's production text (see ``semantic_v3``).

    ``Description: <summary> [Possible identification (unverified): <work>.] [machine-generated ...]``, then
    ``Language:``, ``Document Type:`` and ``Date:`` lines (``standard_date`` = the catalogue's date, else the
    grounded date spans as ``in_text``). An empty summary writes no ``Description:`` line, so only metadata lines
    remain and the text is not eligible for a vector.

    :param json_out: Grounded reply (``GroundingReport.grounded_json``).
    :param catalogue_date: The catalogue's own date of the record, if any.
    :param include_tag: Append :data:`MACHINE_TAG` to the description (always, except for the eligibility test).
    :return: Text (lines joined by newlines).
    :rtype: str
    """
    lines = []
    summary = " ".join(str(json_out.get("summary") or "").split())
    if summary:
        parts = [summary]
        work = json_out.get("identified_work")
        if isinstance(work, Mapping) and work.get("title"):
            label = "Identified work" if work.get("verified") else "Possible identification (unverified)"
            section = f", {work['section']}" if work.get("section") else ""
            parts.append(f"{label}: {work['title']}{section}.")
        if include_tag:
            parts.append(MACHINE_TAG)
        lines.append("Description: " + " ".join(parts))
    languages = [lang for lang in json_out.get("language") or [] if lang != "Unclear"]
    if languages:
        lines.append("Language: " + ", ".join(languages))
    document_type = " ".join(str(json_out.get("document_type") or "").split())
    if document_type:
        lines.append("Document Type: " + document_type)
    dates = list((json_out.get("entities") or {}).get("dates") or [])
    if catalogue_date:
        lines.append(f"Date: standard_date: {catalogue_date}")
    elif dates:
        lines.append("Date: in_text: " + "; ".join(dates[:3]))
    return "\n".join(lines)


def embed_decision(json_out: Mapping[str, Any], catalogue_date: Optional[str] = None) -> Tuple[bool, str]:
    """Whether a description may be embedded, and why not.

    The summary alone must hold :data:`MIN_SUMMARY_CHARS` characters (the semantic rule counts the metadata lines
    too, which would let "Damaged." through), and the semantic rule is applied to the text WITHOUT
    :data:`MACHINE_TAG` (with the tag, a summary that is only a boilerplate label would count as content). A
    confidence under :data:`MIN_EMBED_CONFIDENCE`, or an ``unclear`` genre with no grounded entity, also keeps the
    record vector-less.

    :param json_out: Grounded reply.
    :param catalogue_date: The catalogue's own date, if any.
    :return: ``(eligible, reason)`` with reason ``eligible``, ``short_summary``, ``semantic:<why>``,
        ``low_confidence`` or ``unclear_content``.
    :rtype: Tuple[bool, str]
    """
    if len(" ".join(str(json_out.get("summary") or "").split())) < MIN_SUMMARY_CHARS:
        return False, "short_summary"
    untagged = embed_text_for_version(description_text(json_out, catalogue_date, include_tag=False),
                                      DEFAULT_TEXT_VERSION)
    if not untagged.eligible:
        return False, f"semantic:{semantic_ineligible_reason(untagged.text)}"
    if (json_out.get("confidence") or 0.0) < MIN_EMBED_CONFIDENCE:
        return False, "low_confidence"
    if json_out.get("genre") == "unclear" and not any((json_out.get("entities") or {}).values()):
        return False, "unclear_content"
    return embed_text_for_version(description_text(json_out, catalogue_date), DEFAULT_TEXT_VERSION).eligible, "eligible"


# ----------------------------------------------------------------------------- LM Studio client

Transport = Callable[[str, Dict[str, Any], float], Tuple[int, Any]]


def requests_transport(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, Any]:
    """POST a JSON body; return the status and the decoded body (status 0 on a connection error or timeout).

    :param url: Endpoint.
    :param payload: JSON body.
    :param timeout: Read timeout in seconds (connect timeout: ``lms_transcriber.DEFAULT_CONFIG``).
    :return: ``(status, JSON body or text)``.
    :rtype: Tuple[int, Any]
    """
    try:
        resp = requests.post(url, json=payload, timeout=(DEFAULT_CONFIG.connect_timeout_s, timeout))
    except requests.RequestException as exc:   # server down / timeout: the client retries, then reports
        return 0, f"{type(exc).__name__}: {exc}"
    is_json = resp.headers.get("content-type", "").startswith("application/json")
    return resp.status_code, resp.json() if is_json else resp.text


def text_chat_payload(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
                      extra_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """OpenAI-compatible chat body for a text-only request (``lms_transcriber.chat_payload``'s conventions).

    :param model: LM Studio model key.
    :param messages: Chat messages.
    :param temperature: Sampling temperature.
    :param max_tokens: Output budget (thinking included).
    :param extra_payload: Further fields (``response_format``, ``reasoning_effort``, ...); never ``model`` or
        ``messages``.
    :return: Request body.
    :rtype: Dict[str, Any]
    :raises ValueError: When ``extra_payload`` tries to replace ``model`` or ``messages``.
    """
    payload: Dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                               "messages": messages}
    if extra_payload:
        clash = sorted(set(extra_payload) & {"model", "messages"})
        if clash:
            raise ValueError(f"extra_payload may not replace {clash}")
        payload.update(extra_payload)
    return payload


def json_schema_format(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """``response_format`` asking LM Studio for schema-constrained output.

    :param schema: JSON schema.
    :return: ``{"type": "json_schema", "json_schema": {...}}``.
    :rtype: Dict[str, Any]
    """
    return {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": dict(schema)}}


@dataclass
class ChatResult:
    """Outcome of one chat request (after retries).

    :param content: Reply text, or None on failure.
    :param finish_reason: ``stop`` / ``length`` / ...
    :param schema_used: Whether the accepted request carried the JSON schema.
    :param error: Why there is no content.
    :param usage: Token usage reported by the server.
    """

    content: Optional[str]
    finish_reason: Optional[str]
    schema_used: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LMStudioChatClient:
    """Sequential text-only chat client for LM Studio (no image ever sent).

    :param model: LM Studio model key.
    :param base_url: Server root (``.../v1``).
    :param reasoning: ``off`` or ``on`` (see the module docstring).
    :param temperature: Sampling temperature.
    :param max_tokens: Output budget; None = :data:`MAX_TOKENS` for the reasoning mode.
    :param timeout: Per-request timeout in seconds.
    :param max_retries: Retries of a failed request.
    :param retry_delay: Back-off base in seconds.
    :param transport: ``(url, payload, timeout) -> (status, body)``; tests pass a fake.
    :param sleep: Sleep function (tests pass a no-op).
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_CONFIG.base_url, reasoning: str = "off",
                 temperature: float = TEMPERATURE, max_tokens: Optional[int] = None,
                 timeout: float = REQUEST_TIMEOUT_S, max_retries: int = MAX_RETRIES,
                 retry_delay: float = DEFAULT_CONFIG.retry_delay_s, transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Configure the client (parameters are documented on the class); nothing is sent here."""
        if reasoning not in REASONING_MODES:
            raise ValueError(f"reasoning must be one of {REASONING_MODES}, not {reasoning!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning = reasoning
        self.temperature = temperature
        self.max_tokens = max_tokens or MAX_TOKENS[reasoning]
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transport = transport or requests_transport
        self.sleep = sleep
        self.schema_supported = True

    def payload(self, messages: List[Dict[str, str]], schema: Optional[Mapping[str, Any]],
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Request body for these messages.

        :param messages: Chat messages.
        :param schema: JSON schema for ``response_format``, or None.
        :param max_tokens: Output budget (None = the client's).
        :return: Body.
        :rtype: Dict[str, Any]
        """
        extra = dict(REASONING_FIELDS[self.reasoning])
        if schema is not None:
            extra["response_format"] = json_schema_format(schema)
        return text_chat_payload(self.model, messages, self.temperature, max_tokens or self.max_tokens, extra)

    def chat(self, messages: List[Dict[str, str]], schema: Optional[Mapping[str, Any]] = None) -> ChatResult:
        """Send one chat request, retrying failures; fall back to no schema when the server refuses it.

        A 400 with the schema is retried once without it; when that succeeds the schema is not sent again.
        Another 400 is not retried (the same body would fail again, e.g. a context overflow). A reply cut by the
        token budget (``finish_reason == "length"``: thinking ran long, or the JSON is truncated) is asked again
        with the budget doubled, up to :data:`MAX_TOKENS_CEILING`.

        Every request first yields to a production chat on the shared LM Studio (``lms_yield_gate``, fails open)
        and runs as one of this repo's own calls on the model, so other gated jobs do not take it for prod chat.

        :param messages: Chat messages.
        :param schema: JSON schema for constrained output, or None.
        :return: The result; ``content`` is None on failure (``error`` says why).
        :rtype: ChatResult
        """
        url = f"{self.base_url}/chat/completions"
        use_schema = schema is not None and self.schema_supported
        fell_back = False
        attempt = 0
        max_tokens = self.max_tokens
        while True:
            lms_yield_gate.wait_for_chat_idle(site="describe_from_reads", base_url=self.base_url)
            with lms_yield_gate.own_call(self.model):
                status, body = self.transport(url, self.payload(messages, schema if use_schema else None, max_tokens),
                                              self.timeout)
            if status == 200 and isinstance(body, dict):
                if fell_back:
                    self.schema_supported = False
                result = self._result(body, use_schema)
                if result.finish_reason == "length" and max_tokens < MAX_TOKENS_CEILING:
                    max_tokens = min(2 * max_tokens, MAX_TOKENS_CEILING)
                    print(f"    LM Studio [{self.model}] reply cut by the token budget; asking again with "
                          f"max_tokens={max_tokens}", flush=True)
                    continue
                return result
            text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            error = f"HTTP {status}: {text[:300]}"
            print(f"    LM Studio [{self.model}] {error} (attempt {attempt + 1}/{self.max_retries + 1}"
                  f"{', with schema' if use_schema else ''})", flush=True)
            if status == 400 and use_schema:
                use_schema, fell_back = False, True    # the schema fallback is not counted as a retry
                continue
            if status == 400 or attempt >= self.max_retries:
                return ChatResult(content=None, finish_reason=None, schema_used=use_schema, error=error)
            attempt += 1
            self.sleep(self.retry_delay * attempt)

    @staticmethod
    def _result(body: Mapping[str, Any], schema_used: bool) -> ChatResult:
        """Read the reply out of a 200 response.

        :param body: Response JSON.
        :param schema_used: Whether the request carried the schema.
        :return: The result (``content`` None when the reply is empty).
        :rtype: ChatResult
        """
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = _THINK.sub("", str(message.get("content") or "")).strip()
        if not content and schema_used:
            reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
            if parse_json_object(reasoning, whole=True)[0] is not None:
                content = reasoning     # constrained output that LM Studio put in the reasoning channel
        error = None if content else f"empty content (finish_reason={choice.get('finish_reason')})"
        return ChatResult(content=content or None, finish_reason=choice.get("finish_reason"),
                          schema_used=schema_used, error=error, usage=body.get("usage"))


# ----------------------------------------------------------------------------- one document


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second.

    :return: Timestamp.
    :rtype: str
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def failure_row(doc_id: str, stage: str, error: Any) -> Dict[str, Any]:
    """A ``<out>.failures.jsonl`` row (failed documents are not written to ``--out``, so a re-run retries them).

    :param doc_id: Document.
    :param stage: ``lm_studio``, ``invalid_output`` or ``no_lines``.
    :param error: Message(s).
    :return: Row.
    :rtype: Dict[str, Any]
    """
    return {"doc_id": doc_id, "stage": stage, "error": error, "at": utc_now()}


def output_row(doc: DocumentReads, catalogue: Optional[Mapping[str, Any]], report: GroundingReport, model: str,
               reasoning: str, attempts: int, generated_at: str) -> Dict[str, Any]:
    """The ``--out`` row of one described document (provenance, grounded reply, report, text, eligibility).

    :param doc: Document reads.
    :param catalogue: Its catalogue record (None allowed).
    :param report: Grounding report of the reply.
    :param model: Generator model key.
    :param reasoning: Reasoning mode.
    :param attempts: Requests that produced the reply (2 after a repair).
    :param generated_at: UTC timestamp.
    :return: Row.
    :rtype: Dict[str, Any]
    """
    date = catalogue_date(catalogue)
    eligible, reason = embed_decision(report.grounded_json, date)
    return {
        "doc_id": doc.doc_id, "n_pages": len(doc.pages), "pages_read": doc.n_pages_read, "n_lines": doc.n_lines,
        "n_agreed": doc.n_agreed, "agreed_lines_pct": doc.agreed_lines_pct, "readers": doc.readers,
        "source_files": doc.source_files, "pages": [p.provenance() for p in doc.pages],
        "catalogue": catalogue_metadata(catalogue),
        "model": model, "prompt_version": PROMPT_VERSION, "reasoning": reasoning, "attempts": attempts,
        "generated_at": generated_at, "json": report.grounded_json, "grounding": report.to_dict(),
        "description_text": description_text(report.grounded_json, date), "embed_eligible": eligible,
        "embed_reason": reason, "description_source": DESCRIPTION_SOURCE, "machine_derived": True,
    }


def describe_document(doc: DocumentReads, catalogue: Optional[Mapping[str, Any]], client: Any,
                      max_chars: int = DEFAULT_MAX_CHARS
                      ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Generate, validate (one repair), ground and package one document's description.

    :param doc: Document reads.
    :param catalogue: Its catalogue record (None allowed).
    :param client: An :class:`LMStudioChatClient` (or a fake with ``chat``, ``model``, ``reasoning``).
    :param max_chars: Transcription characters in the prompt.
    :return: ``(row, None)`` or ``(None, failure row)``.
    :rtype: Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
    """
    messages = build_prompt(doc, catalogue, max_chars)
    result = client.chat(messages, OUTPUT_SCHEMA)
    if result.content is None:
        return None, failure_row(doc.doc_id, "lm_studio", result.error)
    obj, errors = parse_and_validate(result.content)
    attempts = 1
    if errors:
        repair = messages + [{"role": "assistant", "content": result.content},
                             {"role": "user", "content": REPAIR_PROMPT.format(
                                 errors="\n".join(f"- {e}" for e in errors[:REPAIR_MAX_ERRORS]))}]
        result = client.chat(repair, OUTPUT_SCHEMA)
        attempts = 2
        if result.content is None:
            return None, failure_row(doc.doc_id, "lm_studio", result.error)
        obj, errors = parse_and_validate(result.content)
        if errors:
            return None, failure_row(doc.doc_id, "invalid_output", errors[:REPAIR_MAX_ERRORS])
    shown, agreed = prompt_transcription(doc, max_chars)
    report = grounding_check(obj, shown, agreed_text=agreed)
    return output_row(doc, catalogue, report, client.model, client.reasoning, attempts, utc_now()), None


def prompt_row(doc: DocumentReads, catalogue: Optional[Mapping[str, Any]], model: str, reasoning: str,
               max_chars: int = DEFAULT_MAX_CHARS) -> Dict[str, Any]:
    """A ``--dry-run`` row: the exact messages one document would be sent, plus its read statistics.

    :param doc: Document reads.
    :param catalogue: Its catalogue record (None allowed).
    :param model: Generator model key.
    :param reasoning: Reasoning mode.
    :param max_chars: Transcription characters in the prompt.
    :return: Row.
    :rtype: Dict[str, Any]
    """
    messages = build_prompt(doc, catalogue, max_chars)
    shown = sum(len(lines) for _, lines in select_prompt_lines(doc, max_chars))
    return {"doc_id": doc.doc_id, "model": model, "reasoning": reasoning, "prompt_version": PROMPT_VERSION,
            "n_pages": len(doc.pages), "pages_read": doc.n_pages_read, "n_lines": doc.n_lines,
            "n_lines_shown": shown, "n_agreed": doc.n_agreed, "agreed_lines_pct": doc.agreed_lines_pct,
            "readers": doc.readers, "source_files": doc.source_files, "catalogue": catalogue_metadata(catalogue),
            "prompt_chars": sum(len(m["content"]) for m in messages), "messages": messages}


# ----------------------------------------------------------------------------- CLI


def done_doc_ids(out: Path) -> Set[str]:
    """Documents already described in ``out`` (the resume key).

    :param out: Output JSONL (may be absent).
    :return: Ids.
    :rtype: Set[str]
    """
    return {json.loads(line)["doc_id"] for line in complete_lines(out)} if out.exists() else set()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON row.

    :param path: File (created with its folder when missing).
    :param row: Row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_doc_ids(doc_ids_path: Optional[str], pool_path: Optional[str],
                    pool_read_class: Optional[str]) -> Optional[Set[str]]:
    """The documents to describe: ``--doc-ids`` and / or the pool's (intersection when both), None = all.

    :param doc_ids_path: ``--doc-ids`` file.
    :param pool_path: ``--pool`` file.
    :param pool_read_class: ``--pool-read-class``.
    :return: Ids, or None for every document with reads.
    :rtype: Optional[Set[str]]
    """
    ids = read_doc_ids(doc_ids_path) if doc_ids_path else None
    if pool_path:
        pool = pool_doc_ids(pool_path, pool_read_class)
        ids = pool if ids is None else ids & pool
    return ids


def median_or_zero(values: Sequence[float]) -> float:
    """Median of a list, 0 when it is empty.

    :param values: Numbers.
    :return: Median.
    :rtype: float
    """
    return float(statistics.median(values)) if values else 0.0


def write_prompts(todo: Sequence[DocumentReads], catalogue: Mapping[str, Mapping[str, Any]], path: Path,
                  model: str, reasoning: str, max_chars: int) -> Dict[str, Any]:
    """``--dry-run``: write every prompt to ``path`` (replaced) and return / print the statistics.

    :param todo: Documents to describe.
    :param catalogue: Catalogue records by id.
    :param path: ``<out>.prompts.jsonl``.
    :param model: Generator model key.
    :param reasoning: Reasoning mode.
    :param max_chars: Transcription characters per prompt.
    :return: ``prompts``, ``skipped_no_lines``, ``with_catalogue``, medians and the truncation count.
    :rtype: Dict[str, Any]
    """
    rows = [prompt_row(doc, catalogue.get(doc.doc_id), model, reasoning, max_chars) for doc in todo if doc.n_lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = {
        "prompts": len(rows), "skipped_no_lines": sum(1 for d in todo if not d.n_lines),
        "with_catalogue": sum(1 for r in rows if r["catalogue"]),
        "median_lines": median_or_zero([r["n_lines"] for r in rows]),
        "median_lines_shown": median_or_zero([r["n_lines_shown"] for r in rows]),
        "median_agreed_pct": median_or_zero([r["agreed_lines_pct"] for r in rows]),
        "agreed_lines_pct_pooled": round(100.0 * sum(r["n_agreed"] for r in rows)
                                         / max(1, sum(r["n_lines"] for r in rows)), 1),
        "docs_without_agreed_line": sum(1 for r in rows if not r["n_agreed"]),
        "truncated": sum(1 for r in rows if r["n_lines_shown"] < r["n_lines"]),
        "median_prompt_chars": median_or_zero([r["prompt_chars"] for r in rows]),
        "max_prompt_chars": max((r["prompt_chars"] for r in rows), default=0), "path": str(path),
    }
    print(f"dry run: {stats['prompts']} prompts -> {path} ({stats['skipped_no_lines']} documents without lines "
          f"skipped); median lines {stats['median_lines']:g} (shown {stats['median_lines_shown']:g}), median agreed "
          f"{stats['median_agreed_pct']:g} % (pooled {stats['agreed_lines_pct_pooled']:g} %), "
          f"{stats['docs_without_agreed_line']} without an agreed line, {stats['truncated']} truncated, "
          f"median prompt {stats['median_prompt_chars']:g} chars (max {stats['max_prompt_chars']})", flush=True)
    return stats


def default_client(args: argparse.Namespace) -> LMStudioChatClient:
    """The LM Studio client a real run uses.

    :param args: Parsed CLI arguments.
    :return: Client.
    :rtype: LMStudioChatClient
    """
    return LMStudioChatClient(model=args.model, base_url=args.base_url, reasoning=args.reasoning,
                              timeout=args.timeout)


def run(args: argparse.Namespace,
        client_factory: Callable[[argparse.Namespace], Any] = default_client) -> Dict[str, Any]:
    """Load reads (and catalogue), then write prompts (``--dry-run``) or describe documents sequentially.

    Resumable: documents already in ``--out`` are skipped; failures go to ``<out>.failures.jsonl`` and are retried
    by the next run. A dry run never creates a client.

    :param args: Parsed CLI arguments.
    :param client_factory: Builds the chat client (tests pass a fake).
    :return: Statistics.
    :rtype: Dict[str, Any]
    """
    doc_ids = resolve_doc_ids(args.doc_ids, args.pool, args.pool_read_class)
    image_map = load_image_map(args.pool) if args.pool else None
    docs = load_document_reads(args.reads, doc_ids, args.max_pages, image_map=image_map,
                               prefer=args.prefer_reader or ())
    out = Path(args.out)
    done = done_doc_ids(out)
    todo = [docs[d] for d in sorted(docs) if d not in done]
    if args.limit:
        todo = todo[:args.limit]
    catalogue = load_catalogue(args.catalogue, {d.doc_id for d in todo}) if args.catalogue else {}
    wanted = "all" if doc_ids is None else str(len(doc_ids))
    print(f"documents wanted {wanted}; with reads {len(docs)}; already in {out.name} "
          f"{len(done & set(docs))}; to do {len(todo)}; catalogue records {len(catalogue)}", flush=True)
    if args.dry_run:
        return write_prompts(todo, catalogue, out.with_suffix(".prompts.jsonl"), args.model, args.reasoning,
                             args.max_chars)
    client = client_factory(args)
    failures = out.with_suffix(".failures.jsonl")
    counts: Counter = Counter()
    for n, doc in enumerate(todo, 1):
        if not doc.n_lines:
            append_jsonl(failures, failure_row(doc.doc_id, "no_lines", "no transcription lines on the used pages"))
            counts["no_lines"] += 1
            continue
        t0 = time.time()
        row, failure = describe_document(doc, catalogue.get(doc.doc_id), client, args.max_chars)
        if row is None:
            append_jsonl(failures, failure)
            counts["failed"] += 1
            print(f"  {n}/{len(todo)} {doc.doc_id}: FAILED {failure['stage']}", flush=True)
            continue
        append_jsonl(out, row)
        counts["described"] += 1
        counts["embed_eligible"] += row["embed_eligible"]
        report = row["grounding"]
        print(f"  {n}/{len(todo)} {doc.doc_id}: {row['json']['genre']} / {row['json']['document_type']}, "
              f"confidence {row['json']['confidence']}, entities kept "
              f"{sum(len(v) for v in report['kept'].values())} dropped {len(report['dropped'])}, "
              f"embed {row['embed_reason']}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {utc_now()}: {dict(counts)} -> {out}", flush=True)
    return dict(counts)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", nargs="+", required=True, help="consensus reads JSONL files (two_reader_lines output)")
    ap.add_argument("--doc-ids", default=None, help="documents to describe: text (one id per line) or JSONL with doc_id")
    ap.add_argument("--pool", default=None,
                    help="pool / job JSONL (doc_id, image_index, image_url per image): wanted documents AND the map "
                         "that re-keys reads to them by exact image URL")
    ap.add_argument("--pool-read-class", default=None,
                    help="with --pool: only documents whose every image row has this reread_class (e.g. already_read)")
    ap.add_argument("--catalogue", default=None,
                    help="merged_shelfmarks.jsonl or an index export (JSONL of hits / _source); read, never fetched")
    ap.add_argument("--out", required=True, help="output JSONL (appended; documents already in it are skipped)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the prompts to <out stem>.prompts.jsonl and stop; no model is called")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="transcription characters per prompt (default %(default)s)")
    ap.add_argument("--prefer-reader", nargs="*", default=None,
                    help="VLM checkpoints to prefer when a page has several reads (default: the data decides)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_CONFIG.base_url)
    ap.add_argument("--reasoning", choices=REASONING_MODES, default="off",
                    help="off: reasoning_effort=low + enable_thinking=false; on: reasoning_effort=xhigh "
                         "(see the module docstring)")
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pool_read_class and not args.pool:
        parser.error("--pool-read-class needs --pool")
    run(args)


if __name__ == "__main__":
    main()
