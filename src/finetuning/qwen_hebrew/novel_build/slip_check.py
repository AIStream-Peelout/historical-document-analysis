"""Shelf-mark slip check of legacy-store photographs filed under a shelf-mark-named file.

Legacy-store photographs whose file is named by a shelf mark (the first ``_``-separated part of the file name is not
all digits) were misfiled at about 8 % in an earlier pilot, while photographs named by a PGP id were not. For every
such photograph in stage1 without a slip outcome, a VLM on LM Studio reads the shelf-mark label or slip in the
photograph (:data:`SLIP_PROMPT`, through :func:`src.models.ocr.lms_transcriber.transcribe_with_lm_studio_meta`,
which holds the LM Studio lock and the yield gate) and the answer is compared with the shelf mark and the canonical
ids the photograph is filed under, in two tiers. Tier 1 goes through the shelf-mark facade
(:func:`~src.datasets.shelfmarks.loose_key` equality, or the answer being one of the filed mark's
:func:`~src.datasets.shelfmarks.variants`): ``match``. Only when it fails, tier 2 (:func:`compare_fuzzy`) reads the
answer as OCR of a mark: the letters a reader confuses with digits are mapped back inside digit runs, separators are
dropped, and the ordered digit groups (with the series letters before them) are compared with those of every
spelling of the filed mark: ``match_fuzzy`` (the same digit groups, the series agreeing or not read) or
``match_fuzzy_partial`` (the read's groups are the first or the last groups of the mark's, the series agreeing). Any
other difference in the digits, or another series, stays ``mismatch``: a neighbouring fragment (``160.26`` for
``160.27``) is a misfiling to look at, not noise to forgive.

Outcomes: ``match`` | ``match_fuzzy`` | ``match_fuzzy_partial`` | ``mismatch`` | ``no_label`` | ``unreadable`` |
``error`` (the LM Studio call failed; retried on the next run). Results go to ``<work dir>/slip_checks.jsonl`` (one
record per photograph sha256, appended as they come, resumable; a fuzzy record carries the normalised strings in its
``fuzzy`` field so it can be audited); every ``mismatch`` and ``unreadable`` goes to the eye queue
(``slip_eye_queue.jsonl`` and ``slip_eye_queue.md``). Nothing is dropped, moved or relabelled here. Run it alone: one
LM Studio consumer at a time. ``--reclassify`` re-runs the comparison over the stored reads (no VLM call).

``--rank-eye-queue`` (no VLM call either) sorts the eye queue by what each mismatch read names. Every ``mismatch`` read
(normalised and raw, under the institution the photograph is filed in) is resolved through the facade
(:func:`~src.datasets.shelfmarks.resolve`) over ONE :class:`~src.datasets.shelfmarks.ShelfmarkIndex` of the merged
corpus, loaded from the cache ``logs/next_round/unseen_v1/rekey.py`` writes. Priority A (``likely_misfiled``): the read
resolves to existing canonical ids and none is the photograph's own. Priority B (``likely_ocr_noise``): it resolves to
nothing, or to the photograph's own ids together with others. A read that resolves only to the photograph's own ids is a
fuzzy match tier 2 missed: it becomes ``match_fuzzy`` with a note. Priority C: ``unreadable``. ``eye_priority`` and
``resolved_ids`` go into both result files and the eye-queue markdown is grouped A / B / C. It starts from
:func:`reclassify_record`, so it is repeatable; a later ``--reclassify`` drops the ranking (run it again).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

from src.datasets import shelfmarks as shelf
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl
from src.models.ocr import lms_transcriber

SLIP_PROMPT = ("Read the shelf-mark label or slip visible in this photograph, if any. Answer with the shelf mark "
               "exactly as written, or NONE.")
PROMPT_NAME = "slip_label_v1"
PROMPT_SHA1 = hashlib.sha1(SLIP_PROMPT.encode("utf-8")).hexdigest()
#: Image origins of the legacy store.
LEGACY_STORE_ORIGINS = ("fjp_local", "fjp_bucket")
OUTCOMES = ("match", "match_fuzzy", "match_fuzzy_partial", "mismatch", "no_label", "unreadable", "error")
#: Outcomes that say the photograph carries the filed mark (strictly, after OCR normalisation, or in part).
MATCH_OUTCOMES = ("match", "match_fuzzy", "match_fuzzy_partial")
#: Outcomes listed in the eye queue (a fuzzy match stays in ``slip_checks.jsonl`` with its normalised strings).
EYE_OUTCOMES = ("mismatch", "unreadable")
#: Version tag of the comparison that decided a record (the strict facade tier, then the OCR-normalised tier).
COMPARISON = "strict+fuzzy_v1"
#: The merged corpus the one shelf-mark index is built from, and the directory its pickle is cached in: the files
#: ``logs/next_round/unseen_v1/rekey.py`` uses, so a single cached index serves both.
MERGED_CORPUS = shelf.MERGED_SHELFMARKS
INDEX_CACHE_DIR = paths.REPO_ROOT / "logs/next_round/unseen_v1/cache"
#: Eye-queue priorities in listing order, with their labels: A the read names another existing mark, B it names no
#: existing mark (or the filed one among others), C the label could not be read.
PRIORITIES: Dict[str, str] = {"A": "likely_misfiled", "B": "likely_ocr_noise", "C": "unreadable"}
#: Record fields the ranking writes; every re-run of the comparison clears them.
RANK_FIELDS = ("eye_priority", "eye_label", "resolved_ids", "resolved_how", "resolved_ambiguous", "mark_distance",
               "rank_note")
#: Start of the ``rank_note`` of a mismatch the index upgraded to ``match_fuzzy``.
UPGRADE_NOTE = "upgraded from mismatch: the read resolves only to the filing's own ids"
#: ``rank_note`` of an A row whose read is a shorter prefix of the filed mark (``160.11`` for ``160.111``): the
#: neighbouring mark is real, but a dropped last digit is the likelier story, so these sort after the other A rows.
TRUNCATION_NOTE = "read_is_a_prefix_of_the_filed_mark"
CHECKS_FILE = "slip_checks.jsonl"
EYE_QUEUE_JSONL = "slip_eye_queue.jsonl"
EYE_QUEUE_MD = "slip_eye_queue.md"
STOP_FILE = "slip_check.stop"
MAX_CONSECUTIVE_ERRORS = 5
DEFAULT_MAX_TOKENS = 256
PROGRESS_EVERY = 25
_ANSWER_PREFIX = re.compile(r"^(?:the\s+)?(?:shelf[\s-]?mark|label|slip|answer)(?:\s+(?:is|reads))?\s*[:：]?\s*", re.I)
_NO_LABEL = re.compile(r"^(?:none|nothing|n/a|no)\b", re.I)
_QUOTES = "`*\"'“”‘’ \t"

#: Letters a reader confuses with a digit, mapped back to it only inside a digit run (:func:`map_digit_confusables`).
DIGIT_CONFUSABLES: Dict[str, str] = {"i": "1", "I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "S": "5"}
#: The confusables that may also start a digit run (``I081``). A leading ``S`` is never a 5: it is the series word
#: (``T-S12.301``, ``AS160``).
LEADING_CONFUSABLES = frozenset("iIl|")
#: A read with no series letters names a mark through its digits alone: at least this many of them (a bare ``2`` names
#: nothing).
MIN_DIGITS_WITHOUT_SERIES = 3
_DIGITS = frozenset("0123456789")
#: A token: a run of letters and digits of any script, or the bar (a ``1`` read as ``|``).
_TOKEN = re.compile(r"(?:[^\W_]|\|)+")
#: The digit runs and the letter runs of a token.
_RUN = re.compile(r"[0-9]+|[^\W\d_]+")
#: The page or piece marker a slip carries beside the mark (``P1``, ``P2``): not part of the shelf mark.
_PAGE_MARKER = re.compile(r"(?<![^\W\d_])[Pp]\s?[0-9]{1,2}(?![^\W_])")
#: The separators between two digit groups (``1-56``, ``160 27``, ``26:53``).
_DIGIT_GAP = re.compile(r"(?<=[0-9])[\s.,:;/\-–—·_]+(?=[0-9])")

#: ``transcribe(model_key, image_path, prompt)`` -> LM Studio completion meta (``text``, ``finish_reason``, ...) or
#: None when the call failed.
Transcribe = Callable[[str, str, str], Optional[Dict[str, Any]]]


@dataclass
class SlipItem:
    """One photograph to check, with every filing it appears under.

    :param sha256: Photograph sha256.
    :param path: Local file.
    :param url: Remote URL, if any.
    :param origin: Image origin.
    :param file_name: The legacy store's file name.
    :param filings: ``{row_id, shelfmark, canonical_ids, keys, spellings}`` per candidate filing the photograph.
    """

    sha256: str
    path: str
    url: Optional[str]
    origin: str
    file_name: str
    filings: List[Dict[str, Any]] = field(default_factory=list)


def lms_transcribe(model_key: str, image_path: str, prompt: str, *,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[Dict[str, Any]]:
    """Send one photograph and a prompt to LM Studio (lock and yield gate included).

    :param model_key: LM Studio model key.
    :param image_path: Image file.
    :param prompt: Prompt.
    :param max_tokens: Output token budget.
    :return: Completion meta (``text``, ``finish_reason``, token counts), or None when the call failed.
    """
    return asyncio.run(lms_transcriber.transcribe_with_lm_studio_meta(model_key, image_path, prompt,
                                                                      max_tokens=max_tokens))


# ----------------------------------------------------------------------------------------------------------- selection

def store_file_name(image: ImageRef) -> str:
    """The legacy store's file name of a photograph: the URL's last part, else the local file name.

    :param image: Stage1 image.
    :return: File name (``""`` when neither is known).
    """
    if image.url:
        return Path(unquote(urlparse(image.url).path)).name
    return Path(image.path).name if image.path else ""


def shelfmark_named(file_name: str) -> bool:
    """True when a file is named by a shelf mark: its first ``_``-separated part is not all digits.

    :param file_name: File name.
    :return: Shelf-mark-named (a PGP-id-named file such as ``12345_1.jpg`` gives False).
    """
    head = Path(file_name).stem.split("_")[0]
    return bool(head) and not head.isdigit()


def institutions(canonical_ids: Iterable[str]) -> List[str]:
    """Institution tokens of canonical ids (used to read a label that names no institution).

    :param canonical_ids: Canonical ids.
    :return: Distinct tokens, in order.
    """
    out: List[str] = []
    for cid in canonical_ids:
        token = shelf.split_id(cid)[0]
        if token and token not in out:
            out.append(token)
    return out


def mark_keys(text: str, institution_tokens: Sequence[str]) -> Set[str]:
    """Facade keys of a mark: ``loose_key`` of every non-lossy variant of every part of a join; a spelling that
    names no institution is keyed under each given institution. A spelling without a digit names no shelf mark
    (with an institution given, the facade would key any words, e.g. ``illegible``).

    :param text: Shelf mark as written (or read).
    :param institution_tokens: Institutions to try for a spelling without one.
    :return: Non-empty keys.
    """
    keys: Set[str] = set()
    for part in shelf.split(text):
        for spelling in shelf.variants(part, lossy=False):
            if not any(ch.isdigit() for ch in spelling):
                continue
            key = shelf.loose_key(spelling)
            if key:
                keys.add(key)
                continue
            keys.update(k for k in (shelf.loose_key(spelling, token) for token in institution_tokens) if k)
    return keys


def _norm(text: str) -> str:
    """Case- and whitespace-folded spelling.

    :param text: Text.
    :return: Folded text.
    """
    return " ".join(text.split()).casefold()


def mark_spellings(shelfmark: Optional[str]) -> List[str]:
    """Folded non-lossy spellings of a filed mark (every part of a join).

    :param shelfmark: Filed shelf mark.
    :return: Sorted distinct case- and whitespace-folded variants (``[]`` for no mark).
    """
    return sorted({_norm(v) for part in shelf.split(shelfmark or "") for v in shelf.variants(part, lossy=False)})


def filing_from_fields(row_id: str, shelfmark: Optional[str], canonical_ids: Sequence[str]) -> Dict[str, Any]:
    """What a photograph filed under a mark and ids should be labelled.

    :param row_id: Candidate row id.
    :param shelfmark: Filed shelf mark.
    :param canonical_ids: Canonical ids of the filing.
    :return: ``{row_id, shelfmark, canonical_ids, keys, spellings}`` (keys: facade keys of the mark and of the ids;
        spellings: folded variants of the mark).
    """
    tokens = institutions(canonical_ids)
    keys = mark_keys(shelfmark, tokens) if shelfmark else set()
    keys.update(k for k in (shelf.loose_key(cid) for cid in canonical_ids) if k)
    return {"row_id": row_id, "shelfmark": shelfmark, "canonical_ids": list(canonical_ids), "keys": sorted(keys),
            "spellings": mark_spellings(shelfmark)}


def filing(cand: Candidate) -> Dict[str, Any]:
    """What a photograph filed under a candidate should be labelled.

    :param cand: Stage1 candidate.
    :return: ``{row_id, shelfmark, canonical_ids, keys, spellings}`` (see :func:`filing_from_fields`).
    """
    return filing_from_fields(cand.row_id, cand.shelfmark, cand.canonical_ids)


def select_images(cands: Sequence[Candidate], done: Set[str],
                  is_file: Callable[[Path], bool] = Path.is_file) -> Tuple[List[SlipItem], Counter]:
    """Legacy-store photographs to check: no slip outcome yet, shelf-mark-named, a local file, not done.

    :param cands: Stage1 candidates.
    :param done: sha256 already checked (any outcome but ``error``).
    :param is_file: File-existence check.
    :return: Items (one per sha256, in first-seen order) and the skip counts by reason.
    """
    items: Dict[str, SlipItem] = {}
    skipped: Counter = Counter()
    for cand in cands:
        for image in cand.images:
            if image.origin not in LEGACY_STORE_ORIGINS:
                continue
            name = store_file_name(image)
            reason = ("has_slip_outcome" if image.slip_outcome else "not_shelfmark_named" if not shelfmark_named(name)
                      else "no_sha256" if not image.sha256 else "already_checked" if image.sha256 in done
                      else "no_local_file" if not image.path or not is_file(Path(image.path)) else None)
            if reason:
                skipped[reason] += 1
                continue
            item = items.setdefault(image.sha256, SlipItem(sha256=image.sha256, path=image.path, url=image.url,
                                                           origin=image.origin, file_name=name))
            if all(f["row_id"] != cand.row_id for f in item.filings):
                item.filings.append(filing(cand))
    return list(items.values()), skipped


# --------------------------------------------------------------------------------------------------------- comparison

def clean_answer(text: str) -> str:
    """The label in a model answer: the first line left after quotes, markup, a ``Shelf mark:`` prefix and full stops
    at either end.

    :param text: Raw answer.
    :return: The label (``""`` when nothing is left).
    """
    for raw in text.splitlines():
        line = raw.strip().strip(_QUOTES)
        if not line or line.startswith("```"):
            continue
        line = _ANSWER_PREFIX.sub("", line).strip().strip(_QUOTES + ".")
        if line:
            return line
    return ""


def is_no_label(label: str) -> bool:
    """True when the answer says there is no label (``NONE``, ``No label visible``, ``N/A``; never with a digit).

    :param label: Cleaned answer.
    :return: No label.
    """
    return not any(ch.isdigit() for ch in label) and bool(_NO_LABEL.match(label))


@dataclass(frozen=True)
class MarkForm:
    """A shelf-mark spelling (or a read) reduced to its series letters and its ordered digit groups.

    :param series: Lower-case letters before the first digit, separators dropped (``tsas`` for ``T-S AS 160.27``,
        ``or``, ``ena``, ``jrl``; ``""`` when the spelling starts with a digit).
    :param parts: From the first digit on, the digit groups in order with the letter groups between them
        (``("8", "j", "5", "1")`` for ``T-S 8J5.1``, ``("1081", "1", "56")`` for ``Or. 1081.1.56``); separators only
        delimit them.
    """

    series: str
    parts: Tuple[str, ...]

    @property
    def squashed(self) -> str:
        """The parts with every separator dropped (``1081156`` for ``Or. 1081.1.56`` and for ``Or. 1081..156``).

        :return: The joined parts.
        """
        return "".join(self.parts)

    @property
    def norm(self) -> str:
        """The normalised string: series letters and squashed parts (``or1081156``).

        :return: Lower-case alphanumerics.
        """
        return self.series + self.squashed

    @property
    def n_digits(self) -> int:
        """How many digits the spelling has.

        :return: Digit count.
        """
        return sum(len(part) for part in self.parts if part[0] in _DIGITS)


def map_digit_confusables(token: str) -> str:
    """Map the letters an OCR reader confuses with digits (:data:`DIGIT_CONFUSABLES`) back to digits, but only inside
    the digit run of one alphanumeric token.

    A confusable counts when it directly follows a digit (``108i`` -> ``1081``, ``160.24S`` -> ``160.245``,
    ``1O5`` -> ``105``) or a confusable already mapped, or when it starts the token and only
    :data:`LEADING_CONFUSABLES` precede the first digit (``I081`` -> ``1081``). Letters glued to the front of a digit
    run stay letters (``AS160``, ``T-S12``: the ``S`` is the series), and a token without a digit is never touched
    (``Misc``, ``IV``).

    :param token: One alphanumeric token (no separators).
    :return: The token with the digit-run confusables mapped.
    """
    first = next((n for n, ch in enumerate(token) if ch in _DIGITS), -1)
    if first < 0:
        return token
    chars = list(token)
    if first and all(ch in LEADING_CONFUSABLES for ch in chars[:first]):
        chars[:first] = ["1"] * first
    for n in range(first + 1, len(chars)):
        if chars[n] in DIGIT_CONFUSABLES and chars[n - 1] in _DIGITS:
            chars[n] = DIGIT_CONFUSABLES[chars[n]]
    return "".join(chars)


def strip_page_marker(text: str) -> str:
    """Drop the page or piece marker a slip carries beside the mark (``T-S Misc 11.17 P2`` -> ``T-S Misc 11.17``).

    :param text: A read label.
    :return: The label without ``P1`` / ``P2`` markers.
    """
    return _PAGE_MARKER.sub(" ", text)


def mark_form(text: str, *, ocr: bool = False) -> MarkForm:
    """Reduce a spelling to its series letters and ordered digit groups.

    With ``ocr`` (a read, not a filed mark) the page marker is dropped first and the digit confusables are mapped
    inside digit runs (:func:`map_digit_confusables`). In both cases the text is lower-cased and every separator
    (spaces, dots, commas, hyphens, slashes, colons, underscores) only delimits the groups.

    :param text: A read label, a filed mark spelling, or the core of a canonical id.
    :param ocr: Treat the text as OCR output.
    :return: The reduced form.
    """
    runs: List[str] = []
    for token in _TOKEN.findall(strip_page_marker(text) if ocr else text):
        runs.extend(_RUN.findall((map_digit_confusables(token) if ocr else token).lower()))
    first = next((n for n, run in enumerate(runs) if run[0] in _DIGITS), len(runs))
    return MarkForm(series="".join(runs[:first]), parts=tuple(runs[first:]))


def series_agreement(read: str, mark: str) -> Optional[str]:
    """How the series letters of a read and of a filed spelling agree.

    :param read: Series letters of the read.
    :param mark: Series letters of the spelling (institution words and class numerals included when the spelling has
        them: ``cambridgecultsas`` or ``parisaiuiib``).
    :return: ``read_has_none`` (the read names no series), ``mark_has_none``, ``same``, ``read_is_tail_of_mark`` (the
        read leaves out an institution word or the front of the series: ``Misc`` for ``T-S Misc``, ``B`` for
        ``II.B.``), ``mark_is_tail_of_read`` (the read adds an institution word); None when they disagree.
    """
    if not read:
        return "read_has_none"
    if not mark:
        return "mark_has_none"
    if read == mark:
        return "same"
    if mark.endswith(read):
        return "read_is_tail_of_mark"
    if read.endswith(mark):
        return "mark_is_tail_of_read"
    return None


def filing_forms(filing_record: Mapping[str, Any]) -> List[Tuple[str, MarkForm]]:
    """Every form a filed mark can be read as: each folded spelling of the mark and the core of each canonical id.

    :param filing_record: A :func:`filing` record (``spellings`` are rebuilt from ``shelfmark`` when missing).
    :return: Distinct ``(spelling, form)`` pairs, spellings first.
    """
    spellings = list(filing_record.get("spellings") or mark_spellings(filing_record.get("shelfmark")))
    cores = [shelf.split_id(cid)[1] for cid in filing_record.get("canonical_ids") or []]
    out: List[Tuple[str, MarkForm]] = []
    seen: Set[MarkForm] = set()
    for text in spellings + cores:
        form = mark_form(text)
        if form.parts and form not in seen:
            seen.add(form)
            out.append((text, form))
    return out


def match_kind(read: MarkForm, mark: MarkForm) -> Optional[Dict[str, Any]]:
    """Whether a read matches one form of a filed mark, and how.

    ``match_fuzzy``: the squashed digit groups are equal (letter groups included) and the series agree or the read
    names none (then with at least :data:`MIN_DIGITS_WITHOUT_SERIES` digits). ``match_fuzzy_partial``: the read names
    a series that agrees and its groups, separators respected, are the first or the last groups of the mark's, fewer
    than all of them. Anything else, a digit that differs included, is no match.

    :param read: The read, from :func:`mark_form` with ``ocr``.
    :param mark: One form of the filed mark.
    :return: ``{"outcome", "series", "partial"}`` (``partial``: ``prefix``, ``suffix`` or None), or None.
    """
    relation = series_agreement(read.series, mark.series)
    if relation is None or not read.parts:
        return None
    if read.squashed == mark.squashed:
        if relation == "read_has_none" and read.n_digits < MIN_DIGITS_WITHOUT_SERIES:
            return None
        return {"outcome": "match_fuzzy", "series": relation, "partial": None}
    count = len(read.parts)
    if relation != "read_has_none" and count < len(mark.parts):
        if mark.parts[:count] == read.parts:
            return {"outcome": "match_fuzzy_partial", "series": relation, "partial": "prefix"}
        if mark.parts[-count:] == read.parts:
            return {"outcome": "match_fuzzy_partial", "series": relation, "partial": "suffix"}
    return None


def compare_fuzzy(label: str, filings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Tier 2: compare a label that failed the strict comparison with every form of every filing, as OCR.

    :param label: Cleaned, non-empty label.
    :param filings: :func:`filing` records.
    :return: ``{outcome, read_norm, read_series, read_groups, marks}``: ``outcome`` is ``match_fuzzy`` when some form
        matches fully, else ``match_fuzzy_partial`` when some form matches in part, else ``mismatch``; ``marks`` lists,
        for the best outcome, the first form of each filing that gave it (``row_id``, ``spelling``, ``norm``,
        ``series``, ``groups``, ``series_agreement``, ``partial``); empty for ``mismatch``.
    """
    read = mark_form(label, ocr=True)
    hits: List[Tuple[str, Dict[str, Any]]] = []
    for record in filings:
        found: Dict[str, Dict[str, Any]] = {}
        for text, form in filing_forms(record):
            kind = match_kind(read, form)
            if kind and kind["outcome"] not in found:
                found[kind["outcome"]] = {"row_id": record["row_id"], "spelling": text, "norm": form.norm,
                                          "series": form.series, "groups": list(form.parts),
                                          "series_agreement": kind["series"], "partial": kind["partial"]}
        hits.extend(found.items())
    outcome = next((o for o in ("match_fuzzy", "match_fuzzy_partial") if any(h[0] == o for h in hits)), "mismatch")
    return {"outcome": outcome, "read_norm": read.norm, "read_series": read.series, "read_groups": list(read.parts),
            "marks": [mark for kind, mark in hits if kind == outcome]}


class Comparison(NamedTuple):
    """How a read label compares with the filings of a photograph.

    :param outcome: One of :data:`OUTCOMES`.
    :param reason: Why.
    :param read_keys: Facade keys of the label.
    :param matched_row_ids: Row ids of the filings the label matched.
    :param fuzzy: Tier 2's audit record (:func:`compare_fuzzy`) when tier 1 failed and tier 2 ran, else None.
    """

    outcome: str
    reason: str
    read_keys: List[str]
    matched_row_ids: List[str]
    fuzzy: Optional[Dict[str, Any]] = None


#: Reason of each outcome of :func:`compare`.
REASONS: Dict[str, str] = {
    "match": "label_matches_filing", "match_fuzzy": "label_matches_filing_after_ocr_normalisation",
    "match_fuzzy_partial": "label_is_part_of_filed_mark", "mismatch": "label_names_another_mark"}


def compare(label: str, filings: Sequence[Mapping[str, Any]]) -> Comparison:
    """Compare a read label with the filings of a photograph, strictly and then, when that fails, as OCR.

    :param label: Cleaned, non-empty label that is not a no-label answer.
    :param filings: :func:`filing` records.
    :return: ``match`` when a key of the label is a key of a filing or the label is one of the filed mark's variants;
        ``unreadable`` when the label gives no key; else tier 2 (:func:`compare_fuzzy`): ``match_fuzzy``,
        ``match_fuzzy_partial`` or ``mismatch``.
    """
    tokens = institutions(cid for f in filings for cid in f["canonical_ids"])
    keys = mark_keys(label, tokens)
    matched = [f["row_id"] for f in filings if keys & set(f["keys"]) or _norm(label) in set(f["spellings"])]
    if matched:
        return Comparison("match", REASONS["match"], sorted(keys), matched)
    if not keys:
        return Comparison("unreadable", "no_shelfmark_parsed", [], [])
    fuzzy = compare_fuzzy(label, filings)
    rows = sorted({mark["row_id"] for mark in fuzzy["marks"]})
    return Comparison(fuzzy["outcome"], REASONS[fuzzy["outcome"]], sorted(keys), rows, fuzzy)


def classify_read(text: str, finish_reason: Optional[str], filings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Decide the outcome of one stored or fresh read.

    :param text: The raw answer.
    :param finish_reason: The completion's finish reason (``length``: the token budget cut the answer).
    :param filings: :func:`filing` records of the photograph.
    :return: The fields the decision sets in a check record: ``label``, ``outcome``, ``reason``, ``read_keys``,
        ``matched_row_ids``, ``fuzzy`` (None unless tier 2 ran).
    """
    label = clean_answer(text)
    decision: Dict[str, Any] = {"label": label, "read_keys": [], "matched_row_ids": [], "fuzzy": None}
    if finish_reason == "length":
        decision.update(outcome="unreadable", reason="answer_cut_by_token_budget")
    elif not label:
        decision.update(outcome="unreadable", reason="empty_answer")
    elif is_no_label(label):
        decision.update(outcome="no_label", reason="model_saw_no_label")
    else:
        result = compare(label, filings)
        decision.update(outcome=result.outcome, reason=result.reason, read_keys=result.read_keys,
                        matched_row_ids=result.matched_row_ids, fuzzy=result.fuzzy)
    return decision


def apply_decision(record: Dict[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    """Put a :func:`classify_read` decision into a check record (the ``fuzzy`` field only when there is one). The
    outcome is decided afresh, so any ranking (:data:`RANK_FIELDS`) the record carried is cleared.

    :param record: Check record, updated in place.
    :param decision: The decision.
    :return: The record.
    """
    record.update({k: v for k, v in decision.items() if k != "fuzzy"}, comparison=COMPARISON)
    for key in RANK_FIELDS:
        record.pop(key, None)
    if decision.get("fuzzy") is None:
        record.pop("fuzzy", None)
    else:
        record["fuzzy"] = decision["fuzzy"]
    return record


def check_image(item: SlipItem, *, model: str, transcribe: Transcribe,
                clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Read the label of one photograph and decide the outcome.

    The LM Studio call is the one place an exception is caught: any failure is recorded as outcome ``error``.

    :param item: Photograph to check.
    :param model: LM Studio model key.
    :param transcribe: The LM Studio call (:data:`Transcribe`).
    :param clock: Seconds clock.
    :return: The check record.
    """
    record: Dict[str, Any] = {
        "sha256": item.sha256, "file_name": item.file_name, "origin": item.origin, "image_path": item.path,
        "url": item.url, "filings": [{k: f[k] for k in ("row_id", "shelfmark", "canonical_ids", "keys")}
                                     for f in item.filings],
        "model": model, "prompt_name": PROMPT_NAME, "prompt_sha1": PROMPT_SHA1, "read": None, "finish_reason": None,
        "label": None, "read_keys": [], "matched_row_ids": [], "outcome": "error", "reason": "", "seconds": None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    started = clock()
    try:
        meta = transcribe(model, item.path, SLIP_PROMPT)
    except Exception as exc:  # noqa: BLE001 - a failed LM Studio call is recorded as an ``error`` outcome
        record.update(reason=f"{type(exc).__name__}: {exc}"[:300], seconds=round(clock() - started, 2))
        return record
    record["seconds"] = round(clock() - started, 2)
    if meta is None:
        record["reason"] = "no_response"
        return record
    text = meta.get("text") or ""
    record.update(read=text, finish_reason=meta.get("finish_reason"))
    return apply_decision(record, classify_read(text, meta.get("finish_reason"), item.filings))


# ----------------------------------------------------------------------------------------------- results, eye queue

def load_checks(path: Path) -> Dict[str, Dict[str, Any]]:
    """Latest check record per photograph.

    :param path: ``slip_checks.jsonl`` (may not exist).
    :return: ``sha256 -> record``.
    """
    return {rec["sha256"]: rec for rec in read_jsonl(path)} if path.is_file() else {}


def eye_entry(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Eye-queue entry of a check record.

    :param record: Check record.
    :return: Outcome, reason, the read and the label, the expected marks and ids, row ids and the image, and the
        ranking fields (:data:`RANK_FIELDS`) when the record has them.
    """
    filings = record.get("filings") or []
    entry = {"sha256": record["sha256"], "outcome": record["outcome"], "reason": record.get("reason"),
             "read": record.get("read"), "label": record.get("label"),
             "expected": [f.get("shelfmark") for f in filings], "canonical_ids": [c for f in filings
                                                                                 for c in f.get("canonical_ids") or []],
             "row_ids": [f.get("row_id") for f in filings], "image_path": record.get("image_path"),
             "file_name": record.get("file_name"), "url": record.get("url")}
    entry.update({key: record[key] for key in RANK_FIELDS if key in record})
    return entry


def queue_order(entry: Mapping[str, Any]) -> Tuple[int, int, int, int, str, str]:
    """Sort key of an eye-queue entry: priority A, B, C (unranked last); within A the nearest marks first and, at one
    distance, the truncated reads (:data:`TRUNCATION_NOTE`) after the rest; then outcome, file name and sha256.

    :param entry: Eye entry.
    :return: The key.
    """
    priority = entry.get("eye_priority")
    distance = entry.get("mark_distance") if priority == "A" else None
    return (list(PRIORITIES).index(priority) if priority in PRIORITIES else len(PRIORITIES),
            distance if distance is not None else 0, int(entry.get("rank_note") == TRUNCATION_NOTE),
            EYE_OUTCOMES.index(entry["outcome"]), entry.get("file_name") or "", entry["sha256"])


def _md(value: Any) -> str:
    """A value as one Markdown table cell.

    :param value: Any value (None -> empty; lists joined with ``; ``).
    :return: Text with ``|`` escaped and whitespace folded.
    """
    if value is None:
        return ""
    text = "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def render_ranked_eye_queue(queue: Sequence[Mapping[str, Any]], checked: int, generated_at: str,
                            upgraded: int = 0) -> str:
    """``slip_eye_queue.md`` of a ranked queue: one table per priority (A, B, C), with counts, each row showing the
    photo path, the expected mark, the read and the resolved ids (entries without a priority come last).

    :param queue: Eye entries in :func:`queue_order`.
    :param checked: Photographs with a check record.
    :param generated_at: Generation time.
    :param upgraded: Mismatches the index upgraded to ``match_fuzzy`` (not listed).
    :return: Markdown.
    """
    groups = {p: [e for e in queue if e.get("eye_priority") == p] for p in PRIORITIES}
    unranked = [e for e in queue if e.get("eye_priority") not in PRIORITIES]
    counts = ", ".join(f"{p} {len(groups[p])}" for p in PRIORITIES)
    lines = ["# Slip-check eye queue", "",
             f"Generated {generated_at} from `{CHECKS_FILE}` ({checked} photographs checked, {len(queue)} here: "
             f"{counts}{f', {len(unranked)} unranked' if unranked else ''}; {upgraded} read(s) the index upgraded to "
             "`match_fuzzy` are not listed). Each legacy-store photograph below is filed under a shelf-mark-named "
             "file. Every read was resolved through the shelf-mark facade over one index of the merged corpus "
             "(normalised and raw read, under the filing's institution). **A**: the read names an existing mark that "
             "is not the filed one (look first, nearest marks first). **B**: it names no existing mark, or the filed "
             "one together with others: most likely OCR noise on the right slip. **C**: the label could not be "
             "read. Nothing was dropped, moved or relabelled: decide by eye."]
    sections = [(f"{p}: {PRIORITIES[p]}", groups[p]) for p in PRIORITIES]
    sections += [("unranked (run `--rank-eye-queue`)", unranked)] if unranked else []
    for title, entries in sections:
        lines += ["", f"## {title} ({len(entries)})", "",
                  "| # | image | expected shelf mark | read | resolved ids | distance | note |",
                  "|---:|---|---|---|---|---:|---|"]
        for n, entry in enumerate(entries, 1):
            expected = list(dict.fromkeys(m for m in entry.get("expected") or [] if m))
            lines.append("| " + " | ".join([str(n), _md(entry.get("image_path")), _md(expected), _md(entry.get("read")),
                                            _md(entry.get("resolved_ids")), _md(entry.get("mark_distance")),
                                            _md(entry.get("rank_note"))]) + " |")
    return "\n".join(lines) + "\n"


def render_eye_queue(queue: Sequence[Mapping[str, Any]], checked: int, generated_at: str, upgraded: int = 0) -> str:
    """``slip_eye_queue.md``: grouped A / B / C when any entry is ranked (:func:`render_ranked_eye_queue`), else one
    flat table.

    :param queue: Eye entries.
    :param checked: Photographs with a check record.
    :param generated_at: Generation time.
    :param upgraded: Mismatches the index upgraded to ``match_fuzzy`` (ranked queues only).
    :return: Markdown.
    """
    if any(entry.get("eye_priority") for entry in queue):
        return render_ranked_eye_queue(queue, checked, generated_at, upgraded)
    lines = ["# Slip-check eye queue", "",
             f"Generated {generated_at} from `{CHECKS_FILE}` ({checked} photographs checked, {len(queue)} here). Each "
             "legacy-store photograph below is filed under a shelf-mark-named file; the label read in the photograph "
             "names another mark (`mismatch`: after OCR normalisation its digit groups still differ from the filed "
             "mark's or it names another series) or could not be read (`unreadable`). Reads that match after OCR "
             "normalisation (`match_fuzzy`, `match_fuzzy_partial`) are not listed: audit them in "
             f"`{CHECKS_FILE}` (`fuzzy` field). Nothing was dropped, moved or relabelled: decide by eye.", "",
             "| # | outcome | reason | read | expected shelf mark | canonical ids | row ids | image |",
             "|---:|---|---|---|---|---|---|---|"]
    for n, entry in enumerate(queue, 1):
        lines.append("| " + " | ".join([str(n)] + [_md(entry.get(k)) for k in (
            "outcome", "reason", "read", "expected", "canonical_ids", "row_ids", "image_path")]) + " |")
    return "\n".join(lines) + "\n"


def write_eye_queue(work_dir: Path, checks: Mapping[str, Mapping[str, Any]],
                    generated_at: Optional[str] = None) -> int:
    """Rewrite the eye queue from every check record (all runs).

    :param work_dir: The build's work dir.
    :param checks: :func:`load_checks` result.
    :param generated_at: Generation time (default: now, UTC).
    :return: Entries written.
    """
    queue = [eye_entry(rec) for rec in checks.values() if rec.get("outcome") in EYE_OUTCOMES]
    queue.sort(key=queue_order)
    write_jsonl(work_dir / EYE_QUEUE_JSONL, queue)
    when = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    upgraded = sum(1 for rec in checks.values() if str(rec.get("rank_note") or "").startswith(UPGRADE_NOTE))
    (work_dir / EYE_QUEUE_MD).write_text(render_eye_queue(queue, len(checks), when, upgraded), encoding="utf-8")
    return len(queue)


def reclassify_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-run the comparison over the read stored in a check record (no VLM call).

    The filings are rebuilt from the stored marks and ids with the current facade; the read, its finish reason and
    every other field stay as they were. A record with no read (outcome ``error``) is returned unchanged.

    :param record: Stored check record.
    :return: A new record whose ``label``, ``outcome``, ``reason``, ``read_keys``, ``matched_row_ids``,
        ``comparison`` and ``fuzzy`` come from the current comparison (:func:`classify_read`).
    """
    updated = dict(record)
    if record.get("read") is None:
        return updated
    filings = [filing_from_fields(f["row_id"], f.get("shelfmark"), f.get("canonical_ids") or [])
               for f in record.get("filings") or []]
    return apply_decision(updated, classify_read(record["read"], record.get("finish_reason"), filings))


def reclassify_checks(work_dir: Path, generated_at: Optional[str] = None) -> Dict[str, Any]:
    """Re-run the comparison over every stored read and rewrite the check file and the eye queue.

    No VLM call is made. The check file is rewritten in place (one record per photograph, first-seen order, written
    beside it and moved over it), so run it while no slip-check run is appending.

    :param work_dir: The build's work dir.
    :param generated_at: Eye-queue generation time (default: now, UTC).
    :return: ``{records, outcomes (count per new outcome), changed (count per old->new outcome), eye_queue}``.
    :raises FileNotFoundError: When the work dir has no check file.
    """
    checks_path = work_dir / CHECKS_FILE
    if not checks_path.is_file():
        raise FileNotFoundError(f"no {CHECKS_FILE} in {work_dir}")
    old = load_checks(checks_path)
    new = {sha: reclassify_record(rec) for sha, rec in old.items()}
    rewrite_checks(checks_path, new.values())
    changed = Counter(f"{old[sha]['outcome']}->{rec['outcome']}" for sha, rec in new.items()
                      if rec["outcome"] != old[sha]["outcome"])
    return {"records": len(new), "outcomes": dict(Counter(rec["outcome"] for rec in new.values())),
            "changed": dict(changed), "eye_queue": write_eye_queue(work_dir, new, generated_at)}


def rewrite_checks(checks_path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Rewrite the check file in place: the records are written beside it and moved over it.

    :param checks_path: ``slip_checks.jsonl``.
    :param records: Every record, in file order.
    """
    staged = checks_path.with_name(checks_path.name + ".tmp")
    write_jsonl(staged, records)
    staged.replace(checks_path)


# ---------------------------------------------------------------------------------------------- eye-queue ranking

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: insertions, deletions and substitutions.

    :param a: One string.
    :param b: The other.
    :return: The distance.
    """
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalise_read_text(label: str) -> str:
    """The read as a spelling the facade can parse: the page marker dropped and the digit confusables mapped as in
    tier 2 (``108i`` is ``1081``), and every separator between two digit groups a dot (``Or. 108i.1-56`` ->
    ``Or. 1081.1.56``: the facade takes ``1-56`` for a numeric range, which resolves only through an ES id of that
    very range). Series words and case are kept.

    :param label: A cleaned read label.
    :return: The normalised spelling, whitespace folded.
    """
    text = _TOKEN.sub(lambda match: map_digit_confusables(match.group()), strip_page_marker(label))
    return " ".join(_DIGIT_GAP.sub(".", text).split())


def mark_distance(label: str, filings: Sequence[Mapping[str, Any]]) -> Optional[int]:
    """How far a read is from the filed mark: the least edit distance between the normalised read (series letters and
    squashed digit groups, tier 2's string) and any form of any filing (``160.26`` against ``160.27``: 1).

    :param label: A cleaned read label.
    :param filings: Stored or fresh filing records.
    :return: The distance, or None when the read has no digit or the filings no form.
    """
    read = mark_form(label, ocr=True)
    forms = [form for record in filings for _, form in filing_forms(record)]
    if not read.parts or not forms:
        return None
    return min(edit_distance(read.norm, form.norm) for form in forms)


def is_truncated_read(label: str, filings: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the normalised read is a shorter prefix of a form of a filed mark (``160.11`` for ``160.111``).

    :param label: A cleaned read label.
    :param filings: Stored or fresh filing records.
    :return: True when some form of some filing starts with the read's string and is longer.
    """
    read = mark_form(label, ocr=True)
    return bool(read.parts) and any(form.norm != read.norm and form.norm.startswith(read.norm)
                                    for record in filings for _, form in filing_forms(record))


def load_shelfmark_index(merged: Path = MERGED_CORPUS, cache_dir: Path = INDEX_CACHE_DIR) -> shelf.ShelfmarkIndex:
    """The one shelf-mark index of the merged corpus, built as ``logs/next_round/unseen_v1/rekey.py`` builds it:
    loaded from the pickle in the same cache directory when its signature (normaliser version, index format, size
    and mtime of the merged file) matches, else built and cached there.

    :param merged: ``merged_shelfmarks.jsonl``.
    :param cache_dir: Directory of the cached index.
    :return: The index.
    """
    return shelf.ShelfmarkIndex(merged, cache_dir=cache_dir)


def resolve_read(label: str, tokens: Sequence[str], index: shelf.ShelfmarkIndex) -> Tuple[List[str], str, bool]:
    """Resolve a read through the facade over one index: the normalised spelling and the raw label, each under every
    given institution.

    :param label: A cleaned read label.
    :param tokens: Institution tokens of the photograph's canonical ids (empty: the facade infers it from the mark).
    :param index: The merged-corpus index.
    :return: ``(ES ids, weakest how, ambiguous)`` over all the tries; ``how`` is ``unmatched`` when nothing resolved.
    """
    ids: Set[str] = set()
    hows: List[str] = []
    ambiguous = False
    for text in dict.fromkeys(t for t in (normalise_read_text(label), label) if t):
        for institution in tokens or [None]:
            result = shelf.resolve(text, index=index, institution=institution)
            if result.cids:
                ids.update(result.cids)
                hows.append(result.how)
                ambiguous = ambiguous or result.ambiguous
    return sorted(ids), (max(hows, key=shelf.HOW_ORDER.index) if hows else "unmatched"), ambiguous


def rank_kind(resolved: Set[str], own: Set[str]) -> Tuple[str, str]:
    """What a read that resolved to ``resolved`` says about a photograph filed under the ids ``own``.

    :param resolved: ES ids the read resolved to.
    :param own: The photograph's own canonical ids.
    :return: ``("A", "")``: the read names existing ids and none is the photograph's own (likely misfiled);
        ``("upgrade", "")``: it names only the photograph's own ids (a fuzzy match tier 2 missed); ``("B", note)``:
        it names nothing, or the photograph's own ids together with others (likely OCR noise).
    """
    if not resolved:
        return "B", ""
    if not resolved & own:
        return "A", ""
    if resolved <= own:
        return "upgrade", ""
    return "B", "resolves_to_own_and_other_ids"


def rank_record(record: Mapping[str, Any], index: shelf.ShelfmarkIndex) -> Dict[str, Any]:
    """Rank one check record for the eye queue.

    ``unreadable`` is priority C. A ``mismatch`` read is resolved (:func:`resolve_read`) and gets priority A or B
    (:func:`rank_kind`) with ``resolved_ids``, ``resolved_how``, ``resolved_ambiguous`` and ``mark_distance``; one that
    resolves only to the photograph's own ids is upgraded to ``match_fuzzy`` with a ``rank_note``. Any other outcome is
    returned without ranking. Earlier ranking fields are dropped first.

    :param record: A check record whose outcome is current (:func:`reclassify_record`).
    :param index: The merged-corpus index.
    :return: A new record.
    """
    ranked = {k: v for k, v in record.items() if k not in RANK_FIELDS}
    if ranked.get("outcome") == "unreadable":
        ranked.update(eye_priority="C", eye_label=PRIORITIES["C"], resolved_ids=[])
        return ranked
    if ranked.get("outcome") != "mismatch":
        return ranked
    filings = ranked.get("filings") or []
    label = ranked.get("label") or ""
    own_ids = [cid for f in filings for cid in f.get("canonical_ids") or []]
    ids, how, ambiguous = resolve_read(label, institutions(own_ids), index)
    kind, note = rank_kind(set(ids), set(own_ids))
    ranked.update(resolved_ids=ids, resolved_how=how, resolved_ambiguous=ambiguous,
                  mark_distance=mark_distance(label, filings))
    if kind == "upgrade":
        rows = sorted(f["row_id"] for f in filings if set(f.get("canonical_ids") or []) & set(ids))
        audit = {**(ranked.get("fuzzy") or {}), "upgraded_by_index": True}
        ranked.update(outcome="match_fuzzy", reason="label_resolves_to_filed_ids_in_index", matched_row_ids=rows,
                      rank_note=f"{UPGRADE_NOTE} ({how})", fuzzy=audit)
        return ranked
    ranked.update(eye_priority=kind, eye_label=PRIORITIES[kind])
    if kind == "A" and is_truncated_read(label, filings):
        note = TRUNCATION_NOTE
    if note:
        ranked["rank_note"] = note
    return ranked


def rank_checks(work_dir: Path, index: shelf.ShelfmarkIndex, generated_at: Optional[str] = None) -> Dict[str, Any]:
    """Rank the eye queue: re-run the comparison over every stored read (:func:`reclassify_record`), rank each record
    (:func:`rank_record`), rewrite the check file and write the eye queue (``slip_eye_queue.jsonl`` and the markdown
    grouped A / B / C). No VLM call; ONE index serves every resolution. Run it while no slip-check run is appending.

    :param work_dir: The build's work dir.
    :param index: The merged-corpus index (:func:`load_shelfmark_index`).
    :param generated_at: Eye-queue generation time (default: now, UTC).
    :return: ``{records, outcomes, changed (old->new outcome counts), priorities (A, B, C counts), upgraded,
        eye_queue}``.
    :raises FileNotFoundError: When the work dir has no check file.
    """
    checks_path = work_dir / CHECKS_FILE
    if not checks_path.is_file():
        raise FileNotFoundError(f"no {CHECKS_FILE} in {work_dir}")
    old = load_checks(checks_path)
    new = {sha: rank_record(reclassify_record(rec), index) for sha, rec in old.items()}
    rewrite_checks(checks_path, new.values())
    priorities = Counter(rec["eye_priority"] for rec in new.values() if rec.get("eye_priority"))
    changed = Counter(f"{old[sha]['outcome']}->{rec['outcome']}" for sha, rec in new.items()
                      if rec["outcome"] != old[sha]["outcome"])
    return {"records": len(new), "outcomes": dict(Counter(rec["outcome"] for rec in new.values())),
            "changed": dict(changed), "priorities": {p: priorities.get(p, 0) for p in PRIORITIES},
            "upgraded": sum(1 for rec in new.values() if str(rec.get("rank_note") or "").startswith(UPGRADE_NOTE)),
            "eye_queue": write_eye_queue(work_dir, new, generated_at)}


def run_slip_checks(work_dir: Path, *, model: str, transcribe: Transcribe = lms_transcribe,
                    limit: Optional[int] = None, stop_file: Optional[Path] = None,
                    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
                    is_file: Callable[[Path], bool] = Path.is_file) -> Dict[str, int]:
    """Check every eligible photograph of stage1 not checked yet; append each record at once; rewrite the eye queue.

    :param work_dir: The build's work dir (reads the stage1 file, writes the check and eye-queue files).
    :param model: LM Studio model key.
    :param transcribe: The LM Studio call (tests stub it).
    :param limit: At most N photographs this run.
    :param stop_file: Stop before the next photograph when this file exists.
    :param max_consecutive_errors: Stop after this many ``error`` outcomes in a row.
    :param is_file: File-existence check.
    :return: Counts: one per outcome checked this run, ``skipped_<reason>``, ``eye_queue``, and ``stopped_by_file``
        or ``stopped_after_errors`` when the run stopped early.
    """
    cands = list(read_jsonl(paths.stage_path(work_dir, "images"), Candidate))
    checks_path = work_dir / CHECKS_FILE
    checks = load_checks(checks_path)
    done = {sha for sha, rec in checks.items() if rec.get("outcome") != "error"}
    items, skipped = select_images(cands, done, is_file)
    counts: Counter = Counter({f"skipped_{reason}": n for reason, n in skipped.items()})
    todo = items if limit is None else items[:limit]
    print(f"slip check: {len(items)} photograph(s) to check, {len(todo)} this run; skipped {dict(skipped)}")
    errors_in_row = 0
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    with checks_path.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(todo, 1):
            if stop_file is not None and stop_file.exists():
                counts["stopped_by_file"] = 1
                break
            record = check_image(item, model=model, transcribe=transcribe)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            checks[item.sha256] = record
            counts[record["outcome"]] += 1
            errors_in_row = errors_in_row + 1 if record["outcome"] == "error" else 0
            if n % PROGRESS_EVERY == 0:
                print(f"  {n}/{len(todo)}: {dict((k, counts[k]) for k in OUTCOMES if counts[k])}")
            if errors_in_row >= max_consecutive_errors:
                counts["stopped_after_errors"] = 1
                print(f"stopping: {errors_in_row} errors in a row (last: {record['reason']})")
                break
    counts["eye_queue"] = write_eye_queue(work_dir, checks)
    return dict(counts)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: run the slip check over stage1 and write the eye queue, or (``--reclassify``) re-run the comparison over
    the stored reads.

    :param argv: Arguments (default: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Read the shelf-mark label in legacy-store photographs named by "
                                                 "shelf mark and compare it with their filing (eye queue; nothing "
                                                 "is moved).")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="The build's work dir.")
    parser.add_argument("--model", default=None,
                        help="LM Studio model key of the VLM that reads the label (required unless --reclassify).")
    parser.add_argument("--reclassify", action="store_true",
                        help=f"Re-run the comparison over the reads already in <work dir>/{CHECKS_FILE} (no VLM call, "
                             "no model needed) and rewrite the outcomes and the eye queue.")
    parser.add_argument("--rank-eye-queue", action="store_true",
                        help="Resolve every mismatch read through the shelf-mark facade over one index of the merged "
                             f"corpus (cached in {INDEX_CACHE_DIR}), rank the eye queue A (likely misfiled) / B "
                             "(likely OCR noise) / C (unreadable), upgrade the reads that resolve only to the filed "
                             "ids to match_fuzzy, and rewrite the check file and the eye queue (no VLM call, no "
                             "model needed; includes the --reclassify step).")
    parser.add_argument("--limit", type=int, default=None, help="At most N photographs this run.")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help=f"Stop before the next photograph when this file exists (default: <work dir>/{STOP_FILE}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Output token budget per read.")
    args = parser.parse_args(argv)
    if args.rank_eye_queue:
        print(f"slip check ranked: {rank_checks(args.work_dir, load_shelfmark_index())}")
        return
    if args.reclassify:
        print(f"slip check reclassified: {reclassify_checks(args.work_dir)}")
        return
    if not args.model:
        parser.error("--model is required unless --reclassify is given")
    stop_file = args.stop_file if args.stop_file is not None else args.work_dir / STOP_FILE

    def transcribe(model_key: str, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """:func:`lms_transcribe` with the CLI's token budget.

        :param model_key: LM Studio model key.
        :param image_path: Image file.
        :param prompt: Prompt.
        :return: Completion meta or None.
        """
        return lms_transcribe(model_key, image_path, prompt, max_tokens=args.max_tokens)

    counts = run_slip_checks(args.work_dir, model=args.model, transcribe=transcribe, limit=args.limit,
                             stop_file=stop_file)
    print(f"slip check done: {counts}")


if __name__ == "__main__":
    main()
