# File name: read_health.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Read-time health check of one grounded VLM page read: did the read fail, and how?

Pure decision logic (no I/O, no model, no ground truth), so the consensus pipeline
(:mod:`src.datasets.consensus.two_reader_lines`) can call it right after a page read and re-read the
page when it fails.  Everything it looks at is what the pipeline holds at that moment: the raw reply
text, the parsed line texts, the Kraken fragment texts of the same image, the token cap of the request
and, when the caller has it, the number of tokens the reply used.

Failure classes (one label per read, first match wins; :attr:`ReadHealth.flags` lists every class that
fired):

``loop``          the reply repeats itself: one 12-letter run covers much of the reply (the offline
                  scorer's ``loop_ratio``), most of the reply's last letters repeat earlier runs (a loop
                  that starts mid-reply and runs to the cap, or whose phrase is too long for
                  ``loop_ratio``), or many reply lines are exact repeats of earlier lines.
``capped``        the reply hit the token cap without looping: it is not a closed JSON array and it used
                  (nearly) every allowed token, so the page's tail is missing.  Without a token count: it stops
                  inside an entry (a reply that forgot only its closing ``]`` ends with ``}``) and is at least
                  ``MIN_CHARS_PER_TOKEN`` characters per allowed token long.
``near_empty``    the reply holds almost no letters although Kraken read text on the image.
``skipped_text``  the reply holds far fewer letters than Kraken read on the same image (whole lines or
                  passages left out).
``parse_loss``    the reply is fine but the pipeline's parser dropped many of its lines (Hebrew
                  abbreviations written with an unescaped ASCII ``"`` break the JSON); the remedy is a
                  tolerant re-parse, not a re-read.

Every threshold is a named module constant with its reason; :class:`Thresholds` bundles them so an
offline calibration can try other values.  Calibrated on held-out pages by
``logs/next_round/failed_reads/validate_read_health.py`` (tuned on one half of the pages, reported on
the other); the numbers are in that folder's ``validation.log``.
"""
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations.arabic_script import repeat_share
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import letters_only
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import loop_ratio

LABEL_OK = "ok"
LABEL_LOOP = "loop"
LABEL_CAPPED = "capped"
LABEL_NEAR_EMPTY = "near_empty"
LABEL_SKIPPED = "skipped_text"
LABEL_PARSE_LOSS = "parse_loss"
FAILURE_LABELS = (LABEL_LOOP, LABEL_CAPPED, LABEL_NEAR_EMPTY, LABEL_SKIPPED, LABEL_PARSE_LOSS)  # precedence order
REREAD_LABELS = (LABEL_LOOP, LABEL_CAPPED, LABEL_NEAR_EMPTY, LABEL_SKIPPED)    # the failures a re-read can fix

# ----------------------------------------------------------------------------- thresholds (one reason each)

LOOP_RATIO = 0.45            # the offline scorer's LOOP_REPEAT_RATIO: one 12-letter run covering >= 45 % of the letters is a loop
TAIL_LETTERS = 600           # a loop starts mid-reply and runs to the cap, so repetition is judged on the reply's last 600 letters
TAIL_REPEAT_SHARE = 0.5      # >= half of the tail's 12-letter runs repeat an earlier run; real text, formulae included, stays far below
DUP_LINES_MIN = 5            # >= 5 later exact repeats of a line ...
DUP_LINE_SHARE = 0.25        # ... making up >= 25 % of the lines is a line loop (short repeated lines are too few letters for the run measures)
CAP_MARGIN_TOKENS = 100      # an unclosed reply within 100 tokens of the cap was cut by the cap (re-tokenised counts differ by a few tokens)
MIN_CHARS_PER_TOKEN = 1.0    # token count unknown: capped replies ran 1.02-2.5 characters per token (a counting loop of digits is the low end)
NEAR_EMPTY_LETTERS = 25      # the offline scorer's ABSTAIN_MIN_CHARS: fewer letters than this is no read at all
MIN_KRAKEN_LETTERS = 100     # near-empty needs an image with text: Kraken read >= 100 letters (an empty reply on a blank side is right)
SKIPPED_RATIO = 0.8          # healthy reads hold >= 0.98x Kraken's letters (5th percentile); below 0.8x the median CER jumps from ~0.2 to >= 0.45 (tuned)
SKIPPED_MIN_KRAKEN_LETTERS = 100   # the ratio is only trusted when Kraken read enough letters to compare against
PARSE_LOSS_MIN_LINES = 2     # >= 2 complete reply entries missing from the parsed lines (one truncated last entry is normal) ...
PARSE_LOSS_SHARE = 0.2       # ... and >= 20 % of the reply's entries: the stored read lost real lines

_LETTER = re.compile(r"[א-תء-ي]")                 # Hebrew and Arabic letters
_ENTRY = re.compile(r'"text"\s*:\s*"(.*?)"\s*,\s*"bbox_2d"\s*:\s*\[', re.S)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")
_RUN = 12                    # run length of the loop measures (the scorer's loop span)


@dataclass(frozen=True)
class Thresholds:
    """Every threshold of the detector (defaults = the module constants).

    :param loop_ratio: Minimum ``loop_ratio`` of the reply's letters for ``loop``.
    :param tail_letters: Letters at the end of the reply judged by ``tail_repeat_share``.
    :param tail_repeat_share: Minimum repeat share of the tail for ``loop``.
    :param dup_lines_min: Minimum later exact line repeats for ``loop`` ...
    :param dup_line_share: ... with at least this share of the lines.
    :param cap_margin_tokens: An unclosed reply within this many tokens of the cap is ``capped``.
    :param min_chars_per_token: Without a token count, an unclosed reply cut inside an entry and at least this
        many characters per allowed token long is ``capped``.
    :param near_empty_letters: Fewer reply letters than this is ``near_empty`` ...
    :param min_kraken_letters: ... when Kraken read at least this many letters.
    :param skipped_ratio: Reply letters below this share of Kraken's letters is ``skipped_text`` ...
    :param skipped_min_kraken_letters: ... when Kraken read at least this many letters.
    :param parse_loss_min_lines: Minimum reply entries missing from the parsed lines for ``parse_loss`` ...
    :param parse_loss_share: ... and minimum share of the reply's entries.
    """

    loop_ratio: float = LOOP_RATIO
    tail_letters: int = TAIL_LETTERS
    tail_repeat_share: float = TAIL_REPEAT_SHARE
    dup_lines_min: int = DUP_LINES_MIN
    dup_line_share: float = DUP_LINE_SHARE
    cap_margin_tokens: int = CAP_MARGIN_TOKENS
    min_chars_per_token: float = MIN_CHARS_PER_TOKEN
    near_empty_letters: int = NEAR_EMPTY_LETTERS
    min_kraken_letters: int = MIN_KRAKEN_LETTERS
    skipped_ratio: float = SKIPPED_RATIO
    skipped_min_kraken_letters: int = SKIPPED_MIN_KRAKEN_LETTERS
    parse_loss_min_lines: int = PARSE_LOSS_MIN_LINES
    parse_loss_share: float = PARSE_LOSS_SHARE


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class ReadFacts:
    """What the detector measures on one read (all of it known at read time).

    :param reply_chars: Characters of the raw reply.
    :param reply_letters: Hebrew and Arabic letters in the raw reply (all of them sit in ``text`` fields, so
        this counts what the model wrote even where the parser lost lines).
    :param reply_tokens: Tokens the reply used, when the caller knows it (else None).
    :param closed: The reply (code fences stripped) ends with ``]``: the model closed the array.
    :param ends_entry: The reply (code fences stripped) ends with ``}``: it stops after a finished entry, not inside one.
    :param n_lines: Parsed lines the pipeline kept.
    :param n_entries: Complete ``{"text": ..., "bbox_2d": [`` entries in the reply, found without JSON parsing.
    :param loop_ratio: Offline scorer's ``loop_ratio`` of the reply's Hebrew-block letters.
    :param tail_repeat_share: ``repeat_share`` of the reply's last ``tail_letters`` letters.
    :param dup_lines: Later exact repeats among the reply's lines (whitespace-normalised).
    :param dup_share: ``dup_lines`` over the number of lines.
    :param kraken_letters: Hebrew and Arabic letters Kraken read on the image.
    :param kraken_lines: Kraken fragments with at least three letters.
    :param letter_ratio: ``reply_letters / kraken_letters`` (None when Kraken read no letters).
    """

    reply_chars: int
    reply_letters: int
    reply_tokens: Optional[int]
    closed: bool
    ends_entry: bool
    n_lines: int
    n_entries: int
    loop_ratio: float
    tail_repeat_share: float
    dup_lines: int
    dup_share: float
    kraken_letters: int
    kraken_lines: int
    letter_ratio: Optional[float]


@dataclass(frozen=True)
class ReadHealth:
    """Verdict on one read.

    :param label: ``ok`` or the first failure class that fired (:data:`FAILURE_LABELS` order).
    :param flags: Every failure class that fired, in precedence order.
    :param facts: The measurements behind the verdict.
    """

    label: str
    flags: Tuple[str, ...]
    facts: ReadFacts

    @property
    def failed(self) -> bool:
        """Whether the read failed in any way.

        :return: True unless the label is ``ok``.
        :rtype: bool
        """
        return self.label != LABEL_OK

    @property
    def reread(self) -> bool:
        """Whether a re-read can fix the failure (a parse loss needs a re-parse instead).

        :return: True for the classes in :data:`REREAD_LABELS`.
        :rtype: bool
        """
        return self.label in REREAD_LABELS

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form.

        :return: ``label``, ``flags`` and the facts.
        :rtype: Dict[str, Any]
        """
        return {"label": self.label, "flags": list(self.flags), **asdict(self.facts)}


# ----------------------------------------------------------------------------- measurements


def letter_count(text: str) -> int:
    """Hebrew and Arabic letters of a text.

    :param text: Any text.
    :type text: str
    :return: Letter count.
    :rtype: int
    """
    return len(_LETTER.findall(text or ""))


def strip_fences(reply: str) -> str:
    """The reply without surrounding whitespace and markdown code fences.

    :param reply: Raw reply.
    :type reply: str
    :return: Stripped reply.
    :rtype: str
    """
    return _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", (reply or "").strip())).strip()


def reply_entries(reply: str) -> List[str]:
    """Text fields of every complete line entry in a grounded reply, found without JSON parsing.

    An entry counts once its ``bbox_2d`` key has opened, so an entry cut by the token cap before its box is
    not counted.  The match runs to the ``", "bbox_2d"`` delimiter, so an unescaped ``"`` inside a Hebrew
    abbreviation (which breaks the JSON parser) does not end the text early.

    :param reply: Raw reply.
    :type reply: str
    :return: Raw (still JSON-escaped) text fields in reply order.
    :rtype: List[str]
    """
    return _ENTRY.findall(reply or "")


def duplicate_lines(texts: Sequence[str]) -> int:
    """Later exact repeats among line texts (whitespace-normalised, empty lines ignored).

    :param texts: Line texts in reading order.
    :type texts: Sequence[str]
    :return: Number of lines that repeat an earlier line.
    :rtype: int
    """
    seen, dups = set(), 0
    for text in texts:
        key = " ".join((text or "").split())
        if not key:
            continue
        if key in seen:
            dups += 1
        seen.add(key)
    return dups


def tail_repeat_share(reply: str, tail_letters: int = TAIL_LETTERS) -> float:
    """``repeat_share`` of the last ``tail_letters`` Hebrew and Arabic letters of a reply.

    :param reply: Raw reply.
    :type reply: str
    :param tail_letters: Letters judged at the end of the reply.
    :type tail_letters: int
    :return: Share of the tail's 12-letter runs that repeat an earlier run of the tail, in [0, 1); 0.0 for
        replies with fewer than three runs' worth of letters.
    :rtype: float
    """
    tail = "".join(_LETTER.findall(reply or ""))[-tail_letters:]
    return repeat_share(tail, span=_RUN) if len(tail) >= 3 * _RUN else 0.0


def read_facts(reply: str, lines: Sequence[str], kraken_texts: Sequence[str],
               reply_tokens: Optional[int] = None, tail_letters: int = TAIL_LETTERS) -> ReadFacts:
    """Measure one read.

    :param reply: Raw VLM reply (None or empty for no reply).
    :type reply: str
    :param lines: Texts of the lines the pipeline parsed from the reply.
    :type lines: Sequence[str]
    :param kraken_texts: Texts of the Kraken fragments of the same image.
    :type kraken_texts: Sequence[str]
    :param reply_tokens: Tokens the reply used, if known (LM Studio ``usage.completion_tokens`` or a
        re-tokenisation); None decides the cap from the reply's length.
    :type reply_tokens: Optional[int]
    :param tail_letters: Letters judged at the end of the reply for ``tail_repeat_share``.
    :type tail_letters: int
    :return: The measurements.
    :rtype: ReadFacts
    """
    reply = reply or ""
    entries = reply_entries(reply)
    texts = entries if len(entries) >= len(lines) else list(lines)
    dups = duplicate_lines(texts)
    reply_letters = letter_count(reply)
    kraken_letters = sum(letter_count(t) for t in kraken_texts)
    return ReadFacts(
        reply_chars=len(reply), reply_letters=reply_letters, reply_tokens=reply_tokens,
        closed=strip_fences(reply).endswith("]"), ends_entry=strip_fences(reply).endswith("}"),
        n_lines=len(lines), n_entries=len(entries),
        loop_ratio=round(loop_ratio(letters_only(reply)), 4),
        tail_repeat_share=round(tail_repeat_share(reply, tail_letters), 4),
        dup_lines=dups, dup_share=round(dups / max(1, len(texts)), 4),
        kraken_letters=kraken_letters, kraken_lines=sum(1 for t in kraken_texts if letter_count(t) >= 3),
        letter_ratio=round(reply_letters / kraken_letters, 4) if kraken_letters else None)


# ----------------------------------------------------------------------------- decision


def is_loop(facts: ReadFacts, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether the reply repeats itself.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param th: Thresholds.
    :type th: Thresholds
    :return: True for a looping reply.
    :rtype: bool
    """
    return (facts.loop_ratio >= th.loop_ratio or facts.tail_repeat_share >= th.tail_repeat_share
            or (facts.dup_lines >= th.dup_lines_min and facts.dup_share >= th.dup_line_share))


def is_capped(facts: ReadFacts, token_cap: int, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether the reply was cut by the token cap.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param token_cap: ``max_tokens`` of the request.
    :type token_cap: int
    :param th: Thresholds.
    :type th: Thresholds
    :return: True when the reply is not a closed array and used (nearly) every allowed token; without a token
        count, when it stops inside an entry and is at least ``min_chars_per_token`` characters per allowed token long.
    :rtype: bool
    """
    if facts.closed:
        return False
    limit = token_cap - th.cap_margin_tokens
    if facts.reply_tokens is not None:
        return facts.reply_tokens >= limit
    return not facts.ends_entry and facts.reply_chars >= th.min_chars_per_token * limit


def is_near_empty(facts: ReadFacts, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether the reply holds almost nothing although the image holds text.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param th: Thresholds.
    :type th: Thresholds
    :return: True for a near-empty reply on an image where Kraken read text.
    :rtype: bool
    """
    return facts.reply_letters < th.near_empty_letters and facts.kraken_letters >= th.min_kraken_letters


def is_skipped_text(facts: ReadFacts, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether the reply holds far fewer letters than Kraken read on the image.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param th: Thresholds.
    :type th: Thresholds
    :return: True when Kraken read enough letters and the reply holds less than ``skipped_ratio`` of them.
    :rtype: bool
    """
    return (facts.kraken_letters >= th.skipped_min_kraken_letters
            and facts.reply_letters < th.skipped_ratio * facts.kraken_letters)


def is_parse_loss(facts: ReadFacts, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether the parser kept far fewer lines than the reply holds.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param th: Thresholds.
    :type th: Thresholds
    :return: True when at least ``parse_loss_min_lines`` and ``parse_loss_share`` of the reply's entries are
        missing from the parsed lines.
    :rtype: bool
    """
    lost = facts.n_entries - facts.n_lines
    return lost >= th.parse_loss_min_lines and lost >= th.parse_loss_share * facts.n_entries


def classify(facts: ReadFacts, token_cap: int, th: Thresholds = DEFAULT_THRESHOLDS) -> Tuple[str, Tuple[str, ...]]:
    """Failure label and every failure flag of one measured read.

    :param facts: Measurements of the read.
    :type facts: ReadFacts
    :param token_cap: ``max_tokens`` of the request.
    :type token_cap: int
    :param th: Thresholds.
    :type th: Thresholds
    :return: ``(label, flags)``; the label is the first flag in :data:`FAILURE_LABELS` order, or ``ok``.
    :rtype: Tuple[str, Tuple[str, ...]]
    """
    fired = {LABEL_LOOP: is_loop(facts, th), LABEL_CAPPED: is_capped(facts, token_cap, th),
             LABEL_NEAR_EMPTY: is_near_empty(facts, th), LABEL_SKIPPED: is_skipped_text(facts, th),
             LABEL_PARSE_LOSS: is_parse_loss(facts, th)}
    flags = tuple(label for label in FAILURE_LABELS if fired[label])
    return (flags[0] if flags else LABEL_OK), flags


def read_health(reply: str, lines: Sequence[str], kraken_texts: Sequence[str], token_cap: int,
                reply_tokens: Optional[int] = None, th: Thresholds = DEFAULT_THRESHOLDS) -> ReadHealth:
    """Measure and judge one read.

    :param reply: Raw VLM reply.
    :type reply: str
    :param lines: Texts of the lines the pipeline parsed from the reply.
    :type lines: Sequence[str]
    :param kraken_texts: Texts of the Kraken fragments of the same image.
    :type kraken_texts: Sequence[str]
    :param token_cap: ``max_tokens`` of the request.
    :type token_cap: int
    :param reply_tokens: Tokens the reply used, if known.
    :type reply_tokens: Optional[int]
    :param th: Thresholds.
    :type th: Thresholds
    :return: Label, flags and facts.
    :rtype: ReadHealth
    """
    facts = read_facts(reply, lines, kraken_texts, reply_tokens, th.tail_letters)
    label, flags = classify(facts, token_cap, th)
    return ReadHealth(label, flags, facts)


def classify_read(reply: str, lines: Sequence[str], kraken_texts: Sequence[str], token_cap: int,
                  reply_tokens: Optional[int] = None, th: Thresholds = DEFAULT_THRESHOLDS) -> str:
    """Failure class of one read, or ``ok`` (the one-call form of :func:`read_health`).

    :param reply: Raw VLM reply.
    :type reply: str
    :param lines: Texts of the lines the pipeline parsed from the reply.
    :type lines: Sequence[str]
    :param kraken_texts: Texts of the Kraken fragments of the same image.
    :type kraken_texts: Sequence[str]
    :param token_cap: ``max_tokens`` of the request.
    :type token_cap: int
    :param reply_tokens: Tokens the reply used, if known.
    :type reply_tokens: Optional[int]
    :param th: Thresholds.
    :type th: Thresholds
    :return: One of :data:`FAILURE_LABELS` or ``ok``.
    :rtype: str
    """
    return read_health(reply, lines, kraken_texts, token_cap, reply_tokens, th).label


def health_of_entry(entry: Mapping[str, Any], token_cap: int, reply_tokens: Optional[int] = None,
                    th: Thresholds = DEFAULT_THRESHOLDS) -> ReadHealth:
    """Judge a read stored in the pipeline's raw cache (``ai_reads/raw/<doc>__<image>__<model>.json``).

    :param entry: Raw-cache entry: ``vlm_raw``, ``vlm_lines`` (``{text, box}``) and ``frags`` (the Kraken
        fragments read with it).
    :type entry: Mapping[str, Any]
    :param token_cap: ``max_tokens`` the read was made with.
    :type token_cap: int
    :param reply_tokens: Tokens the reply used, if known.
    :type reply_tokens: Optional[int]
    :param th: Thresholds.
    :type th: Thresholds
    :return: Label, flags and facts.
    :rtype: ReadHealth
    """
    lines = [ln.get("text") or "" for ln in entry.get("vlm_lines") or []]
    kraken = [f.get("text") or "" for f in entry.get("frags") or []]
    return read_health(entry.get("vlm_raw") or "", lines, kraken, token_cap, reply_tokens, th)
