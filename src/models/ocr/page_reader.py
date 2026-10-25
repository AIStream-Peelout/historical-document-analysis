# File name: page_reader.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Production page reader: one whole-page read of a manuscript image through LM Studio.

The reader wraps :func:`lms_transcriber.transcribe_with_lm_studio_meta` with the settings the prompt
ablation of 2026-10-25 recommended (``logs/prompt_ablation_1024/REPORT_2026-10-25.md``):

1. **One script-neutral prompt** for every page, :data:`prompts.SCRIPT_NEUTRAL_PROMPT`: the fragment prompt
   with "handwritten text in its original script" in place of the script assertion. Asserting the script
   made the reader write Arabic on Hebrew-script sides (60-93 % of them) and gained nothing on Arabic pages.
   The script-asserting prompts (``FRAGMENT_TRANSCRIBE_PROMPT``, ``ARABIC_FRAGMENT_PROMPT``) are reachable
   only with ``legacy=True``, for reproducing past evaluations.
2. **An output budget of 1,200 tokens** (:data:`READ_MAX_TOKENS`): a page read needs 170-305 tokens; the
   budget only bounds the cost of a loop.
3. **Generation metadata in every stored read** (``finish_reason`` and the token counts of
   :func:`lms_transcriber.completion_meta`). A ``length`` finish (the budget cut the generation) is a loop:
   the text is cut where the loop guard (``loop_guard_outputs.guard_text``, run over Hebrew and Arabic
   letters together) finds the loop start, the read is marked ``looped``, and by default the page is
   re-requested once at temperature 0 (:func:`choose_read` keeps the better of the two).
4. **An ink-density pre-gate**: before the model is called the ink inside the image's fragment mask is
   measured as the coverage tool does (``arabic_bench_coverage``: backing removed, black-hat ink, the
   8 / 16 per-mille thresholds). An image the tool classes ``blank_or_faint`` is stored as an empty read
   with ``reason = "blank"`` (the equivalent of a ``[NO_READABLE_TEXT]`` marker, which these checkpoints
   never produce) and is not sent to the model. A failed mask or a dark paper fails open: the page is read.

The decisions are pure functions (:func:`select_prompt`, :func:`settle_read`, :func:`choose_read`,
:func:`gate_decision`) and are tested without a model; :func:`read_page` composes them with the LM Studio
call and the ink measure, both injectable. Every read is a dict with :data:`READ_FIELDS`; ``reason`` is None
for a read, ``"blank"`` (gated), ``"empty"`` (the model generated nothing) or ``"failed"`` (no answer from the
server after the transcriber's retries; ``text`` None, so a resumable runner reads the page again).
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import arabic_bench_coverage as abc
from src.datasets.evaluations.helper_eval_scripts import loop_guard_outputs as lg
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import LOOP_REPEAT_SHARE, MIN_ANSWER_LETTERS
from src.finetuning.qwen_hebrew.prompts import LEGACY_SCRIPT_PROMPTS, SCRIPT_NEUTRAL_PROMPT
from src.models.ocr.lms_transcriber import DEFAULT_CONFIG, transcribe_with_lm_studio_meta

READ_MAX_TOKENS = 1200
RETRY_TEMPERATURE = 0.0
PROMPT_SCRIPT_NEUTRAL = "script_neutral"
STOP, LENGTH = "stop", "length"
REASON_BLANK, REASON_EMPTY, REASON_FAILED = "blank", "empty", "failed"
READ_FIELDS = ("text", "raw_text", "reason", "blank", "looped", "cut_at", "letters_removed", "repeat_share", "finish_reason",
               "prompt_tokens", "completion_tokens", "total_tokens", "attempts", "retried", "prompt", "max_tokens", "temperature",
               "model", "ink", "secs")


@dataclass(frozen=True)
class ReaderSettings:
    """How a page is read.

    :param max_tokens: Output token budget of one request.
    :param temperature: Sampling temperature of the first request (the benchmarks' 0.1).
    :param retry_on_length: Re-request a page once when the budget cut the first generation.
    :param retry_temperature: Temperature of that retry.
    :param blank_gate: Measure the ink first and skip the model on a blank or faint image.
    :param image: Thresholds of the ink measure (the coverage tool's).
    :param run_len: Loop guard: length of a letter run.
    :param window: Loop guard: runs that must mostly repeat from the loop start on.
    :param min_repeat: Loop guard: share of those runs that must repeat.
    :param loop_gate: A ``stop`` read is cut only when its ``repeat_share`` exceeds this (the scorer's loop flag);
        a ``length`` read is always tried.
    """
    max_tokens: int = READ_MAX_TOKENS
    temperature: float = DEFAULT_CONFIG.temperature
    retry_on_length: bool = True
    retry_temperature: float = RETRY_TEMPERATURE
    blank_gate: bool = True
    image: abc.ImageSettings = field(default_factory=abc.ImageSettings)
    run_len: int = lg.RUN_LEN
    window: int = lg.WINDOW
    min_repeat: float = lg.MIN_REPEAT
    loop_gate: float = LOOP_REPEAT_SHARE


DEFAULT_SETTINGS = ReaderSettings()


# --- pure decisions ---------------------------------------------------------------------------------------------------

def select_prompt(script_hint: Optional[str] = None, legacy: bool = False) -> Tuple[str, str]:
    """The prompt a page is read with.

    :param script_hint: What the pipeline believes the page's script is (``"hebrew"``, ``"arabic"``, anything else or
        None). The production prompt does not depend on it: the model reads the script off the image.
    :type script_hint: Optional[str]
    :param legacy: Reproduce a past evaluation with the prompt that asserted ``script_hint``.
    :type legacy: bool
    :return: ``(name, prompt text)``: ``("script_neutral", SCRIPT_NEUTRAL_PROMPT)``, or with ``legacy``
        ``("legacy_<script>", the asserting prompt)``.
    :rtype: Tuple[str, str]
    :raises ValueError: With ``legacy`` and a hint no past prompt asserted.
    """
    if not legacy:
        return PROMPT_SCRIPT_NEUTRAL, SCRIPT_NEUTRAL_PROMPT
    script = (script_hint or "").lower()
    if script not in LEGACY_SCRIPT_PROMPTS:
        raise ValueError(f"no script-asserting prompt for {script_hint!r}; the past evaluations used {sorted(LEGACY_SCRIPT_PROMPTS)}")
    return f"legacy_{script}", LEGACY_SCRIPT_PROMPTS[script]


def script_letters(text: str) -> str:
    """Hebrew and Arabic letters of a text, in order (the letters ``arabic_script.repeat_share`` counts).

    :param text: Any text.
    :type text: str
    :return: Letters only.
    :rtype: str
    """
    return "".join(ar.SCRIPT_LETTER.findall(ar.normalize_arabic(text)))


def settle_read(text: Optional[str], finish_reason: Optional[str], settings: ReaderSettings = DEFAULT_SETTINGS) -> Dict[str, Any]:
    """Loop handling of one generation: what is stored as the read, and whether it looped.

    A ``length`` finish is treated as a loop: the loop guard is applied without its gate (the text is cut
    where a loop starts, if one is found; a long page cut by the budget without a loop keeps its text) and
    the read is marked ``looped``. Any other finish is cut only when its ``repeat_share`` exceeds
    ``settings.loop_gate``, the scorer's loop flag, and is marked ``looped`` in that case.

    :param text: The generation's text (None when the request failed).
    :type text: Optional[str]
    :param finish_reason: ``"stop"``, ``"length"`` or whatever the server said (None when absent).
    :type finish_reason: Optional[str]
    :param settings: Reader settings (loop guard parameters).
    :type settings: ReaderSettings
    :return: ``text`` (the read to store), ``raw_text`` (the generation when it was cut, else None), ``looped``,
        ``cut_at`` (character offset of the cut, None without one), ``letters_removed`` and ``repeat_share``
        (of the generation).
    :rtype: Dict[str, Any]
    """
    if text is None:
        return {"text": None, "raw_text": None, "looped": False, "cut_at": None, "letters_removed": 0, "repeat_share": 0.0}
    length = finish_reason == LENGTH
    share = ar.repeat_share(text)
    guarded, cut_at, cut = lg.guard_text(text, settings.run_len, settings.window, settings.min_repeat,
                                         None if length else settings.loop_gate, script_letters)
    return {"text": guarded, "raw_text": text if cut else None, "looped": length or share > settings.loop_gate, "cut_at": cut_at,
            "letters_removed": len(script_letters(text)) - len(script_letters(guarded)), "repeat_share": round(share, 4)}


def _read_key(read: Mapping[str, Any]) -> Tuple[bool, int]:
    """Order of two reads of one page: a finished generation that holds a read first, then the letters kept."""
    letters = len(script_letters(read.get("text") or ""))
    return read.get("finish_reason") == STOP and letters >= MIN_ANSWER_LETTERS, letters


def choose_read(first: Mapping[str, Any], retry: Mapping[str, Any]) -> Dict[str, Any]:
    """Which of a page's two reads to keep.

    A read counts as *finished* when the model emitted its end token and the text holds at least
    ``MIN_ANSWER_LETTERS`` (the scorer's minimum for an answer): a retry that stops at once with nothing,
    or with a few letters, must not replace the text read before a loop.

    :param first: The settled first read (:func:`settle_read` fields plus the completion metadata).
    :type first: Mapping[str, Any]
    :param retry: The settled retry.
    :type retry: Mapping[str, Any]
    :return: The retry when it is finished and the first read is not, or when neither (or both) is and it
        kept at least as many letters; otherwise the first read.
    :rtype: Dict[str, Any]
    """
    return dict(retry) if _read_key(retry) >= _read_key(first) else dict(first)


def gate_decision(ink: Mapping[str, Any], settings: ReaderSettings = DEFAULT_SETTINGS) -> Tuple[bool, str]:
    """Whether an image is skipped as blank, from its ink measures.

    :param ink: :func:`measure_ink` result.
    :type ink: Mapping[str, Any]
    :param settings: Reader settings (image thresholds).
    :type settings: ReaderSettings
    :return: ``(skip, ink class)``: skip only for the coverage tool's ``blank_or_faint``; ``text`` and
        ``unknown`` (mask failed, dark paper, ink outside the fragment) are read.
    :rtype: Tuple[bool, str]
    """
    ink_class = abc.ink_class(ink, settings.image)
    return ink_class == abc.BLANK, ink_class


# --- the image measure and the model call -------------------------------------------------------------------------------

def measure_ink(image_path: str, settings: ReaderSettings = DEFAULT_SETTINGS) -> Dict[str, Any]:
    """Ink density of an image inside its fragment mask, as the coverage tool measures it.

    :param image_path: Image file.
    :type image_path: str
    :param settings: Reader settings (image thresholds).
    :type settings: ReaderSettings
    :return: ``arabic_bench_coverage.ink_measures`` of the fragment mask plus ``loose_ink_permille`` (the ink inside
        the loose mask).
    :rtype: Dict[str, Any]
    """
    rgb = abc.load_rgb(Path(image_path), settings.image.long_side)
    ink, _ = abc.ink_measures(rgb, abc.fragment_mask(rgb, settings.image), settings.image)
    loose, _ = abc.ink_measures(rgb, abc.loose_mask(rgb, settings.image), settings.image)
    ink["loose_ink_permille"] = loose["ink_permille"]
    return ink


def _record(model_id: str, prompt_name: str, settings: ReaderSettings, started: float, **fields: Any) -> Dict[str, Any]:
    """A read with every :data:`READ_FIELDS` key (defaults: an empty, unread page)."""
    record: Dict[str, Any] = {"text": None, "raw_text": None, "reason": None, "blank": False, "looped": False, "cut_at": None,
                              "letters_removed": 0, "repeat_share": 0.0, "finish_reason": None, "prompt_tokens": None,
                              "completion_tokens": None, "total_tokens": None, "attempts": 0, "retried": False,
                              "prompt": prompt_name, "max_tokens": settings.max_tokens, "temperature": settings.temperature,
                              "model": model_id, "ink": None, "secs": None}
    record.update(fields)
    record["secs"] = round(time.time() - started, 1)
    return record


async def read_page(model_id: str, image_path: str, settings: ReaderSettings = DEFAULT_SETTINGS,
                    script_hint: Optional[str] = None, legacy: bool = False,
                    transcribe: Optional[Callable[..., Awaitable[Optional[Dict[str, Any]]]]] = None,
                    measure: Optional[Callable[[str, ReaderSettings], Dict[str, Any]]] = None,
                    **transcribe_kwargs: Any) -> Dict[str, Any]:
    """Read one page image: ink gate, one model request, loop handling, at most one retry.

    :param model_id: LM Studio model key.
    :type model_id: str
    :param image_path: Image file.
    :type image_path: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param script_hint: See :func:`select_prompt` (ignored unless ``legacy``).
    :type script_hint: Optional[str]
    :param legacy: Reproduce a past evaluation with the script-asserting prompt.
    :type legacy: bool
    :param transcribe: ``(model_id, image_path, prompt, temperature=..., max_tokens=..., **kwargs) -> completion meta
        or None`` (default :func:`lms_transcriber.transcribe_with_lm_studio_meta`; injectable for tests).
    :param measure: ``(image_path, settings) -> ink measures`` (default :func:`measure_ink`).
    :param transcribe_kwargs: Passed through to ``transcribe`` (``base_url``, ``timeout``, ...).
    :type transcribe_kwargs: Any
    :return: The read (:data:`READ_FIELDS`): ``text`` is the stored read (``""`` for a blank page or an empty
        generation, None when the request failed), ``reason`` None, ``"blank"``, ``"empty"`` or ``"failed"``,
        ``ink`` the measures with their ``class``.
    :rtype: Dict[str, Any]
    """
    transcribe = transcribe or transcribe_with_lm_studio_meta
    measure = measure or measure_ink
    prompt_name, prompt = select_prompt(script_hint, legacy)
    started = time.time()
    ink: Optional[Dict[str, Any]] = None
    if settings.blank_gate:
        ink = dict(measure(image_path, settings))
        skip, ink["class"] = gate_decision(ink, settings)
        if skip:
            return _record(model_id, prompt_name, settings, started, text="", reason=REASON_BLANK, blank=True, ink=ink)
    meta = await transcribe(model_id, image_path, prompt, temperature=settings.temperature, max_tokens=settings.max_tokens,
                            **transcribe_kwargs)
    if meta is None:
        return _record(model_id, prompt_name, settings, started, reason=REASON_FAILED, attempts=1, ink=ink)
    read = _settled(meta, settings.temperature, settings)
    attempts, retried = 1, False
    if read["finish_reason"] == LENGTH and settings.retry_on_length:
        attempts, retried = 2, True
        again = await transcribe(model_id, image_path, prompt, temperature=settings.retry_temperature,
                                 max_tokens=settings.max_tokens, **transcribe_kwargs)
        if again is not None:
            read = choose_read(read, _settled(again, settings.retry_temperature, settings))
    return _record(model_id, prompt_name, settings, started, attempts=attempts, retried=retried, ink=ink,
                   reason=REASON_EMPTY if read["text"] == "" else None,
                   **{k: read[k] for k in ("text", "raw_text", "looped", "cut_at", "letters_removed", "repeat_share", "finish_reason",
                                           "prompt_tokens", "completion_tokens", "total_tokens", "temperature")})


def _settled(meta: Mapping[str, Any], temperature: float, settings: ReaderSettings) -> Dict[str, Any]:
    """A completion's metadata with its loop handling; an empty or blank generation is the empty read ``""``, not a failure."""
    return dict(meta, temperature=temperature, **settle_read((meta.get("text") or "").strip(), meta.get("finish_reason"), settings))
