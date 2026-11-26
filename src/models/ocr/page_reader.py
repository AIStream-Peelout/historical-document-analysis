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

**Decode options** (2026-10-29, :mod:`src.models.ocr.reader_options`; all off by default, so the default reader
behaves exactly as above): a marker-aware loop detector that also retries a looping ``stop`` read
(``loop_detector="markers"``), an orientation probe before the read (``orientation="probe"``, or ``"gated"``: only
after an upright read that hit the token limit, looped or came back as gap markers), band crops of the
page joined into one read (``crops="always"`` / ``"auto"``) and a script post-filter with a retry
(``script_filter=True``). They are measured in ``logs/next_round/arabic_decode_1029/REPORT.md``.
"""
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from PIL import Image

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import arabic_bench_coverage as abc
from src.datasets.evaluations.helper_eval_scripts import loop_guard_outputs as lg
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import LOOP_REPEAT_SHARE, MIN_ANSWER_LETTERS
from src.finetuning.qwen_hebrew.prompts import LEGACY_SCRIPT_PROMPTS, SCRIPT_NEUTRAL_PROMPT
from src.models.ocr import reader_options as ro
from src.models.ocr.lms_transcriber import DEFAULT_CONFIG, transcribe_with_lm_studio_meta

READ_MAX_TOKENS = 1200
RETRY_TEMPERATURE = 0.0
PROMPT_SCRIPT_NEUTRAL = "script_neutral"
STOP, LENGTH = "stop", "length"
REASON_BLANK, REASON_EMPTY, REASON_FAILED = "blank", "empty", "failed"
READ_FIELDS = ("text", "raw_text", "reason", "blank", "looped", "cut_at", "letters_removed", "repeat_share", "finish_reason",
               "prompt_tokens", "completion_tokens", "total_tokens", "attempts", "retried", "prompt", "max_tokens", "temperature",
               "model", "ink", "secs",
               # decode options (defaults when an option is off)
               "first_finish_reason", "markers", "unit_repeat_share", "stripped_chars", "stripped", "rotation", "orientation",
               "crop", "options")


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
    :param loop_detector: ``"letters"`` (production: runs of Hebrew and Arabic letters; only a ``length`` finish is
        retried) or ``"markers"`` (runs of every non-space character with each ``[...]`` one unit; a ``stop`` read
        flagged as a loop is retried too).
    :param script_filter: Strip characters outside Arabic, Hebrew, digits and punctuation from the stored read
        (:func:`reader_options.filter_script`) and retry a read that lost ``filter_retry_chars`` or more.
    :param filter_retry_chars: See ``script_filter``.
    :param orientation: ``"off"``, ``"probe"`` (:func:`reader_options.choose_orientation` over four short reads before
        every read) or ``"gated"`` (read upright; probe only a read :func:`reader_options.suspect_read` flags, and read
        again turned when a turned probe wins).
    :param suspect_marker_share: ``gated``: gap markers per word that make an upright read suspect.
    :param probe_max_tokens: Output budget of one orientation probe.
    :param orientation_margin: Nats per token a turned probe must beat the upright probe by.
    :param orientation_statistic: Probe score compared (``mean_logprob`` or ``first_logprob``).
    :param min_probe_script_share: A probe whose output is under this share of Hebrew / Arabic letters cannot win.
    :param crops: ``"off"``, ``"always"`` or ``"auto"`` (:func:`reader_options.crop_decision`).
    :param crop_overlap: Share of the fragment box height two neighbouring bands both contain.
    :param crop_target_mp: Megapixels per band before a box is cut into more bands (the processor's budget).
    :param crop_max_bands: Upper bound on bands.
    :param crop_min_mp: ``auto``: crop an image of at least this many megapixels (default 1000: never).
    :param crop_max_density: ``auto``: crop a page whose read has fewer letters per ink unit than this.
    :param crop_min_ratio: The joined band read must keep this share of the whole-page read's letters.
    :param view_dir: Where turned and cropped views are written (None: a folder in the system temp directory).
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
    loop_detector: str = ro.LOOP_LETTERS
    script_filter: bool = False
    filter_retry_chars: int = ro.FILTER_RETRY_CHARS
    orientation: str = ro.ORIENTATION_OFF
    suspect_marker_share: float = ro.SUSPECT_MARKER_SHARE
    probe_max_tokens: int = ro.PROBE_MAX_TOKENS
    orientation_margin: float = ro.ORIENTATION_MARGIN
    orientation_statistic: str = "mean_logprob"
    min_probe_script_share: float = ro.MIN_PROBE_SCRIPT_SHARE
    crops: str = ro.CROPS_OFF
    crop_overlap: float = ro.CROP_OVERLAP
    crop_target_mp: float = ro.CROP_TARGET_MP
    crop_max_bands: int = ro.CROP_MAX_BANDS
    crop_min_mp: float = 1000.0            # size trigger off: alone it never helped (arabic_decode_1029 tune_crops.md)
    crop_max_density: float = 2.5          # tuned 2026-10-29 on the Arabic benchmark (plateau 2.5-3.0; 2.0 loss-free)
    crop_min_ratio: float = ro.CROP_MIN_RATIO
    view_dir: Optional[str] = None


DEFAULT_SETTINGS = ReaderSettings()


def options_tag(settings: ReaderSettings) -> str:
    """The decode options a reader runs with, as a short tag (``""`` for the production defaults).

    :param settings: Reader settings.
    :type settings: ReaderSettings
    :return: ``+``-joined option names, e.g. ``"markers+probe+crops:auto+filter"``.
    :rtype: str
    """
    parts = []
    if settings.loop_detector != ro.LOOP_LETTERS:
        parts.append(settings.loop_detector)
    if settings.orientation != ro.ORIENTATION_OFF:
        parts.append(settings.orientation)
    if settings.crops != ro.CROPS_OFF:
        parts.append(f"crops:{settings.crops}")
    if settings.script_filter:
        parts.append("filter")
    return "+".join(parts)


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


def settle_read_markers(text: Optional[str], finish_reason: Optional[str],
                        settings: ReaderSettings = DEFAULT_SETTINGS) -> Dict[str, Any]:
    """:func:`settle_read` with the marker-aware loop detector (``loop_detector="markers"``).

    The guard runs on :func:`reader_options.loop_units` (every non-space character, a gap marker as one unit), so a
    ``[...]`` flood or a repeat of digits, punctuation or Latin text is a loop too. A read is ``looped`` when its
    budget ran out, when the repeat share of its units or of its letters exceeds ``settings.loop_gate`` (the
    production gate), or when the loop guard's window rule finds a loop on the units whose tail is mostly
    non-letters (:func:`reader_options.non_letter_loop`: a flood that the share gate misses after a long real
    text). A looped read is cut where the unit loop starts (or, when the unit rule finds no start, where the letter
    rule does).

    :param text: The generation's text (None when the request failed).
    :type text: Optional[str]
    :param finish_reason: The server's finish reason.
    :type finish_reason: Optional[str]
    :param settings: Reader settings (loop guard parameters).
    :type settings: ReaderSettings
    :return: :func:`settle_read`'s fields plus ``unit_repeat_share`` and ``markers``.
    :rtype: Dict[str, Any]
    """
    if text is None:
        return {"text": None, "raw_text": None, "looped": False, "cut_at": None, "letters_removed": 0, "repeat_share": 0.0,
                "unit_repeat_share": 0.0, "markers": 0}
    units, offsets = ro.loop_units(text)
    unit_share, letter_share = ro.unit_repeat_share(units, settings.run_len), ar.repeat_share(text)
    start = lg.loop_start(units, settings.run_len, settings.window, settings.min_repeat)
    looped = (finish_reason == LENGTH or unit_share > settings.loop_gate or letter_share > settings.loop_gate
              or ro.non_letter_loop(units, start))
    guarded, cut_at = text, None
    if looped:
        if start is not None:
            cut_at = offsets[start]
            guarded = text[:cut_at].rstrip()
        else:
            guarded, cut_at, _ = lg.guard_text(text, settings.run_len, settings.window, settings.min_repeat, None, script_letters)
    return {"text": guarded, "raw_text": text if cut_at is not None else None, "looped": looped, "cut_at": cut_at,
            "letters_removed": len(script_letters(text)) - len(script_letters(guarded)), "repeat_share": round(letter_share, 4),
            "unit_repeat_share": round(unit_share, 4), "markers": ro.marker_count(text)}


def needs_retry(read: Mapping[str, Any], settings: ReaderSettings = DEFAULT_SETTINGS) -> bool:
    """Whether a settled first read is requested once more at ``settings.retry_temperature``.

    :param read: The settled first read (:func:`_settled` fields).
    :type read: Mapping[str, Any]
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :return: Production: a ``length`` finish (with ``retry_on_length``). With ``loop_detector="markers"`` also any
        read flagged as a loop; with ``script_filter`` also a read that lost ``filter_retry_chars`` characters.
    :rtype: bool
    """
    if not settings.retry_on_length:
        return False
    if read.get("finish_reason") == LENGTH:
        return True
    if settings.loop_detector == ro.LOOP_MARKERS and read.get("looped"):
        return True
    return bool(settings.script_filter and (read.get("stripped_chars") or 0) >= settings.filter_retry_chars)


def _read_key(read: Mapping[str, Any]) -> Tuple[bool, int]:
    """Order of two reads of one page: a finished generation that holds a read first, then the letters kept."""
    letters = len(script_letters(read.get("text") or ""))
    return read.get("finish_reason") == STOP and letters >= MIN_ANSWER_LETTERS, letters


def _option_key(read: Mapping[str, Any], settings: ReaderSettings) -> Tuple[bool, bool, bool, int]:
    """:func:`_read_key` when a decode option is on: a finished read with letters, then not a loop, then not a read
    that needed the script filter, then the letters kept."""
    finished, letters = _read_key(read)
    filtered = bool(settings.script_filter and (read.get("stripped_chars") or 0) >= settings.filter_retry_chars)
    return finished, not read.get("looped"), not filtered, letters


def choose_read_options(first: Mapping[str, Any], retry: Mapping[str, Any], settings: ReaderSettings) -> Dict[str, Any]:
    """:func:`choose_read` for a reader with decode options: also prefers a read that is not a loop and a read the
    script filter did not have to clean (:func:`_option_key`); ties keep the retry, as :func:`choose_read` does.

    :param first: The settled first read.
    :type first: Mapping[str, Any]
    :param retry: The settled retry.
    :type retry: Mapping[str, Any]
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :return: The read kept.
    :rtype: Dict[str, Any]
    """
    return dict(retry) if _option_key(retry, settings) >= _option_key(first, settings) else dict(first)


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
                              "model": model_id, "ink": None, "secs": None, "first_finish_reason": None, "markers": 0,
                              "unit_repeat_share": None, "stripped_chars": 0, "stripped": None, "rotation": 0, "orientation": None,
                              "crop": None, "options": options_tag(settings)}
    record.update(fields)
    record["secs"] = round(time.time() - started, 1)
    return record


Transcribe = Callable[..., Awaitable[Optional[Dict[str, Any]]]]
_SETTLED_KEYS = ("text", "raw_text", "looped", "cut_at", "letters_removed", "repeat_share", "finish_reason", "prompt_tokens",
                 "completion_tokens", "total_tokens", "temperature", "first_finish_reason", "markers", "unit_repeat_share",
                 "stripped_chars", "stripped")


def _view_dir(settings: ReaderSettings) -> Path:
    """Directory of turned and cropped views (``settings.view_dir`` or a folder in the system temp directory)."""
    return Path(settings.view_dir) if settings.view_dir else Path(tempfile.gettempdir()) / "page_reader_views"


async def read_view(model_id: str, image_path: str, prompt: str, settings: ReaderSettings, transcribe: Transcribe,
                    **transcribe_kwargs: Any) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    """One read of one image file: a request, loop handling (and the script filter), at most one retry.

    With every decode option off this is exactly the production read: a ``length`` finish is retried once at
    ``settings.retry_temperature`` and :func:`choose_read` keeps the better read.

    :param model_id: LM Studio model key.
    :type model_id: str
    :param image_path: Image (or view) file.
    :type image_path: str
    :param prompt: Prompt text.
    :type prompt: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param transcribe: See :func:`read_page`.
    :type transcribe: Transcribe
    :param transcribe_kwargs: Passed through to ``transcribe``.
    :type transcribe_kwargs: Any
    :return: ``(settled read or None when the first request failed, attempts, retried)``.
    :rtype: Tuple[Optional[Dict[str, Any]], int, bool]
    """
    meta = await transcribe(model_id, image_path, prompt, temperature=settings.temperature, max_tokens=settings.max_tokens,
                            **transcribe_kwargs)
    if meta is None:
        return None, 1, False
    read = _settled(meta, settings.temperature, settings)
    read["first_finish_reason"] = read["finish_reason"]
    if not needs_retry(read, settings):
        return read, 1, False
    again = await transcribe(model_id, image_path, prompt, temperature=settings.retry_temperature, max_tokens=settings.max_tokens,
                             **transcribe_kwargs)
    if again is not None:
        first_finish = read["finish_reason"]
        retry = _settled(again, settings.retry_temperature, settings)
        read = choose_read(read, retry) if not options_tag(settings) else choose_read_options(read, retry, settings)
        read["first_finish_reason"] = first_finish
    return read, 2, True


async def probe_orientation(model_id: str, image_path: str, prompt: str, settings: ReaderSettings, transcribe: Transcribe,
                            **transcribe_kwargs: Any) -> Tuple[int, Dict[str, Any]]:
    """Read the page briefly in each of the four orientations and choose the one it is read in.

    Each probe is a request at temperature 0 with ``settings.probe_max_tokens`` and token log-probabilities.

    :param model_id: LM Studio model key.
    :type model_id: str
    :param image_path: Image file.
    :type image_path: str
    :param prompt: Prompt text.
    :type prompt: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param transcribe: See :func:`read_page`.
    :type transcribe: Transcribe
    :param transcribe_kwargs: Passed through (``extra_payload`` gains ``logprobs``).
    :type transcribe_kwargs: Any
    :return: ``(rotation, {"reason", "statistic", "margin", "probes": {rotation: probe_features}})``.
    :rtype: Tuple[int, Dict[str, Any]]
    """
    kwargs = dict(transcribe_kwargs)
    kwargs["extra_payload"] = dict(kwargs.get("extra_payload") or {}, logprobs=True)
    probes: Dict[int, Dict[str, Any]] = {}
    for rotation in ro.ROTATIONS:
        view = ro.rotated_view(image_path, rotation, _view_dir(settings))
        meta = await transcribe(model_id, view, prompt, temperature=0.0, max_tokens=settings.probe_max_tokens, **kwargs)
        probes[rotation] = ro.probe_features(meta)
    rotation, reason = ro.choose_orientation(probes, settings.orientation_margin, settings.min_probe_script_share,
                                             settings.orientation_statistic)
    return rotation, {"reason": reason, "statistic": settings.orientation_statistic, "margin": settings.orientation_margin,
                      "probes": {str(r): p for r, p in probes.items()}}


async def _with_bands(model_id: str, path: str, read: Dict[str, Any], rotation: int, prompt: str, settings: ReaderSettings,
                      transcribe: Transcribe, ink: Optional[Mapping[str, Any]],
                      **transcribe_kwargs: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], int]:
    """A settled read of one orientation with the band-crop read applied (when crops are on and triggered).

    :param model_id: LM Studio model key.
    :type model_id: str
    :param path: The image in this orientation.
    :type path: str
    :param read: The settled whole-page read of ``path``.
    :type read: Dict[str, Any]
    :param rotation: The orientation (view-name prefix of the bands).
    :type rotation: int
    :param prompt: Prompt text.
    :type prompt: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param transcribe: See :func:`read_page`.
    :type transcribe: Transcribe
    :param ink: Ink measures of the page.
    :type ink: Optional[Mapping[str, Any]]
    :param transcribe_kwargs: Passed through to ``transcribe``.
    :type transcribe_kwargs: Any
    :return: ``(read, crop, band requests)``: the joined band read replaces ``text`` (the whole read kept as
        ``raw_text``) when the bands are accepted; ``crop`` is None when the page was not cropped.
    :rtype: Tuple[Dict[str, Any], Optional[Dict[str, Any]], int]
    """
    if settings.crops == ro.CROPS_OFF:
        return read, None, 0
    crop = await read_bands(model_id, path, read, prompt, settings, transcribe, ink, f"r{rotation}_", **transcribe_kwargs)
    if crop is None:
        return read, None, 0
    if crop["accepted"]:
        read = dict(read, text=crop["text"], looped=crop["looped"],
                    raw_text=read["raw_text"] if read["raw_text"] is not None else read["text"])
    return read, crop, crop["requests"]


def gate_view(read: Mapping[str, Any], crop: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What :func:`reader_options.suspect_read` judges: the whole read, or the joined band read when bands replaced it
    (a ``length`` finish then means a band hit the limit).

    :param read: The settled read (after the band read was applied).
    :type read: Mapping[str, Any]
    :param crop: The band-crop record, None when the page was not cropped.
    :type crop: Optional[Mapping[str, Any]]
    :return: ``text``, ``looped`` and ``first_finish_reason``.
    :rtype: Dict[str, Any]
    """
    if not (crop and crop.get("accepted")):
        return {k: read.get(k) for k in ("text", "looped", "finish_reason", "first_finish_reason")}
    hit = any(b.get("first_finish_reason") == LENGTH for b in crop.get("bands") or [])
    return {"text": read.get("text"), "looped": crop.get("looped"), "first_finish_reason": LENGTH if hit else STOP}


async def _gated_turn(model_id: str, image_path: str, read: Dict[str, Any], crop: Optional[Dict[str, Any]], attempts: int,
                      retried: bool, prompt: str, settings: ReaderSettings, transcribe: Transcribe,
                      ink: Optional[Mapping[str, Any]], **transcribe_kwargs: Any
                      ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], int, bool, int, Dict[str, Any]]:
    """``orientation="gated"``: probe the orientation only when the upright read (after its band crops, when crops are
    on) is still suspect, and read the page again turned when a turned probe wins by the margin.

    The turned read (with its own band crops) replaces the upright one; when its request fails the upright read is kept.
    Crops come first so that a page the bands already read is never turned (T-S NS 113.14: upright, its whole read
    loops, the bands read it).

    :param model_id: LM Studio model key.
    :type model_id: str
    :param image_path: Upright image file.
    :type image_path: str
    :param read: The settled upright read (band read applied).
    :type read: Dict[str, Any]
    :param crop: The upright band-crop record (None when not cropped).
    :type crop: Optional[Dict[str, Any]]
    :param attempts: Requests the upright read and its bands took.
    :type attempts: int
    :param retried: Whether the upright read was retried.
    :type retried: bool
    :param prompt: Prompt text.
    :type prompt: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param transcribe: See :func:`read_page`.
    :type transcribe: Transcribe
    :param ink: Ink measures of the page (the band trigger).
    :type ink: Optional[Mapping[str, Any]]
    :param transcribe_kwargs: Passed through to ``transcribe``.
    :type transcribe_kwargs: Any
    :return: ``(read, crop, attempts, retried, rotation, orientation)``: ``orientation`` holds ``gate``
        (:func:`reader_options.suspect_read`'s reason), the probe fields when probed, and ``turned_read_failed``.
    :rtype: Tuple[Dict[str, Any], Optional[Dict[str, Any]], int, bool, int, Dict[str, Any]]
    """
    suspect, why = ro.suspect_read(gate_view(read, crop), settings.suspect_marker_share)
    orientation: Dict[str, Any] = {"gate": why}
    if not suspect:
        return read, crop, attempts, retried, 0, orientation
    rotation, probed = await probe_orientation(model_id, image_path, prompt, settings, transcribe, **transcribe_kwargs)
    orientation.update(probed)
    if rotation == 0:
        return read, crop, attempts, retried, 0, orientation
    view = ro.rotated_view(image_path, rotation, _view_dir(settings))
    turned, more, turned_retried = await read_view(model_id, view, prompt, settings, transcribe, **transcribe_kwargs)
    if turned is None:
        orientation["turned_read_failed"] = True
        return read, crop, attempts + more, retried, 0, orientation
    turned, turned_crop, bands = await _with_bands(model_id, view, turned, rotation, prompt, settings, transcribe, ink,
                                                   **transcribe_kwargs)
    return turned, turned_crop, attempts + more + bands, turned_retried, rotation, orientation


async def read_bands(model_id: str, image_path: str, whole: Mapping[str, Any], prompt: str, settings: ReaderSettings,
                     transcribe: Transcribe, ink: Optional[Mapping[str, Any]], tag: str = "",
                     **transcribe_kwargs: Any) -> Optional[Dict[str, Any]]:
    """The band-crop read of a page when :func:`reader_options.crop_decision` asks for one.

    :param model_id: LM Studio model key.
    :type model_id: str
    :param image_path: Image in its reading orientation.
    :type image_path: str
    :param whole: The settled whole-page read of the same image.
    :type whole: Mapping[str, Any]
    :param prompt: Prompt text.
    :type prompt: str
    :param settings: Reader settings.
    :type settings: ReaderSettings
    :param transcribe: See :func:`read_page`.
    :type transcribe: Transcribe
    :param ink: Ink measures of the page (the ``auto`` density trigger), None when not measured.
    :type ink: Optional[Mapping[str, Any]]
    :param tag: View-name prefix (the rotation).
    :type tag: str
    :param transcribe_kwargs: Passed through to ``transcribe``.
    :type transcribe_kwargs: Any
    :return: None when the page is not cropped; else ``{"reason", "box", "accepted", "accept_reason", "text"
        (joined), "dropped_lines", "density", "bands": [{"box", "text", "finish_reason", "first_finish_reason", "looped",
        "attempts", ...}], "requests"}``.
    :rtype: Optional[Dict[str, Any]]
    """
    with Image.open(image_path) as im:
        pixels = im.size[0] * im.size[1]
    letters = len(script_letters(whole.get("text") or ""))
    crop, reason = ro.crop_decision(settings.crops, pixels, letters, ink, settings.crop_min_mp, settings.crop_max_density)
    if not crop:
        return None
    box = ro.fragment_box(image_path)
    views = ro.band_views(image_path, box, _view_dir(settings), settings.crop_overlap, settings.crop_target_mp,
                          settings.crop_max_bands, tag)
    bands, requests, failed, looped = [], 0, False, False
    for band_box, path in views:
        read, attempts, retried = await read_view(model_id, path, prompt, settings, transcribe, **transcribe_kwargs)
        requests += attempts
        if read is None:
            failed = True
            bands.append({"box": list(band_box), "text": None, "failed": True, "attempts": attempts})
            continue
        looped = looped or bool(read["looped"])
        bands.append({"box": list(band_box), "attempts": attempts, "retried": retried, "failed": False,
                      **{k: read.get(k) for k in ("text", "finish_reason", "first_finish_reason", "looped", "cut_at",
                                                  "completion_tokens", "stripped_chars", "markers")}})
    joined, dropped = ro.merge_bands([b.get("text") or "" for b in bands])
    accepted, why = ro.accept_bands(whole, len(script_letters(joined)), looped, failed, settings.crop_min_ratio)
    return {"reason": reason, "box": list(box) if box else None, "accepted": accepted, "accept_reason": why, "text": joined,
            "dropped_lines": dropped, "density": ro.letters_per_ink(letters, ink), "bands": bands, "requests": requests,
            "looped": looped}


async def read_page(model_id: str, image_path: str, settings: ReaderSettings = DEFAULT_SETTINGS,
                    script_hint: Optional[str] = None, legacy: bool = False,
                    transcribe: Optional[Transcribe] = None,
                    measure: Optional[Callable[[str, ReaderSettings], Dict[str, Any]]] = None,
                    **transcribe_kwargs: Any) -> Dict[str, Any]:
    """Read one page image: ink gate, one model request, loop handling, at most one retry.

    With decode options on (:class:`ReaderSettings`), an orientation probe runs before the read (the read is then
    made on the turned image; ``gated``: after a suspect upright read, which a turned read then replaces), the retry also fires on a marker loop or a filtered read, and a band-crop read may
    replace the whole-page read after it. ``attempts`` counts every request but the probes.

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
        ``ink`` the measures with their ``class``; ``rotation`` / ``orientation`` (the probes), ``crop`` (the band
        reads and whether they were kept) and ``stripped`` when those options are on.
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
    rotation, orientation, path = 0, None, image_path
    if settings.orientation == ro.ORIENTATION_PROBE:
        rotation, orientation = await probe_orientation(model_id, image_path, prompt, settings, transcribe, **transcribe_kwargs)
        path = ro.rotated_view(image_path, rotation, _view_dir(settings))
    read, attempts, retried = await read_view(model_id, path, prompt, settings, transcribe, **transcribe_kwargs)
    if read is None:
        return _record(model_id, prompt_name, settings, started, reason=REASON_FAILED, attempts=attempts, ink=ink,
                       rotation=rotation, orientation=orientation)
    read, crop, bands = await _with_bands(model_id, path, read, rotation, prompt, settings, transcribe, ink, **transcribe_kwargs)
    attempts += bands
    if settings.orientation == ro.ORIENTATION_GATED:
        read, crop, attempts, retried, rotation, orientation = await _gated_turn(
            model_id, image_path, read, crop, attempts, retried, prompt, settings, transcribe, ink, **transcribe_kwargs)
    return _record(model_id, prompt_name, settings, started, attempts=attempts, retried=retried, ink=ink,
                   reason=REASON_EMPTY if read["text"] == "" else None, rotation=rotation, orientation=orientation, crop=crop,
                   **{k: read[k] for k in _SETTLED_KEYS if k in read})


def _settled(meta: Mapping[str, Any], temperature: float, settings: ReaderSettings) -> Dict[str, Any]:
    """A completion's metadata with its loop handling (and the script filter when on); an empty or blank generation is
    the empty read ``""``, not a failure."""
    text = (meta.get("text") or "").strip()
    if settings.loop_detector == ro.LOOP_MARKERS:
        fields = settle_read_markers(text, meta.get("finish_reason"), settings)
    else:
        fields = settle_read(text, meta.get("finish_reason"), settings)
        fields["markers"] = ro.marker_count(text)
    if settings.script_filter and fields["text"]:
        fields["text"], fields["stripped_chars"], fields["stripped"] = ro.filter_script(fields["text"])
    meta = {k: v for k, v in meta.items() if k != "token_logprobs"}
    return dict(meta, temperature=temperature, **fields)
