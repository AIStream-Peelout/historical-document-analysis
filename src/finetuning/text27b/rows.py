# File name: rows.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Row schema, prompts, JSONL I/O, token counts and the chat-template render contract (text27b).

A training row is ``{"messages": [system, user, assistant], "meta": {...}}`` (the v0_1 schema,
extended for v0_2; see :class:`Meta`). Every row is rendered with the local Qwen3.8 chat template
and ``enable_thinking=False``: the system turn then carries no "Reasoning effort" sentence and
the assistant turn opens with exactly one empty think block (``<think>\\n\\n</think>\\n\\n``).

Public API (other text27b modules import these names):

Constants
    :data:`BUILD`, :data:`PGP_FAMILIES`, :data:`LITERATURE_FAMILIES`, :data:`CANON_FAMILIES`, :data:`FAMILIES`,
    :data:`CONDITIONS`, :data:`READ_LIKE_CONDITIONS`, :data:`SPLITS`, :data:`TRAIN_SPLITS`,
    :data:`EVAL_SPLITS`, :data:`EXPOSURE_LEVELS`, :data:`LICENCE_CLASSES`, :data:`V01_META_FIELDS`,
    :data:`EMPTY_THINK`, :data:`EFFORT_MARKER`.
Prompts
    :data:`SYSTEM`, :data:`SCRIPT_NAME`, :data:`INSTRUCTION`, :data:`COND_NOTE_V0` (v0_1 text,
    verbatim), :data:`READ_NOTE_V2` (the one amended note shared by every read-like condition,
    plan §9.1), :data:`COND_NOTE` (v0_2 mapping), :func:`user_prompt`, :func:`make_messages` (every message content
    NFC-normalised here, the one place rows get their text: the tokenizer normalises to NFC, so a decomposed
    target would not survive the harness mask guardrail), :func:`nfc`, :func:`non_nfc_roles`.
Types
    :class:`Message`, :class:`Meta`, :class:`Row`, :class:`RenderError`.
I/O
    :func:`read_jsonl` (streaming), :func:`load_jsonl`, :func:`write_jsonl` (atomic; scrubs project-name strings from
    ``meta.target_raw`` through :func:`scrub_meta`).
Tokens and rendering
    :func:`get_tokenizer`, :func:`count_tokens`, :func:`count_tokens_batch`, :func:`load_chat_template`,
    :func:`render_chat`, :func:`check_render`, :func:`split_prompt_completion`, :func:`tokens_of_row`.
Identity
    :func:`row_id`, :func:`fingerprint`, :func:`new_meta`, :func:`validate_row`.

The tokenizer and template locations come from :func:`paths.load_paths` (``tokenizer_json``,
``chat_template``) unless a path is passed explicitly.
"""
from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypedDict, Union

import orjson

from src.finetuning.text27b import paths as _paths

# ------------------------------------------------------------------------------------ constants
BUILD = "v0_2"
#: Families built from PGP material (meta.family == meta.task).
PGP_FAMILIES: Tuple[str, ...] = ("translate_en", "translate_he", "catalogue_json", "describe_pgp",
                                 "extract_relations", "analyse_document")
#: Families built from the academic literature.
LITERATURE_FAMILIES: Tuple[str, ...] = ("domain_reading", "rag_context")
#: Canon-comparison families (:mod:`canon_diff`): KTIV human transcriptions aligned to Sefaria canonical texts; the
#: targets are computed by deterministic alignment (no model output).
CANON_FAMILIES: Tuple[str, ...] = ("canon_identify", "canon_diff")
FAMILIES: Tuple[str, ...] = PGP_FAMILIES + LITERATURE_FAMILIES + CANON_FAMILIES
CONDITIONS: Tuple[str, ...] = ("edition", "noised_edition", "read", "read_hard", "read_partial",
                               "edition_rescript", "literature")
#: Conditions whose input is (or imitates) a machine read; they share one prompt note.
READ_LIKE_CONDITIONS: Tuple[str, ...] = ("noised_edition", "read", "read_hard", "read_partial")
SPLITS: Tuple[str, ...] = ("train", "dev", "test_indist", "heldout_vision", "heldout", "probe")
TRAIN_SPLITS: Tuple[str, ...] = ("train",)
EVAL_SPLITS: Tuple[str, ...] = ("dev", "test_indist", "heldout_vision", "heldout", "probe")
EXPOSURE_LEVELS: Tuple[str, ...] = ("page", "pgpid", "fragment", "none")
LICENCE_CLASSES: Tuple[str, ...] = ("pd", "cc-by", "cc-by-nc", "copyright", "unknown")
#: Meta fields every v0_1 row carried (kept in v0_2).
V01_META_FIELDS: Tuple[str, ...] = ("task", "lang_class", "condition", "pgpid", "shelfmark", "shelfmark_group",
                                    "split", "held_out_vision_benchmark", "target_chars", "target_tokens",
                                    "tokens", "input_cer", "input_source", "input_truncated", "licence",
                                    "provenance", "source")
#: The empty think block the template writes before every assistant answer with thinking off.
EMPTY_THINK = "<think>\n\n</think>\n\n"
#: Start of the sentence the template injects into the system turn when thinking is on.
EFFORT_MARKER = "Reasoning effort"
_IM_END = "<|im_end|>\n"
_ASSISTANT_OPEN = "<|im_start|>assistant\n"

# -------------------------------------------------------------------------------------- prompts
SYSTEM = ("You are an expert philologist and historian of the Cairo Genizah, fluent in Judaeo-Arabic, "
          "Arabic, Hebrew and Aramaic documents. Follow the instruction exactly and do not invent "
          "facts the document or the supplied context does not support.")
SCRIPT_NAME: Dict[str, str] = {"hebrew": "Hebrew script", "arabic": "Arabic script", "mixed": "mixed script"}
INSTRUCTION: Dict[str, str] = {
    "translate_en": "Translate this Genizah document ({script}) into English. Follow the line order of the "
                    "original, keep side and margin labels, and mark lost text with [...].",
    "translate_he": "Translate this Genizah document ({script}) into Modern Hebrew. Follow the line order of "
                    "the original, keep side and margin labels, and mark lost text with [...].",
    "catalogue_json": "Catalogue this Genizah document ({script}). Return only a JSON object with the keys "
                      "document_type, languages, date, origin, destination, places_mentioned, people, tags "
                      "and summary. Use null or [] for anything the cataloguer could not establish.",
    "describe_pgp": "Write a catalogue description of this Genizah document ({script}) in the style of the "
                    "Princeton Geniza Project: what kind of document it is, who wrote to whom or who the "
                    "parties are, where and when if known, and what it is about.",
    "extract_relations": "Extract the people and places this Genizah document ({script}) is linked to. Return "
                         "only a JSON object {{\"relations\": [{{\"subject\", \"predicate\", \"object\"}}]}}. "
                         "Use THIS_DOCUMENT as the subject of document-level relations. Predicates: origin, "
                         "destination, mentions_place, possibly_mentions_place, sender, recipient, "
                         "mentions_person, has_social_role.",
    "analyse_document": "Write the scholarly introduction and commentary for this Genizah document ({script}) "
                        "as an editor would in a published edition: what the document is, who the people "
                        "are, where and when it was written, its context and significance, and notes on "
                        "difficult words and readings.",
}
#: v0_1 condition notes, verbatim (``noised_edition`` deliberately shares the ``read`` note).
COND_NOTE_V0: Dict[str, str] = {
    "edition": "The text below is a scholarly edition. Square brackets mark restorations or lacunae; "
               "lines starting with § are side or margin labels.",
    "read": "The text below is an automatic transcription made from photographs of the manuscript. It "
            "contains reading errors (wrong, missing or merged letters) and may run sides together; "
            "work from what the document most plausibly says.",
}
COND_NOTE_V0["noised_edition"] = COND_NOTE_V0["read"]
#: The v0_2 read note (plan §9.1): one text for every read-like condition, since serving cannot
#: tell a good read from a failing one; it now also warns of skipped lines and neighbouring text.
READ_NOTE_V2 = ("The text below is an automatic transcription made from photographs of the manuscript. It "
                "contains reading errors (wrong, missing or merged letters), may run sides together, skip "
                "lines or include text from a neighbouring document; work from what the document most "
                "plausibly says.")
#: v0_2 condition notes. ``edition_rescript`` (B9: an Arabic-script edition re-written in Hebrew
#: letters) uses the edition note so the model treats it like a Judaeo-Arabic edition.
COND_NOTE: Dict[str, str] = {"edition": COND_NOTE_V0["edition"], "edition_rescript": COND_NOTE_V0["edition"]}
COND_NOTE.update({c: READ_NOTE_V2 for c in READ_LIKE_CONDITIONS})


# ----------------------------------------------------------------------------------------- types
class Message(TypedDict):
    """One chat turn."""

    role: str
    content: str


class Meta(TypedDict, total=False):
    """Row metadata (v0_1 fields plus the v0_2 additions of the build brief, rule 5)."""

    # v0_1
    task: str
    lang_class: str
    condition: str
    pgpid: Optional[str]
    shelfmark: Optional[str]
    shelfmark_group: Optional[str]
    split: str
    held_out_vision_benchmark: bool
    target_chars: int
    target_tokens: int
    tokens: int
    input_cer: Optional[float]
    input_source: Optional[str]
    input_truncated: bool
    licence: Optional[str]
    provenance: Optional[str]
    source: Optional[str]
    edition_available: Optional[bool]
    read_model: Optional[str]
    read_coverage: Optional[float]
    read_agreed_share: Optional[float]
    # v0_2
    family: str
    noise_band: Optional[str]
    noise_class: Optional[str]
    reader_exposed: Optional[bool]
    exposure_level: Optional[str]
    target_raw: Optional[str]
    strip_ops: List[str]
    pgp_description_dup: bool
    lit_sources: List[Dict[str, Any]]
    work: Optional[str]
    page_range: Optional[str]
    licence_class: Optional[str]
    twin_row_id: Optional[str]
    build: str
    row_id: str


class Row(TypedDict):
    """A training / eval row."""

    messages: List[Message]
    meta: Meta


class RenderError(ValueError):
    """The rendered chat string breaks the render contract (think block, effort sentence, suffix)."""


# ------------------------------------------------------------------------------------- prompts
def user_prompt(family: str, condition: str, script: str, text: str, extra: str = "",
                notes: Optional[Dict[str, str]] = None) -> str:
    """User message for a PGP family / condition (the v0 builder's layout).

    :param family: Task family (a key of :data:`INSTRUCTION`).
    :param condition: Input condition (a key of ``notes``).
    :param script: Script of the input (``hebrew`` / ``arabic`` / ``mixed``; unknown → Hebrew script).
    :param text: Document text placed inside ``<document>`` tags.
    :param extra: Additional blocks (context, translation) placed after the document.
    :param notes: Condition-note mapping; defaults to the v0_2 :data:`COND_NOTE`.
    :returns: The prompt.
    :raises KeyError: On an unknown family or condition.
    """
    notes = COND_NOTE if notes is None else notes
    head = INSTRUCTION[family].format(script=SCRIPT_NAME.get(script, "Hebrew script"))
    body = f"{head}\n\n{notes[condition]}\n\n<document>\n{text}\n</document>"
    return body + (f"\n\n{extra}" if extra else "")


def nfc(text: str) -> str:
    """NFC normal form of a text (the form the tokenizer normalises to).

    :param text: Any text.
    :returns: The NFC-normalised text.
    """
    return unicodedata.normalize("NFC", text)


def make_messages(user: str, assistant: str, system: str = SYSTEM) -> List[Message]:
    """Build the three-turn message list (contents stripped, as the template trims them, and NFC-normalised).

    Every family builder creates its rows here, so this is the one place message text is normalised (F3): token
    counts computed from these messages are the counts of the NFC text the harness tokenizes.

    :param user: User turn.
    :param assistant: Assistant target.
    :param system: System turn.
    :returns: ``[system, user, assistant]`` messages.
    """
    return [{"role": "system", "content": nfc(system.strip())}, {"role": "user", "content": nfc(user.strip())},
            {"role": "assistant", "content": nfc(assistant.strip())}]


def non_nfc_roles(row: Dict[str, Any]) -> List[str]:
    """Roles of a row whose message content is not in NFC (the NFC gate of the mixture, re-scan and Hub checks), plus
    ``meta.target_raw`` when the pre-strip target is not NFC (F12).

    :param row: Chat row.
    :returns: Roles with non-NFC content (empty = the row passes).
    """
    out = [str(m.get("role")) for m in row.get("messages") or []
           if isinstance(m.get("content"), str) and not unicodedata.is_normalized("NFC", m["content"])]
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    raw = meta.get("target_raw")
    if isinstance(raw, str) and not unicodedata.is_normalized("NFC", raw):
        out.append("meta.target_raw")
    return out


# ------------------------------------------------------------------------------------------- I/O
def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Stream the objects of a JSONL file (blank lines skipped).

    :param path: File path.
    :returns: Iterator over parsed rows.
    """
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a whole JSONL file.

    :param path: File path.
    :returns: Parsed rows.
    """
    return list(read_jsonl(path))


def _scrub_value(value: Any, strip: Any) -> Any:
    """Apply a string scrub to every string inside a JSON value.

    :param value: JSON value.
    :param strip: ``str -> str`` scrub.
    :returns: The scrubbed copy.
    """
    if isinstance(value, str):
        return strip(value)
    if isinstance(value, list):
        return [_scrub_value(v, strip) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_value(v, strip) for k, v in value.items()}
    return value


def scrub_meta(row: Any) -> Any:
    """Remove project-name strings from a row's ``meta.target_raw`` (F4: no such string anywhere in a row, meta included)
    and NFC-normalise it (F12: the pre-strip target is NFC like the messages).

    The pre-strip target is kept for the audit, but the strings the strips removed from the target are removed
    from it too (:func:`textnorm.strip_source_names`). A JSON ``target_raw`` (catalogue / relations targets) is scrubbed
    string by string and re-serialised, so it stays valid JSON. Rows without a string ``target_raw`` are returned
    unchanged.

    :param row: A row (any other JSON object passes through).
    :returns: The same object, its ``meta.target_raw`` scrubbed in place when needed.
    """
    meta = row.get("meta") if isinstance(row, dict) else None
    raw = meta.get("target_raw") if isinstance(meta, dict) else None
    if isinstance(raw, str) and raw:
        from src.finetuning.text27b import textnorm as _tn  # sibling module; imported lazily (it loads Levenshtein)
        if not unicodedata.is_normalized("NFC", raw):
            raw = meta["target_raw"] = unicodedata.normalize("NFC", raw)
        if not _tn.has_source_names(raw):
            return row
        strip = (lambda t: _tn.strip_source_names(t)[0])
        obj = None
        if raw.lstrip()[:1] in ("{", "["):
            try:
                obj = orjson.loads(raw)
            except orjson.JSONDecodeError:
                obj = None
        if isinstance(obj, (dict, list)):
            meta["target_raw"] = orjson.dumps(_scrub_value(obj, strip)).decode("utf-8")
        else:
            meta["target_raw"] = strip(raw)
    return row


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows to a JSONL file atomically (temporary file in the same directory, then rename).

    Every row goes through :func:`scrub_meta` (project-name strings removed from ``meta.target_raw``, which is also
    NFC-normalised).

    :param path: Destination file (parent directories are created).
    :param rows: Rows to write (any iterable, consumed once).
    :returns: Number of rows written.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    n = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            for row in rows:
                fh.write(orjson.dumps(scrub_meta(row)))
                fh.write(b"\n")
                n += 1
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return n


# -------------------------------------------------------------------------------------- tokens
def _default_path(key: str) -> Path:
    """Look up a path entry in the build configuration.

    :param key: ``tokenizer_json`` or ``chat_template``.
    :returns: The configured path.
    """
    return _paths.load_paths().get(key)


@functools.lru_cache(maxsize=4)
def _tokenizer_at(path: str) -> Any:
    """Load a ``tokenizers`` tokenizer (cached per path).

    :param path: ``tokenizer.json`` path.
    :returns: A :class:`tokenizers.Tokenizer`.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def get_tokenizer(path: Optional[Union[str, Path]] = None) -> Any:
    """Return the Qwen3.8 tokenizer (local ``tokenizer.json``).

    :param path: Explicit ``tokenizer.json``; defaults to ``paths.tokenizer_json``.
    :returns: A :class:`tokenizers.Tokenizer`.
    :raises FileNotFoundError: When the file does not exist.
    """
    p = Path(path) if path is not None else _default_path("tokenizer_json")
    if not p.exists():
        raise FileNotFoundError(f"tokenizer not found: {p}")
    return _tokenizer_at(str(p))


def count_tokens(text: str, tokenizer_path: Optional[Union[str, Path]] = None) -> int:
    """Exact token count of ``text`` (no special tokens added; special-token strings in the text count as one).

    :param text: Any text.
    :param tokenizer_path: Explicit ``tokenizer.json``; defaults to ``paths.tokenizer_json``.
    :returns: Number of tokens.
    """
    if not text:
        return 0
    return len(get_tokenizer(tokenizer_path).encode(text, add_special_tokens=False).ids)


def count_tokens_batch(texts: Sequence[str], tokenizer_path: Optional[Union[str, Path]] = None) -> List[int]:
    """Exact token counts of many texts in one batched call (same numbers as :func:`count_tokens`).

    :param texts: Texts.
    :param tokenizer_path: Explicit ``tokenizer.json``; defaults to ``paths.tokenizer_json``.
    :returns: Token counts in input order.
    """
    if not texts:
        return []
    enc = get_tokenizer(tokenizer_path).encode_batch(list(texts), add_special_tokens=False)
    return [len(e.ids) for e in enc]


# ------------------------------------------------------------------------------------ template
def _raise_exception(message: str) -> None:
    """Template helper mirroring ``transformers``' ``raise_exception``.

    :param message: Error message from the template.
    :returns: Never returns.
    :raises RenderError: Always.
    """
    raise RenderError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: Optional[int] = None, separators: Any = None,
            sort_keys: bool = False) -> str:
    """JSON filter with the ``transformers`` chat-template signature.

    :param value: Value to serialise.
    :param ensure_ascii: Escape non-ASCII characters.
    :param indent: Indentation.
    :param separators: Item / key separators.
    :param sort_keys: Sort object keys.
    :returns: JSON text.
    """
    import json

    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


@functools.lru_cache(maxsize=4)
def _template_at(path: str, mtime_ns: int) -> Any:
    """Compile a chat template the way ``transformers`` does (sandboxed, trim/lstrip blocks).

    :param path: ``chat_template.jinja`` path.
    :param mtime_ns: Modification time (cache key).
    :returns: A compiled :class:`jinja2.Template`.
    """
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    with open(path, encoding="utf-8") as fh:
        return env.from_string(fh.read())


def load_chat_template(path: Optional[Union[str, Path]] = None) -> Any:
    """Return the compiled local Qwen3.8 chat template.

    :param path: Explicit ``chat_template.jinja``; defaults to ``paths.chat_template``.
    :returns: A compiled :class:`jinja2.Template`.
    :raises FileNotFoundError: When the file does not exist.
    """
    p = Path(path) if path is not None else _default_path("chat_template")
    if not p.exists():
        raise FileNotFoundError(f"chat template not found: {p}")
    return _template_at(str(p), p.stat().st_mtime_ns)


def check_render(rendered: str, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool = False) -> None:
    """Assert the render contract for ``enable_thinking=False``.

    * no "Reasoning effort" sentence anywhere;
    * exactly one ``<think>\\n\\n</think>\\n\\n`` per assistant turn (plus one for the generation
      prompt) and no other ``<think>`` tag;
    * a row ending in an assistant turn renders to ``... <think>\\n\\n</think>\\n\\n{content}<|im_end|>\\n``
      with that empty block the only one before the content; a generation prompt ends with
      ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.

    :param rendered: Output of the template.
    :param messages: The messages that were rendered.
    :param add_generation_prompt: Whether a generation prompt was requested.
    :returns: None.
    :raises RenderError: On any violation.
    """
    if EFFORT_MARKER in rendered:
        raise RenderError("rendered chat contains the reasoning-effort sentence")
    n_assist = sum(1 for m in messages if m.get("role") == "assistant")
    want = n_assist + (1 if add_generation_prompt else 0)
    if rendered.count(EMPTY_THINK) != want or rendered.count("<think>") != want:
        raise RenderError(f"expected {want} empty think block(s), found {rendered.count(EMPTY_THINK)} "
                          f"empty / {rendered.count('<think>')} total")
    if add_generation_prompt:
        if not rendered.endswith(_ASSISTANT_OPEN + EMPTY_THINK):
            raise RenderError("generation prompt does not end with the empty think block")
        return
    last = messages[-1] if messages else {}
    if last.get("role") == "assistant":
        content = str(last.get("content") or "").strip()
        tail = EMPTY_THINK + content + _IM_END
        if not rendered.endswith(tail):
            raise RenderError("rendered row does not end with <empty think> + assistant content")


def render_chat(messages: Sequence[Dict[str, Any]], enable_thinking: bool = False, add_generation_prompt: bool = False,
                template_path: Optional[Union[str, Path]] = None, check: bool = True,
                reasoning_effort: Optional[str] = None) -> str:
    """Render messages with the local chat template.

    :param messages: Chat turns (``role`` / ``content``).
    :param enable_thinking: Template flag; training rows use False.
    :param add_generation_prompt: Append the assistant header (for prompts without a target).
    :param template_path: Explicit ``chat_template.jinja``; defaults to ``paths.chat_template``.
    :param check: Run :func:`check_render` (only meaningful with ``enable_thinking=False``).
    :param reasoning_effort: Template variable for thinking renders (``low`` / ``medium`` / ``xhigh``; None leaves the
        template default). Only passed when given, so training renders are unchanged.
    :returns: The rendered string.
    :raises RenderError: When ``check`` is on and the contract is broken.
    """
    template = load_chat_template(template_path)
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}
    rendered = template.render(messages=list(messages), enable_thinking=enable_thinking,
                               add_generation_prompt=add_generation_prompt, tools=None, **extra)
    if check and not enable_thinking:
        check_render(rendered, messages, add_generation_prompt)
    return rendered


def split_prompt_completion(messages: Sequence[Dict[str, Any]],
                            template_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """Split a rendered row into the masked prompt and the supervised completion.

    The prompt is everything up to and including the empty think block; the completion is the
    assistant content plus ``<|im_end|>\\n``. ``prompt + completion`` equals :func:`render_chat`.

    :param messages: Chat turns ending in the assistant target.
    :param template_path: Explicit ``chat_template.jinja``.
    :returns: ``(prompt, completion)``.
    :raises RenderError: When the last turn is not an assistant turn or the split does not reproduce the render.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise RenderError("last message must be the assistant target")
    full = render_chat(messages, template_path=template_path)
    completion = str(messages[-1].get("content") or "").strip() + _IM_END
    prompt = full[: len(full) - len(completion)]
    if not prompt.endswith(_ASSISTANT_OPEN + EMPTY_THINK):
        raise RenderError("prompt part does not end with the assistant header and empty think block")
    return prompt, completion


def tokens_of_row(row: Dict[str, Any], tokenizer_path: Optional[Union[str, Path]] = None,
                  template_path: Optional[Union[str, Path]] = None) -> int:
    """Exact token length of a rendered row (``enable_thinking=False``).

    :param row: A row with ``messages``.
    :param tokenizer_path: Explicit ``tokenizer.json``.
    :param template_path: Explicit ``chat_template.jinja``.
    :returns: Number of tokens of the full rendered conversation.
    """
    return count_tokens(render_chat(row["messages"], template_path=template_path), tokenizer_path)


# ------------------------------------------------------------------------------------ identity
def row_id(family: str, key: Any, condition: str, read_model: Optional[str] = None, *extra: Any) -> str:
    """Stable row id: sha1 of ``family|key|condition|read_model[|extra...]``, first 16 hex characters.

    :param family: Row family.
    :param key: pgpid for PGP families, page key for literature rows.
    :param condition: Input condition.
    :param read_model: Reader of the input (None / '' for non-read conditions).
    :param extra: Optional further discriminators (e.g. a distractor flag); omitted when not given.
    :returns: 16-character hex id.
    """
    parts = [str(family), "" if key is None else str(key), str(condition), read_model or ""]
    parts.extend(str(e) for e in extra)
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


#: Default fingerprint columns: the three message contents.
DEFAULT_FINGERPRINT_KEYS: Tuple[str, ...] = ("system", "user", "assistant")


def _column(row: Dict[str, Any], key: str) -> str:
    """Text of one fingerprint column of a row.

    :param row: Row.
    :param key: A role name (``system`` / ``user`` / ``assistant``) or ``meta.<field>``.
    :returns: The column text ('' when absent; non-strings JSON-encoded).
    """
    if key.startswith("meta."):
        value = (row.get("meta") or {}).get(key[5:])
    else:
        value = next((m.get("content") for m in row.get("messages") or [] if m.get("role") == key), None)
    if value is None:
        return ""
    return value if isinstance(value, str) else orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def fingerprint(rows: Iterable[Dict[str, Any]], keys: Sequence[str] = DEFAULT_FINGERPRINT_KEYS) -> str:
    """Order-sensitive sha256 over the text columns of rows.

    :param rows: Rows in their materialised order.
    :param keys: Columns: role names and/or ``meta.<field>``.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    for row in rows:
        for k in keys:
            h.update(k.encode("utf-8"))
            h.update(b"\x1f")
            h.update(_column(row, k).encode("utf-8"))
            h.update(b"\x1e")
        h.update(b"\x1d")
    return h.hexdigest()


def new_meta(family: str, condition: str, split: str, key: Any, read_model: Optional[str] = None,
             **fields: Any) -> Meta:
    """Start a v0_2 meta block with the identity fields filled in.

    Sets ``family``, ``task`` (= family unless given), ``condition``, ``split``, ``read_model``,
    ``build`` and ``row_id`` (from family, key, condition, read model); other fields pass through.

    :param family: Row family.
    :param condition: Input condition.
    :param split: Split name.
    :param key: pgpid or page key used in the row id.
    :param read_model: Reader of the input, if any.
    :param fields: Further meta fields.
    :returns: The meta dict.
    """
    meta: Dict[str, Any] = {"family": family, "task": fields.pop("task", family), "condition": condition,
                            "split": split, "read_model": read_model, "build": BUILD,
                            "row_id": row_id(family, key, condition, read_model)}
    meta.update(fields)
    return meta  # type: ignore[return-value]


def validate_row(row: Dict[str, Any], require_v01: bool = True) -> List[str]:
    """Check a row against the v0_2 schema.

    :param row: Row to check.
    :param require_v01: Also require every v0_1 meta field (:data:`V01_META_FIELDS`).
    :returns: Problems found (empty when the row is valid).
    """
    problems: List[str] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        problems.append("messages must be [system, user, assistant]")
    else:
        for m in msgs:
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                problems.append(f"empty {m.get('role')} content")
            elif m["content"] != m["content"].strip():
                problems.append(f"{m.get('role')} content has surrounding whitespace (the template trims it)")
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return problems + ["meta missing"]
    for key, allowed in (("family", FAMILIES), ("condition", CONDITIONS), ("split", SPLITS)):
        if meta.get(key) not in allowed:
            problems.append(f"meta.{key}={meta.get(key)!r} not in {allowed}")
    if meta.get("exposure_level") is not None and meta["exposure_level"] not in EXPOSURE_LEVELS:
        problems.append(f"meta.exposure_level={meta['exposure_level']!r}")
    if meta.get("licence_class") is not None and meta["licence_class"] not in LICENCE_CLASSES:
        problems.append(f"meta.licence_class={meta['licence_class']!r}")
    if meta.get("build") != BUILD:
        problems.append(f"meta.build={meta.get('build')!r} != {BUILD!r}")
    rid = meta.get("row_id")
    if not (isinstance(rid, str) and len(rid) == 16 and all(c in "0123456789abcdef" for c in rid)):
        problems.append("meta.row_id must be 16 hex characters")
    if require_v01:
        missing = [k for k in V01_META_FIELDS if k not in meta]
        if missing:
            problems.append(f"missing v0_1 meta fields {missing}")
    return problems
