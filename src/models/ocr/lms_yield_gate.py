"""Yield to prod chat: hold this repo's LM Studio / GPU batch work while a production chat turn is active.

The Mac Studio's LM Studio serves production chat (api.cairogenizah.ai) with three models: the router
``qwen/qwen3-4b-2507``, synthesis ``qwen/qwen3.8-27b`` and verifier ``qwen/qwen3.6-35b-a3b``. This repo's VLM
batches, benchmarks and LLM jobs share that GPU with no priority, which made every chat call 2.4-3.2x slower
(genizah_search RAG audit, 2026-10). Before each unit of GPU work (one page, one LLM call, one eval row, one stage)
a hooked call site asks this gate whether a prod chat is active, by reading ``lms ps --json`` (local, read-only,
about 0.2 s, cached in-process for 2 s):

* a chat model is **busy** when its ``status`` is not ``idle`` or ``queued > 0``;
* a chat model is **recent** when ``lastUsedTime`` (epoch ms; null while a request is in flight, set to the
  request's end when the model returns to idle) is younger than ``LMS_YIELD_RECENT_S`` (180 s bridges the gaps
  between the router, retrieval, synthesis, verify and repair steps of one chat turn);
* signals caused by this repo's own chat-model calls are discounted through the own-call registry
  (:func:`own_call`), so sibling jobs never yield to each other;
* non-chat models (the VLM checkpoints) never hold.

Modes, from ``LMS_YIELD_MODE`` (process environment first, then the repo ``.env``, default ``dryrun``):

* ``off``: return at once. No ``lms ps``, no log line, no own-call markers.
* ``dryrun``: evaluate once per unit, log what the gate would have done, never wait.
* ``on``: while a chat is active, poll every ``LMS_YIELD_POLL_S`` (3 s) for at most ``LMS_YIELD_MAX_WAIT_S``
  (900 s; must stay below the shortest VLM TTL, 3,600 s today, so values >= 3000 are rejected), then proceed with
  that one unit. The next unit re-checks.

The gate fails open and never raises into its caller: a failed ``lms ps`` (missing CLI, non-zero exit, timeout,
bad JSON), an unreadable registry or an unwritable log lets the unit proceed. Three try/excepts implement this
(:func:`read_lms_ps`, :func:`_guarded_io` for the registry, ``.env`` and summary files, :func:`_append_jsonl`);
everything else is non-raising by construction (regex-validated config, string-only host checks, no DNS). A
``base_url`` that is not this machine (the AMD workstation, say) logs ``skip_remote`` once per host and never holds.

Files, under ``LMS_YIELD_DIR`` (default ``<repo>/logs/lms_yield``, gitignored):

* ``yield_gate.jsonl``: one record per gate call, ``ts, pid, argv0, site, mode, decision
  (clear|hold|error|skip_remote), reasons, ignored_own, error, waited_s, polls, capped, would_wait_s_est, ps_ms``
  (plus ``host`` for ``skip_remote``). ``ts`` is the start of the gate call; ``argv0`` is the basename of
  ``sys.argv[0]``. Each line is written by one ``write()`` to an ``O_APPEND`` file, so several processes can append.
* ``own_calls.jsonl``: one record per own chat-model call, ``ts, pid, argv0, model, start_ms, end_ms``. It is the
  permanent record of which LM Studio requests were this repo's own (the markers below are pruned).
* ``own_calls/``: one live marker per own call, kept until ``recent_s + 60 s`` after the call ends.

CLI (from the repo root)::

    .venv/bin/python -m src.models.ocr.lms_yield_gate --check             # exit 0 clear, 3 hold (diagnostic, any mode)
    .venv/bin/python -m src.models.ocr.lms_yield_gate --wait --site NAME  # gate one shell stage; honours the mode
    .venv/bin/python -m src.models.ocr.lms_yield_gate --summarize [--since 2026-10-11T00:00]

Known limitations:

1. An own call on a chat model masks prod on that same model. While any sibling process has an in-flight own call
   on model M, a prod request on M cannot be told apart from it: busy is ignored for M, and ``lastUsedTime`` is null
   while M is in flight. So a sibling 35B judge, KG or bibliography call masks the prod verify, ``describe_from_reads``
   on the 27B masks prod synthesis, and coreference batches on the 4B mask the prod router. The prod turn's signals
   on the other chat models still hold the gate, the 180 s recency covers the gaps, and the sibling's next unit sees
   the chat's ``lastUsedTime`` when it differs from its own ``end_ms`` by more than 5 s. The exact fix is a prod-side
   ``/chat/activity`` lease (v2); :func:`chat_active` is the single seam for it.
2. A VLM page already in flight when a chat starts is not interrupted (p50 29 s, p90 55 s); only the next unit
   waits. Retries inside one ``transcribe_with_lm_studio`` call are not re-gated (one check per call).
3. The first chat signal is the 4B router call, so the turn's retrieval before it is not covered.
4. Anything that touches a chat model counts as chat: Isaac's own tests, prod eval runs, a watchdog sending
   inference. The dry-run review shows whether this causes false holds.
5. ``own_call`` is a no-op in ``off`` mode, so a chat-model job started with ``LMS_YIELD_MODE=off`` while other
   processes run ``on`` looks like prod chat to them. To stop one job from yielding, remove its gate call and keep
   its ``own_call`` wrapper instead.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (Any, Awaitable, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set,
                    Tuple, TypeVar)

from dotenv import dotenv_values

# ============================================================================
# Constants
# ============================================================================

CHAT_MODELS: Tuple[str, ...] = ("qwen/qwen3-4b-2507", "qwen/qwen3.8-27b", "qwen/qwen3.6-35b-a3b")
"""Prod chat models (router, synthesis, verifier) served by the Studio's LM Studio."""

MODES: Tuple[str, ...] = ("off", "dryrun", "on")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_STATE_DIR = REPO_ROOT / "logs" / "lms_yield"
DEFAULT_LMS_CLI = "~/.lmstudio/bin/lms"

GATE_LOG = "yield_gate.jsonl"
OWN_CALLS_LOG = "own_calls.jsonl"
OWN_CALLS_DIR = "own_calls"

PS_CACHE_S = 2.0
"""Seconds an ``lms ps`` answer is reused within one process (shorter than the 3 s poll)."""

MARKER_GRACE_S = 60.0
"""A finished own-call marker is kept for ``recent_s`` plus this, so the ``lastUsedTime`` match still works."""

FUTURE_TOL_MS = 60_000
"""A ``lastUsedTime`` further than this in the future is implausible and ignored (fail open)."""

SPAN_GAP_S = 300.0
"""``--summarize``: records of one pid further apart than this are separate spans and do not count as active."""

MAX_PID = 2 ** 31 - 1

_NUMBER_RE = re.compile(r"\d+(\.\d+)?")
_HOST_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://)?(?:[^@/]*@)?(\[[^\]/]*\]|[^:/?#]*)")
_SLUG_RE = re.compile(r"[^A-Za-z0-9.-]+")
_CHAT_BY_TAIL: Dict[str, str] = {model.rsplit("/", 1)[-1]: model for model in CHAT_MODELS}

# Numeric settings: env key -> (YieldConfig field, default, lowest, highest accepted value).
_NUMERIC_SETTINGS: Dict[str, Tuple[str, float, float, float]] = {
    "LMS_YIELD_RECENT_S": ("recent_s", 180.0, 10.0, 900.0),
    "LMS_YIELD_POLL_S": ("poll_s", 3.0, 1.0, 60.0),
    "LMS_YIELD_MAX_WAIT_S": ("max_wait_s", 900.0, 0.0, 2999.0),
}
_ENV_KEYS: Tuple[str, ...] = ("LMS_YIELD_MODE", "LMS_YIELD_DIR", "LMS_CLI", *_NUMERIC_SETTINGS)

T = TypeVar("T")
PsReader = Callable[[str, float], Optional[List[Dict[str, Any]]]]


# ============================================================================
# Data types
# ============================================================================

@dataclass(frozen=True)
class YieldConfig:
    """Gate settings; one per process (see :func:`get_config`).

    :ivar mode: ``off``, ``dryrun`` or ``on`` (env ``LMS_YIELD_MODE``).
    :ivar recent_s: A chat model used this recently counts as active (env ``LMS_YIELD_RECENT_S``, 10-900).
    :ivar poll_s: Seconds between polls in ``on`` mode (env ``LMS_YIELD_POLL_S``, 1-60).
    :ivar max_wait_s: Cap on one wait; then the unit proceeds (env ``LMS_YIELD_MAX_WAIT_S``, 0-2999, below the VLM TTL).
    :ivar own_end_tol_ms: A ``lastUsedTime`` within this of an own call's end is that own call.
    :ivar ps_timeout_s: Timeout of one ``lms ps --json`` run.
    :ivar lms_cli: Path of the ``lms`` CLI (env ``LMS_CLI``).
    :ivar state_dir: Directory of the logs and the own-call registry (env ``LMS_YIELD_DIR``).
    """

    mode: str = "dryrun"
    recent_s: float = 180.0
    poll_s: float = 3.0
    max_wait_s: float = 900.0
    own_end_tol_ms: int = 5000
    ps_timeout_s: float = 10.0
    lms_cli: str = os.path.expanduser(DEFAULT_LMS_CLI)
    state_dir: Path = DEFAULT_STATE_DIR


@dataclass(frozen=True)
class GateDecision:
    """Whether a prod chat is active right now.

    :ivar hold: True when this unit should yield.
    :ivar reasons: Chat signals that hold, e.g. ``("qwen/qwen3.8-27b:generating", "qwen/qwen3-4b-2507:recent 42s")``.
    :ivar ignored_own: Chat-model signals discarded because they are this repo's own calls.
    :ivar error: Set when ``lms ps`` failed; ``hold`` is then False (fail open).
    :ivar would_wait_s_est: Lower bound of the wait when holding: ``recent_s - age`` of the youngest recency signal,
        or ``recent_s`` when a chat model is still busy; None when not holding.
    """

    hold: bool
    reasons: Tuple[str, ...] = ()
    ignored_own: Tuple[str, ...] = ()
    error: Optional[str] = None
    would_wait_s_est: Optional[float] = None


@dataclass(frozen=True)
class OwnCall:
    """One of this repo's own chat-model calls, as read back from the registry.

    :ivar pid: Process that made the call.
    :ivar model: Canonical chat-model id (one of :data:`CHAT_MODELS`).
    :ivar start_ms: Call start, epoch ms.
    :ivar end_ms: Call end, epoch ms; None while the call is in flight.
    :ivar argv0: Basename of the calling process's ``sys.argv[0]``.
    """

    pid: int
    model: str
    start_ms: int
    end_ms: Optional[int] = None
    argv0: str = ""


@dataclass(frozen=True)
class Span:
    """A stretch of time one process would have waited (dryrun) or did wait (``on``), for ``--summarize``.

    :ivar pid: Process id.
    :ivar start: Span start, epoch seconds.
    :ivar end: Span end, epoch seconds.
    :ivar holds: Number of hold records in the span.
    :ivar reasons: Reasons of the first hold record.
    :ivar actual: True for a real ``on``-mode wait, False for a dryrun would-wait.
    """

    pid: int
    start: float
    end: float
    holds: int
    reasons: Tuple[str, ...]
    actual: bool = False

    @property
    def seconds(self) -> float:
        """Span length.

        :returns: ``end - start``, never negative.
        :rtype: float
        """
        return max(0.0, self.end - self.start)


# ============================================================================
# Per-process state
# ============================================================================

_config: Optional[YieldConfig] = None
_ps_cache: Optional[Tuple[float, Optional[List[Dict[str, Any]]]]] = None
_warned: Set[str] = set()
_skip_logged: Set[str] = set()
_marker_counter = itertools.count()


def reset_caches() -> None:
    """Forget the cached config, the cached ``lms ps`` answer and the once-per-process warnings (for tests)."""
    global _config, _ps_cache
    _config = None
    _ps_cache = None
    _warned.clear()
    _skip_logged.clear()


def _warn_once(key: str, message: str) -> None:
    """Print ``message`` the first time ``key`` is seen in this process.

    :param key: Deduplication key.
    :param message: Line to print.
    """
    if key in _warned:
        return
    _warned.add(key)
    print(message, flush=True)


def _clear_ps_cache() -> None:
    """Drop the cached ``lms ps`` answer so the next gate call reads a fresh one."""
    global _ps_cache
    _ps_cache = None


# ============================================================================
# Guarded I/O (the three try/excepts)
# ============================================================================

def _guarded_io(op: Callable[[], T]) -> Tuple[Optional[T], Optional[Exception]]:
    """Run one local file or JSON operation of the registry, the ``.env`` read or the summarizer.

    A failure means "nothing known" to the caller (no own calls, no ``.env`` value, a skipped log line).

    :param op: Zero-argument callable doing the I/O.
    :returns: ``(result, None)`` on success, ``(None, error)`` on ``OSError`` or ``ValueError`` (which covers
        ``json.JSONDecodeError`` and ``UnicodeDecodeError``).
    :rtype: Tuple[Optional[T], Optional[Exception]]
    """
    try:
        return op(), None
    except (OSError, ValueError) as exc:
        return None, exc


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line with a single ``write()``; on failure print one warning per path and carry on.

    :param path: JSONL file (its directory is created when missing).
    :param record: JSON-serialisable record.
    """
    line = (json.dumps(record, separators=(",", ":"), default=str) + "\n").encode("ascii")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab", buffering=0) as handle:
            handle.write(line)
    except OSError as exc:
        _warn_once(f"log:{path}", f"    ✗ [lms_yield_gate] cannot append to {path} ({exc}); continuing without it")


def read_lms_ps(lms_cli: str, timeout_s: float) -> Optional[List[Dict[str, Any]]]:
    """Run ``lms ps --json`` (read-only) and return its entries, one per loaded model.

    This is the fail-open point: a missing CLI, a non-zero exit, a timeout, invalid JSON or a non-list answer all
    return None (one warning per failure kind per process), and the gate then lets the unit proceed.

    :param lms_cli: Path of the ``lms`` CLI.
    :param timeout_s: Seconds before the run is abandoned.
    :returns: The entries (dicts only), or None on any failure.
    :rtype: Optional[List[Dict[str, Any]]]
    """
    try:
        proc = subprocess.run([lms_cli, "ps", "--json"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, check=False)
        data = json.loads(proc.stdout) if proc.returncode == 0 else None
    except Exception as exc:  # noqa: BLE001 -- the fail-open point: any failure means "no answer", never a crash
        _warn_once(f"ps:{type(exc).__name__}",
                   f"    ✗ [lms_yield_gate] lms ps failed ({type(exc).__name__}: {exc}); proceeding (fail open)")
        return None
    if proc.returncode != 0 or not isinstance(data, list):
        problem = f"exit code {proc.returncode}" if proc.returncode != 0 else f"a {type(data).__name__}, not a list"
        _warn_once(f"ps:{problem}", f"    ✗ [lms_yield_gate] lms ps returned {problem}; proceeding (fail open)")
        return None
    return [entry for entry in data if isinstance(entry, dict)]


# ============================================================================
# Config
# ============================================================================

def _env_file_values(env_file: Path) -> Dict[str, str]:
    """Read the gate's own keys from a dotenv file without touching ``os.environ``.

    :param env_file: Path of the dotenv file.
    :returns: The ``LMS_YIELD_*`` / ``LMS_CLI`` values found; empty when the file is missing or unreadable.
    :rtype: Dict[str, str]
    """
    values, error = _guarded_io(lambda: dotenv_values(env_file))
    if error is not None or not values:
        return {}
    return {key: str(value) for key, value in values.items() if key in _ENV_KEYS and value is not None}


def _number_setting(key: str, raw: Optional[str], default: float, low: float, high: float) -> float:
    """Validate one numeric setting with a regex and bounds; a bad value gives the default and one warning.

    :param key: Env key (for the warning).
    :param raw: The configured text, or None when unset.
    :param default: Built-in default.
    :param low: Lowest accepted value.
    :param high: Highest accepted value.
    :returns: The accepted value or the default.
    :rtype: float
    """
    if raw is None:
        return default
    text = raw.strip()
    if _NUMBER_RE.fullmatch(text) is None or not low <= float(text) <= high:
        hint = " (must stay below the 3600 s VLM TTL)" if key == "LMS_YIELD_MAX_WAIT_S" else ""
        _warn_once(f"config:{key}", f"    ✗ [lms_yield_gate] {key}={raw!r} is not a number in "
                                    f"{low:g}-{high:g}{hint}; using {default:g}")
        return default
    return float(text)


def _mode_setting(raw: Optional[str]) -> str:
    """Validate ``LMS_YIELD_MODE``; anything but off/dryrun/on gives ``dryrun`` and one warning.

    :param raw: The configured text, or None when unset.
    :returns: ``off``, ``dryrun`` or ``on``.
    :rtype: str
    """
    if raw is None:
        return "dryrun"
    mode = raw.strip().lower()
    if mode in MODES:
        return mode
    _warn_once("config:LMS_YIELD_MODE", f"    ✗ [lms_yield_gate] LMS_YIELD_MODE={raw!r} is not off|dryrun|on; "
                                        "using dryrun")
    return "dryrun"


def _path_setting(key: str, raw: Optional[str], default: str) -> str:
    """Validate a path setting; an empty value gives the default and one warning.

    :param key: Env key (for the warning).
    :param raw: The configured text, or None when unset.
    :param default: Built-in default.
    :returns: The path text with ``~`` expanded.
    :rtype: str
    """
    text = default if raw is None else raw.strip()
    if not text:
        _warn_once(f"config:{key}", f"    ✗ [lms_yield_gate] {key} is empty; using {default}")
        text = default
    return os.path.expanduser(text)


def load_config(env: Optional[Mapping[str, str]] = None, env_file: Optional[Path] = None) -> YieldConfig:
    """Build the gate config: process environment first, then the repo ``.env``, then the defaults. Never raises.

    The ``.env`` file is read with ``dotenv_values`` (no ``load_dotenv``, ``os.environ`` is never changed), because
    most hooked entry points never load it themselves. Invalid values fall back to the default with one warning per
    key per process.

    :param env: Environment mapping; defaults to ``os.environ``.
    :param env_file: Dotenv file; defaults to ``<repo>/.env``.
    :returns: The validated config.
    :rtype: YieldConfig
    """
    env = os.environ if env is None else env
    file_values = _env_file_values(DEFAULT_ENV_FILE if env_file is None else Path(env_file))
    raw: Dict[str, Optional[str]] = {}
    for key in _ENV_KEYS:
        value = env.get(key)
        value = file_values.get(key) if value is None else value
        raw[key] = None if value is None else str(value)
    numbers = {field: _number_setting(key, raw[key], default, low, high)
               for key, (field, default, low, high) in _NUMERIC_SETTINGS.items()}
    state_dir = Path(_path_setting("LMS_YIELD_DIR", raw["LMS_YIELD_DIR"], str(DEFAULT_STATE_DIR)))
    if not state_dir.is_absolute():
        state_dir = REPO_ROOT / state_dir
    return YieldConfig(mode=_mode_setting(raw["LMS_YIELD_MODE"]),
                       lms_cli=_path_setting("LMS_CLI", raw["LMS_CLI"], DEFAULT_LMS_CLI),
                       state_dir=state_dir, **numbers)


def get_config() -> YieldConfig:
    """Return this process's config, loading it on first use (the ``.env`` file is read once per process).

    :returns: The cached config.
    :rtype: YieldConfig
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ============================================================================
# Pure helpers: models, hosts, numbers
# ============================================================================

def chat_model_id(model: Any) -> Optional[str]:
    """Return the canonical prod chat-model id for ``model``, with or without the ``qwen/`` publisher prefix.

    :param model: A model id as passed by callers or reported by ``lms ps``.
    :returns: One of :data:`CHAT_MODELS`, or None for any other model.
    :rtype: Optional[str]
    """
    if not isinstance(model, str):
        return None
    return _CHAT_BY_TAIL.get(model.strip().lower().rsplit("/", 1)[-1])


def is_chat_model(model: Any) -> bool:
    """Suffix-tolerant test for a prod chat model (``qwen3.6-35b-a3b`` == ``qwen/qwen3.6-35b-a3b``).

    :param model: Model id.
    :returns: True for one of the three prod chat models.
    :rtype: bool
    """
    return chat_model_id(model) is not None


def _host_of(base_url: Any) -> str:
    """Extract the lower-cased host of a URL or ``host:port`` string with one regex (no urllib, no DNS).

    :param base_url: URL text.
    :returns: The host without brackets, or an empty string.
    :rtype: str
    """
    match = _HOST_RE.match(str(base_url or "").strip())
    return (match.group(1) if match else "").strip("[]").lower()


def is_local_lms(base_url: Optional[str]) -> bool:
    """True when ``base_url`` points at this machine's LM Studio. String comparison only; never raises.

    None or empty counts as local (the callers' default is ``http://localhost:1234/v1``). The LAN IP deliberately
    does not count; such a URL logs ``skip_remote`` with its host.

    :param base_url: ``http://host:port/v1``, ``host:port``, None, ...
    :returns: Whether the gate applies to this URL.
    :rtype: bool
    """
    if not base_url:
        return True
    text = str(base_url).strip()
    if text == "::1":
        return True
    host = _host_of(text)
    name = socket.gethostname().lower()
    return bool(host) and host in {"localhost", "127.0.0.1", "::1", name, name + ".local", name.removesuffix(".local")}


def _as_number(value: Any) -> Optional[float]:
    """Return a finite JSON number as float, else None (bools, strings, huge ints and NaN are rejected).

    :param value: Value from an ``lms ps`` entry or a marker.
    :returns: The number or None.
    :rtype: Optional[float]
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value) if abs(value) < 2 ** 62 else None
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _is_int(value: Any) -> bool:
    """True for a real ``int`` (not a bool).

    :param value: Value to test.
    :returns: Whether it is an int.
    :rtype: bool
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _now_ms(clock: Callable[[], float]) -> int:
    """Current time in epoch milliseconds from an injectable clock.

    :param clock: Callable returning epoch seconds.
    :returns: Epoch milliseconds.
    :rtype: int
    """
    return int(clock() * 1000)


def _iso(seconds: float) -> str:
    """Local ISO-8601 timestamp with offset and milliseconds.

    :param seconds: Epoch seconds.
    :returns: e.g. ``2026-10-11T09:15:02.123-04:00``.
    :rtype: str
    """
    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _argv0() -> str:
    """Basename of ``sys.argv[0]`` (identifies the job next to its pid, keeps log lines short).

    :returns: e.g. ``two_reader_lines.py``.
    :rtype: str
    """
    argv = getattr(sys, "argv", None) or [""]
    first = str(argv[0])
    return Path(first).name or first


# ============================================================================
# The decision (pure)
# ============================================================================

def _entry_chat_model(entry: Mapping[str, Any]) -> Optional[str]:
    """Match an ``lms ps`` entry to a chat model on ``identifier``, ``modelKey`` or ``indexedModelIdentifier``.

    :param entry: One ``lms ps --json`` entry.
    :returns: Canonical chat-model id, or None for a non-chat model.
    :rtype: Optional[str]
    """
    for key in ("identifier", "modelKey", "indexedModelIdentifier"):
        model = chat_model_id(entry.get(key))
        if model is not None:
            return model
    return None


def decide(entries: Sequence[Mapping[str, Any]], own_calls: Sequence[OwnCall], now_ms: int,
           cfg: YieldConfig) -> GateDecision:
    """Decide from ``lms ps`` entries and the own-call registry whether a prod chat is active. Pure.

    A chat model holds when busy (``status`` not idle, or ``queued > 0``) unless one of this repo's own calls on it
    is in flight, or when recently used (``lastUsedTime`` within ``recent_s``) unless an own call on it ended within
    ``own_end_tol_ms`` of that ``lastUsedTime``. Discounted signals are reported in ``ignored_own``.

    :param entries: ``lms ps --json`` entries.
    :param own_calls: Live own-call markers (all processes).
    :param now_ms: Current epoch milliseconds.
    :param cfg: Gate config.
    :returns: The decision.
    :rtype: GateDecision
    """
    reasons: List[str] = []
    ignored: List[str] = []
    busy_hold = False
    windows_left: List[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model = _entry_chat_model(entry)
        if model is None:
            continue
        mine = [call for call in own_calls if chat_model_id(call.model) == model]
        status = str(entry.get("status") or "idle")
        queued_value = _as_number(entry.get("queued"))
        queued = int(queued_value) if queued_value is not None and queued_value > 0 else 0
        if status.lower() != "idle" or queued > 0:
            parts = [status] if status.lower() != "idle" else []
            if queued:
                parts.append(f"queued {queued}")
            signal = f"{model}:{' '.join(parts)}"
            if any(call.end_ms is None for call in mine):
                ignored.append(signal)
            else:
                reasons.append(signal)
                busy_hold = True
        last_used = _as_number(entry.get("lastUsedTime"))
        if last_used is None:
            continue
        age_ms = now_ms - last_used
        if not -FUTURE_TOL_MS <= age_ms < cfg.recent_s * 1000.0:
            continue
        age_s = max(age_ms, 0.0) / 1000.0
        signal = f"{model}:recent {int(age_s)}s"
        if any(call.end_ms is not None and abs(last_used - call.end_ms) <= cfg.own_end_tol_ms for call in mine):
            ignored.append(signal)
        else:
            reasons.append(signal)
            windows_left.append(cfg.recent_s - age_s)
    if not reasons:
        return GateDecision(hold=False, ignored_own=tuple(ignored))
    estimate = cfg.recent_s if busy_hold else max(windows_left)
    return GateDecision(hold=True, reasons=tuple(reasons), ignored_own=tuple(ignored),
                        would_wait_s_est=round(estimate, 1))


# ============================================================================
# Own-call registry
# ============================================================================

def _read_json_file(path: Path) -> Tuple[Any, Optional[Exception]]:
    """Read and parse one JSON file through the I/O guard.

    :param path: File to read.
    :returns: ``(data, None)`` or ``(None, error)``.
    :rtype: Tuple[Any, Optional[Exception]]
    """
    return _guarded_io(lambda: json.loads(path.read_text(encoding="utf-8")))


def _remove_file(path: Path) -> None:
    """Delete a file through the I/O guard (missing is fine).

    :param path: File to delete.
    """
    _guarded_io(lambda: path.unlink(missing_ok=True))


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Write a JSON file atomically (tmp file, then ``os.replace``) through the I/O guard.

    :param path: Destination.
    :param data: JSON-serialisable mapping.
    """
    tmp = path.with_name(path.name + ".tmp")

    def write() -> None:
        """Create the directory, write the tmp file and move it into place."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)

    _guarded_io(write)


def _pid_alive(pid: int) -> bool:
    """Signal-0 liveness probe: ``ProcessLookupError`` means dead; anything else (``PermissionError``) alive.

    :param pid: Process id (already validated as 1..MAX_PID).
    :returns: Whether the process may still be running.
    :rtype: bool
    """
    if pid == os.getpid():
        return True
    _, error = _guarded_io(lambda: os.kill(pid, 0))
    return not isinstance(error, ProcessLookupError)


def _own_call_from_json(data: Any) -> Optional[OwnCall]:
    """Validate a marker's JSON; anything malformed is None (the marker is then deleted).

    :param data: Parsed marker.
    :returns: The own call, or None.
    :rtype: Optional[OwnCall]
    """
    if not isinstance(data, dict):
        return None
    pid, start_ms, end_ms = data.get("pid"), data.get("start_ms"), data.get("end_ms")
    model = chat_model_id(data.get("model"))
    if not _is_int(pid) or not 0 < pid <= MAX_PID or model is None or not _is_int(start_ms):
        return None
    if end_ms is not None and not _is_int(end_ms):
        return None
    return OwnCall(pid=pid, model=model, start_ms=start_ms, end_ms=end_ms, argv0=str(data.get("argv0") or ""))


def load_own_calls(cfg: YieldConfig, now_ms: int) -> List[OwnCall]:
    """Read the own-call registry, pruning as it goes; a registry failure means "no own calls known".

    Deletes markers whose process is dead, finished markers older than ``recent_s + 60 s`` and unreadable or
    corrupt files.

    :param cfg: Gate config (``state_dir``, ``recent_s``).
    :param now_ms: Current epoch milliseconds.
    :returns: The live markers of every process.
    :rtype: List[OwnCall]
    """
    directory = cfg.state_dir / OWN_CALLS_DIR
    paths, error = _guarded_io(lambda: sorted(directory.glob("*.json")))
    if error is not None or not paths:
        return []
    expiry_ms = (cfg.recent_s + MARKER_GRACE_S) * 1000.0
    calls: List[OwnCall] = []
    for path in paths:
        data, read_error = _read_json_file(path)
        call = None if read_error is not None else _own_call_from_json(data)
        expired = call is not None and call.end_ms is not None and now_ms - call.end_ms > expiry_ms
        if call is None or expired or not _pid_alive(call.pid):
            _remove_file(path)
            continue
        calls.append(call)
    return calls


@contextmanager
def own_call(model: str, *, cfg: Optional[YieldConfig] = None,
             clock: Callable[[], float] = time.time) -> Iterator[None]:
    """Mark the wrapped block as one of this repo's own calls on a prod chat model.

    Writes ``<state_dir>/own_calls/<pid>-<n>-<model>.json`` (``end_ms`` null) on entry and rewrites it with
    ``end_ms`` on exit; on exit also appends ``{ts, pid, argv0, model, start_ms, end_ms}`` to ``own_calls.jsonl``
    (also when the block raises) and clears the ``lms ps`` cache, so the next gate call never judges the just-ended
    call from a snapshot taken while it was in flight. A no-op for non-chat models (the VLM path pays nothing) and
    in ``off`` mode. Never raises on its own.

    :param model: Model id the wrapped block sends its request to.
    :param cfg: Gate config; defaults to the process config.
    :param clock: Epoch-seconds clock (injectable for tests).
    :returns: A context manager.
    :rtype: Iterator[None]
    """
    canonical = chat_model_id(model)
    if canonical is None:
        yield
        return
    config = cfg or get_config()
    if config.mode == "off":
        yield
        return
    pid = os.getpid()
    marker = config.state_dir / OWN_CALLS_DIR / f"{pid}-{next(_marker_counter)}-{_SLUG_RE.sub('_', canonical)}.json"
    record: Dict[str, Any] = {"pid": pid, "model": canonical, "start_ms": _now_ms(clock), "end_ms": None,
                              "argv0": _argv0()}
    _write_json_atomic(marker, record)
    try:
        yield
    finally:
        record["end_ms"] = _now_ms(clock)
        _write_json_atomic(marker, record)
        _append_jsonl(config.state_dir / OWN_CALLS_LOG,
                      {"ts": _iso(record["end_ms"] / 1000.0), "pid": pid, "argv0": record["argv0"],
                       "model": canonical, "start_ms": record["start_ms"], "end_ms": record["end_ms"]})
        _clear_ps_cache()


# ============================================================================
# Checking and waiting
# ============================================================================

def _read_ps_cached(cfg: YieldConfig, reader: PsReader,
                    clock: Callable[[], float]) -> Tuple[Optional[List[Dict[str, Any]]], float]:
    """``lms ps`` entries through the 2 s in-process cache.

    :param cfg: Gate config (CLI path, timeout).
    :param reader: ``read_lms_ps`` or a test double.
    :param clock: Epoch-seconds clock.
    :returns: ``(entries or None, milliseconds spent reading; 0 on a cache hit)``.
    :rtype: Tuple[Optional[List[Dict[str, Any]]], float]
    """
    global _ps_cache
    now = clock()
    cached = _ps_cache
    if cached is not None and 0.0 <= now - cached[0] < PS_CACHE_S:
        return cached[1], 0.0
    started = time.perf_counter()
    entries = reader(cfg.lms_cli, cfg.ps_timeout_s)
    _ps_cache = (now, entries)
    return entries, (time.perf_counter() - started) * 1000.0


def _check(cfg: YieldConfig, reader: PsReader, clock: Callable[[], float],
           registry: Optional[Sequence[OwnCall]] = None) -> Tuple[GateDecision, float]:
    """One evaluation: ``lms ps`` (cached), the registry, then :func:`decide`.

    :param cfg: Gate config.
    :param reader: ``read_lms_ps`` or a test double.
    :param clock: Epoch-seconds clock.
    :param registry: Own calls to use instead of reading the registry directory.
    :returns: ``(decision, milliseconds spent on lms ps)``.
    :rtype: Tuple[GateDecision, float]
    """
    entries, spent_ms = _read_ps_cached(cfg, reader, clock)
    if entries is None:
        return GateDecision(hold=False, error="lms ps failed (fail open)"), spent_ms
    now_ms = _now_ms(clock)
    own = list(registry) if registry is not None else load_own_calls(cfg, now_ms)
    return decide(entries, own, now_ms, cfg), spent_ms


def chat_active(*, cfg: Optional[YieldConfig] = None, reader: Optional[PsReader] = None,
                clock: Callable[[], float] = time.time,
                registry: Optional[Sequence[OwnCall]] = None) -> GateDecision:
    """Is a prod chat active right now? The single seam for a future prod-side activity lease.

    :param cfg: Gate config; defaults to the process config.
    :param reader: ``lms ps`` reader; defaults to :func:`read_lms_ps`.
    :param clock: Epoch-seconds clock.
    :param registry: Own calls to use instead of reading (and pruning) the registry directory.
    :returns: The decision; ``hold=False`` with ``error`` set when ``lms ps`` failed.
    :rtype: GateDecision
    """
    return _check(cfg or get_config(), reader or read_lms_ps, clock, registry)[0]


def _record(site: str, cfg: YieldConfig, label: str, decision: Optional[GateDecision], *, ts: float,
            waited_s: float = 0.0, polls: int = 0, capped: bool = False, ps_ms: float = 0.0,
            error: Optional[str] = None, host: Optional[str] = None) -> Dict[str, Any]:
    """Build one ``yield_gate.jsonl`` record.

    :param site: Call site name.
    :param cfg: Gate config.
    :param label: ``clear``, ``hold``, ``error`` or ``skip_remote``.
    :param decision: The first decision of the gate call (None for ``skip_remote``).
    :param ts: Gate-call start, epoch seconds.
    :param waited_s: Seconds actually waited.
    :param polls: Polls after the first check.
    :param capped: True when the wait hit ``max_wait_s``.
    :param ps_ms: Milliseconds spent on ``lms ps``.
    :param error: Error to record (first or mid-wait ``lms ps`` failure).
    :param host: Parsed host, for ``skip_remote`` only.
    :returns: The record.
    :rtype: Dict[str, Any]
    """
    record: Dict[str, Any] = {
        "ts": _iso(ts), "pid": os.getpid(), "argv0": _argv0(), "site": site, "mode": cfg.mode,
        "decision": label,
        "reasons": list(decision.reasons) if decision else [],
        "ignored_own": list(decision.ignored_own) if decision else [],
        "error": error,
        "waited_s": round(waited_s, 1), "polls": polls, "capped": capped,
        "would_wait_s_est": decision.would_wait_s_est if decision else None,
        "ps_ms": round(ps_ms, 1),
    }
    if host is not None:
        record["host"] = host
    return record


def _prepare(site: str, base_url: Optional[str], cfg: Optional[YieldConfig],
             clock: Callable[[], float]) -> Optional[YieldConfig]:
    """Common start of a gate call: handle ``off`` and remote URLs.

    :param site: Call site name.
    :param base_url: LM Studio URL the unit will use (None = local).
    :param cfg: Gate config or None for the process config.
    :param clock: Epoch-seconds clock.
    :returns: The config to evaluate with, or None when the call returns 0.0 at once.
    :rtype: Optional[YieldConfig]
    """
    config = cfg or get_config()
    if config.mode == "off":
        return None
    wait_note = ("logging what it would do, never waiting" if config.mode == "dryrun" else
                 f"yielding to prod chat (poll {config.poll_s:g} s, cap {config.max_wait_s:g} s)")
    _warn_once("announce", f"    ⏸ [lms_yield_gate] mode={config.mode}: {wait_note}; "
                           f"log {config.state_dir / GATE_LOG}")
    if is_local_lms(base_url):
        return config
    host = _host_of(base_url)
    if host not in _skip_logged:
        _skip_logged.add(host)
        _append_jsonl(config.state_dir / GATE_LOG,
                      _record(site, config, "skip_remote", None, ts=clock(), host=host))
    return None


class _GateCall:
    """State of one gate call, shared by the sync and the async driver."""

    def __init__(self, site: str, cfg: YieldConfig, clock: Callable[[], float]) -> None:
        """Start the call.

        :param site: Call site name.
        :param cfg: Gate config (mode not ``off``).
        :param clock: Epoch-seconds clock.
        """
        self.site = site
        self.cfg = cfg
        self.clock = clock
        self.ts = clock()
        self.first: Optional[GateDecision] = None
        self.last = GateDecision(hold=False)
        self.polls = 0
        self.capped = False
        self.waiting = False
        self.ps_ms = 0.0

    def observe(self, result: Tuple[GateDecision, float]) -> None:
        """Take one check result.

        :param result: ``(decision, ms spent on lms ps)`` from :func:`_check`.
        """
        decision, spent_ms = result
        self.ps_ms += spent_ms
        if self.first is None:
            self.first = decision
        else:
            self.polls += 1
        self.last = decision

    def next_sleep(self) -> Optional[float]:
        """Seconds to sleep before the next poll, or None when the unit should proceed now.

        :returns: Sleep length or None.
        :rtype: Optional[float]
        """
        if self.cfg.mode != "on" or self.last.error is not None or not self.last.hold:
            return None
        waited = self.clock() - self.ts
        if waited >= self.cfg.max_wait_s:
            self.capped = True
            return None
        if not self.waiting:
            self.waiting = True
            print(f"    ⏸ [{self.site}] yielding to prod chat ({', '.join(self.last.reasons)})", flush=True)
        return min(self.cfg.poll_s, self.cfg.max_wait_s - waited)

    def finish(self) -> float:
        """Print the end line (after a real wait), log the record and return the seconds waited.

        :returns: Seconds waited (0.0 when the unit did not wait).
        :rtype: float
        """
        first = self.first or GateDecision(hold=False)
        waited = max(0.0, self.clock() - self.ts) if self.waiting else 0.0
        if self.waiting:
            note = " (cap reached)" if self.capped else (" (lms ps failed, fail open)" if self.last.error else "")
            print(f"    ▶ [{self.site}] proceeding after {waited:.0f} s{note}", flush=True)
        label = "error" if first.error else ("hold" if first.hold else "clear")
        _append_jsonl(self.cfg.state_dir / GATE_LOG,
                      _record(self.site, self.cfg, label, first, ts=self.ts, waited_s=waited, polls=self.polls,
                              capped=self.capped, ps_ms=self.ps_ms, error=first.error or self.last.error))
        return waited


def wait_for_chat_idle(site: str, *, base_url: Optional[str] = None, cfg: Optional[YieldConfig] = None,
                       reader: Optional[PsReader] = None, clock: Callable[[], float] = time.time,
                       sleep: Callable[[float], None] = time.sleep) -> float:
    """Gate one unit of GPU work (sync callers). Never raises; see the module docstring for the modes.

    :param site: Call site name for the log, e.g. ``"llm_client"``.
    :param base_url: LM Studio URL the unit will use; a non-local URL never holds (``skip_remote``).
    :param cfg: Gate config; defaults to the process config.
    :param reader: ``lms ps`` reader; defaults to :func:`read_lms_ps`.
    :param clock: Epoch-seconds clock.
    :param sleep: Sleep function.
    :returns: Seconds waited.
    :rtype: float
    """
    config = _prepare(site, base_url, cfg, clock)
    if config is None:
        return 0.0
    read = reader or read_lms_ps
    call = _GateCall(site, config, clock)
    call.observe(_check(config, read, clock))
    pause = call.next_sleep()
    while pause is not None:
        sleep(pause)
        call.observe(_check(config, read, clock))
        pause = call.next_sleep()
    return call.finish()


async def await_chat_idle(site: str, *, base_url: Optional[str] = None, cfg: Optional[YieldConfig] = None,
                          reader: Optional[PsReader] = None, clock: Callable[[], float] = time.time,
                          sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> float:
    """Gate one unit of GPU work (async callers). ``lms ps`` runs in a worker thread, so concurrent coroutines
    (e.g. the Kraken prefetch of ``two_reader_lines``) keep running. Never raises.

    :param site: Call site name for the log, e.g. ``"transcribe:<model>"``.
    :param base_url: LM Studio URL the unit will use; a non-local URL never holds (``skip_remote``).
    :param cfg: Gate config; defaults to the process config.
    :param reader: ``lms ps`` reader; defaults to :func:`read_lms_ps`.
    :param clock: Epoch-seconds clock.
    :param sleep: Async sleep function.
    :returns: Seconds waited.
    :rtype: float
    """
    config = _prepare(site, base_url, cfg, clock)
    if config is None:
        return 0.0
    read = reader or read_lms_ps
    call = _GateCall(site, config, clock)
    call.observe(await asyncio.to_thread(_check, config, read, clock))
    pause = call.next_sleep()
    while pause is not None:
        await sleep(pause)
        call.observe(await asyncio.to_thread(_check, config, read, clock))
        pause = call.next_sleep()
    return call.finish()


# ============================================================================
# --summarize
# ============================================================================

def _parse_ts(value: Any) -> Optional[float]:
    """Parse an ISO timestamp (naive = local time) to epoch seconds.

    :param value: Timestamp text.
    :returns: Epoch seconds, or None when unparsable.
    :rtype: Optional[float]
    """
    if not isinstance(value, str):
        return None
    parsed, error = _guarded_io(lambda: datetime.fromisoformat(value))
    if error is not None or parsed is None:
        return None
    return (parsed if parsed.tzinfo else parsed.astimezone()).timestamp()


def read_jsonl(path: Path, since: Optional[float] = None) -> List[Dict[str, Any]]:
    """Read one of the gate's JSONL logs, skipping bad lines; each record gets ``_t`` (epoch seconds of ``ts``).

    :param path: JSONL file.
    :param since: Keep only records at or after this epoch second.
    :returns: Records in file order.
    :rtype: List[Dict[str, Any]]
    """
    text, error = _guarded_io(lambda: path.read_text(encoding="utf-8", errors="replace"))
    if error is not None or not text:
        return []
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        data, bad = _guarded_io(lambda: json.loads(line))
        if bad is not None or not isinstance(data, dict) or not _is_int(data.get("pid")):
            continue
        moment = _parse_ts(data.get("ts"))
        if moment is None or (since is not None and moment < since):
            continue
        data["_t"] = moment
        records.append(data)
    return records


def _by_pid(records: Iterable[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    """Group records by pid, each group sorted by time.

    :param records: Records with ``pid`` and ``_t``.
    :returns: pid -> records.
    :rtype: Dict[int, List[Mapping[str, Any]]]
    """
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(int(record["pid"]), []).append(record)
    for group in groups.values():
        group.sort(key=lambda record: float(record["_t"]))
    return groups


def would_wait_spans(records: Iterable[Mapping[str, Any]]) -> Dict[int, List[Span]]:
    """Reconstruct per-pid wait spans from gate records.

    Dryrun: consecutive ``hold`` records of one pid (no other record of that pid in between, gaps <= 300 s) form one
    span from the first hold to the last hold plus its ``would_wait_s_est``, cut at the pid's next record (which saw
    the chat gone). ``on`` mode: each record with ``waited_s > 0`` is an actual span ``[ts, ts + waited_s]``.

    :param records: Records from :func:`read_jsonl`.
    :returns: pid -> spans in time order.
    :rtype: Dict[int, List[Span]]
    """
    spans: Dict[int, List[Span]] = {}
    for pid, group in _by_pid(records).items():
        found: List[Span] = []
        current: Optional[Dict[str, Any]] = None
        for record in group:
            moment = float(record["_t"])
            waited = _as_number(record.get("waited_s")) or 0.0
            reasons = tuple(str(reason) for reason in record.get("reasons") or ())
            is_dry_hold = record.get("decision") == "hold" and record.get("mode") == "dryrun"
            if is_dry_hold and current is not None and moment - current["last"] <= SPAN_GAP_S:
                current.update(last=moment, estimate=_as_number(record.get("would_wait_s_est")),
                               holds=current["holds"] + 1)
                continue
            if current is not None:
                found.append(_close_span(pid, current, bound=moment))
            current = None
            if is_dry_hold:
                current = {"start": moment, "last": moment, "holds": 1, "reasons": reasons,
                           "estimate": _as_number(record.get("would_wait_s_est"))}
            elif record.get("mode") == "on" and waited > 0:
                found.append(Span(pid, moment, moment + waited, 1, reasons, actual=True))
        if current is not None:
            found.append(_close_span(pid, current, bound=None))
        spans[pid] = found
    return spans


def _close_span(pid: int, current: Mapping[str, Any], bound: Optional[float]) -> Span:
    """Close an open dryrun span: last hold plus its estimate, but no later than the pid's next record.

    :param pid: Process id.
    :param current: Open span (``start``, ``last``, ``holds``, ``estimate``, ``reasons``).
    :param bound: Time of the pid's next record, or None at the end of the log.
    :returns: The span.
    :rtype: Span
    """
    end = current["last"] + (current["estimate"] or 0.0)
    if bound is not None:
        end = min(end, bound)
    return Span(pid, current["start"], max(end, current["last"]), current["holds"], current["reasons"])


def _day(moment: float) -> str:
    """Local calendar day of an epoch second.

    :param moment: Epoch seconds.
    :returns: ``YYYY-MM-DD``.
    :rtype: str
    """
    return datetime.fromtimestamp(moment).strftime("%Y-%m-%d")


def daily_ratios(records: Iterable[Mapping[str, Any]],
                 spans: Mapping[int, Sequence[Span]]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    """Active seconds and would-wait seconds per (pid, day), for the <= 3 % review check.

    Active time sums the gaps of <= 300 s between consecutive records of one pid (on the day of the earlier
    record); would-wait time sums that pid's dryrun spans by their start day.

    :param records: Records from :func:`read_jsonl`.
    :param spans: Output of :func:`would_wait_spans`.
    :returns: (pid, day) -> (active_s, would_wait_s).
    :rtype: Dict[Tuple[int, str], Tuple[float, float]]
    """
    active: Dict[Tuple[int, str], float] = {}
    for pid, group in _by_pid(records).items():
        for earlier, later in zip(group, group[1:]):
            gap = float(later["_t"]) - float(earlier["_t"])
            if 0.0 <= gap <= SPAN_GAP_S:
                key = (pid, _day(float(earlier["_t"])))
                active[key] = active.get(key, 0.0) + gap
    waits: Dict[Tuple[int, str], float] = {}
    for pid, pid_spans in spans.items():
        for span in pid_spans:
            if not span.actual:
                key = (pid, _day(span.start))
                waits[key] = waits.get(key, 0.0) + span.seconds
    return {key: (active.get(key, 0.0), waits.get(key, 0.0)) for key in sorted(set(active) | set(waits))}


def render_summary(state_dir: Path, since: Optional[float] = None) -> str:
    """Human-readable dry-run summary of ``yield_gate.jsonl`` and ``own_calls.jsonl``.

    :param state_dir: The gate's state directory.
    :param since: Only records at or after this epoch second.
    :returns: The summary text.
    :rtype: str
    """
    gate_path, own_path = state_dir / GATE_LOG, state_dir / OWN_CALLS_LOG
    records = read_jsonl(gate_path, since)
    if not records:
        return f"no gate records in {gate_path}" + (f" since {_iso(since)}" if since is not None else "")
    spans = would_wait_spans(records)
    ratios = daily_ratios(records, spans)
    lines = [f"{gate_path}: {len(records)} records, {_iso(min(r['_t'] for r in records))} .. "
             f"{_iso(max(r['_t'] for r in records))}"]
    for pid, group in _by_pid(records).items():
        counts: Dict[str, int] = {}
        for record in group:
            counts[str(record.get("decision"))] = counts.get(str(record.get("decision")), 0) + 1
        waited = sum(_as_number(record.get("waited_s")) or 0.0 for record in group)
        capped = sum(1 for record in group if record.get("capped") is True)
        sites = sorted({str(record.get("site")) for record in group})
        lines.append(f"pid {pid} {group[-1].get('argv0')}: {len(group)} records "
                     f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}); actual waits {waited:.0f} s, "
                     f"capped {capped}; first {_iso(float(group[0]['_t']))}, last {_iso(float(group[-1]['_t']))}; "
                     f"sites {', '.join(sites[:4])}{' ...' if len(sites) > 4 else ''}")
        for (ratio_pid, day), (active_s, wait_s) in ratios.items():
            if ratio_pid == pid:
                share = f"{100.0 * wait_s / active_s:.2f} %" if active_s > 0 else "n/a"
                lines.append(f"  {day}: active {active_s:.0f} s, would-wait {wait_s:.0f} s = {share}")
        for span in spans.get(pid, []):
            kind = "waited" if span.actual else "would-wait"
            lines.append(f"  {kind} {_iso(span.start)} -> {_iso(span.end)} ({span.seconds:.0f} s, {span.holds} hold"
                         f"{'s' if span.holds != 1 else ''}): {', '.join(span.reasons)}")
        hosts = sorted({str(record.get("host")) for record in group if record.get("decision") == "skip_remote"})
        if hosts:
            lines.append(f"  skip_remote hosts: {', '.join(hosts)}")
    own = read_jsonl(own_path, since)
    totals: Dict[Tuple[int, str, str], List[float]] = {}
    for record in own:
        key = (int(record["pid"]), str(record.get("argv0")), str(record.get("model")))
        start, end = _as_number(record.get("start_ms")), _as_number(record.get("end_ms"))
        totals.setdefault(key, []).append(max(0.0, (end - start) / 1000.0) if start and end else 0.0)
    lines.append(f"{own_path}: {len(own)} own chat-model calls")
    for (pid, argv0, model), durations in sorted(totals.items()):
        lines.append(f"  pid {pid} {argv0} {model}: {len(durations)} calls, {sum(durations):.0f} s")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def _since_arg(text: str) -> float:
    """argparse type for ``--since``: ISO date/time (naive = local) to epoch seconds.

    :param text: ISO text, e.g. ``2026-10-11T00:00``.
    :returns: Epoch seconds.
    :rtype: float
    :raises ValueError: On unparsable text (argparse turns it into a usage error).
    """
    parsed = datetime.fromisoformat(text)
    return (parsed if parsed.tzinfo else parsed.astimezone()).timestamp()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``--check``, ``--wait --site NAME`` or ``--summarize [--since ISO]``.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit code: ``--check`` 0 clear / 3 hold; otherwise 0.
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="python -m src.models.ocr.lms_yield_gate",
                                     description="Yield-to-prod-chat gate for LM Studio batch work.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true",
                        help="print whether a prod chat is active now (any mode); exit 0 clear, 3 hold")
    action.add_argument("--wait", action="store_true",
                        help="gate one stage of a shell script; honours LMS_YIELD_MODE, always exits 0")
    action.add_argument("--summarize", action="store_true", help="summarise yield_gate.jsonl and own_calls.jsonl")
    parser.add_argument("--site", default="cli:wait", help="site name logged by --wait")
    parser.add_argument("--base-url", default=None, help="LM Studio URL the stage will use (default: local)")
    parser.add_argument("--since", type=_since_arg, default=None, help="--summarize: ISO start, naive = local")
    args = parser.parse_args(argv)
    cfg = get_config()
    if args.check:
        decision = chat_active(cfg=cfg)
        state = "HOLD" if decision.hold else ("clear (lms ps failed, fail open)" if decision.error else "clear")
        details = [", ".join(decision.reasons)] if decision.reasons else []
        details += [f"ignored own: {', '.join(decision.ignored_own)}"] if decision.ignored_own else []
        print(f"mode={cfg.mode} {state}" + (f": {'; '.join(details)}" if details else ""), flush=True)
        return 3 if decision.hold else 0
    if args.wait:
        waited = wait_for_chat_idle(args.site, base_url=args.base_url, cfg=cfg)
        print(f"mode={cfg.mode} waited {waited:.0f} s", flush=True)
        return 0
    print(render_summary(cfg.state_dir, since=args.since), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
