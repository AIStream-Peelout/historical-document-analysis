"""Tests for the yield-to-prod-chat gate (src/models/ocr/lms_yield_gate.py).

Fake clocks, sleeps and ``lms ps`` readers throughout; the only real subprocesses are tiny shell scripts standing in
for the ``lms`` CLI. ``tests/conftest.py`` pins the gate off with a temporary state dir; tests here pass an explicit
:class:`YieldConfig` (or set the env) to turn it on.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.models.ocr import lms_yield_gate as g
from src.models.ocr.lms_yield_gate import GateDecision, OwnCall, YieldConfig

NOW_MS = 1_791_700_000_000
NOW_S = NOW_MS / 1000.0
FOUR_B, TWENTY_SEVEN_B, THIRTY_FIVE_B = g.CHAT_MODELS
VLM = "qwen3-vl-8b-heb-v22i-a50"

GATE_FIELDS = {"ts", "pid", "argv0", "site", "mode", "decision", "reasons", "ignored_own", "error", "waited_s",
               "polls", "capped", "would_wait_s_est", "ps_ms"}
OWN_FIELDS = {"ts", "pid", "argv0", "model", "start_ms", "end_ms"}


def entry(identifier: str, status: str = "idle", queued: int = 0, last: Optional[int] = None) -> Dict[str, Any]:
    """One ``lms ps --json`` entry.

    :param identifier: Model id (used for identifier, modelKey and indexedModelIdentifier).
    :param status: LM Studio status.
    :param queued: Queued requests.
    :param last: ``lastUsedTime`` in epoch ms (None while in flight).
    :returns: The entry.
    """
    return {"identifier": identifier, "modelKey": identifier, "indexedModelIdentifier": identifier,
            "status": status, "queued": queued, "lastUsedTime": last, "parallel": 4}


def quiet(now_ms: int = NOW_MS) -> List[Dict[str, Any]]:
    """All chat models idle and last used 10 minutes ago, plus a VLM that is generating.

    :param now_ms: Reference time.
    :returns: Entries.
    """
    old = now_ms - 600_000
    return [entry(FOUR_B, last=old), entry(TWENTY_SEVEN_B, last=old), entry(THIRTY_FIVE_B, last=old),
            entry(VLM, status="generating")]


def with_entry(replacement: Dict[str, Any], now_ms: int = NOW_MS) -> List[Dict[str, Any]]:
    """:func:`quiet` with the entry of one model replaced.

    :param replacement: New entry (matched on its identifier).
    :param now_ms: Reference time.
    :returns: Entries.
    """
    return [replacement if e["identifier"] == replacement["identifier"] else e for e in quiet(now_ms)]


HOLD_27B = with_entry(entry(TWENTY_SEVEN_B, status="generating"))


class FakeTime:
    """Fake epoch clock whose sleeps (sync and async) advance it."""

    def __init__(self, start: float = NOW_S) -> None:
        """:param start: Initial epoch seconds."""
        self.now = start
        self.sleeps: List[float] = []

    def clock(self) -> float:
        """:returns: Current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """:param seconds: Advance the clock by this much."""
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        """:param seconds: Advance the clock by this much."""
        self.sleep(seconds)


class ScriptedReader:
    """``lms ps`` reader returning scripted answers in order, repeating the last one."""

    def __init__(self, *answers: Optional[List[Dict[str, Any]]]) -> None:
        """:param answers: Entries lists, or None for a failed ``lms ps``."""
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, lms_cli: str, timeout_s: float) -> Optional[List[Dict[str, Any]]]:
        """:returns: The next scripted answer."""
        self.calls += 1
        return self.answers[min(self.calls, len(self.answers)) - 1]


def gate_lines(state_dir: Path) -> List[Dict[str, Any]]:
    """Parse every ``yield_gate.jsonl`` line.

    :param state_dir: Gate state dir.
    :returns: Records.
    """
    path = state_dir / g.GATE_LOG
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_marker(cfg: YieldConfig, name: str, **data: Any) -> Path:
    """Write a raw own-call marker file.

    :param cfg: Config (state dir).
    :param name: File name.
    :param data: Marker JSON fields.
    :returns: The marker path.
    """
    directory = cfg.state_dir / g.OWN_CALLS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(data))
    return path


# ── decide ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_decide_all_idle_with_old_last_use_is_clear() -> None:
    """Idle chat models used 10 minutes ago do not hold."""
    assert g.decide(quiet(), [], NOW_MS, YieldConfig()) == GateDecision(hold=False)


def test_decide_27b_generating_holds() -> None:
    """A generating synthesis model holds; the estimate is a full recency window."""
    decision = g.decide(HOLD_27B, [], NOW_MS, YieldConfig())
    assert decision.hold and decision.reasons == (f"{TWENTY_SEVEN_B}:generating",)
    assert decision.would_wait_s_est == 180.0


def test_decide_4b_queued_holds() -> None:
    """A queued request on the router holds even while it reports idle."""
    decision = g.decide(with_entry(entry(FOUR_B, queued=1, last=NOW_MS - 600_000)), [], NOW_MS, YieldConfig())
    assert decision.hold and decision.reasons == (f"{FOUR_B}:queued 1",)


def test_decide_35b_used_42s_ago_holds() -> None:
    """Recency holds, with the age in the reason and the rest of the window as the estimate."""
    decision = g.decide(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS - 42_000)), [], NOW_MS, YieldConfig())
    assert decision.hold and decision.reasons == (f"{THIRTY_FIVE_B}:recent 42s",)
    assert decision.would_wait_s_est == 138.0


def test_decide_35b_used_200s_ago_is_clear() -> None:
    """Outside the 180 s window there is no hold."""
    assert not g.decide(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS - 200_000)), [], NOW_MS, YieldConfig()).hold


def test_decide_null_last_used_while_idle_is_clear() -> None:
    """A null lastUsedTime is not an error and not a hold."""
    assert not g.decide(with_entry(entry(THIRTY_FIVE_B, last=None)), [], NOW_MS, YieldConfig()).hold


def test_decide_vlm_activity_never_holds() -> None:
    """VLM checkpoints, generating or just used, are not prod chat."""
    entries = quiet() + [entry("qwen3-vl-8b-heb-v21b-step1200", queued=2, last=NOW_MS - 1_000)]
    assert g.decide(entries, [], NOW_MS, YieldConfig()) == GateDecision(hold=False)


def test_decide_bare_model_id_matches() -> None:
    """The 35B reported without the qwen/ prefix still counts, under its canonical id."""
    decision = g.decide([entry("qwen3.6-35b-a3b", status="processingPrompt")], [], NOW_MS, YieldConfig())
    assert decision.reasons == (f"{THIRTY_FIVE_B}:processingPrompt",)


def test_is_chat_model_is_suffix_tolerant() -> None:
    """Publisher prefix and case do not matter; other models and suffixed names do not match."""
    assert g.is_chat_model("qwen3.6-35b-a3b") and g.is_chat_model(THIRTY_FIVE_B)
    assert g.is_chat_model(" QWEN/Qwen3-4B-2507 ") and g.chat_model_id("qwen3.8-27b") == TWENTY_SEVEN_B
    assert not g.is_chat_model(VLM) and not g.is_chat_model("isaacmg/qwen3-vl-8b-heb-v21b-step1200")
    assert not g.is_chat_model("qwen3.6-35b-a3b-mlx") and not g.is_chat_model(None)


def test_own_inflight_marker_masks_busy_on_the_same_model() -> None:
    """Our own in-flight 35B call is not a chat; the signal is reported as ignored."""
    own = [OwnCall(pid=1, model="qwen3.6-35b-a3b", start_ms=NOW_MS - 5_000)]
    decision = g.decide(with_entry(entry(THIRTY_FIVE_B, status="generating")), own, NOW_MS, YieldConfig())
    assert not decision.hold and decision.ignored_own == (f"{THIRTY_FIVE_B}:generating",)


def test_own_marker_ending_within_5s_of_last_use_masks_recency() -> None:
    """lastUsedTime within own_end_tol_ms of our call's end is that call."""
    own = [OwnCall(pid=1, model=THIRTY_FIVE_B, start_ms=NOW_MS - 60_000, end_ms=NOW_MS - 17_000)]
    decision = g.decide(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS - 20_000)), own, NOW_MS, YieldConfig())
    assert not decision.hold and decision.ignored_own == (f"{THIRTY_FIVE_B}:recent 20s",)


def test_own_marker_ending_30s_off_still_holds() -> None:
    """A use 30 s after our call ended is someone else's (prod)."""
    own = [OwnCall(pid=1, model=THIRTY_FIVE_B, start_ms=NOW_MS - 90_000, end_ms=NOW_MS - 50_000)]
    decision = g.decide(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS - 20_000)), own, NOW_MS, YieldConfig())
    assert decision.hold and decision.ignored_own == ()


def test_own_marker_on_27b_does_not_mask_the_4b() -> None:
    """Masking is per model."""
    own = [OwnCall(pid=1, model=TWENTY_SEVEN_B, start_ms=NOW_MS - 5_000)]
    decision = g.decide(with_entry(entry(FOUR_B, status="generating")), own, NOW_MS, YieldConfig())
    assert decision.hold and decision.reasons == (f"{FOUR_B}:generating",) and decision.ignored_own == ()


def test_masked_27b_turn_is_still_caught_by_the_4b_router_recency() -> None:
    """Known limitation 1, softened: the prod router's recency still holds while our 27B call masks synthesis."""
    entries = [entry(FOUR_B, last=NOW_MS - 30_000), entry(TWENTY_SEVEN_B, status="generating"),
               entry(THIRTY_FIVE_B, last=NOW_MS - 600_000)]
    own = [OwnCall(pid=1, model=TWENTY_SEVEN_B, start_ms=NOW_MS - 10_000)]
    decision = g.decide(entries, own, NOW_MS, YieldConfig())
    assert decision.hold and decision.reasons == (f"{FOUR_B}:recent 30s",)
    assert decision.ignored_own == (f"{TWENTY_SEVEN_B}:generating",)


# ── own-call registry ───────────────────────────────────────────────────────────────────────────────────────────

def test_own_call_writes_then_finalises_its_marker(tmp_path: Path) -> None:
    """The marker is written with end_ms null on entry and rewritten with end_ms on exit."""
    clock = FakeTime()
    with g.own_call("qwen3.6-35b-a3b", cfg=YieldConfig(state_dir=tmp_path), clock=clock.clock):
        (marker,) = (tmp_path / g.OWN_CALLS_DIR).glob("*.json")
        live = json.loads(marker.read_text())
        clock.now += 7
    assert marker.name.startswith(f"{os.getpid()}-")
    assert live == {"pid": os.getpid(), "model": THIRTY_FIVE_B, "start_ms": NOW_MS, "end_ms": None,
                    "argv0": g._argv0()}
    assert json.loads(marker.read_text())["end_ms"] == NOW_MS + 7_000
    assert not list((tmp_path / g.OWN_CALLS_DIR).glob("*.tmp"))


def test_own_call_appends_exactly_one_record_even_when_the_block_raises(tmp_path: Path) -> None:
    """own_calls.jsonl gets one well-formed line per call, written in a finally."""
    with pytest.raises(RuntimeError):
        with g.own_call("qwen3.8-27b", cfg=YieldConfig(state_dir=tmp_path)):
            raise RuntimeError("boom")
    (line,) = (tmp_path / g.OWN_CALLS_LOG).read_text().splitlines()
    record = json.loads(line)
    assert set(record) == OWN_FIELDS and record["model"] == TWENTY_SEVEN_B and record["pid"] == os.getpid()
    assert record["end_ms"] >= record["start_ms"]


def test_own_call_is_a_noop_for_a_vlm_and_in_off_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The VLM path pays nothing (not even a config load); off mode writes nothing."""
    monkeypatch.setattr(g, "get_config", lambda: pytest.fail("VLM own_call must not load the config"))
    with g.own_call(VLM):
        pass
    with g.own_call(THIRTY_FIVE_B, cfg=YieldConfig(mode="off", state_dir=tmp_path)):
        pass
    assert list(tmp_path.iterdir()) == []


def test_lms_ps_cache_is_two_seconds_and_own_call_exit_clears_it(tmp_path: Path) -> None:
    """Gate calls 1 s apart share one lms ps; 3 s apart read again; an own call's exit forces a fresh read."""
    cfg, clock, reader = YieldConfig(mode="dryrun", state_dir=tmp_path), FakeTime(), ScriptedReader(quiet())
    gate = dict(cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep)
    g.wait_for_chat_idle("a", **gate)
    clock.now += 1
    g.wait_for_chat_idle("a", **gate)
    assert reader.calls == 1
    clock.now += 2
    g.wait_for_chat_idle("a", **gate)
    assert reader.calls == 2
    with g.own_call(THIRTY_FIVE_B, cfg=cfg, clock=clock.clock):
        pass
    g.wait_for_chat_idle("a", **gate)
    assert reader.calls == 3


def test_dead_pid_markers_are_pruned(tmp_path: Path) -> None:
    """A marker of a finished process is deleted; ours is kept."""
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    cfg = YieldConfig(state_dir=tmp_path)
    dead = write_marker(cfg, f"{finished.pid}-0-m.json", pid=finished.pid, model=THIRTY_FIVE_B,
                        start_ms=NOW_MS - 1_000, end_ms=None)
    mine = write_marker(cfg, f"{os.getpid()}-0-m.json", pid=os.getpid(), model=THIRTY_FIVE_B,
                        start_ms=NOW_MS - 1_000, end_ms=None)
    assert [call.pid for call in g.load_own_calls(cfg, NOW_MS)] == [os.getpid()]
    assert not dead.exists() and mine.exists()


def test_expired_markers_are_pruned(tmp_path: Path) -> None:
    """A finished marker older than recent_s + 60 s goes; a fresh one stays."""
    cfg = YieldConfig(state_dir=tmp_path)
    old = write_marker(cfg, "a.json", pid=os.getpid(), model=THIRTY_FIVE_B, start_ms=NOW_MS - 300_000,
                       end_ms=NOW_MS - 241_000)
    fresh = write_marker(cfg, "b.json", pid=os.getpid(), model=THIRTY_FIVE_B, start_ms=NOW_MS - 30_000,
                         end_ms=NOW_MS - 10_000)
    calls = g.load_own_calls(cfg, NOW_MS)
    assert [call.end_ms for call in calls] == [NOW_MS - 10_000]
    assert not old.exists() and fresh.exists()


def test_corrupt_markers_are_skipped_and_removed(tmp_path: Path) -> None:
    """Unparsable or malformed markers are dropped without breaking the gate."""
    cfg = YieldConfig(mode="dryrun", state_dir=tmp_path)
    directory = tmp_path / g.OWN_CALLS_DIR
    directory.mkdir(parents=True)
    (directory / "x.json").write_text("{not json")
    write_marker(cfg, "y.json", pid="abc", model=THIRTY_FIVE_B, start_ms=1)
    write_marker(cfg, "z.json", pid=os.getpid(), model=VLM, start_ms=1)
    assert g.load_own_calls(cfg, NOW_MS) == [] and list(directory.iterdir()) == []
    assert g.chat_active(cfg=cfg, reader=ScriptedReader(HOLD_27B), clock=FakeTime().clock).hold


def test_gate_ignores_this_processes_inflight_own_call(tmp_path: Path) -> None:
    """End to end: our own 35B call in flight does not make the gate hold."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    reader = ScriptedReader(with_entry(entry(THIRTY_FIVE_B, status="generating")))
    with g.own_call(THIRTY_FIVE_B, cfg=cfg, clock=clock.clock):
        waited = g.wait_for_chat_idle("x", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep)
    (record,) = gate_lines(tmp_path)
    assert waited == 0.0 and record["decision"] == "clear"
    assert record["ignored_own"] == [f"{THIRTY_FIVE_B}:generating"]


def test_gate_ignores_the_recency_of_a_just_finished_own_call(tmp_path: Path) -> None:
    """End to end: lastUsedTime set by our own call that just ended does not hold the next unit."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    with g.own_call("qwen3.6-35b-a3b", cfg=cfg, clock=clock.clock):
        clock.now += 5
    clock.now += 1
    reader = ScriptedReader(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS + 5_000 - 150)))
    assert g.wait_for_chat_idle("x", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep) == 0.0
    assert gate_lines(tmp_path)[0]["ignored_own"] == [f"{THIRTY_FIVE_B}:recent 1s"]


# ── read_lms_ps ─────────────────────────────────────────────────────────────────────────────────────────────────

def fake_cli(tmp_path: Path, body: str) -> str:
    """Write an executable shell script standing in for ``lms``.

    :param tmp_path: Directory.
    :param body: Script body.
    :returns: Script path.
    """
    path = tmp_path / "lms"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("body", ["exit 3", "echo 'not json'", "echo '{\"identifier\": \"x\"}'"])
def test_read_lms_ps_bad_answers_return_none(tmp_path: Path, body: str) -> None:
    """Non-zero exit, invalid JSON and a dict instead of a list all fail open."""
    assert g.read_lms_ps(fake_cli(tmp_path, body), 5.0) is None


def test_read_lms_ps_missing_cli_returns_none(tmp_path: Path) -> None:
    """A missing CLI fails open."""
    assert g.read_lms_ps(str(tmp_path / "missing" / "lms"), 5.0) is None


def test_read_lms_ps_timeout_returns_none(tmp_path: Path) -> None:
    """A hung CLI is abandoned at the timeout."""
    started = time.monotonic()
    assert g.read_lms_ps(fake_cli(tmp_path, "exec sleep 5"), 0.3) is None
    assert time.monotonic() - started < 4


def test_read_lms_ps_parses_the_entry_list(tmp_path: Path) -> None:
    """A JSON array comes back with non-dict items dropped."""
    cli = fake_cli(tmp_path, "echo '[{\"identifier\": \"m\", \"status\": \"idle\"}, 5]'")
    assert g.read_lms_ps(cli, 5.0) == [{"identifier": "m", "status": "idle"}]


def test_chat_active_fails_open_when_lms_ps_fails(tmp_path: Path) -> None:
    """No answer from lms ps means hold=False with the error recorded."""
    decision = g.chat_active(cfg=YieldConfig(state_dir=tmp_path, lms_cli=str(tmp_path / "nope")))
    assert decision.hold is False and decision.error
    assert g.chat_active(cfg=YieldConfig(state_dir=tmp_path), reader=ScriptedReader(None)).error


# ── wait_for_chat_idle / await_chat_idle ───────────────────────────────────────────────────────────────────────

def test_off_mode_never_reads_lms_ps(tmp_path: Path) -> None:
    """off returns at once in both forms, with no lms ps and no log line."""
    cfg, clock, reader = YieldConfig(mode="off", state_dir=tmp_path), FakeTime(), ScriptedReader(HOLD_27B)
    assert g.wait_for_chat_idle("s", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep) == 0.0
    assert asyncio.run(g.await_chat_idle("s", cfg=cfg, reader=reader, clock=clock.clock,
                                         sleep=clock.async_sleep)) == 0.0
    assert reader.calls == 0 and list(tmp_path.iterdir()) == []


def test_the_suite_runs_with_the_gate_off(isolate_lms_yield_gate: Path) -> None:
    """conftest pins the env: the process config is off and points at a temporary directory."""
    reader = ScriptedReader(HOLD_27B)
    assert g.wait_for_chat_idle("s", reader=reader) == 0.0 and reader.calls == 0
    assert g.get_config().mode == "off" and g.get_config().state_dir == isolate_lms_yield_gate


def test_dryrun_logs_the_hold_and_never_sleeps(tmp_path: Path) -> None:
    """dryrun records what it would have done, with the would-wait lower bound."""
    cfg, clock = YieldConfig(mode="dryrun", state_dir=tmp_path), FakeTime()
    reader = ScriptedReader(with_entry(entry(THIRTY_FIVE_B, last=NOW_MS - 42_000)))
    assert g.wait_for_chat_idle("judge", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep) == 0.0
    assert asyncio.run(g.await_chat_idle("judge", cfg=cfg, reader=reader, clock=clock.clock,
                                         sleep=clock.async_sleep)) == 0.0
    assert clock.sleeps == []
    for record in gate_lines(tmp_path):
        assert record["decision"] == "hold" and record["mode"] == "dryrun" and record["would_wait_s_est"] == 138.0
        assert record["waited_s"] == 0 and record["polls"] == 0 and record["capped"] is False
        assert record["reasons"] == [f"{THIRTY_FIVE_B}:recent 42s"] and record["site"] == "judge"


def test_on_mode_polls_every_3s_until_clear(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """on sleeps in poll_s steps while the chat is active and returns the seconds waited."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    reader = ScriptedReader(HOLD_27B, HOLD_27B, HOLD_27B, quiet())
    assert g.wait_for_chat_idle("page", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep) == 9.0
    assert clock.sleeps == [3.0, 3.0, 3.0]
    (record,) = gate_lines(tmp_path)
    assert record["decision"] == "hold" and record["waited_s"] == 9.0 and record["polls"] == 3
    assert record["capped"] is False and record["error"] is None
    out = capsys.readouterr().out
    assert f"⏸ [page] yielding to prod chat ({TWENTY_SEVEN_B}:generating)" in out
    assert "▶ [page] proceeding after 9 s" in out


def test_on_mode_async_polls_until_clear(tmp_path: Path) -> None:
    """The async form waits the same way (lms ps in a worker thread, async sleep)."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    reader = ScriptedReader(HOLD_27B, HOLD_27B, quiet())
    waited = asyncio.run(g.await_chat_idle("page", cfg=cfg, reader=reader, clock=clock.clock,
                                           sleep=clock.async_sleep))
    assert waited == 6.0 and clock.sleeps == [3.0, 3.0] and gate_lines(tmp_path)[0]["polls"] == 2


def test_on_mode_proceeds_at_the_cap(tmp_path: Path) -> None:
    """900 s of continuous chat: the unit proceeds with capped=true."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    waited = g.wait_for_chat_idle("page", cfg=cfg, reader=ScriptedReader(HOLD_27B), clock=clock.clock,
                                  sleep=clock.sleep)
    (record,) = gate_lines(tmp_path)
    assert waited == 900.0 and len(clock.sleeps) == 300 and record["capped"] is True and record["polls"] == 300


def test_reader_error_mid_wait_proceeds(tmp_path: Path) -> None:
    """An lms ps failure during a wait ends it (fail open) and is recorded."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    reader = ScriptedReader(HOLD_27B, None)
    assert g.wait_for_chat_idle("page", cfg=cfg, reader=reader, clock=clock.clock, sleep=clock.sleep) == 3.0
    (record,) = gate_lines(tmp_path)
    assert record["decision"] == "hold" and record["error"] and record["capped"] is False


def test_first_read_error_is_an_error_record(tmp_path: Path) -> None:
    """A failed first lms ps proceeds at once and logs decision=error."""
    cfg, clock = YieldConfig(mode="on", state_dir=tmp_path), FakeTime()
    assert g.wait_for_chat_idle("p", cfg=cfg, reader=ScriptedReader(None), clock=clock.clock,
                                sleep=clock.sleep) == 0.0
    (record,) = gate_lines(tmp_path)
    assert record["decision"] == "error" and record["error"] and clock.sleeps == []


def test_remote_base_url_skips_without_reading_lms_ps(tmp_path: Path) -> None:
    """A base_url on another machine logs skip_remote once per host and never reads lms ps."""
    cfg, clock, reader = YieldConfig(mode="on", state_dir=tmp_path), FakeTime(), ScriptedReader(HOLD_27B)
    for _ in range(2):
        assert g.wait_for_chat_idle("llm_client", base_url="http://192.168.1.50:1234/v1", cfg=cfg, reader=reader,
                                    clock=clock.clock, sleep=clock.sleep) == 0.0
    assert asyncio.run(g.await_chat_idle("x", base_url="amd-box.local:1234", cfg=cfg, reader=reader,
                                         clock=clock.clock, sleep=clock.async_sleep)) == 0.0
    records = gate_lines(tmp_path)
    assert reader.calls == 0 and clock.sleeps == []
    assert [(r["decision"], r["host"]) for r in records] == [("skip_remote", "192.168.1.50"),
                                                              ("skip_remote", "amd-box.local")]


def test_jsonl_records_parse_with_the_documented_field_set(tmp_path: Path) -> None:
    """Every record has exactly the documented fields (plus host for skip_remote) and stays short."""
    clock = FakeTime()
    for mode, answer, base_url in [("dryrun", quiet(), None), ("dryrun", HOLD_27B, None), ("on", None, None),
                                   ("on", HOLD_27B, "http://10.0.0.9:1234/v1")]:
        g.reset_caches()
        g.wait_for_chat_idle("transcribe:qwen3-vl-8b-heb-v22i-a50", base_url=base_url,
                             cfg=YieldConfig(mode=mode, state_dir=tmp_path), reader=ScriptedReader(answer),
                             clock=clock.clock, sleep=clock.sleep)
    lines = (tmp_path / g.GATE_LOG).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["decision"] for r in records] == ["clear", "hold", "error", "skip_remote"]
    for record in records:
        expected = GATE_FIELDS | ({"host"} if record["decision"] == "skip_remote" else set())
        assert set(record) == expected and record["pid"] == os.getpid()
    assert max(len(line.encode()) + 1 for line in lines) < 512


def test_unwritable_state_dir_never_raises(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A state dir that cannot be created costs a warning, not the unit."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    cfg, clock = YieldConfig(mode="dryrun", state_dir=blocker / "state"), FakeTime()
    assert g.wait_for_chat_idle("x", cfg=cfg, reader=ScriptedReader(HOLD_27B), clock=clock.clock,
                                sleep=clock.sleep) == 0.0
    with g.own_call(THIRTY_FIVE_B, cfg=cfg):
        pass
    assert "cannot append" in capsys.readouterr().out


# ── --summarize and the CLI ─────────────────────────────────────────────────────────────────────────────────────

def gate_record(pid: int, moment: float, decision: str, mode: str = "dryrun", est: Optional[float] = None,
                waited: float = 0.0) -> Dict[str, Any]:
    """A synthetic ``yield_gate.jsonl`` record.

    :param pid: Process id.
    :param moment: Epoch seconds.
    :param decision: clear/hold/error/skip_remote.
    :param mode: Gate mode.
    :param est: would_wait_s_est.
    :param waited: waited_s.
    :returns: Record.
    """
    return {"ts": g._iso(moment), "pid": pid, "argv0": f"job{pid}.py", "site": "s", "mode": mode,
            "decision": decision, "reasons": [f"{TWENTY_SEVEN_B}:generating"] if decision == "hold" else [],
            "ignored_own": [], "error": None, "waited_s": waited, "polls": 0, "capped": False,
            "would_wait_s_est": est, "ps_ms": 200.0}


def test_summarize_reconstructs_would_wait_spans_per_pid(tmp_path: Path) -> None:
    """Consecutive holds of one pid form one span; a clear, another pid or a > 300 s gap ends it."""
    t0 = NOW_S
    records = [gate_record(1, t0, "clear"), gate_record(1, t0 + 60, "hold", est=120),
               gate_record(2, t0 + 70, "clear"), gate_record(1, t0 + 120, "hold", est=60),
               gate_record(1, t0 + 200, "clear"), gate_record(1, t0 + 1000, "hold", est=150),
               gate_record(2, t0, "hold", est=30), gate_record(2, t0 + 400, "hold", est=30),
               gate_record(3, t0, "hold", mode="on", waited=74.0)]
    (tmp_path / g.GATE_LOG).write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n")
    spans = g.would_wait_spans(g.read_jsonl(tmp_path / g.GATE_LOG))
    assert [(s.start - t0, s.end - t0, s.holds) for s in spans[1]] == [(60, 180, 2), (1000, 1150, 1)]
    assert [(s.start - t0, s.end - t0) for s in spans[2]] == [(0, 30), (400, 430)]
    assert [(s.seconds, s.actual) for s in spans[3]] == [(74.0, True)]
    ratios = g.daily_ratios(g.read_jsonl(tmp_path / g.GATE_LOG), spans)
    assert ratios[(1, g._day(t0))] == (200.0, 270.0)
    text = g.render_summary(tmp_path)
    assert "pid 1 job1.py: 5 records (clear 2, hold 3)" in text and "would-wait" in text and "waited" in text


def test_summarize_since_filters_and_cli_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    """--summarize honours --since; an empty log is reported, not an error."""
    monkeypatch.setenv("LMS_YIELD_DIR", str(tmp_path))
    g.reset_caches()
    assert g.main(["--summarize"]) == 0 and "no gate records" in capsys.readouterr().out
    (tmp_path / g.GATE_LOG).write_text(json.dumps(gate_record(1, NOW_S, "hold", est=10)) + "\n")
    assert g.main(["--summarize"]) == 0 and "pid 1 job1.py: 1 records" in capsys.readouterr().out
    assert g.main(["--summarize", "--since", g._iso(NOW_S + 1)]) == 0
    assert "no gate records" in capsys.readouterr().out


def test_check_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """--check exits 3 on hold and 0 when clear or when lms ps fails (fail open)."""
    now_ms = int(time.time() * 1000)
    for answer, code, word in [(with_entry(entry(TWENTY_SEVEN_B, status="generating"), now_ms), 3, "HOLD"),
                               (quiet(now_ms), 0, "clear"), (None, 0, "fail open")]:
        g.reset_caches()
        monkeypatch.setattr(g, "read_lms_ps", lambda cli, timeout, answer=answer: answer)
        assert g.main(["--check"]) == code
        assert word in capsys.readouterr().out


def test_wait_cli_honours_off_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """--wait in off mode returns 0 without reading lms ps."""
    monkeypatch.setattr(g, "read_lms_ps", lambda cli, timeout: pytest.fail("off mode must not read lms ps"))
    assert g.main(["--wait", "--site", "stage"]) == 0 and "waited 0 s" in capsys.readouterr().out


# ── never raise: hosts and config ───────────────────────────────────────────────────────────────────────────────

def test_is_local_lms_is_string_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local forms are recognised without any DNS lookup; malformed input still gives a bool."""
    def no_dns(*args: Any, **kwargs: Any) -> None:
        """Fail the test on any resolver call."""
        raise AssertionError("DNS lookup attempted")

    monkeypatch.setattr(socket, "gethostbyname", no_dns)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    name = socket.gethostname()
    bare = name.removesuffix(".local")
    for url in [None, "", "http://localhost:1234/v1", "localhost:1234", "127.0.0.1:1234", "http://[::1]:1234", "::1",
                name, f"http://{bare}:1234/v1", f"{bare}.local:1234", f"http://{bare.upper()}.LOCAL:1234/v1"]:
        assert g.is_local_lms(url) is True, url
    for url in ["http://192.168.1.50:1234/v1", "amd-box.local:1234"]:
        assert g.is_local_lms(url) is False, url
    for url in ["http://[::1:1234", "://", "@@@", "http://user:pw@:1234", " "]:
        assert isinstance(g.is_local_lms(url), bool), url


def test_load_config_precedence(tmp_path: Path) -> None:
    """Process env beats .env, .env beats the default; a missing file gives defaults; os.environ is untouched."""
    env_file = tmp_path / ".env"
    env_file.write_text("LMS_YIELD_MODE=on\nLMS_YIELD_RECENT_S=120\nSOME_SECRET=not-for-the-gate\n")
    before = dict(os.environ)
    from_file = g.load_config(env={}, env_file=env_file)
    assert from_file.mode == "on" and from_file.recent_s == 120.0
    assert g.load_config(env={"LMS_YIELD_MODE": "off"}, env_file=env_file).mode == "off"
    defaults = g.load_config(env={}, env_file=tmp_path / "missing.env")
    assert defaults == YieldConfig(lms_cli=os.path.expanduser("~/.lmstudio/bin/lms"), state_dir=g.DEFAULT_STATE_DIR)
    assert defaults.mode == "dryrun" and dict(os.environ) == before


@pytest.mark.parametrize("key, value, field, default", [
    ("LMS_YIELD_RECENT_S", "abc", "recent_s", 180.0),
    ("LMS_YIELD_POLL_S", "-1", "poll_s", 3.0),
    ("LMS_YIELD_MAX_WAIT_S", "1e9", "max_wait_s", 900.0),
    ("LMS_YIELD_RECENT_S", "", "recent_s", 180.0),
    ("LMS_YIELD_MAX_WAIT_S", "3600", "max_wait_s", 900.0),
])
def test_load_config_rejects_bad_numbers_with_one_warning(tmp_path: Path, capsys: pytest.CaptureFixture, key: str,
                                                          value: str, field: str, default: float) -> None:
    """Bad or out-of-range numbers fall back to the default with one warning per key per process."""
    for _ in range(2):
        assert getattr(g.load_config(env={key: value}, env_file=tmp_path / "none.env"), field) == default
    assert capsys.readouterr().out.count(key) == 1


def test_load_config_mode_and_valid_values(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Mode is case-insensitive; an unknown mode is dryrun with one warning; valid numbers and dirs pass."""
    missing = tmp_path / "none.env"
    assert g.load_config(env={"LMS_YIELD_MODE": "ON"}, env_file=missing).mode == "on"
    assert g.load_config(env={"LMS_YIELD_MODE": " Off "}, env_file=missing).mode == "off"
    assert capsys.readouterr().out == ""
    assert g.load_config(env={"LMS_YIELD_MODE": "bogus"}, env_file=missing).mode == "dryrun"
    assert capsys.readouterr().out.count("LMS_YIELD_MODE") == 1
    cfg = g.load_config(env={"LMS_YIELD_MAX_WAIT_S": "2999", "LMS_YIELD_POLL_S": " 5 ",
                             "LMS_YIELD_DIR": "logs/elsewhere"}, env_file=missing)
    assert (cfg.max_wait_s, cfg.poll_s, cfg.state_dir) == (2999.0, 5.0, g.REPO_ROOT / "logs" / "elsewhere")


# ── hook placement (fake transports and lms ps; no LM Studio traffic) ──────────────────────────────────────────────

class FakeResponse:
    """aiohttp-style response with a fixed chat-completions body."""

    status = 200

    def raise_for_status(self) -> None:
        """Never fails."""

    async def json(self) -> Dict[str, Any]:
        """:returns: A one-choice completion."""
        return {"choices": [{"message": {"content": " ok "}}]}

    async def __aenter__(self) -> "FakeResponse":
        """:returns: Self."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Nothing to close."""


def fake_session(state_dir: Path, seen: List[List[Any]]) -> type:
    """aiohttp ``ClientSession`` stand-in that records the own-call markers present at POST time.

    :param state_dir: Gate state dir.
    :param seen: Receives one list of marker dicts per POST.
    :returns: The session class.
    """
    class Session:
        """Fake session."""

        def __init__(self, timeout: Any = None) -> None:
            """:param timeout: Ignored."""

        def post(self, url: str, json: Dict[str, Any], timeout: Any = None) -> FakeResponse:
            """:returns: A fake response, after recording the live markers."""
            seen.append([g._read_json_file(p)[0] for p in sorted((state_dir / g.OWN_CALLS_DIR).glob("*.json"))])
            return FakeResponse()

        async def __aenter__(self) -> "Session":
            """:returns: Self."""
            return self

        async def __aexit__(self, *exc: Any) -> None:
            """Nothing to close."""

    return Session


def dryrun_hooks(monkeypatch: pytest.MonkeyPatch, answer: Optional[List[Dict[str, Any]]]) -> None:
    """Switch the process gate to dryrun with a fake ``lms ps`` (the hooks pass no reader).

    :param monkeypatch: Pytest monkeypatch fixture.
    :param answer: What the fake ``lms ps`` returns.
    """
    monkeypatch.setenv("LMS_YIELD_MODE", "dryrun")
    monkeypatch.setattr(g, "read_lms_ps", lambda cli, timeout: answer)
    g.reset_caches()


def test_complete_text_hook_gates_then_marks_the_request_as_own(isolate_lms_yield_gate: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """complete_text: one gate record before the POST, the POST inside an own-call marker, one own_calls line."""
    from src.models.ocr import lms_transcriber as lms
    state, seen = isolate_lms_yield_gate, []
    dryrun_hooks(monkeypatch, quiet(int(time.time() * 1000)))
    monkeypatch.setattr(lms.aiohttp, "ClientSession", fake_session(state, seen))
    assert asyncio.run(lms.complete_text("qwen3.6-35b-a3b", "judge this")) == "ok"
    ((marker,),) = seen
    assert marker["model"] == THIRTY_FIVE_B and marker["end_ms"] is None
    (record,) = gate_lines(state)
    assert record["site"] == "complete_text:qwen3.6-35b-a3b" and record["decision"] == "clear"
    assert len((state / g.OWN_CALLS_LOG).read_text().splitlines()) == 1


def test_transcribe_hook_gates_without_an_own_call(isolate_lms_yield_gate: Path, monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    """transcribe_with_lm_studio (VLM): gated once, logged in dryrun, never registered as an own call."""
    from PIL import Image

    from src.models.ocr import lms_transcriber as lms
    state, seen = isolate_lms_yield_gate, []
    dryrun_hooks(monkeypatch, with_entry(entry(TWENTY_SEVEN_B, status="generating"), int(time.time() * 1000)))
    monkeypatch.setattr(lms.aiohttp, "ClientSession", fake_session(state, seen))
    image = tmp_path / "page.jpg"
    Image.new("RGB", (8, 8)).save(image)
    assert asyncio.run(lms.transcribe_with_lm_studio(VLM, str(image), "read")) == "ok"
    assert seen == [[]] and not (state / g.OWN_CALLS_LOG).exists()
    (record,) = gate_lines(state)
    assert record["site"] == f"transcribe:{VLM}" and record["decision"] == "hold" and record["waited_s"] == 0


def test_coreference_hook_gates_then_marks_the_request_as_own(isolate_lms_yield_gate: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """coreference_resolver._call_lms: gate record, POST inside an own-call marker on the 4B, payload unchanged."""
    from unittest import mock

    from src.models.llm.academic import coreference_resolver as cr
    state, seen, bodies = isolate_lms_yield_gate, [], []
    dryrun_hooks(monkeypatch, quiet(int(time.time() * 1000)))

    def post(url: str, json: Dict[str, Any], timeout: int) -> mock.MagicMock:
        """Record markers and body; answer with one valid completion."""
        seen.append([g._read_json_file(p)[0] for p in sorted((state / g.OWN_CALLS_DIR).glob("*.json"))])
        bodies.append(json)
        response = mock.MagicMock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        return response

    monkeypatch.setattr(cr.requests, "post", post)
    assert cr._call_lms("names", "http://localhost:1234/v1", FOUR_B, 5) == "{}"
    ((marker,),) = seen
    assert marker["model"] == FOUR_B and marker["end_ms"] is None
    assert bodies[0]["response_format"] == {"type": "text"}
    (record,) = gate_lines(state)
    assert record["site"] == "coreference_resolver" and record["decision"] == "clear"
