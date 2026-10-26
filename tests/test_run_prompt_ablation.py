# File name: test_run_prompt_ablation.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the Arabic-script prompt-ablation runner (prompts, resume, yielding, safeguards) with a fake model."""
import asyncio
import json
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_prompt_ablation as rpa

PANEL = [{"doc_id": "a", "image_index": 0, "image_file": "a__0.jpg", "image_class": "covered"},
         {"doc_id": "a", "image_index": 1, "image_file": "a__1.jpg", "image_class": "hebrew_side"},
         {"doc_id": "b", "image_index": 0, "image_file": "b__0.jpg", "image_class": "blank_faint"}]
ROW_FIELDS = {"doc_id", "image_index", "image_file", "image_class", "text", "finish_reason", "prompt_tokens", "completion_tokens",
              "secs", "model", "prompt", "max_tokens", "temperature"}


def _meta(text: Optional[str], finish: str = "stop") -> Dict[str, Any]:
    """A :func:`lms_transcriber.completion_meta`-shaped result.

    :param text: Answer text.
    :type text: Optional[str]
    :param finish: Finish reason.
    :type finish: str
    :return: Meta dict.
    :rtype: Dict[str, Any]
    """
    return {"text": text, "finish_reason": finish, "prompt_tokens": 900, "completion_tokens": 12, "total_tokens": 912}


def _rows(path: Path) -> List[Dict[str, Any]]:
    """JSON lines of a file.

    :param path: File.
    :type path: Path
    :return: Rows.
    :rtype: List[Dict[str, Any]]
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- prompts and paths ------------------------------------------------------------------------------------------------

def test_prompt_variants() -> None:
    from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import ARABIC_FRAGMENT_PROMPT
    trained = rpa.build_prompt(rpa.TRAINED)
    assert trained == ARABIC_FRAGMENT_PROMPT
    noscript = rpa.build_prompt(rpa.NO_SCRIPT)
    assert rpa.NO_SCRIPT_CLAUSE in noscript and "Arabic" not in noscript and "Hebrew" not in noscript
    assert noscript == trained.replace("handwritten Arabic script", rpa.NO_SCRIPT_CLAUSE)   # nothing but the clause changes
    stopping = rpa.build_prompt(rpa.STOPPING)
    assert stopping == rpa.STOPPING_PROMPT and rpa.ABSTAIN_MARKER in stopping
    assert set(rpa.VARIANTS) == {"trained", "noscript", "stopping"}
    with pytest.raises(ValueError):
        rpa.build_prompt("arabic")


def test_output_path(tmp_path: Path) -> None:
    assert rpa.output_path(tmp_path, "qwen/qwen3-vl-8b", "noscript", 1200) == tmp_path / "qwen_qwen3-vl-8b__noscript__mt1200.jsonl"
    assert rpa.output_path(tmp_path, "m", "trained", 2500).name == "m__trained__mt2500.jsonl"


def test_pending_jobs(tmp_path: Path) -> None:
    out = tmp_path / "arm.jsonl"
    assert rpa.pending_jobs(PANEL, out) == (PANEL, 0)
    out.write_text("\n".join(json.dumps(r) for r in [{"doc_id": "a", "image_index": 0, "text": "نص"},
                                                     {"doc_id": "a", "image_index": 1, "text": None},
                                                     {"doc_id": "b", "image_index": "0", "text": ""}]) + "\n\n", encoding="utf-8")
    jobs, answered = rpa.pending_jobs(PANEL, out)
    assert jobs == [PANEL[1]] and answered == 2                  # a null text is retried; an empty string is an answer
    assert rpa.pending_jobs(PANEL, out, limit=1) == ([], 1)
    assert rpa.pending_jobs(PANEL, out, limit=2) == ([PANEL[1]], 1)


# --- LM Studio state --------------------------------------------------------------------------------------------------

def test_busy_others() -> None:
    models = [{"identifier": "mine", "status": "generating"}, {"identifier": "chat", "status": "PROCESSINGPROMPT"},
              {"identifier": "idle", "status": "IDLE"}, {"identifier": "quiet"}, {"status": "loading"}]
    assert rpa.busy_others(models, "mine") == ["chat", "?"]
    assert rpa.busy_others([{"identifier": "mine", "status": "generating"}], "mine") == []
    assert rpa.busy_others([{"identifier": "x", "status": "Idle"}], "mine") == []
    assert rpa.busy_others([], "mine") == []


def test_model_info() -> None:
    models = [{"identifier": "other", "status": "idle"},
              {"identifier": "mine", "status": "idle", "contextLength": 8192, "maxContextLength": 262144, "parallel": 1,
               "quantization": "Q8_0", "path": "ignored", "sizeBytes": 1}]
    assert rpa.model_info(models, "mine") == {"identifier": "mine", "contextLength": 8192, "maxContextLength": 262144,
                                              "status": "idle", "parallel": 1, "quantization": "Q8_0"}
    assert rpa.model_info([{"identifier": "mine"}], "mine")["contextLength"] is None
    assert rpa.model_info(models, "absent") is None


def _script(tmp_path: Path, name: str, body: str) -> Path:
    """An executable shell script standing in for ``lms``.

    :param tmp_path: Directory.
    :type tmp_path: Path
    :param name: File name.
    :type name: str
    :param body: Shell lines after the shebang.
    :type body: str
    :return: Its path.
    :rtype: Path
    """
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_lms_models(tmp_path: Path) -> None:
    assert rpa.lms_models(tmp_path / "missing" / "lms") == []
    listed = _script(tmp_path, "lms_ok", """echo '[{"identifier": "m", "status": "idle", "contextLength": 4096}]'""")
    assert rpa.lms_models(listed) == [{"identifier": "m", "status": "idle", "contextLength": 4096}]
    assert rpa.lms_models(_script(tmp_path, "lms_garbage", "echo 'No models are loaded'")) == []
    assert rpa.lms_models(_script(tmp_path, "lms_fails", "echo '[]'; exit 3")) == []


def test_lms_models_bracketed_non_json_is_empty(tmp_path: Path) -> None:
    assert rpa.lms_models(_script(tmp_path, "lms_log", "echo '[warn] daemon starting'")) == []
    assert rpa.lms_models(_script(tmp_path, "lms_obj", "echo '{\"a\": 1}'")) == []


# --- run --------------------------------------------------------------------------------------------------------------

class FakeModel:
    """A fake ``transcribe_with_lm_studio_meta`` that records its calls."""

    def __init__(self, answers: Optional[Dict[str, Optional[Dict[str, Any]]]] = None) -> None:
        """
        :param answers: Meta per image file name (default: an Arabic answer for every image).
        :type answers: Optional[Dict[str, Optional[Dict[str, Any]]]]
        """
        self.answers = answers or {}
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, model: str, image_path: str, prompt: str, temperature: float = 0.0,
                       max_tokens: int = 0) -> Optional[Dict[str, Any]]:
        """Answer one image.

        :param model: Model key.
        :type model: str
        :param image_path: Image path.
        :type image_path: str
        :param prompt: Prompt.
        :type prompt: str
        :param temperature: Temperature.
        :type temperature: float
        :param max_tokens: Budget.
        :type max_tokens: int
        :return: Meta or None.
        :rtype: Optional[Dict[str, Any]]
        """
        name = Path(image_path).name
        self.calls.append({"model": model, "image": image_path, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens})
        return self.answers.get(name, _meta("بسم الله"))


def _run(tmp_path: Path, transcribe: FakeModel, served: bool = True, status: Any = None, **kwargs: Any) -> Dict[str, int]:
    """Run one arm with fakes.

    :param tmp_path: Benchmark and output root.
    :type tmp_path: Path
    :param transcribe: Fake model.
    :type transcribe: FakeModel
    :param served: What ``served_check`` answers.
    :type served: bool
    :param status: ``status_check`` (default: nothing loaded).
    :type status: Any
    :return: Run stats.
    :rtype: Dict[str, int]
    """
    return asyncio.run(rpa.run(PANEL, "qwen/m", "trained", 1200, tmp_path / "out", tmp_path, transcribe=transcribe,
                               served_check=lambda m: served, free_check=lambda: 99.0, status_check=status or (lambda: []),
                               prompt_text="x", wait_s=0.0, **kwargs))


def test_run_decodes_every_image_once_writes_rows_and_resumes(tmp_path: Path) -> None:
    model = FakeModel({"a__1.jpg": _meta("كلام طويل", "length")})
    stats = _run(tmp_path, model, temperature=0.2)
    assert stats == {"answered": 3, "failed": 0, "skipped": 0, "stopped": 0, "yield_s": 0}
    assert [Path(c["image"]) for c in model.calls] == [tmp_path / "images" / r["image_file"] for r in PANEL]
    assert all(c["prompt"] == "x" and c["max_tokens"] == 1200 and c["temperature"] == 0.2 and c["model"] == "qwen/m" for c in model.calls)
    out = rpa.output_path(tmp_path / "out", "qwen/m", "trained", 1200)
    rows = _rows(out)
    assert [set(r) for r in rows] == [ROW_FIELDS] * 3
    first, second = rows[0], rows[1]
    assert {k: v for k, v in first.items() if k != "secs"} == {
        "doc_id": "a", "image_index": 0, "image_file": "a__0.jpg", "image_class": "covered", "text": "بسم الله",
        "finish_reason": "stop", "prompt_tokens": 900, "completion_tokens": 12, "model": "qwen/m", "prompt": "trained",
        "max_tokens": 1200, "temperature": 0.2}
    assert isinstance(first["secs"], float) and second["finish_reason"] == "length" and second["text"] == "كلام طويل"
    again = _run(tmp_path, model)
    assert again == {"answered": 0, "failed": 0, "skipped": 3, "stopped": 0, "yield_s": 0} and len(model.calls) == 3
    assert len(_rows(out)) == 3


def test_run_limit(tmp_path: Path) -> None:
    model = FakeModel()
    assert _run(tmp_path, model, limit=2)["answered"] == 2 and len(model.calls) == 2


def test_run_counts_a_failed_request_with_the_model_served_and_retries_it(tmp_path: Path) -> None:
    model = FakeModel({"a__1.jpg": None, "b__0.jpg": _meta(None, "length")})
    stats = _run(tmp_path, model)
    assert stats == {"answered": 1, "failed": 2, "skipped": 0, "stopped": 0, "yield_s": 0}
    rows = _rows(rpa.output_path(tmp_path / "out", "qwen/m", "trained", 1200))
    assert rows[1]["text"] is None and rows[1]["finish_reason"] is None and rows[1]["completion_tokens"] is None
    assert rows[2]["text"] is None and rows[2]["finish_reason"] == "length"
    model.answers = {}
    assert _run(tmp_path, model) == {"answered": 2, "failed": 0, "skipped": 1, "stopped": 0, "yield_s": 0}
    assert [Path(c["image"]).name for c in model.calls[3:]] == ["a__1.jpg", "b__0.jpg"]


def test_run_stops_when_the_model_is_gone(tmp_path: Path) -> None:
    model = FakeModel({"a__1.jpg": None})
    stats = _run(tmp_path, model, served=False)
    assert stats == {"answered": 1, "failed": 0, "skipped": 0, "stopped": 1, "yield_s": 0}
    assert len(model.calls) == 2                                      # nothing after the request that found no model
    rows = _rows(rpa.output_path(tmp_path / "out", "qwen/m", "trained", 1200))
    assert [(r["doc_id"], r["image_index"]) for r in rows] == [("a", 0)]   # and no row for it


def test_run_yields_while_another_model_is_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        """Record a wait without waiting.

        :param seconds: Requested wait.
        :type seconds: float
        """
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(rpa.asyncio, "sleep", fake_sleep)
    states = iter([[{"identifier": "chat", "status": "generating"}, {"identifier": "qwen/m", "status": "generating"}],
                   [{"identifier": "chat", "status": "idle"}]])
    status_calls: List[int] = []

    def status() -> List[Dict[str, Any]]:
        """Busy on the first check, idle after.

        :return: Loaded models.
        :rtype: List[Dict[str, Any]]
        """
        status_calls.append(1)
        return next(states, [])

    model = FakeModel()
    stats = asyncio.run(rpa.run(PANEL, "qwen/m", "trained", 1200, tmp_path / "out", tmp_path, transcribe=model,
                                served_check=lambda m: True, free_check=lambda: 99.0, status_check=status, prompt_text="x",
                                wait_s=2.0))
    assert stats["yield_s"] == 2 and sleeps == [2.0]                  # one wait, before the first request
    assert stats["answered"] == 3 and len(model.calls) == 3           # the request still happened
    assert len(status_calls) == 4                                     # busy, idle, then one check per later image


def test_run_waits_on_the_disk_floor(tmp_path: Path) -> None:
    readings = [5.0, 5.0, 50.0]
    checks: List[float] = []

    def free() -> float:
        """Below the floor twice (check and log line), then above it.

        :return: GiB free.
        :rtype: float
        """
        checks.append(readings[min(len(checks), len(readings) - 1)])
        return checks[-1]

    model = FakeModel()
    stats = asyncio.run(rpa.run(PANEL[:1], "m", "stopping", 2500, tmp_path / "out", tmp_path, transcribe=model,
                                served_check=lambda m: True, free_check=free, status_check=lambda: [],
                                prompt_text="x", wait_s=0.0, min_free_gb=10.0))
    assert stats["answered"] == 1 and checks == [5.0, 5.0, 50.0]      # waited once, then decoded
