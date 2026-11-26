# File name: test_run_reader_decode_arms.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the decode-arm runner: the request cache, the orientation calibration and a resumable arm run (no model)."""
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from src.datasets.evaluations.helper_eval_scripts import run_reader_decode_arms as rd
from src.models.ocr import page_reader as pr
from src.models.ocr import reader_options as ro

TEXT = "بسم الله الرحمن الرحيم\nالى الشيخ ابو سعد اطال الله بقاه وادام عزه"


class Counter:
    """A fake transcriber that counts its calls."""

    def __init__(self, text: Optional[str] = TEXT) -> None:
        self.text, self.calls = text, 0

    async def __call__(self, model_id: str, image_path: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1200,
                       **kwargs: Any) -> Optional[Dict[str, Any]]:
        self.calls += 1
        if self.text is None:
            return None
        return {"text": self.text, "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _image(path: Path, colour: str = "white") -> str:
    Image.new("RGB", (30, 20), colour).save(path)
    return str(path)


def test_the_cache_answers_a_repeated_request_and_keys_on_the_image_bytes(tmp_path: Path) -> None:
    a, b = _image(tmp_path / "a.png"), _image(tmp_path / "b.png", "black")
    fake = Counter()
    cache = rd.RequestCache(tmp_path / "c.jsonl", transcribe=fake)
    first = asyncio.run(cache("m", a, "p", temperature=0.1, max_tokens=1200))
    again = asyncio.run(cache("m", a, "p", temperature=0.1, max_tokens=1200))
    assert first == again and fake.calls == 1 and (cache.hits, cache.misses) == (1, 1)
    asyncio.run(cache("m", a, "p", temperature=0.0, max_tokens=1200))
    asyncio.run(cache("m", a, "p", temperature=0.1, max_tokens=48, extra_payload={"logprobs": True}))
    asyncio.run(cache("m", b, "p", temperature=0.1, max_tokens=1200))
    assert fake.calls == 4
    reloaded = rd.RequestCache(tmp_path / "c.jsonl", transcribe=Counter(), cache_only=True)
    assert asyncio.run(reloaded("m", a, "p", temperature=0.0, max_tokens=1200))["text"] == TEXT
    with pytest.raises(rd.CacheMiss):
        asyncio.run(reloaded("m", a, "other prompt", temperature=0.0, max_tokens=1200))


def test_failures_are_not_cached_and_no_cache_reads_fresh(tmp_path: Path) -> None:
    a = _image(tmp_path / "a.png")
    failing = Counter(None)
    cache = rd.RequestCache(tmp_path / "c.jsonl", transcribe=failing)
    assert asyncio.run(cache("m", a, "p")) is None and not (tmp_path / "c.jsonl").exists()
    fresh = Counter()
    asyncio.run(rd.RequestCache(tmp_path / "c.jsonl", transcribe=Counter()).__call__("m", a, "p"))
    uncached = rd.RequestCache(tmp_path / "c.jsonl", transcribe=fresh, enabled=False)
    asyncio.run(uncached("m", a, "p"))
    asyncio.run(uncached("m", a, "p"))
    assert fresh.calls == 2 and uncached.hits == 0


def _probe(mean: float) -> Dict[str, Any]:
    return {"ok": True, "mean_logprob": mean, "first_logprob": mean, "script_share": 0.9, "repeat_share": 0.0}


def test_presented_probes_shift_the_turns() -> None:
    absolute = {0: _probe(-0.1), 90: _probe(-1.0), 180: _probe(-2.0), 270: _probe(-3.0)}
    shown = rd.presented_probes(absolute, 90)        # arrived turned 90: a further 270 brings it upright
    assert shown[270]["mean_logprob"] == -0.1 and shown[0]["mean_logprob"] == -1.0


def test_calibration_counts_and_chooses_a_margin_that_keeps_upright_pages() -> None:
    clear = {"probes": {"0": _probe(-0.2), "90": _probe(-1.0), "180": _probe(-0.9), "270": _probe(-1.1)}}
    close = {"probes": {"0": _probe(-0.5), "90": _probe(-1.0), "180": _probe(-0.6), "270": _probe(-1.1)}}
    outcome = rd.calibration_outcomes([clear], "mean_logprob", 0.3)
    assert outcome == {"upright_kept": 1, "upright_total": 1, "turned_fixed": 3, "turned_total": 3}
    assert rd.calibration_outcomes([close], "mean_logprob", 0.3)["turned_fixed"] == 2     # 180 -> 0 needs a 0.1 lead only
    result = rd.calibrate([clear, close], ("mean_logprob",), (0.0, 0.05, 0.1, 0.2, 0.5, 1.0))
    assert result["safe"] and result["outcome"]["upright_kept"] == 2 and result["outcome"]["turned_fixed"] == 6
    assert result["margin"] in (0.0, 0.05)


def test_settings_from_args_reads_the_calibration(tmp_path: Path) -> None:
    (tmp_path / "cal.json").write_text(json.dumps({"statistic": "first_logprob", "margin": 0.45}))
    args = rd.argparse.Namespace(loop_detector="markers", script_filter=True, orientation="probe", orientation_margin=0.3,
                                 orientation_statistic="mean_logprob", orientation_margin_from=str(tmp_path / "cal.json"),
                                 crops="auto", crop_min_mp=40.0, crop_max_density=1.5)
    settings = rd.settings_from_args(args, tmp_path)
    assert (settings.orientation_margin, settings.orientation_statistic) == (0.45, "first_logprob")
    assert pr.options_tag(settings) == "markers+probe+crops:auto+filter" and settings.view_dir == str(tmp_path / "views")


def test_run_arm_writes_one_row_per_image_and_resumes(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    panel = [{"doc_id": f"D{i}", "image_index": 0, "image_file": f"D{i}.png", "image_class": "covered"} for i in range(3)]
    for row, colour in zip(panel, ("white", "grey", "black")):     # distinct bytes: one request each
        _image(tmp_path / "images" / row["image_file"], colour)
    fake = Counter()
    cache = rd.RequestCache(tmp_path / "c.jsonl", transcribe=fake)
    settings = replace(pr.DEFAULT_SETTINGS, blank_gate=False, view_dir=str(tmp_path / "views"))
    kwargs = dict(free_check=lambda: 100.0, served_check=lambda m: True, unload=lambda m: "")
    stats = asyncio.run(rd.run_arm(panel, "m", "arm0", tmp_path, tmp_path, settings, cache, limit=2, **kwargs))
    assert stats["answered"] == 2 and stats["misses"] == 2
    stats = asyncio.run(rd.run_arm(panel, "m", "arm0", tmp_path, tmp_path, settings, cache, **kwargs))
    assert stats["skipped"] == 2 and stats["answered"] == 1
    rows = [json.loads(l) for l in (tmp_path / "arm0.jsonl").read_text().splitlines()]
    assert [r["doc_id"] for r in rows] == ["D0", "D1", "D2"] and all(r["text"] == TEXT and r["arm"] == "arm0" for r in rows)
    assert set(pr.READ_FIELDS) <= set(rows[0])
    stats = asyncio.run(rd.run_arm(panel, "m", "arm1", tmp_path, tmp_path, replace(settings, loop_detector=ro.LOOP_MARKERS), cache,
                                   **kwargs))
    assert stats["hits"] == 3 and stats["misses"] == 0 and fake.calls == 3


def test_a_cache_only_replay_never_unloads(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    panel = [{"doc_id": "D0", "image_index": 0, "image_file": "D0.png", "image_class": "covered"}]
    _image(tmp_path / "images" / "D0.png", "white")
    settings = replace(pr.DEFAULT_SETTINGS, blank_gate=False, view_dir=str(tmp_path / "views"))
    asyncio.run(rd.run_arm(panel, "m", "arm0", tmp_path, tmp_path, settings, rd.RequestCache(tmp_path / "c.jsonl", transcribe=Counter()),
                           free_check=lambda: 100.0, served_check=lambda m: True, unload=lambda m: ""))
    def unload(model: str) -> str:
        raise AssertionError("a cache-only replay must not unload")
    replay = rd.RequestCache(tmp_path / "c.jsonl", cache_only=True)
    stats = asyncio.run(rd.run_arm(panel, "m", "replay", tmp_path, tmp_path, settings, replay,
                                   free_check=lambda: 30.0, served_check=lambda m: True, unload=unload, unload_below_gb=40.0,
                                   min_free_gb=10.0))
    assert stats["answered"] == 1 and stats["hits"] == 1 and stats["unloads"] == 0


def test_unload_refuses_production_models() -> None:
    with pytest.raises(ValueError):
        rd.unload_own_model("qwen/qwen3.6-35b-a3b")
