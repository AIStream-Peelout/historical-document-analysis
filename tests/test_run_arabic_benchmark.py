# File name: test_run_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script benchmark runner (prompt variants, resume, safeguards) with a fake model."""
import asyncio
import json

import pytest

from src.datasets.consensus.consensus_gate import build_fragment_prompt
from src.datasets.evaluations.helper_eval_scripts import run_arabic_benchmark as rab


@pytest.fixture()
def bench(tmp_path):
    records = [{"id": "Cambridge_CUL_T_S_Ar_38_31", "images": [{"file": "a__0.jpg", "image_index": 0}, {"file": "a__1.jpg", "image_index": 1}],
                "gt": {"sections": []}},
               {"id": "Oxford_Bodleian_MS_heb_d_80_43", "images": [{"file": "b__0.jpg", "image_index": 0}], "gt": {"sections": []}}]
    with open(tmp_path / "benchmark.jsonl", "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return tmp_path


def test_prompt_variants():
    doc = "Cambridge_CUL_T_S_Ar_38_31"
    assert rab.build_prompt(doc, "standard") == build_fragment_prompt(doc)       # the benchmarked prompt, untouched
    arabic = rab.build_prompt(doc, "arabic")
    assert "The text is written in Arabic script." in arabic and "Hebrew" not in arabic and "nikud" not in arabic
    assert arabic.splitlines()[0] == build_fragment_prompt(doc).splitlines()[0]  # same framing line
    assert arabic.endswith("Return ONLY the transcription with no commentary.")
    trained = rab.build_prompt(doc, "trained")                                    # the training rows' own wording
    assert "handwritten Arabic script" in trained and "Hebrew" not in trained and trained != arabic
    assert set(rab.PROMPT_VARIANTS) == {"arabic", "standard", "trained"}
    with pytest.raises(ValueError):
        rab.build_prompt(doc, "hebrew")


def test_output_path_is_a_file_name(tmp_path):
    assert rab.output_path(tmp_path, "qwen/qwen3-vl-8b", "arabic") == tmp_path / "outputs" / "qwen_qwen3-vl-8b__arabic.jsonl"


def test_run_decodes_every_image_once_and_resumes(bench):
    calls = []

    async def fake(model, image_path, prompt, max_tokens=None):
        calls.append((model, image_path.rsplit("/", 1)[-1], max_tokens, "Arabic script" in prompt))
        return None if image_path.endswith("a__1.jpg") and len(calls) < 3 else "بسم الله"

    stats = asyncio.run(rab.run(bench, "qwen/m", "arabic", transcribe=fake, served_check=lambda m: True, free_check=lambda: 99.0))
    assert stats == {"answered": 2, "failed": 1, "skipped": 0, "stopped": 0}
    assert calls == [("qwen/m", "a__0.jpg", 2500, True), ("qwen/m", "a__1.jpg", 2500, True), ("qwen/m", "b__0.jpg", 2500, True)]
    stats = asyncio.run(rab.run(bench, "qwen/m", "arabic", transcribe=fake, served_check=lambda m: True, free_check=lambda: 99.0))
    assert stats == {"answered": 1, "failed": 0, "skipped": 2, "stopped": 0} and calls[-1][1] == "a__1.jpg"   # only the failed image again
    rows = [json.loads(l) for l in rab.output_path(bench, "qwen/m", "arabic").read_text(encoding="utf-8").splitlines()]
    assert [(r["doc_id"], r["image_index"], r["text"]) for r in rows][1] == ("Cambridge_CUL_T_S_Ar_38_31", 1, None)
    assert rab.pending_jobs(rab.load_benchmark(bench), rab.output_path(bench, "qwen/m", "arabic")) == ([], 3)
    standard = asyncio.run(rab.run(bench, "qwen/m", "standard", limit=1, transcribe=fake, served_check=lambda m: True, free_check=lambda: 99.0))
    assert standard["answered"] == 2 and calls[-1][3] is False              # its own file, the Hebrew-script prompt, first document only


def test_run_stops_when_the_model_is_gone_and_waits_on_the_disk_floor(bench):
    async def dead(model, image_path, prompt, max_tokens=None):
        return None

    stats = asyncio.run(rab.run(bench, "gone", "arabic", transcribe=dead, served_check=lambda m: False, free_check=lambda: 99.0))
    assert stats == {"answered": 0, "failed": 0, "skipped": 0, "stopped": 1}
    assert rab.output_path(bench, "gone", "arabic").read_text() == ""       # nothing recorded for the request that found no model

    free = iter([5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

    async def ok(model, image_path, prompt, max_tokens=None):
        return "نص"

    stats = asyncio.run(rab.run(bench, "m", "arabic", limit=1, transcribe=ok, served_check=lambda m: True,
                                free_check=lambda: next(free), wait_s=0.0))
    assert stats["answered"] == 2
