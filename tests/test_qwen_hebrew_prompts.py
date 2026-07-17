"""Prompt integrity tests for the Qwen3-VL Hebrew fine-tuning pipeline.

The training module keeps a byte-copy of the eval's section-extraction
prompts (the eval file can't be imported in the training venv because it
pulls wandb/kraken/google-cloud-vision). These tests pin the copies to the
originals — if the eval prompts change, training prompts must be re-synced.
"""

import pytest

from src.finetuning.qwen_hebrew.prompts import (
    CROP_TRANSCRIBE_PROMPTS,
    SECTION_EXTRACT_PROMPTS,
    SECTIONS,
    prompt_for,
)


def test_section_extract_prompts_byte_equal_to_eval() -> None:
    """Training Task-B prompts must be byte-identical to the eval prompts."""
    eval_module = pytest.importorskip(
        "src.datasets.evaluations.talmud_evaluation",
        reason="eval-only dependencies not installed in this venv",
    )
    assert SECTION_EXTRACT_PROMPTS == eval_module.SECTION_EXTRACT_PROMPTS, (
        "SECTION_EXTRACT_PROMPTS drifted from talmud_evaluation.py — "
        "the eval wording is empirically load-bearing; re-sync the copy."
    )


def test_all_sections_covered() -> None:
    """Both prompt families must cover exactly the three sections."""
    assert set(CROP_TRANSCRIBE_PROMPTS) == set(SECTIONS)
    assert set(SECTION_EXTRACT_PROMPTS) == set(SECTIONS)


def test_crop_prompts_share_core_instructions() -> None:
    """Crop prompts must keep the eval's anti-hallucination instructions."""
    for section, prompt in CROP_TRANSCRIBE_PROMPTS.items():
        assert "Do NOT correct or complete from memory" in prompt, section
        assert "[?]" in prompt, section
        assert "line breaks" in prompt, section


def test_prompt_for_dispatch() -> None:
    """``prompt_for`` must dispatch to the right family."""
    assert prompt_for("crop_transcribe", "gemara") == CROP_TRANSCRIBE_PROMPTS["gemara"]
    assert prompt_for("page_extract", "rashi") == SECTION_EXTRACT_PROMPTS["rashi"]
    with pytest.raises(KeyError):
        prompt_for("unknown_task", "gemara")
