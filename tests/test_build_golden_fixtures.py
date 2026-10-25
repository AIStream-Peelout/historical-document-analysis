# File name: test_build_golden_fixtures.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""The golden fixtures hold no edition text: :func:`src.embeddings.build_golden_fixtures.mask_edition_text`.

The fixtures live in a public repo, so the transcription and translation values of their records are masked letter by
letter. These tests pin the mask's shape rules and check that every committed row is already masked.
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.embeddings.build_golden_fixtures import (
    DEFAULT_OUT_DIR,
    V3_FIXTURE_NAME,
    V4_FIXTURE_NAME,
    fixture_rows,
    mask_edition_text,
    mask_existing,
    write_jsonl,
)

SAMPLE = "\n".join([
    "Document ID: Example_1",
    "Shelf Mark: T-S 99.1",
    "Description: Letter about a shipment of flax.",
    "Transcription: [Editor: Example, Editor] Editor: Gil, Moshe כתאבי אליך מן אלאסכנדריה [..] יום 3",
    "Translation: My lord, I write to you from Alexandria on day 3.",
    "Language: Judaeo-Arabic",
])


def _rows(name: str) -> List[Dict[str, object]]:
    """Rows of a committed fixture.

    :param name: Fixture file name.
    :return: Parsed rows.
    """
    with open(DEFAULT_OUT_DIR / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_mask_keeps_shape_and_credits() -> None:
    """Length, non-letters, labels, other fields and editor credits survive; the edition letters do not."""
    masked = mask_edition_text(SAMPLE, "Example_1")
    assert len(masked) == len(SAMPLE)
    old_lines, new_lines = SAMPLE.split("\n"), masked.split("\n")
    assert [ln for ln in new_lines if not ln.startswith(("Transcription:", "Translation:"))] == \
        [ln for ln in old_lines if not ln.startswith(("Transcription:", "Translation:"))]
    transcription = new_lines[3]
    assert transcription.startswith("Transcription: [Editor: Example, Editor] Editor: Gil, Moshe ")
    assert "כתאבי" not in transcription and "אלאסכנדריה" not in transcription
    head, tail = transcription.split("[..]")
    assert tail.endswith(" 3") and "יום" not in tail  # brackets and digits stay, the letters of "יום" do not
    assert new_lines[4].startswith("Translation: ") and "Alexandria" not in new_lines[4]
    for old, new in zip(SAMPLE, masked):
        assert old.isalpha() == new.isalpha()
        assert old.isupper() == new.isupper()


def test_mask_is_idempotent_and_record_specific() -> None:
    """Masking twice changes nothing; another record id gives other letters."""
    once = mask_edition_text(SAMPLE, "Example_1")
    assert mask_edition_text(once, "Example_1") == once
    assert mask_edition_text(SAMPLE, "Example_2") != once


def test_text_without_edition_fields_is_unchanged() -> None:
    """A record with no transcription or translation is returned as is."""
    text = "Document ID: Example_3\nDescription: כתאבי אליך\nAlternate titles: [פיוט] אשמים קלים"
    assert mask_edition_text(text, "Example_3") == text


@pytest.mark.parametrize("name", [V4_FIXTURE_NAME, V3_FIXTURE_NAME])
def test_committed_fixtures_hold_no_edition_text(name: str) -> None:
    """Every committed row is already masked: masking its text again leaves it unchanged."""
    rows = _rows(name)
    assert rows
    unmasked = [r["doc_id"] for r in rows if mask_edition_text(r["v8_text"], r["doc_id"]) != r["v8_text"]]
    assert unmasked == []


def test_mask_existing_rewrites_both_fixtures(tmp_path: Path) -> None:
    """``--mask-existing`` masks the stored texts and recomputes both rows from them."""
    v4, v3 = fixture_rows(SAMPLE, "Example_1", "random")
    write_jsonl(tmp_path / V4_FIXTURE_NAME, [v4])
    write_jsonl(tmp_path / V3_FIXTURE_NAME, [v3])
    counts = mask_existing(tmp_path)
    assert counts["rows"] == 1 and counts["masked"] == 1
    new_v4 = json.loads((tmp_path / V4_FIXTURE_NAME).read_text(encoding="utf-8"))
    new_v3 = json.loads((tmp_path / V3_FIXTURE_NAME).read_text(encoding="utf-8"))
    expected_v4, expected_v3 = fixture_rows(mask_edition_text(SAMPLE, "Example_1"), "Example_1", "random")
    assert new_v4 == expected_v4 and new_v3 == expected_v3
    assert mask_existing(tmp_path)["masked"] == 0
