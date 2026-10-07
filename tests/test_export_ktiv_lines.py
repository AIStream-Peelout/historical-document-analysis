"""Unit tests for the Kraken line exporter's pure helpers."""
import numpy as np

from src.finetuning.kraken.export_ktiv_lines import (
    clean_text,
    column_pitch,
    ink_centre,
    out_of_codec,
    split_manuscripts,
)
from src.finetuning.qwen_hebrew.ktiv_layout import GAP_TOKEN


def test_clean_text_rejects_gaps_and_uncertain_words() -> None:
    assert clean_text(f"אמר רבי {GAP_TOKEN} יוחנן") is None
    assert clean_text("אמר ר..י יוחנן") is None
    assert clean_text("אמר [רבי] יוחנן") is None
    assert clean_text("ז @֒ ואילו") is None
    assert clean_text("אב") is None


def test_clean_text_normalises_whitespace_and_nfkd() -> None:
    assert clean_text("  אמר   ר'  יוחנן ") == "אמר ר' יוחנן"
    # NFKD decomposes precomposed shin-dot forms; plain letters are unchanged.
    assert clean_text("שלום") == "שלום"


def test_out_of_codec_lists_unknown_characters_once() -> None:
    codec = frozenset("אבגד '")
    assert out_of_codec("אב גד", codec) == []
    assert out_of_codec("אᴗב ᴗ", codec) == ["ᴗ"]


def test_column_pitch_uses_centre_spacing_not_box_height() -> None:
    # Generous boxes (200 px) on a 160 px pitch.
    lines = [{"box": (0, y, 100, y + 200)} for y in (0, 160, 320, 480)]
    assert column_pitch(lines, h_med=200.0) == 160.0
    assert column_pitch(lines[:1], h_med=200.0) == 0.8 * 200.0


def test_ink_centre_snaps_to_the_dark_band_near_the_box_centre() -> None:
    page = np.full((400, 300), 230, dtype=np.uint8)
    page[150:190, :] = 40           # this line's ink, centre 170
    page[310:350, :] = 40           # the next line, centre 330 (pitch 160)
    # Box centre 20 px off the ink -> re-centred on the ink.
    assert abs(ink_centre(page, 0, 300, 150.0, 160.0) - 170.0) <= 4
    # A box centred between the lines but nearer to this one stays on this one.
    assert abs(ink_centre(page, 0, 300, 230.0, 160.0) - 170.0) <= 4


def test_ink_centre_keeps_the_box_centre_on_a_blank_strip() -> None:
    page = np.full((100, 50), 230, dtype=np.uint8)
    assert ink_centre(page, 0, 50, 50.0, 30.0) == 50.0


def test_split_manuscripts_is_seeded_and_sized() -> None:
    ms = [str(i) for i in range(100)]
    a = split_manuscripts(ms, 0.05, 1); b = split_manuscripts(ms, 0.05, 1)
    assert a == b and len(a) == 5
    assert split_manuscripts(["x"], 0.05, 1) == {"x"}
