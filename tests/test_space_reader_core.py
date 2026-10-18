# File name: test_space_reader_core.py
# Date: 9/10/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""The Space's model-free core: prompts byte-identical to training, incremental parsing, drawing."""
import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "spaces/genizah_reader"))
import reader_core as rc  # noqa: E402
from src.finetuning.qwen_hebrew import build_ktiv_dataset as bkd  # noqa: E402
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT  # noqa: E402


def test_prompts_match_training_wording():
    assert rc.PROMPTS["transcribe"] == FRAGMENT_TRANSCRIBE_PROMPT
    assert rc.PROMPTS["lines"] == bkd._GROUNDED_PROMPT
    assert rc.PROMPTS["locate"] == bkd._LOCATE_PROMPT


def test_parse_lines_incremental_and_truncated():
    partial = '[{"text": "אבג", "bbox_2d": [100, 100, 900, 150]}, {"text": "דהו", "bbox_2d": [100, 160, 9'
    lines, complete = rc.parse_lines(partial)
    assert [l["text"] for l in lines] == ["אבג"] and not complete
    full = partial + '00, 210]}]'
    lines, complete = rc.parse_lines(full)
    assert len(lines) == 2 and complete
    assert rc.parse_lines('[{"text": "", "bbox_2d": [0, 0, 5, 5]}]')[0] == []       # empty text dropped
    assert rc.parse_lines('[{"text": "x", "bbox_2d": [50, 50, 50, 90]}]')[0] == []  # zero width dropped


def test_parse_locate_and_clamp():
    assert rc.parse_locate('{"bbox_2d": [120, 30, 1040, -5]}') == [120, 0, 1000, 30]
    assert rc.parse_locate("no box here") is None
    assert rc.parse_locate('{"bbox_2d": [10, 10, 10, 50]}') is None


def test_looping_detector():
    assert not rc.looping("שורה אחת בלבד")
    assert rc.looping("אמן קצרה ולא קטופה " * 12)


def test_draw_boxes_scales_to_image():
    im = Image.new("RGB", (400, 200), "white")
    out = rc.draw_boxes(im, [[0, 0, 500, 1000]], ["a"])
    assert out.size == (400, 200) and out.getpixel((1, 100)) != (255, 255, 255)   # left half outlined
    assert out.getpixel((399, 199)) == (255, 255, 255)                            # right edge untouched
