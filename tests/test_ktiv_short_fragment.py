# File name: test_ktiv_short_fragment.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the v4 ``short_fragment`` family (task ``page_short``) in build_ktiv_dataset.

Gate decided 2026-09-14: pages under the 150-letter gate with >= 40 Hebrew
letters and a damage share <= 0.30 (gap tokens + tokens with dots/brackets,
over all tokens) get the page-transcription row only; the image must be in
frame and the manuscript must pass decontamination like any other.
"""
import io
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew import build_ktiv_dataset as bkd
from src.finetuning.qwen_hebrew.ktiv_layout import reconstruct_page
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

SYS = "990000000000000007"
ALEPHBET = "אבגדהוזחטיכלמנסעפצקרשת"


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> Dict:
    """One AnnotationPage item with a rectangular SvgSelector.

    :param text: Word text.
    :param x0: Left edge.
    :param y0: Top edge.
    :param x1: Right edge.
    :param y1: Bottom edge.
    :return: Annotation item dict.
    """
    path = f"M{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} {x0},{y0}z"
    return {"id": "1 oldVer", "type": "Annotation",
            "body": {"type": "TextualBody", "value": text},
            "target": {"selector": {"type": "SvgSelector",
                                    "value": f"<svg><path d=\"{path}\"/></svg>"}}}


def _page(lines: List[List[str]], right: float = 1900.0) -> List[Dict]:
    """Lay out words right-to-left, 100 px line pitch, generous word spacing.

    :param lines: Words per line in reading order.
    :param right: Right edge of the first word on every line.
    :return: AnnotationPage items.
    """
    items = []
    for li, words in enumerate(lines):
        x1 = right
        for w in words:
            items.append(_item(w, x1 - 300, 100 + li * 100, x1, 160 + li * 100))
            x1 -= 340
    return items


def _words(n_lines: int, per_line: int, length: int, offset: int) -> List[List[str]]:
    """Distinct Hebrew words cut from a rotating alphabet.

    :param n_lines: Lines.
    :param per_line: Words per line.
    :param length: Letters per word.
    :param offset: Alphabet rotation (keeps pages' letter runs apart).
    :return: Words per line.
    """
    out, k = [], offset
    for _ in range(n_lines):
        line = []
        for _ in range(per_line):
            line.append("".join(ALEPHBET[(k + j) % len(ALEPHBET)] for j in range(length)))
            k += 3
        out.append(line)
    return out


def _short_clean() -> List[List[str]]:
    """48-letter page, one restored token of 12 (damage share 1/12)."""
    lines = _words(4, 3, 4, offset=1)
    lines[0][1] = f"[{lines[0][1][:2]}]{lines[0][1][2:]}"
    return lines


def _short_damaged() -> List[List[str]]:
    """48-letter page with 5 of 12 tokens carrying dots (damage share 5/12)."""
    lines = _words(4, 3, 4, offset=2)
    for li, wi in ((0, 0), (1, 1), (2, 2), (3, 0), (3, 2)):
        w = lines[li][wi]
        lines[li][wi] = w[:2] + "." + w[2:]
    return lines


def _write_fixture(tmp_path: Path, pages: List[Tuple[str, List[Dict]]]) -> Dict:
    """Write one manuscript's image zip and return its API-shape bundle.

    :param tmp_path: KTIV directory stand-in.
    :param pages: ``(fl, items)`` per page; every page image is 2000x1000.
    :return: Bundle dict with ``sys_num`` attached.
    """
    with zipfile.ZipFile(tmp_path / f"ktiv_PNX_MANUSCRIPTS{SYS}-1_images.zip", "w") as zf:
        for fl, _items in pages:
            buf = io.BytesIO()
            Image.new("RGB", (2000, 1000), "white").save(buf, "JPEG")
            zf.writestr(f"{fl}.jpg", buf.getvalue())
    return {"sys_num": SYS, "source": "nli_ktiv_viewer",
            "pages": [{"fl": fl, "annotation_page": {"items": items}}
                      for fl, items in pages]}


@pytest.fixture
def manuscript(tmp_path: Path) -> Dict:
    """A manuscript with one normal page and four pages under the letter gate."""
    pages = [("FL1", _page(_words(8, 5, 5, offset=0))),     # 200 letters: normal
             ("FL2", _page(_short_clean())),                  # short, passes
             ("FL3", _page(_short_damaged())),                # short, too damaged
             ("FL4", _page(_words(3, 2, 3, offset=4))),       # 18 letters
             ("FL5", _page(_short_clean(), right=2300.0))]    # short, boxes out of frame
    return _write_fixture(tmp_path, pages)


def _build(tmp_path: Path, bundles: List[Dict], shingles: set,
           short_fragments: bool) -> Tuple[List[Dict], Dict]:
    """Run build_rows with a fixed seed into ``tmp_path/images``."""
    return bkd.build_rows(bundles, tmp_path, tmp_path / "images", shingles,
                          random.Random(bkd.SPLIT_SEED), short_fragments=short_fragments)


# ── damage share ─────────────────────────────────────────────────────────────

def test_damage_share_counts_gaps_dots_and_brackets():
    assert bkd.page_damage_share("אבג [...] ד.ה [ו]ז חטי") == pytest.approx(3 / 5)


def test_damage_share_counts_each_token_once():
    assert bkd.page_damage_share("אב [ג.] דה") == pytest.approx(1 / 3)


def test_damage_share_splits_lines_and_handles_clean_and_empty_pages():
    assert bkd.page_damage_share("אבג\nד.ה") == pytest.approx(0.5)
    assert bkd.page_damage_share("אבג דהו\nזחט") == 0.0
    assert bkd.page_damage_share(bkd.GAP_TOKEN) == 1.0
    assert bkd.page_damage_share("") == 1.0


def test_damage_share_of_fixture_pages():
    clean = reconstruct_page(_page(_short_clean()))
    damaged = reconstruct_page(_page(_short_damaged()))
    assert bkd.page_damage_share(clean["text"]) == pytest.approx(1 / 12)
    assert bkd.page_damage_share(damaged["text"]) == pytest.approx(5 / 12)


# ── gate ─────────────────────────────────────────────────────────────────────

def test_gate_boundaries_are_inclusive():
    assert bkd.short_fragment_gate(bkd.SHORT_MIN_LETTERS, bkd.SHORT_MAX_DAMAGE_SHARE, 1)
    assert bkd.short_fragment_gate(bkd.MIN_LETTERS - 1, 0.0, 3)


@pytest.mark.parametrize("letters, damage, n_lines, reason", [
    (39, 0.0, 5, "too_few_letters"),
    (150, 0.0, 5, "not_short"),       # belongs to fragment_transcribe
    (100, 0.31, 5, "too_damaged"),
    (100, 0.10, 0, "too_few_lines"),
])
def test_gate_rejections(letters, damage, n_lines, reason):
    assert bkd.short_fragment_reason(letters, damage, n_lines) == reason
    assert not bkd.short_fragment_gate(letters, damage, n_lines)


def test_image_frame_reason():
    lines = [{"box": (100, 100, 1900, 160)}]
    assert bkd.image_frame_reason(2000, 1000, lines) is None
    assert bkd.image_frame_reason(2000, 399, lines) == "image_too_small"
    assert bkd.image_frame_reason(1800, 1000, lines) == "boxes_out_of_frame"
    assert bkd.image_frame_reason(1880, 1000, lines) is None     # within 2% slack


# ── build_rows integration ───────────────────────────────────────────────────

def test_build_rows_emits_one_page_short_row(tmp_path, manuscript):
    rows, stats = _build(tmp_path, [manuscript], set(), short_fragments=True)
    short = [r for r in rows if r["task"] == "page_short"]
    page = next(r for r in rows if r["task"] == "fragment_transcribe")
    assert len(short) == 1
    row = short[0]
    assert row["stem"] == f"ktiv_{SYS}_FL2"
    assert row["section"] == "page"
    assert row["question"] == FRAGMENT_TRANSCRIBE_PROMPT == page["question"]
    assert row["label_source"] == page["label_source"]
    assert row["answer"] == reconstruct_page(_page(_short_clean()))["text"]
    assert Path(row["image"]).exists()
    assert (row["image_width"], row["image_height"]) == (2000, 1000)
    assert stats["page_pass"] == 1 and stats["page_too_few_letters"] == 4
    assert stats["page_short_pass"] == 2          # FL2 + FL5 pass the text gate
    assert stats["page_short_too_damaged"] == 1
    assert stats["page_short_too_few_letters"] == 1
    assert stats["page_short_boxes_out_of_frame"] == 1
    assert stats["rows_page_short"] == 1 and stats["ms_page_short"] == 1
    assert stats["ms_kept"] == 1 and "ms_short_only" not in stats


def test_short_fragments_leave_other_families_untouched(tmp_path, manuscript):
    rows_on, stats_on = _build(tmp_path, [manuscript], set(), short_fragments=True)
    rows_off, stats_off = _build(tmp_path, [manuscript], set(), short_fragments=False)
    assert [r for r in rows_on if r["task"] != "page_short"] == rows_off
    assert not any(k.startswith(("page_short", "rows_page_short", "ms_page_short"))
                   for k in stats_off)
    for key, value in stats_off.items():
        assert stats_on[key] == value


def test_contaminated_short_page_drops_its_manuscript(tmp_path, manuscript):
    short_text = reconstruct_page(_page(_short_clean()))["text"]
    letters = "".join(bkd._HEB_RE.findall(short_text))
    n = bkd.DECONTAM_SHINGLE
    shingles = {letters[i:i + n] for i in range(len(letters) - n + 1)}
    normal_text = reconstruct_page(_page(_words(8, 5, 5, offset=0)))["text"]
    assert bkd.shingle_hits(normal_text, shingles) < bkd.DECONTAM_MIN_HITS
    rows, stats = _build(tmp_path, [manuscript], shingles, short_fragments=True)
    assert rows == []
    assert stats["ms_contaminated"] == 1 and stats["ms_contaminated_by_short"] == 1
    assert stats["page_short_contaminated"] == 2   # FL2 and FL5 (same text)
    assert stats["contaminated_sys_nums"] == [SYS]
    rows_off, stats_off = _build(tmp_path, [manuscript], shingles, short_fragments=False)
    assert rows_off and stats_off["ms_kept"] == 1


def test_short_only_manuscript_is_kept(tmp_path):
    bundle = _write_fixture(tmp_path, [("FL2", _page(_short_clean()))])
    rows, stats = _build(tmp_path, [bundle], set(), short_fragments=True)
    assert [r["task"] for r in rows] == ["page_short"]
    assert stats["ms_short_only"] == 1 and "ms_kept" not in stats
    rows_off, _ = _build(tmp_path, [bundle], set(), short_fragments=False)
    assert rows_off == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
