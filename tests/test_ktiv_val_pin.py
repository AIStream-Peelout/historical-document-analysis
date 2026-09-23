# File name: test_ktiv_val_pin.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for pinning the KTIV val split across rebuilds (``--val-manuscripts``).

A rebuild must not move a manuscript between splits: pinned manuscripts stay
in val, the previous build's other manuscripts stay in train, and only
manuscripts new since then are drawn (seeded) into val.
"""
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest
from datasets import load_from_disk
from PIL import Image

from src.finetuning.qwen_hebrew import build_ktiv_dataset as bkd

ALEPHBET = "אבגדהוזחטיכלמנסעפצקרשת"


def _rows(manuscripts: List[str], per_ms: int = 3) -> List[Dict]:
    """Minimal rows, several per manuscript, incl. a page_short row each.

    :param manuscripts: sys_nums.
    :param per_ms: Rows per manuscript besides the page_short row.
    :return: Rows with ``stem`` and ``task``.
    """
    rows = []
    for m in manuscripts:
        rows += [{"stem": f"ktiv_{m}_FL{j}", "task": "fragment_transcribe"} for j in range(per_ms)]
        rows.append({"stem": f"ktiv_{m}_FL9", "task": "page_short"})
    return rows


OLD = [f"99{i:04d}" for i in range(200)]
NEW = [f"98{i:04d}" for i in range(60)]


def _historical(ms: List[str], frac: float, seed: int) -> set:
    """The pre-pin rule, written out independently."""
    ms = sorted(set(ms))
    rng = random.Random(seed)
    rng.shuffle(ms)
    return set(ms[:max(1, int(len(ms) * frac))])


def test_unpinned_split_is_the_historical_rule():
    ms = OLD + NEW
    assert bkd.choose_val_manuscripts(ms, 0.05, 7) == _historical(ms, 0.05, 7)


def test_pin_keeps_old_manuscripts_in_place_and_draws_new_ones():
    old_val = bkd.choose_val_manuscripts(OLD, 0.05, bkd.SPLIT_SEED)
    present = OLD[5:] + NEW                      # a few old manuscripts dropped out
    val = bkd.choose_val_manuscripts(present, 0.05, bkd.SPLIT_SEED,
                                     pinned=old_val, known=set(OLD))
    assert old_val & set(present) <= val                  # every surviving pinned ms
    assert not (val & (set(OLD) - old_val))               # no old train ms moves to val
    drawn = val - old_val
    assert drawn <= set(NEW) and len(drawn) == max(1, int(len(NEW) * 0.05))
    again = bkd.choose_val_manuscripts(present, 0.05, bkd.SPLIT_SEED,
                                       pinned=old_val, known=set(OLD))
    assert again == val                                   # seeded


def test_plain_list_pin_is_the_whole_val_set():
    pinned = set(OLD[:7])
    assert bkd.choose_val_manuscripts(OLD + NEW, 0.05, 1, pinned=pinned) == pinned


def test_split_keeps_every_row_of_a_manuscript_together():
    rows = _rows(OLD + NEW)
    pinned = set(OLD[:10])
    train, val = bkd.split_by_manuscript(rows, 0.05, bkd.SPLIT_SEED, pinned, set(OLD))
    val_ms = {bkd.stem_manuscript(r["stem"]) for r in val}
    train_ms = {bkd.stem_manuscript(r["stem"]) for r in train}
    assert not val_ms & train_ms and len(train) + len(val) == len(rows)
    assert pinned <= val_ms
    assert sum(r["task"] == "page_short" for r in val) == len(val_ms)


def test_written_pin_round_trips(tmp_path: Path):
    rows = _rows(OLD)
    train, val = bkd.split_by_manuscript(rows, 0.05, bkd.SPLIT_SEED)
    path = tmp_path / "val_manuscripts.json"
    path.write_text(json.dumps(bkd.val_pin_record(train, val, {"rule": "test"})))
    pinned, known = bkd.load_val_pin(path)
    assert known == set(OLD)
    # same manuscripts -> identical split; plus new ones -> old rows never move
    assert bkd.split_by_manuscript(rows, 0.05, 123, pinned, known) == (train, val)
    train2, val2 = bkd.split_by_manuscript(_rows(OLD + NEW), 0.05, 123, pinned, known)
    assert [r for r in val2 if bkd.stem_manuscript(r["stem"]) in known] == val
    assert [r for r in train2 if bkd.stem_manuscript(r["stem"]) in known] == train
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(sorted(pinned)))
    assert bkd.load_val_pin(list_path) == (pinned, None)


# ── build() wiring: --val-manuscripts -> split -> val_manuscripts.json ──────

SYS = ["990000000000000011", "990000000000000012", "990000000000000013"]


def _page_items(offset: int) -> List[Dict]:
    """One 200-letter, 8-line page of distinct 5-letter words (passes every gate).

    :param offset: Alphabet rotation, different per manuscript.
    :return: AnnotationPage items.
    """
    items, k = [], offset
    for li in range(8):
        x1 = 1900.0
        for _ in range(5):
            word = "".join(ALEPHBET[(k + j) % len(ALEPHBET)] for j in range(5))
            path = f"M{x1 - 300},{100 + li * 100} {x1},{100 + li * 100} {x1},{160 + li * 100} " \
                   f"{x1 - 300},{160 + li * 100}z"
            items.append({"id": "1 oldVer", "body": {"value": word},
                          "target": {"selector": {"value": f"<svg><path d=\"{path}\"/></svg>"}}})
            x1 -= 340
            k += 3
    return items


@pytest.fixture
def ktiv_dir(tmp_path: Path) -> Path:
    """Three API-shape manuscripts with one page and one page image each."""
    root = tmp_path / "ktiv"
    root.mkdir()
    for n, sys_num in enumerate(SYS):
        stem = f"ktiv_PNX_MANUSCRIPTS{sys_num}-1"
        bundle = {"source": "nli_ktiv_viewer", "doc_id": sys_num,
                  "pages": [{"fl": "FL1", "annotation_page": {"items": _page_items(n)}}]}
        (root / f"{stem}_transcription.json").write_text(json.dumps(bundle, ensure_ascii=False))
        buf = io.BytesIO()
        Image.new("RGB", (2000, 1000), "white").save(buf, "JPEG")
        with zipfile.ZipFile(root / f"{stem}_images.zip", "w") as zf:
            zf.writestr("FL1.jpg", buf.getvalue())
    return root


def test_build_pins_val_and_writes_the_next_pin(ktiv_dir: Path, tmp_path: Path):
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"val_manuscripts": [SYS[1]], "known_manuscripts": SYS}))
    out = tmp_path / "out"
    bkd.build(ktiv_dir, tmp_path / "images", out, val_manuscripts=pin)
    dsd = load_from_disk(str(out))
    assert {bkd.stem_manuscript(s) for s in dsd["val"]["stem"]} == {SYS[1]}
    assert {bkd.stem_manuscript(s) for s in dsd["train"]["stem"]} == {SYS[0], SYS[2]}
    written = json.loads((out / "val_manuscripts.json").read_text())
    assert written["val_manuscripts"] == [SYS[1]] and written["known_manuscripts"] == SYS
    stats = json.loads((out / "stats.json").read_text())
    assert stats["val_pin"]["pinned_present"] == 1 and stats["val_pin"]["new_manuscripts"] == 0


def test_build_rejects_a_pin_that_empties_val(ktiv_dir: Path, tmp_path: Path):
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps(["990000000000000999"]))
    with pytest.raises(ValueError, match="matches no manuscript"):
        bkd.build(ktiv_dir, tmp_path / "images", tmp_path / "out", val_manuscripts=pin)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
