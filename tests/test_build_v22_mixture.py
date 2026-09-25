# File name: test_build_v22_mixture.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the v22 pilot mixture builder on tiny synthetic images-once sources.

Four small DatasetDicts in the KTIV row schema (KTIV with transcription and
grounding families; editions and QA split over several ``train_*`` splits)
are saved, exported images-once and mixed. The mixture must repeat
over-subscribed pools evenly, shuffle reproducibly, copy only the images its
rows reference, load through ``ImagesOnceDataset`` and keep forbidden
sources out of its card.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from src.finetuning.qwen_hebrew import build_v22_mixture as mix
from src.finetuning.qwen_hebrew import images_once
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES

LOCATE_Q = 'Locate the phrase "שלום" on this page. Respond with ONLY {"bbox_2d": [x1, y1, x2, y2]}.'
READ_BOX_Q = "Transcribe ONLY the text inside the region bbox_2d = [1, 2, 30, 40]."
BOX_A = '{"bbox_2d": [1, 2, 30, 40]}'
VAL_ROWS = {"ktiv": 2, "pgp_editions": 1, "documentary_grounding": 1, "pgp_qa": 5}


def _img(root: Path, i: int) -> Path:
    """Write (once) a small JPEG whose bytes are distinct per ``i``.

    :param root: Fixture directory.
    :param i: Image id.
    :return: The JPEG's path.
    """
    path = root / "img" / f"{i}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        colour = ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256)
        Image.new("RGB", (40 + i, 30), colour).save(path, "JPEG", quality=90)
    return path


def _row(image: Path, task: str, stem: str, question: str, answer: str) -> Dict:
    """One row in the shared feature schema.

    :param image: Image file.
    :param task: Task family.
    :param stem: Row id (unique across the fixture).
    :param question: Prompt.
    :param answer: Target.
    :return: Row dict.
    """
    with Image.open(image) as im:
        w, h = im.size
    return {"image": str(image), "question": question, "answer": answer, "task": task,
            "section": "page", "stem": stem, "label_source": "synthetic",
            "target_chars": len(answer), "target_tokens": 0, "image_width": w, "image_height": h}


def _export(root: Path, name: str, splits: Dict[str, List[Dict]]) -> Path:
    """Save a DatasetDict and export it images-once.

    :param root: Fixture directory.
    :param name: Dataset directory name.
    :param splits: split -> rows.
    :return: The export directory.
    """
    ds_dir = root / name
    DatasetDict({s: Dataset.from_list(rows, features=FEATURES)
                 for s, rows in splits.items()}).save_to_disk(str(ds_dir))
    return mix.ensure_export(ds_dir, mix.export_dir_for(ds_dir))


@pytest.fixture
def sources(tmp_path: Path) -> Dict[str, Path]:
    """dataset -> images-once export of the four tiny sources."""
    t = "Transcribe the page."
    ktiv = {"train": [_row(_img(tmp_path, 0), "fragment_transcribe", "k0_page", t, "שורה"),
                      _row(_img(tmp_path, 0), "locate", "k0_loc", LOCATE_Q, BOX_A),
                      _row(_img(tmp_path, 1), "fragment_transcribe", "k1_page", t, "טקסט"),
                      _row(_img(tmp_path, 1), "read_box", "k1_box", READ_BOX_Q, "מילה"),
                      _row(_img(tmp_path, 2), "line_transcribe", "k2_line", t, "שורה"),
                      _row(_img(tmp_path, 3), "locate", "k3_loc", LOCATE_Q, BOX_A)],
            "val": [_row(_img(tmp_path, 4), "fragment_transcribe", "k4_page", t, "עמוד"),
                    _row(_img(tmp_path, 4), "locate", "k4_loc", LOCATE_Q, BOX_A)]}
    editions = {"train_page": [_row(_img(tmp_path, 10), "fragment_transcribe", "e10_page", t, "מהדורה"),
                               _row(_img(tmp_path, 11), "fragment_transcribe", "e11_page", t, "מהדורה")],
                "train_line": [_row(_img(tmp_path, 10), "line_by_number", "e10_line", t, "שורה")],
                "val": [_row(_img(tmp_path, 12), "fragment_transcribe", "e12_page", t, "מהדורה")]}
    grounding = {"train": [_row(_img(tmp_path, 20), "locate", "g20_loc", LOCATE_Q, BOX_A),
                           _row(_img(tmp_path, 21), "locate", "g21_loc", LOCATE_Q, BOX_A)],
                 "val": [_row(_img(tmp_path, 22), "locate", "g22_loc", LOCATE_Q, BOX_A)]}
    qa = {"train_qa_date": [_row(_img(tmp_path, 30), "qa_date", "q30", "When?", '{"line": 1, "text": "תשרי"}')],
          "train_qa_place": [_row(_img(tmp_path, 31), "qa_place", "q31", "Where?", '{"line": 2, "text": "פסטאט"}')],
          "val": [_row(_img(tmp_path, 32), "qa_date", "q32", "When?", '{"line": 1, "text": "ניסן"}')]}
    return {"ktiv": _export(tmp_path, "ktiv", ktiv),
            "pgp_editions": _export(tmp_path, "editions", editions),
            "documentary_grounding": _export(tmp_path, "grounding", grounding),
            "pgp_qa": _export(tmp_path, "qa", qa)}


def _build(sources: Dict[str, Path], out: Path, **overrides) -> Dict:
    """Build a 20-row mixture of the fixture.

    :param sources: dataset -> export.
    :param out: Destination.
    :param overrides: build_mixture keyword overrides.
    :return: build_mixture's result.
    """
    kwargs = {"train_rows": 20, "seed": 3407, "val_rows": VAL_ROWS, "workers": 2, **overrides}
    return mix.build_mixture(out, sources, **kwargs)


def _rows(out: Path, split: str) -> List[Dict]:
    """(stem, source) of a built split, in stored order.

    :param out: Mixture directory.
    :param split: ``train`` or ``val``.
    :return: Row dicts.
    """
    return pq.read_table(out / "rows" / f"{split}.parquet").select(["stem", "source"]).to_pylist()


def test_plan_quotas():
    assert mix.plan_quotas(mix.DEFAULT_SHARES, 8000) == {
        "ktiv_transcription": 2400, "pgp_editions": 2000, "ktiv_grounding": 1200,
        "documentary_grounding": 800, "pgp_qa": 1600}
    with pytest.raises(ValueError, match="unknown"):
        mix.plan_quotas({"ktiv": 1.0}, 10)
    with pytest.raises(ValueError, match="sum"):
        mix.plan_quotas({"pgp_qa": 0.5}, 10)


def test_quota_above_pool_repeats_rows_evenly():
    idx, info = mix.sample_source(3, 8, np.random.default_rng(0))
    counts = Counter(idx.tolist())
    assert sorted(counts) == [0, 1, 2] and sorted(counts.values()) == [2, 3, 3]
    assert info == {"pool": 3, "quota": 8, "taken": 8, "unique_rows": 3, "full_passes": 2,
                    "partial_rows": 2, "passes": 2.6667}
    idx, info = mix.sample_source(10, 4, np.random.default_rng(0))
    assert len(set(idx.tolist())) == 4 and info["full_passes"] == 0 and info["partial_rows"] == 4
    with pytest.raises(ValueError, match="empty pool"):
        mix.sample_source(0, 1, np.random.default_rng(0))


def test_classification_and_unknown_task():
    assert mix.classify_task("locate") == "grounding"
    assert mix.classify_task("read_box_word") == "grounding"
    assert mix.classify_task("region_transcribe") == "transcription"
    with pytest.raises(ValueError, match="unclassified"):
        mix.classify_task("brand_new_task")
    rows = {"section": ["x"], "question": ["q"], "answer": ["a"]}
    with pytest.raises(ValueError, match="unclassified"):
        mix.ktiv_task_report(pa.table({"task": ["brand_new_task"], **rows}))
    with pytest.raises(ValueError, match="fragment_transcribe"):      # rows contradict the bucket
        mix.ktiv_task_report(pa.table({"task": ["fragment_transcribe"], **rows, "answer": [BOX_A]}))


def test_build_quotas_passes_and_train_splits(sources, tmp_path):
    result = _build(sources, tmp_path / "out")
    comps = result["mixture"]["components"]
    assert {c: i["taken"] for c, i in comps.items()} == {
        "ktiv_transcription": 6, "pgp_editions": 5, "ktiv_grounding": 3,
        "documentary_grounding": 2, "pgp_qa": 4}
    assert result["mixture"]["train_splits"]["pgp_editions"] == ["train_page", "train_line"]
    assert comps["pgp_editions"]["pool"] == 3 and comps["pgp_qa"]["pool"] == 2
    assert comps["pgp_qa"]["full_passes"] == 2 and comps["pgp_editions"]["partial_rows"] == 2
    train = pq.read_table(tmp_path / "out" / "rows" / "train.parquet")
    for comp, info in comps.items():
        counts = Counter(r["stem"] for r in train.select(["stem", "source"]).to_pylist()
                         if r["source"] == comp)
        q, n = info["quota"], info["pool"]
        assert len(counts) == min(q, n)
        assert set(counts.values()) <= {q // n, -(-q // n)}
    for task, source in zip(train["task"].to_pylist(), train["source"].to_pylist()):
        if source.startswith("ktiv_"):
            assert source == f"ktiv_{mix.classify_task(task)}"
    assert result["mixture"]["val"]["pgp_qa"] == {"pool": 1, "requested": 5, "taken": 1}
    assert result["manifest"]["splits"]["val"]["rows"] == 5
    assert result["mixture"]["buckets"]["train"] == {"grounding": 5, "qa": 4, "transcription": 11}


def test_shuffle_is_deterministic_with_the_seed(sources, tmp_path):
    for name, seed in (("a", 3407), ("b", 3407), ("c", 1)):
        _build(sources, tmp_path / name, seed=seed)
    for split in ("train", "val"):
        assert _rows(tmp_path / "a", split) == _rows(tmp_path / "b", split)
    assert _rows(tmp_path / "a", "train") != _rows(tmp_path / "c", "train")
    order = [r["source"] for r in _rows(tmp_path / "a", "train")]
    assert order != sorted(order, key=list(mix.COMPONENTS).index)     # interleaved, not blocks


def test_only_referenced_images_are_copied(sources, tmp_path):
    out = tmp_path / "out"
    _build(sources, out, shares={"ktiv_transcription": 0.5, "pgp_editions": 0.5}, val_rows={"ktiv": 1})
    referenced = set()
    for split in ("train", "val"):
        referenced |= set(pq.read_table(out / "rows" / f"{split}.parquet")[images_once.SHA_COLUMN]
                          .to_pylist())
    stored = {p.name for p in (out / "images").iterdir()}
    assert stored == {f"{s}.jpg" for s in referenced}
    for unused in ("documentary_grounding", "pgp_qa"):
        assert not stored & {p.name for p in (sources[unused] / "images").iterdir()}


def test_truncated_copy_is_rewritten(sources, tmp_path):
    out = tmp_path / "out"
    _build(sources, out)
    victim = sorted((out / "images").iterdir())[0]
    good = victim.read_bytes()
    victim.write_bytes(good[:10])                                      # interrupted copy
    _build(sources, out)
    assert victim.read_bytes() == good


def test_rows_load_through_images_once_dataset(sources, tmp_path):
    out = tmp_path / "out"
    _build(sources, out)
    originals = {}
    for export in sources.values():
        for split in mix.export_splits(export) + ["val"]:
            ds = images_once.ImagesOnceDataset(export / "rows" / f"{split}.parquet", export / "images")
            originals.update({ds[i]["stem"]: ds[i] for i in range(len(ds))})
    for split, n_rows in (("train", 20), ("val", 5)):
        ds = images_once.ImagesOnceDataset(out / "rows" / f"{split}.parquet", out / "images",
                                           check_files=True)
        assert len(ds) == n_rows
        for i in range(len(ds)):
            item = ds[i]
            want = originals[item["stem"]]
            assert list(item) == list(want) + ["source"]
            assert {k: v for k, v in item.items() if k not in ("image", "source")} == \
                {k: v for k, v in want.items() if k != "image"}
            assert item["image"].tobytes() == want["image"].tobytes()
            assert item["source"] in mix.COMPONENTS
    check = mix.self_check(out)
    assert check["train"]["rows"] == 20 and check["train"]["keys"][-1] == "source"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_images"] == len(list((out / "images").iterdir()))


def test_card_credits_sources_and_refuses_forbidden_names(sources, tmp_path):
    out = tmp_path / "out"
    result = _build(sources, out)
    card = (out / "README.md").read_text()
    assert "NLI-KTIV" in card and "Princeton Geniza Project" in card
    assert not any(word in card.lower() for word in mix.FORBIDDEN_CARD_STRINGS)
    bad = json.loads(json.dumps(result["mixture"]))
    bad["config"]["dirs"]["pgp_editions"] = "/data/FJP_editions"
    with pytest.raises(ValueError, match="fjp"):
        mix.write_card(tmp_path, bad, result["manifest"])


def test_current_export_is_reused(sources, tmp_path):
    export = sources["pgp_qa"]
    stamp = (export / "manifest.json").stat().st_mtime_ns
    assert mix.export_is_current(tmp_path / "qa", export)
    mix.ensure_export(tmp_path / "qa", export)
    assert (export / "manifest.json").stat().st_mtime_ns == stamp
    assert not mix.export_is_current(tmp_path / "grounding", export)    # another source's export


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
