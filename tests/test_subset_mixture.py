# File name: test_subset_mixture.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the ablation subset of a built mixture (v23h = v23a without its Arabic-script rows).

A tiny v23a-like mixture (KTIV + full documentary pages + two Arabic components) is built with the real builder;
the subset must keep the other rows in their stored order with the same schema metadata, copy only the images
those rows reference, leave the source untouched, record per-component counts and the step arithmetic, and load
through ``ImagesOnceDataset``.
"""
import json
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from src.finetuning.qwen_hebrew import build_v22_mixture as mix
from src.finetuning.qwen_hebrew import images_once
from src.finetuning.qwen_hebrew import subset_mixture as sm
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES

HEB_Q = "Transcribe the handwritten Hebrew script on this page."
AR_Q = "Transcribe the handwritten Arabic script on this page."
SHARES = {"ktiv_transcription": 0.3, "pgp_edition_pages": 0.2, "arabic_editions": 0.3, "arabic_agapet": 0.2}
VAL_ROWS = {"ktiv": 1, "pgp_edition_pages": 1, "arabic_editions": 1, "agapet": 1}
ARABIC = ("arabic_agapet", "arabic_editions")


def _img(root: Path, i: int) -> Path:
    """Write (once) a small JPEG whose bytes are distinct per ``i``.

    :param root: Fixture directory.
    :param i: Image id.
    :return: The JPEG's path.
    """
    path = root / "img" / f"{i}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40 + i, 30), ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256)).save(path, "JPEG", quality=90)
    return path


def _row(image: Path, stem: str, question: str, answer: str) -> Dict:
    """One page-transcription row in the shared feature schema.

    :param image: Image file.
    :param stem: Row id.
    :param question: Prompt.
    :param answer: Target.
    :return: Row dict.
    """
    with Image.open(image) as im:
        w, h = im.size
    return {"image": str(image), "question": question, "answer": answer, "task": "fragment_transcribe",
            "section": "page", "stem": stem, "label_source": "synthetic", "target_chars": len(answer),
            "target_tokens": 0, "image_width": w, "image_height": h}


def _export(root: Path, name: str, splits: Dict[str, List[Dict]]) -> Path:
    """Save a DatasetDict and export it images-once.

    :param root: Fixture directory.
    :param name: Dataset directory name.
    :param splits: split -> rows.
    :return: The export directory.
    """
    DatasetDict({s: Dataset.from_list(rows, features=FEATURES) for s, rows in splits.items()}).save_to_disk(str(root / name))
    return mix.ensure_export(root / name, mix.export_dir_for(root / name))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A built 40-row v23a-like mixture with a train variant file, as on the NAS."""
    rows = {
        "ktiv": {"train": [_row(_img(tmp_path, i), f"k{i}", HEB_Q, "שורה של טקסט") for i in range(4)],
                 "val": [_row(_img(tmp_path, 9), "k9", HEB_Q, "עמוד")]},
        "pgp_edition_pages": {"train": [_row(_img(tmp_path, i), f"p{i}", HEB_Q, "כתאבי אליך") for i in (10, 11, 12)]
                              + [_row(_img(tmp_path, 13), "p13", HEB_Q, "כתאבי אליך كتابي اليك")],     # a mixed-script page
                              "val": [_row(_img(tmp_path, 19), "p19", HEB_Q, "שלום")]},
        "arabic_editions": {"train": [_row(_img(tmp_path, i), f"a{i}", AR_Q, "كتابي اليك اطال الله بقاك") for i in (20, 21)],
                            "val": [_row(_img(tmp_path, 29), "a29", AR_Q, "وسلامتك من مصر")]},
        "agapet": {"train": [_row(_img(tmp_path, i), f"g{i}", AR_Q, "بسم الاب والابن") for i in (30, 31, 32)],
                   "val": [_row(_img(tmp_path, 39), "g39", AR_Q, "والروح القدس")]},
    }
    exports = {d: _export(tmp_path, d, splits) for d, splits in rows.items()}
    src = tmp_path / "genizah_v23a_like"
    mix.build_mixture(src, exports, train_rows=40, seed=3407, shares=SHARES, val_rows=VAL_ROWS, workers=2)
    mix.write_train_variant(src, "without_outside", ["arabic_agapet"])
    return src


def _snapshot(root: Path) -> Dict[str, bytes]:
    """Every file under a directory with its bytes.

    :param root: Directory.
    :return: relative path -> content.
    """
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_resolve_drop():
    present = ["ktiv_transcription", "pgp_edition_pages", "arabic_editions", "arabic_agapet"]
    assert sm.resolve_drop(present, drop_prefix="arabic_") == ["arabic_agapet", "arabic_editions"]
    assert sm.resolve_drop(present, ["arabic_agapet"]) == ["arabic_agapet"]
    with pytest.raises(ValueError, match="have no train rows"):
        sm.resolve_drop(present, ["arabic_muharaf"])
    with pytest.raises(ValueError, match="nothing to drop"):
        sm.resolve_drop(present, drop_prefix="vqa_")
    with pytest.raises(ValueError, match="every component"):
        sm.resolve_drop(present, ["ktiv_transcription", "pgp_edition_pages"], "arabic_")


def test_keep_rows_and_steps():
    table = pa.table({"source": ["a", "b", "a", "c"], "stem": ["1", "2", "3", "4"]})
    assert sm.keep_rows(table, {"a"})["stem"].to_pylist() == ["2", "4"]
    assert sm.steps_for(5640) == 705 and sm.steps_for(12000) == 1500 and sm.steps_for(6960) == 870 and sm.steps_for(15) == 1


def test_subset_keeps_order_meta_and_only_its_images(source, tmp_path):
    before = _snapshot(source)
    out = tmp_path / "genizah_v23h_like"
    result = sm.write_subset_mixture(source, out, drop_prefix="arabic_", provenance={"hub_repo": "x/y", "revision": "a" * 40},
                                      workers=2)
    assert _snapshot(source) == before                                           # the source is only read
    for split in ("train", "val"):
        full = pq.read_table(source / "rows" / f"{split}.parquet")
        kept = pq.read_table(out / "rows" / f"{split}.parquet")
        assert kept.to_pylist() == [r for r in full.to_pylist() if not r["source"].startswith("arabic_")]
        assert kept.schema.metadata == full.schema.metadata
    train = pq.read_table(out / "rows" / "train.parquet")
    assert train.num_rows == 20 and set(train["source"].to_pylist()) == {"ktiv_transcription", "pgp_edition_pages"}
    wanted = {s for split in ("train", "val") for s in pq.read_table(out / "rows" / f"{split}.parquet")["image_sha1"].to_pylist()}
    assert {p.stem for p in (out / "images").iterdir()} == wanted                # only the kept rows' images
    for split, check in mix.self_check(out).items():
        assert check["source"] in ("ktiv_transcription", "pgp_edition_pages")
    ds = images_once.ImagesOnceDataset(out / "rows" / "train.parquet", out / "images", check_files=True)
    assert len(ds) == 20 and ds[0]["image"].size[1] == 30
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["train"]["rows"] == 20 and manifest["n_images"] == len(wanted)
    assert sum(s["unique_images"] for s in manifest["splits"].values()) == len(wanted)   # what the notebook's data cell expects
    assert set(manifest["source"]) == {"ktiv", "pgp_edition_pages"}
    assert result["manifest"] == manifest


def test_derivation_counts_steps_and_audit(source, tmp_path):
    out = tmp_path / "sub"
    d = sm.write_subset_mixture(source, out, drop_prefix="arabic_", workers=2, grad_accum=4)["derivation"]
    assert d == json.loads((out / "derivation.json").read_text())
    assert d["dropped_components"] == list(ARABIC) and d["seed"] == 3407
    assert d["train"]["rows_before"] == 40 and d["train"]["rows_after"] == 20
    assert d["val"]["rows_before"] == 4 and d["val"]["rows_after"] == 2
    pc = d["per_component"]
    assert pc["ktiv_transcription"]["train_rows_after"] == pc["ktiv_transcription"]["train_rows_before"] == 12
    assert pc["pgp_edition_pages"]["train_rows_after"] == 8 and pc["pgp_edition_pages"]["share_after"] == 0.4
    assert pc["arabic_agapet"]["train_rows_after"] == 0 and not pc["arabic_agapet"]["kept"] and pc["arabic_agapet"]["passes"] == 0
    src_mix = json.loads((source / "mixture.json").read_text())
    assert pc["pgp_edition_pages"]["passes"] == src_mix["components"]["pgp_edition_pages"]["passes"]   # exposure unchanged
    s = d["schedule"]
    assert (s["steps_before"], s["steps_after"], s["warmup_steps_before"], s["warmup_steps_after"]) == (10, 5, 1, 1)
    assert s["rows_left_over"] == 0 and "20 rows / (1 x 4) = 5 steps" in s["arithmetic"]
    audit = d["arabic_script_audit"]                                              # the mixed-script page stays, counted
    n_mixed = sum(1 for r in pq.read_table(out / "rows" / "train.parquet").to_pylist() if r["stem"] == "p13")
    assert audit["rows_with_arabic_letters"] == n_mixed == 2 and audit["arabic_letters"] == 2 * len("كتابياليك")
    assert audit["stems_by_component"] == {"pgp_edition_pages": ["p13"]}
    assert d["source"]["train_parquet_sha256"] == sm.file_sha256(source / "rows" / "train.parquet")


def test_mixture_json_and_card_describe_the_subset(source, tmp_path):
    out = tmp_path / "genizah_v23h_like"
    sm.write_subset_mixture(source, out, drop_prefix="arabic_", provenance={"hub_repo": "isaacmg/src", "revision": "b" * 40},
                            workers=2)
    m = json.loads((out / "mixture.json").read_text())
    assert set(m["components"]) == {"ktiv_transcription", "pgp_edition_pages"} and "train_variants" not in m
    assert m["config"]["name"] == "genizah_v23h_like" and m["config"]["train_rows"] == 20 and m["config"]["seed"] == 3407
    assert set(m["config"]["dirs"]) == set(m["val"]) == {"ktiv", "pgp_edition_pages"}
    assert m["realised_shares"]["components"] == {"ktiv_transcription": 0.6, "pgp_edition_pages": 0.4}
    assert m["derived_from"]["dropped_components"] == list(ARABIC) and m["val_missing_components"] == []
    card = (out / "README.md").read_text()
    assert "## Derived from genizah_v23a_like" in card and "`isaacmg/src` (revision `" + "b" * 40 + "`)" in card
    assert "20 of 40 train" in card and "Agapet" not in card and "arabic_agapet" in card      # named as dropped, not credited
    assert "Quota = round(share x 40) in the source build" in card and "round(share x 20)" not in card
    assert "derivation.json      rows before / after" in card and "by `src/finetuning/qwen_hebrew/subset_mixture.py`" in card
    with pytest.raises(ValueError, match="exactly once"):
        sm.adapt_card("no builder card here", json.loads((out / "derivation.json").read_text()))
    assert not any(w in card.lower() for w in mix.FORBIDDEN_CARD_STRINGS)
    assert "train_without_outside" not in card                                     # the source's variant file is not carried


def test_refuses_an_output_inside_the_source(source):
    with pytest.raises(ValueError, match="must not be the source"):
        sm.write_subset_mixture(source, source / "sub", drop_prefix="arabic_")
    with pytest.raises(ValueError, match="must not be the source"):
        sm.write_subset_mixture(source, source, drop_prefix="arabic_")


def test_cli(source, tmp_path, capsys):
    out = tmp_path / "cli"
    sm.main(["--src", str(source), "--out", str(out), "--drop", "arabic_agapet,arabic_editions",
             "--source-repo", "isaacmg/src", "--source-revision", "c" * 40, "--workers", "2"])
    printed = capsys.readouterr().out
    assert '"steps_after": 2' in printed and "self-check train" in printed
    assert json.loads((out / "derivation.json").read_text())["source"]["revision"] == "c" * 40
