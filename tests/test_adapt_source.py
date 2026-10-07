# File name: test_adapt_source.py
# Date: 9/28/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for adapting older sources (synthetic renders, Talmud replay) to the KTIV row schema.

Tiny DatasetDicts shaped like the real sources are adapted through the CLI:
a synthetic-like one (``eval`` split, ``font/style/mode`` columns, no length
columns, saved with ``save_to_disk``) and a Talmud-like one (KTIV columns in
another order, a ``smoke`` split, read from Hub-style parquet shards). The
adapted splits must hold exactly the KTIV columns in the KTIV order with the
KTIV types, untouched image bytes, and only the rows the filter keeps; the
Talmud filter file must reproduce the v21b notebook's lambdas.
"""
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from datasets import Image as ImageFeature
from PIL import Image

from src.finetuning.qwen_hebrew import adapt_source, images_once
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES
from src.finetuning.qwen_hebrew.build_v22_mixture import check_layouts

REPO = Path(__file__).resolve().parents[1]
TALMUD_FILTER = REPO / "src/finetuning/qwen_hebrew/filters/talmud_v21b.json"

SYNTH_FEATURES = Features({
    "image": ImageFeature(), "question": Value("string"), "answer": Value("string"),
    "task": Value("string"), "section": Value("string"), "stem": Value("string"),
    "label_source": Value("string"), "font": Value("string"), "style": Value("string"),
    "mode": Value("string"), "image_width": Value("int32"), "image_height": Value("int32")})
TALMUD_ORDER = ["answer", "section", "stem", "label_source", "target_chars", "target_tokens",
                "image", "question", "task", "image_width", "image_height"]
TALMUD_FEATURES = Features({c: FEATURES[c] for c in TALMUD_ORDER})


def _jpeg(shade: int, size: Tuple[int, int] = (24, 16)) -> bytes:
    """Solid grey JPEG bytes.

    :param shade: Grey level.
    :param size: (width, height).
    :return: Encoded bytes.
    """
    buf = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _synth_row(i: int, answer: str) -> Dict:
    """One synthetic-render row.

    :param i: Row number (image shade, stem).
    :param answer: Target text.
    :return: Row dict.
    """
    return {"image": {"bytes": _jpeg(10 * i), "path": None}, "question": "Transcribe.",
            "answer": answer, "task": "synthetic_transcribe", "section": "synthetic",
            "stem": f"synth_{i:03d}", "label_source": "synthetic", "font": "NotoRashi",
            "style": "clean", "mode": "shuffled_words", "image_width": 24, "image_height": 16}


def _talmud_row(i: int, task: str, section: str, stem: str) -> Dict:
    """One Talmud-replay row (KTIV columns, Talmud order).

    :param i: Row number (image shade, answer).
    :param task: Task family.
    :param section: Section name.
    :param stem: Page stem (``<volume>_<page>``).
    :return: Row dict.
    """
    answer = f"טקסט {i}"
    return {"answer": answer, "section": section, "stem": stem, "label_source": "talmud_gt",
            "target_chars": len(answer), "target_tokens": 7, "image": {"bytes": _jpeg(5 * i), "path": None},
            "question": f"prompt {task}", "task": task, "image_width": 24, "image_height": 16}


# (task, section, stem, kept by the v21b filter)
TALMUD_ROWS: List[Tuple[str, str, str, bool]] = [
    ("crop_transcribe", "gemara", "01_2", True),
    ("crop_transcribe", "gemara", "16_4", True),        # the volume exclusion is pages-only
    ("crop_transcribe", "rashi", "01_2", False),
    ("page_extract", "gemara", "01_2", False),
    ("page_extract", "rashi", "01_2", True),
    ("page_extract", "tosafot", "02_7b", True),
    ("page_extract", "rashi", "16_4", False),           # volume 16
    ("page_extract", "tosafot", "05_3b", False),        # volume 05
    ("page_extract", "rashi", "160_1", True),           # volume 160, not 16
    ("page_extract", "tosafot", "5_1", True),           # volume 5, not 05
    ("region_transcribe", "rashi", "01_2", False),
]


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    """Saved synthetic-like DatasetDict (train 3 rows, eval 2 rows)."""
    dsd = DatasetDict({
        "train": Dataset.from_list([_synth_row(i, "אבג" * (i + 1)) for i in range(3)],
                                   features=SYNTH_FEATURES),
        "eval": Dataset.from_list([_synth_row(i, "דה") for i in range(3, 5)], features=SYNTH_FEATURES)})
    dsd.save_to_disk(str(tmp_path / "synth"))
    return tmp_path / "synth"


@pytest.fixture
def talmud_parquet(tmp_path: Path) -> Path:
    """Hub-style parquet shards of a Talmud-like dataset (train in two shards, val, smoke)."""
    rows = [_talmud_row(i, *r[:3]) for i, r in enumerate(TALMUD_ROWS)]
    data = tmp_path / "talmud" / "data"
    data.mkdir(parents=True)
    shards = {"train-00000-of-00002": rows[:6], "train-00001-of-00002": rows[6:],
              "val-00000-of-00001": rows[3:6], "smoke-00000-of-00001": rows[:1]}
    for name, part in shards.items():
        Dataset.from_list(part, features=TALMUD_FEATURES).to_parquet(str(data / f"{name}.parquet"))
    return data


def _raw_images(ds: Dataset) -> List[bytes]:
    """Stored image bytes of every row, undecoded.

    :param ds: Dataset with an Image column.
    :return: Bytes per row.
    """
    return [cell["bytes"] for cell in ds.cast_column("image", ImageFeature(decode=False))["image"]]


def notebook_keeps(task: str, section: str, stem: str) -> bool:
    """The v21b notebook's Talmud selection (cell 2), transcribed literally.

    :param task: Row task.
    :param section: Row section.
    :param stem: Row stem.
    :return: True when the row is in gemara_crops or pages_small.
    """
    gemara_crops = task == "crop_transcribe" and section == "gemara"
    pages_small = (task == "page_extract" and section in ("rashi", "tosafot")
                   and stem.split("_")[0] not in ("16", "05"))
    return gemara_crops or pages_small


def test_talmud_filter_file_reproduces_the_notebook():
    row_filter = adapt_source.compile_filter(json.loads(TALMUD_FILTER.read_text()))
    stems = ["16", "16_2", "16_2b", "05", "05_3", "160_2", "016_2", "16b_2", "5_2", "05b_1",
             "01_16", "01_05", "x_16", "", "_16", "16__2", "1605_2"]
    for task in ("crop_transcribe", "page_extract", "region_transcribe"):
        for section in ("gemara", "rashi", "tosafot", "page"):
            for stem in stems:
                row = {"task": task, "section": section, "stem": stem}
                assert row_filter.keeps(row) == notebook_keeps(task, section, stem), row
    assert [row_filter.keeps({"task": t, "section": s, "stem": st}) for t, s, st, _ in TALMUD_ROWS] == \
        [keep for *_, keep in TALMUD_ROWS]


def test_compile_filter_semantics():
    row_filter = adapt_source.compile_filter({"any": [
        {"task": "a", "stem_regex": "^x"},
        {"task": ["b", "c"], "section_not_regex": "bad"}]})
    assert row_filter.columns == ("section", "stem", "task")
    assert row_filter.keeps({"task": "a", "section": "bad", "stem": "x1"})
    assert not row_filter.keeps({"task": "a", "section": "ok", "stem": "yx"})
    assert row_filter.keeps({"task": "c", "section": "fine", "stem": "y"})
    assert not row_filter.keeps({"task": "c", "section": "so bad", "stem": "y"})
    assert not row_filter.keeps({"task": "d", "section": "fine", "stem": "x"})
    # batched datasets filter: one list per column, in row_filter.columns order
    assert row_filter.batch_mask(["s", "bad", "s"], ["x", "x", "x"], ["a", "c", "d"]) == [True, False, False]


@pytest.mark.parametrize("spec", [
    {}, {"any": []}, {"any": [{}]}, {"any": [{"task": []}]}, {"all": [{"task": "a"}]},
    {"any": [{"stem_regex": 5}]}, {"any": ["task"]}])
def test_bad_filter_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        adapt_source.compile_filter(spec)


def test_split_planning_and_pairs():
    assert adapt_source.parse_pairs(["eval=val", "a=b=c"], "--x") == {"eval": "val", "a": "b=c"}
    for bad in (["eval"], ["=val"], ["eval="], ["a=b", "a=c"]):
        with pytest.raises(ValueError):
            adapt_source.parse_pairs(bad, "--x")
    assert adapt_source.plan_splits(["train", "eval"], {"eval": "val"}) == {"train": "train", "val": "eval"}
    assert adapt_source.plan_splits(["smoke", "train", "val"], {}, ["train", "val"]) == \
        {"train": "train", "val": "val"}
    with pytest.raises(ValueError, match="rename missing"):
        adapt_source.plan_splits(["train"], {"eval": "val"})
    with pytest.raises(ValueError, match="collide"):
        adapt_source.plan_splits(["train", "eval", "val"], {"eval": "val"})
    with pytest.raises(ValueError, match="cannot keep"):
        adapt_source.plan_splits(["train", "eval"], {}, ["train", "val"])


def test_parquet_split_files(talmud_parquet):
    files = adapt_source.parquet_split_files(talmud_parquet)
    assert list(files) == ["smoke", "train", "val"]
    assert [Path(f).name for f in files["train"]] == ["train-00000-of-00002.parquet",
                                                      "train-00001-of-00002.parquet"]


def test_adapt_synthetic_like_source(synth_dir, tmp_path):
    out = tmp_path / "adapted"
    adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val",
                       "--drop-columns", "font", "style", "--drop-columns", "mode",
                       "--meta", "hub_repo=isaacmg/synthetic_hebrew_v3", "--out", str(out)])
    assert images_once.saved_splits(out) == ["train", "val"]
    src, got = load_from_disk(str(synth_dir)), load_from_disk(str(out))
    for split, source in (("train", "train"), ("val", "eval")):
        ds = got[split]
        assert ds.column_names == list(FEATURES)
        assert ds.features.to_dict() == FEATURES.to_dict()
        assert len(ds) == len(src[source])
        assert ds["target_chars"][:] == [len(a) for a in src[source]["answer"]]
        assert ds["target_tokens"][:] == [0] * len(ds)
        assert _raw_images(ds) == _raw_images(src[source])          # image bytes untouched
        for col in ("question", "answer", "task", "section", "stem", "label_source",
                    "image_width", "image_height"):
            assert ds[col][:] == src[source][col][:]
    stats = json.loads((out / "stats.json").read_text())
    assert stats["split_sources"] == {"train": "train", "val": "eval"}
    assert stats["dropped_columns"] == ["font", "style", "mode"]
    assert stats["meta"] == {"hub_repo": "isaacmg/synthetic_hebrew_v3"}
    assert stats["splits"]["val"]["added_columns"] == ["target_chars", "target_tokens"]
    assert stats["splits"]["train"]["rows_in"] == stats["splits"]["train"]["rows_out"] == 3


def test_adapt_talmud_like_parquet_with_filter(talmud_parquet, tmp_path):
    out = tmp_path / "adapted"
    adapt_source.main(["--parquet-dir", str(talmud_parquet), "--keep-splits", "train", "val",
                       "--filter", str(TALMUD_FILTER), "--cache-dir", str(tmp_path / "cache"),
                       "--out", str(out)])
    assert images_once.saved_splits(out) == ["train", "val"]              # smoke dropped
    got = load_from_disk(str(out))
    source = {name: Dataset.from_parquet([str(p) for p in sorted(talmud_parquet.glob(f"{name}-*.parquet"))],
                                         cache_dir=str(tmp_path / "cache2"))
              for name in ("train", "val")}
    for split, ds in got.items():
        assert ds.column_names == list(FEATURES)
        assert ds.features.to_dict() == FEATURES.to_dict()
        want = source[split].filter(lambda t, s, st: notebook_keeps(t, s, st),
                                    input_columns=["task", "section", "stem"])
        assert len(ds) == len(want) > 0
        for col in list(FEATURES)[1:]:
            assert ds[col][:] == want[col][:]                             # values kept, rows in order
        assert _raw_images(ds) == _raw_images(want)
    assert got["train"]["stem"][:] == [st for t, s, st, keep in TALMUD_ROWS if keep]
    assert got["train"]["target_tokens"][:] == [7] * len(got["train"])     # existing columns kept
    stats = json.loads((out / "stats.json").read_text())
    assert stats["dropped_splits"] == ["smoke"] and stats["source_format"] == "parquet"
    assert stats["splits"]["train"] == {
        "rows_in": 11, "rows_out": 6, "added_columns": [],
        "tasks": {"crop_transcribe": 2, "page_extract": 4},
        "sections": {"gemara": 2, "rashi": 2, "tosafot": 2}}
    assert stats["filter"] == json.loads(TALMUD_FILTER.read_text())


def test_label_source_override(synth_dir, tmp_path):
    out = tmp_path / "adapted"
    adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val",
                       "--drop-columns", "font", "style", "mode", "--label-source", "synthetic_v3",
                       "--out", str(out)])
    got = load_from_disk(str(out))
    assert got["train"].column_names == list(FEATURES)
    assert set(got["train"]["label_source"][:]) == set(got["val"]["label_source"][:]) == {"synthetic_v3"}


def test_saved_source_is_left_untouched(synth_dir, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in synth_dir.rglob("*")}
    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps({"any": [{"stem_regex": "[024]$"}]}))
    adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val", "--filter", str(rule),
                       "--drop-columns", "font", "style", "mode", "--out", str(tmp_path / "adapted")])
    assert load_from_disk(str(tmp_path / "adapted"))["train"]["stem"][:] == ["synth_000", "synth_002"]
    assert {p: p.stat().st_mtime_ns for p in synth_dir.rglob("*")} == before   # no cache files written


def test_schema_mismatches_fail_loudly(synth_dir, tmp_path):
    with pytest.raises(ValueError, match="extra \\['font', 'style', 'mode'\\]"):
        adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val",
                           "--out", str(tmp_path / "a")])
    with pytest.raises(ValueError, match="cannot drop missing"):
        adapt_source.main(["--src", str(synth_dir), "--drop-columns", "font", "style", "mode", "colour",
                           "--out", str(tmp_path / "b")])
    wide = load_from_disk(str(synth_dir))["train"].remove_columns(["font", "style", "mode"])
    wide = wide.cast_column("image_width", Value("int64"))
    with pytest.raises(ValueError, match="types differ"):
        adapt_source.adapt_split(wide)
    with pytest.raises(ValueError, match="kept no rows"):
        adapt_source.adapt_dataset(DatasetDict({"train": wide.cast_column("image_width", Value("int32"))}),
                                   row_filter=adapt_source.compile_filter({"any": [{"task": "nope"}]}))


def test_output_dir_must_be_new(synth_dir, tmp_path):
    out = tmp_path / "adapted"
    out.mkdir()
    (out / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val",
                           "--drop-columns", "font", "style", "mode", "--out", str(out)])


def test_adapted_export_joins_a_ktiv_export(synth_dir, tmp_path):
    ktiv = DatasetDict({split: Dataset.from_list([
        {"image": {"bytes": _jpeg(200 + i), "path": None}, "question": "q", "answer": "א",
         "task": "fragment_transcribe", "section": "page", "stem": f"ktiv_{split}_{i}",
         "label_source": "ktiv_nli", "target_chars": 1, "target_tokens": 0,
         "image_width": 24, "image_height": 16} for i in range(2)], features=FEATURES)
        for split in ("train", "val")})
    ktiv.save_to_disk(str(tmp_path / "ktiv"))
    adapt_source.main(["--src", str(synth_dir), "--rename-split", "eval=val",
                       "--drop-columns", "font", "style", "mode", "--out", str(tmp_path / "synth_ktiv")])
    images_once.export_images_once(tmp_path / "ktiv", tmp_path / "ktiv_once")
    manifest = images_once.export_images_once(tmp_path / "synth_ktiv", tmp_path / "synth_once")
    assert (tmp_path / "synth_once" / "stats.json").exists()               # provenance travels
    assert manifest["splits"]["val"]["rows"] == 2
    tables = [pq.read_table(tmp_path / d / "rows" / "train.parquet") for d in ("ktiv_once", "synth_once")]
    check_layouts([json.loads(t.schema.metadata[images_once.META_KEY]) for t in tables])
    joined = pa.concat_tables([t.replace_schema_metadata(None) for t in tables])   # one schema
    assert joined.num_rows == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
