# File name: test_build_v22_mixture.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the v22 mixture builder on tiny synthetic images-once sources.

Four small DatasetDicts in the KTIV row schema (KTIV with transcription and
grounding families; editions and QA split over several ``train_*`` splits)
are saved, exported images-once and mixed, plus two extra sources supplied
with ``--source`` (Talmud replay and synthetic renders). The mixture must
repeat over-subscribed pools evenly, shuffle reproducibly, copy only the
images its rows reference, never put a val row on a train image, load
through ``ImagesOnceDataset``, keep the pilot's defaults when no extra source
is given, credit and rights-note every source it uses and keep forbidden
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
EXPORT_NAMES = {"ktiv": "ktiv", "pgp_editions": "editions", "documentary_grounding": "grounding",
                "pgp_qa": "qa", "talmud": "talmud", "synthetic": "synthetic"}
PILOT_DATASETS = ["ktiv", "pgp_editions", "documentary_grounding", "pgp_qa"]
# recorded in genizah_v22_pilot/mixture.json (built 2026-09-24 with `--workers 32` only)
PILOT_CONFIG_SHARES = {"ktiv_transcription": 0.3, "pgp_editions": 0.25, "ktiv_grounding": 0.15,
                       "documentary_grounding": 0.1, "pgp_qa": 0.2}
PILOT_CONFIG_VAL_ROWS = {"ktiv": 60, "pgp_editions": 60, "documentary_grounding": 40, "pgp_qa": 40}
SIX_SHARES = {"ktiv_transcription": 0.25, "pgp_editions": 0.2, "ktiv_grounding": 0.15,
              "documentary_grounding": 0.1, "pgp_qa": 0.1, "talmud_replay": 0.1, "synthetic": 0.1}
SIX_VAL_ROWS = {**VAL_ROWS, "talmud": 2, "synthetic": 1}


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


def _source_rows(tmp_path: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """dataset -> split -> rows of the four tiny sources (no val row on a train image).

    :param tmp_path: Fixture directory.
    :return: The rows.
    """
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
    return {"ktiv": ktiv, "pgp_editions": editions, "documentary_grounding": grounding, "pgp_qa": qa}


def _export_all(tmp_path: Path, rows: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Path]:
    """Export every source.

    :param tmp_path: Fixture directory.
    :param rows: dataset -> split -> rows.
    :return: dataset -> images-once export.
    """
    return {d: _export(tmp_path, EXPORT_NAMES[d], splits) for d, splits in rows.items()}


def _extra_rows(tmp_path: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """dataset -> split -> rows of tiny Talmud-replay and synthetic sources (KTIV row schema).

    :param tmp_path: Fixture directory.
    :return: The rows (no val row on a train image).
    """
    t = "Transcribe the page."
    talmud = {"train": [_row(_img(tmp_path, 40), "talmud_page", "t40", t, "גמרא"),
                        _row(_img(tmp_path, 41), "talmud_page", "t41", t, "משנה")],
              "val": [_row(_img(tmp_path, i), "talmud_page", f"t{i}", t, "תלמוד") for i in (42, 43, 44)]}
    synthetic = {"train": [_row(_img(tmp_path, i), "synthetic_render", f"s{i}", t, "אבגד") for i in (50, 51, 52)],
                 "val": [_row(_img(tmp_path, i), "synthetic_render", f"s{i}", t, "הוזח") for i in (53, 54)]}
    return {"talmud": talmud, "synthetic": synthetic}


@pytest.fixture
def sources(tmp_path: Path) -> Dict[str, Path]:
    """dataset -> images-once export of the four tiny sources."""
    return _export_all(tmp_path, _source_rows(tmp_path))


@pytest.fixture
def full_sources(tmp_path: Path) -> Dict[str, Path]:
    """dataset -> images-once export of the four pilot sources plus Talmud replay and synthetic."""
    return _export_all(tmp_path, {**_source_rows(tmp_path), **_extra_rows(tmp_path)})


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
    assert result["mixture"]["val"]["pgp_qa"] == {"pool": 1, "eligible": 1, "requested": 5, "taken": 1,
                                                  "dropped": 0, "refilled": 0}
    assert result["manifest"]["val_dedupe"]["dropped_rows"] == 0
    assert result["manifest"]["splits"]["val"]["rows"] == 5
    assert result["mixture"]["buckets"]["train"] == {"grounding": 5, "qa": 4, "transcription": 11}


def test_refill_val_draw_replaces_blocked_rows_while_the_pool_lasts():
    rng = np.random.default_rng(0)
    blocked = np.array([True, False, False, False, True])
    idx, dropped = mix.refill_val_draw(np.array([0, 1, 2]), blocked, 3, rng)
    assert idx.tolist() == [1, 2, 3] and dropped.tolist() == [0]      # row 3: the one clean spare
    blocked = np.array([True, False, True, True, True])
    idx, dropped = mix.refill_val_draw(np.array([0, 1, 2]), blocked, 3, rng)
    assert idx.tolist() == [1] and dropped.tolist() == [0, 2]          # exhausted: smaller count
    idx, dropped = mix.refill_val_draw(np.array([3, 1]), np.zeros(5, dtype=bool), 2, rng)
    assert idx.tolist() == [1, 3] and dropped.tolist() == []           # nothing blocked: draw stands


def _seed_drawing(dataset: str, pool: int, wanted: int, row: int) -> int:
    """First seed whose initial val draw of a dataset takes a given pool row.

    :param dataset: Dataset key.
    :param pool: Val pool size.
    :param wanted: Val rows requested.
    :param row: Pool row that must be drawn.
    :return: The seed.
    """
    return next(seed for seed in range(1000)
                if row in mix.component_rng(seed, f"val/{dataset}").choice(pool, size=wanted,
                                                                           replace=False))


def test_val_never_shares_a_train_image(tmp_path):
    rows = _source_rows(tmp_path)
    t = "Transcribe the page."
    # documentary val row 1 sits on an editions train page; three clean rows can replace it
    rows["documentary_grounding"]["val"] = [
        _row(_img(tmp_path, i), "locate", stem, LOCATE_Q, BOX_A)
        for i, stem in ((22, "g22_loc"), (10, "g10_clash"), (23, "g23_loc"), (24, "g24_loc"))]
    # editions val: one clean row + one on a KTIV train page -> nothing left to refill with
    rows["pgp_editions"]["val"].append(_row(_img(tmp_path, 0), "fragment_transcribe", "e0_clash", t, "טקסט"))
    sources = _export_all(tmp_path, rows)
    seed = _seed_drawing("documentary_grounding", pool=4, wanted=2, row=1)
    val_rows = {"ktiv": 2, "pgp_editions": 2, "documentary_grounding": 2, "pgp_qa": 1}
    result = _build(sources, tmp_path / "out", seed=seed, val_rows=val_rows)

    train = pq.read_table(tmp_path / "out" / "rows" / "train.parquet")
    val = pq.read_table(tmp_path / "out" / "rows" / "val.parquet")
    assert not set(val[images_once.SHA_COLUMN].to_pylist()) & set(train[images_once.SHA_COLUMN].to_pylist())
    stems = val["stem"].to_pylist()
    assert "g10_clash" not in stems and "e0_clash" not in stems
    infos = result["mixture"]["val"]
    assert infos["documentary_grounding"] == {"pool": 4, "eligible": 3, "requested": 2, "taken": 2,
                                              "dropped": 1, "refilled": 1}
    assert infos["pgp_editions"] == {"pool": 2, "eligible": 1, "requested": 2, "taken": 1,
                                     "dropped": 1, "refilled": 0}
    dedupe = json.loads((tmp_path / "out" / "manifest.json").read_text())["val_dedupe"]
    assert dedupe == result["manifest"]["val_dedupe"]
    assert sorted(dedupe["dropped_stems"]) == ["e0_clash", "g10_clash"]
    assert dedupe["dropped_rows"] == 2 and dedupe["refilled_rows"] == 1 and dedupe["rule"]
    assert result["manifest"]["splits"]["val"]["rows"] == len(stems) == 2 + 1 + 2 + 1
    assert result["stats"]["val"]["by_source"]["documentary_grounding"] == 2


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


def _cli_args(sources: Dict[str, Path], tmp_path: Path, out: Path, *extra: str) -> List[str]:
    """Command line of a 20-row fixture build (pilot DatasetDicts exported as needed).

    :param sources: dataset -> images-once export.
    :param tmp_path: Fixture directory (holds the saved DatasetDicts).
    :param out: Mixture directory.
    :param extra: Further arguments.
    :return: argv for :func:`mix.main`.
    """
    return ["--out", str(out), "--train-rows", "20", "--workers", "2", "--ktiv-dir", str(sources["ktiv"]),
            "--editions-dir", str(tmp_path / "editions"), "--grounding-dir", str(tmp_path / "grounding"),
            "--qa-dir", str(tmp_path / "qa"), *extra]


def test_parse_sources() -> None:
    """``--source NAME=DIR`` takes the extra datasets only, each once."""
    assert mix.parse_sources(["talmud=/nas/t", " synthetic = ~/s "]) == {
        "talmud": Path("/nas/t"), "synthetic": Path("~/s").expanduser()}
    assert mix.parse_sources([]) == {}
    for spec, match in (("talmud", "NAME=DIR"), ("talmud=", "NAME=DIR"), ("=/nas/t", "NAME=DIR"),
                        ("talmud_replay=/nas/t", "unknown dataset 'talmud_replay'"),
                        ("ktiv=/nas/k", "--ktiv-dir"), ("pgp_qa=/nas/q", "--qa-dir")):
        with pytest.raises(ValueError, match=match):
            mix.parse_sources([spec])
    with pytest.raises(ValueError, match="twice"):
        mix.parse_sources(["talmud=/nas/a", "talmud=/nas/b"])


def test_pilot_call_keeps_the_pilot_defaults() -> None:
    """Without ``--source`` the pilot's command line resolves to its recorded shares, val rows, quotas."""
    a = mix.build_parser().parse_args(["--workers", "32"])
    assert a.source == [] and a.shares is None and a.val_rows is None
    datasets = [*mix.PILOT_FLAGS, *mix.parse_sources(a.source)]
    assert datasets == PILOT_DATASETS
    shares, val_rows, quotas = mix.resolve_plan(datasets, a.train_rows, a.shares, a.val_rows)
    assert list(shares.items()) == list(PILOT_CONFIG_SHARES.items())
    assert list(val_rows.items()) == list(PILOT_CONFIG_VAL_ROWS.items())
    assert quotas == {"ktiv_transcription": 2400, "pgp_editions": 2000, "ktiv_grounding": 1200,
                      "documentary_grounding": 800, "pgp_qa": 1600}


def test_default_shares_and_val_rows_follow_the_supplied_sources() -> None:
    """Each supplied extra dataset adds its default share (pilot shares scaled) and val rows."""
    six = PILOT_DATASETS + ["talmud", "synthetic"]
    shares = mix.default_shares(six)
    assert list(shares)[-2:] == ["talmud_replay", "synthetic"]
    assert shares["talmud_replay"] == 0.08 and shares["synthetic"] == 0.07
    assert sum(shares.values()) == pytest.approx(1.0)
    for comp, share in mix.DEFAULT_SHARES.items():
        assert shares[comp] == pytest.approx(share * 0.85)
    only_talmud = mix.default_shares(PILOT_DATASETS + ["talmud"])
    assert "synthetic" not in only_talmud and sum(only_talmud.values()) == pytest.approx(1.0)
    assert mix.default_val_rows(six) == {**PILOT_CONFIG_VAL_ROWS, "talmud": 30, "synthetic": 20}
    assert mix.default_val_rows(PILOT_DATASETS + ["synthetic"]) == {**PILOT_CONFIG_VAL_ROWS, "synthetic": 20}
    assert mix.plan_quotas(shares, 20000)["talmud_replay"] == 1600


def test_share_or_val_rows_without_a_source_is_an_error(sources, tmp_path) -> None:
    """Shares or val rows needing an unsupplied or unfinished source fail before anything is written."""
    shares = {**mix.DEFAULT_SHARES, "pgp_qa": 0.1, "talmud_replay": 0.1}
    with pytest.raises(ValueError, match=r"talmud_replay needs dataset 'talmud' \(--source talmud=DIR\)"):
        _build(sources, tmp_path / "out", shares=shares)
    with pytest.raises(ValueError, match=r"val rows ask for dataset 'synthetic'"):
        _build(sources, tmp_path / "out", val_rows={**VAL_ROWS, "synthetic": 2})
    with pytest.raises(FileNotFoundError, match="talmud: .* not a finished images-once export"):
        _build({**sources, "talmud": tmp_path / "talmud_in_progress"}, tmp_path / "out", shares=shares)
    # the CLI checks the plan before any export step: the pilot dirs need not even exist
    nowhere = str(tmp_path / "nowhere")
    with pytest.raises(ValueError, match=r"--source talmud=DIR"):
        mix.main(["--out", str(tmp_path / "cli"), "--shares", json.dumps(shares), "--ktiv-dir", nowhere,
                  "--editions-dir", nowhere, "--grounding-dir", nowhere, "--qa-dir", nowhere])
    with pytest.raises(FileNotFoundError, match="talmud: "):
        mix.main(_cli_args(sources, tmp_path, tmp_path / "cli", "--source", f"talmud={nowhere}"))
    assert not (tmp_path / "out").exists() and not (tmp_path / "cli").exists()


def test_extra_sources_are_mixed_at_the_requested_shares(full_sources, tmp_path) -> None:
    """``--source`` Talmud and synthetic rows join at their shares, each row under its one component."""
    out = tmp_path / "cli"
    mix.main(_cli_args(full_sources, tmp_path, out, "--source", f"talmud={full_sources['talmud']}",
                       "--source", f"synthetic={full_sources['synthetic']}",
                       "--shares", json.dumps(SIX_SHARES), "--val-rows", json.dumps(SIX_VAL_ROWS)))
    mixture = json.loads((out / "mixture.json").read_text())
    comps = mixture["components"]
    assert {c: i["taken"] for c, i in comps.items()} == {
        "ktiv_transcription": 5, "pgp_editions": 4, "ktiv_grounding": 3, "documentary_grounding": 2,
        "pgp_qa": 2, "talmud_replay": 2, "synthetic": 2}
    assert (comps["talmud_replay"]["dataset"], comps["talmud_replay"]["bucket"]) == ("talmud", "transcription")
    assert (comps["synthetic"]["dataset"], comps["synthetic"]["pool"]) == ("synthetic", 3)
    assert mixture["config"]["dirs"]["talmud"] == str(full_sources["talmud"])
    assert mixture["train_splits"]["synthetic"] == ["train"]
    train = _rows(out, "train")
    assert {r["stem"] for r in train if r["source"] == "talmud_replay"} == {"t40", "t41"}
    synthetic = [r["stem"] for r in train if r["source"] == "synthetic"]
    assert len(set(synthetic)) == 2 and set(synthetic) <= {"s50", "s51", "s52"}
    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"]["unique_images_by_source"]["talmud_replay"] == 2
    assert stats["train"]["unique_images_by_source"]["ktiv_transcription"] == 3      # pages 0, 1, 2
    assert comps["synthetic"]["unique_images"] == 2
    assert stats["train"]["by_bucket"] == {"grounding": 5, "qa": 2, "transcription": 13}
    assert {r["source"] for r in _rows(out, "val")} == set(comps)
    direct = tmp_path / "direct"                            # the CLI builds what the direct call builds
    _build(full_sources, direct, shares=SIX_SHARES, val_rows=SIX_VAL_ROWS)
    for split in ("train", "val"):
        assert _rows(out, split) == _rows(direct, split)
    assert mix.self_check(out)["train"]["rows"] == 20


def test_val_rows_of_extra_sources_avoid_train_images(tmp_path) -> None:
    """Talmud and synthetic val rows are drawn, deduped against train images and refilled like any source."""
    rows = {**_source_rows(tmp_path), **_extra_rows(tmp_path)}
    t = "Transcribe the page."
    # talmud val row 0 sits on a Talmud train page (40); three clean rows can replace it
    rows["talmud"]["val"].insert(0, _row(_img(tmp_path, 40), "talmud_page", "t40_clash", t, "גמרא"))
    # synthetic val: two clean rows + one on a KTIV train page (0), all three asked for
    rows["synthetic"]["val"].append(_row(_img(tmp_path, 0), "synthetic_render", "s0_clash", t, "טקסט"))
    sources = _export_all(tmp_path, rows)
    seed = _seed_drawing("talmud", pool=4, wanted=2, row=0)
    result = _build(sources, tmp_path / "out", seed=seed, shares=SIX_SHARES,
                    val_rows={**VAL_ROWS, "talmud": 2, "synthetic": 3})

    train = pq.read_table(tmp_path / "out" / "rows" / "train.parquet")
    val = pq.read_table(tmp_path / "out" / "rows" / "val.parquet")
    assert not set(val[images_once.SHA_COLUMN].to_pylist()) & set(train[images_once.SHA_COLUMN].to_pylist())
    infos = result["mixture"]["val"]
    assert infos["talmud"] == {"pool": 4, "eligible": 3, "requested": 2, "taken": 2, "dropped": 1, "refilled": 1}
    assert infos["synthetic"] == {"pool": 3, "eligible": 2, "requested": 3, "taken": 2, "dropped": 1,
                                  "refilled": 0}
    assert sorted(result["manifest"]["val_dedupe"]["dropped_stems"]) == ["s0_clash", "t40_clash"]
    assert result["stats"]["val"]["by_source"]["talmud_replay"] == 2
    assert result["stats"]["val"]["by_source"]["synthetic"] == 2
    assert result["mixture"]["val_missing_components"] == []


def test_default_val_rows_cover_the_extra_sources(full_sources, tmp_path) -> None:
    """With defaults, the extra sources get their default shares and val rows (capped by their pools)."""
    result = _build(full_sources, tmp_path / "out", shares=None, val_rows=None)
    cfg = result["mixture"]["config"]
    assert cfg["shares"] == mix.default_shares(full_sources) and cfg["val_rows"]["talmud"] == 30
    val = result["mixture"]["val"]
    assert (val["talmud"]["requested"], val["talmud"]["taken"]) == (30, 3)
    assert (val["synthetic"]["requested"], val["synthetic"]["taken"]) == (20, 2)
    for comp, info in result["mixture"]["components"].items():
        assert info["quota"] == round(cfg["shares"][comp] * 20)


def test_card_lists_used_components_and_rights_notes(full_sources, tmp_path) -> None:
    """The card lists every component with train rows and rights-notes each extra source it uses."""
    result = _build(full_sources, tmp_path / "full", shares=SIX_SHARES, val_rows=SIX_VAL_ROWS)
    card = (tmp_path / "full" / "README.md").read_text()
    for comp, i in result["mixture"]["components"].items():
        assert (f"| `{comp}` | {i['bucket']} | {i['dataset']} | {i['pool']:,} | {i['share']:.4g} | "
                f"{i['taken']:,} |") in card
        assert card.count(f"| `{comp}` |") == 1
    assert "# Genizah v22 training mixture: full (images-once)" in card and "6 source datasets" in card
    for note in ("HebrewBooks.org", "© Moznaim Publishers", '"No commercial use allowed"', "stays private",
                 "generated renders", "CC-BY compatible", "NLI-KTIV", "Princeton Geniza Project"):
        assert note in card
    assert not any(word in card.lower() for word in mix.FORBIDDEN_CARD_STRINGS)

    _build({d: full_sources[d] for d in PILOT_DATASETS}, tmp_path / "pilot")
    card = (tmp_path / "pilot" / "README.md").read_text()
    assert "Moznaim" not in card and "CC-BY" not in card and "4 source datasets" in card

    # Talmud supplied but unused (share 0, no val rows): no row, no note
    shares = {**SIX_SHARES, "talmud_replay": 0.0, "synthetic": 0.2}
    _build(full_sources, tmp_path / "no_talmud", shares=shares, val_rows={**VAL_ROWS, "synthetic": 1})
    card = (tmp_path / "no_talmud" / "README.md").read_text()
    assert "`talmud_replay`" not in card and "Moznaim" not in card and "CC-BY compatible" in card
    assert "5 source datasets" in card


def test_layouts_tolerate_column_order_and_name_the_culprit(full_sources, tmp_path) -> None:
    """An extra export with the KTIV columns in another order mixes; a real difference names its source."""
    base = {"columns": ["image", "question", "answer"], "image_column": "image", "image_mode": None}
    mix.check_layouts([base, {**base, "columns": ["answer", "image", "question"]}], ["ktiv", "talmud"])
    with pytest.raises(ValueError, match=r"talmud lacks \['answer'\] and adds \['tractate'\]"):
        mix.check_layouts([base, {**base, "columns": ["image", "question", "tractate"]}], ["ktiv", "talmud"])
    with pytest.raises(ValueError, match="image_mode: synthetic has 'RGB', ktiv has None"):
        mix.check_layouts([base, {**base, "image_mode": "RGB"}], ["ktiv", "synthetic"])
    for path in (full_sources["talmud"] / "rows").glob("*.parquet"):      # "answer" moved last
        table = pq.read_table(path)
        meta = json.loads(table.schema.metadata[images_once.META_KEY])
        meta["columns"] = [c for c in meta["columns"] if c != "answer"] + ["answer"]
        table = table.select([c for c in table.column_names if c != "answer"] + ["answer"])
        pq.write_table(table.replace_schema_metadata(
            {images_once.META_KEY: json.dumps(meta, ensure_ascii=False).encode()}), path)
    out = tmp_path / "out"
    _build(full_sources, out, shares=SIX_SHARES, val_rows=SIX_VAL_ROWS)
    ktiv_columns = pq.read_table(full_sources["ktiv"] / "rows" / "train.parquet").column_names
    assert pq.read_table(out / "rows" / "train.parquet").column_names == ktiv_columns + ["source"]
    ds = images_once.ImagesOnceDataset(out / "rows" / "train.parquet", out / "images")
    talmud = [ds[i] for i in range(len(ds)) if ds[i]["source"] == "talmud_replay"]
    assert {(item["stem"], item["answer"], item["question"]) for item in talmud} == {
        ("t40", "גמרא", "Transcribe the page."), ("t41", "משנה", "Transcribe the page.")}


def test_each_non_ktiv_dataset_feeds_exactly_one_component(monkeypatch) -> None:
    """Every non-KTIV dataset maps all its rows to its one component; a second entry is refused."""
    table = pa.table({"task": ["talmud_page", "anything"]})
    assert set(mix.TASK_SPLIT_DATASETS) == {"ktiv", "vqa"}
    datasets = {d for d, _ in mix.COMPONENTS.values()} - set(mix.TASK_SPLIT_DATASETS)
    assert datasets == {"pgp_editions", "documentary_grounding", "pgp_qa", "talmud", "synthetic", "pgp_edition_pages",
                        "arabic_editions", "agapet", "muharaf", "baybars", "iskandar"}
    for dataset in datasets:
        (comp,) = set(mix.row_components(dataset, table))
        assert mix.COMPONENTS[comp][0] == dataset
    assert mix.row_components("talmud", table) == ["talmud_replay", "talmud_replay"]
    assert mix.row_components("agapet", table) == ["arabic_agapet", "arabic_agapet"]         # outside Arabic sets: one each
    assert {mix.COMPONENTS[c] for c in mix.COMPONENTS if c.startswith("arabic_")} == {
        ("arabic_editions", "transcription"), ("agapet", "transcription"), ("muharaf", "transcription"),
        ("baybars", "transcription"), ("iskandar", "transcription")}
    assert set(mix.RIGHTS_NOTES) >= {"arabic_editions", "agapet", "muharaf", "baybars", "iskandar"}
    assert not any(bad in note.lower() for note in mix.RIGHTS_NOTES.values() for bad in mix.FORBIDDEN_CARD_STRINGS)
    with pytest.raises(ValueError, match="unknown dataset"):
        mix.row_components("talmud_replay", table)
    monkeypatch.setitem(mix.COMPONENTS, "synthetic_boxes", ("synthetic", "grounding"))
    with pytest.raises(ValueError, match="exactly one"):
        mix.row_components("synthetic", table)


def test_mixture_stats_count_unique_images_per_component() -> None:
    """Stats report distinct images per component and skip missing target lengths."""
    table = pa.table({"source": ["talmud_replay", "talmud_replay", "synthetic"], "task": ["p", "p", "s"],
                      "target_chars": pa.array([4, None, 6], pa.int32()),
                      images_once.SHA_COLUMN: ["a", "a", "b"]})
    stats = mix.mixture_stats({"train": table})["train"]
    assert stats["unique_images_by_source"] == {"synthetic": 1, "talmud_replay": 1}
    assert stats["unique_images"] == 2 and stats["by_bucket"] == {"transcription": 3}
    assert stats["mean_target_chars"] == {"synthetic": 6.0, "talmud_replay": 4.0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_transcription_rows_with_editors_marks_never_enter_the_mixture(tmp_path) -> None:
    rows = _source_rows(tmp_path)
    t = "Transcribe the page."
    rows["pgp_editions"]["train_page"] += [_row(_img(tmp_path, 13), "fragment_transcribe", "e13_page", t, "שמואל $דויד$"),
                                           _row(_img(tmp_path, 14), "fragment_transcribe", "e14_page", t, "דינרין ____")]
    rows["pgp_editions"]["val"] += [_row(_img(tmp_path, 15), "fragment_transcribe", "e15_page", t, "כסף & ב")]
    result = _build(_export_all(tmp_path, rows), tmp_path / "mix")
    train = pq.read_table(tmp_path / "mix" / "rows" / "train.parquet").to_pylist()
    val = pq.read_table(tmp_path / "mix" / "rows" / "val.parquet").to_pylist()
    assert not {"e13_page", "e14_page", "e15_page"} & {r["stem"] for r in train + val}
    assert {"e10_page", "e11_page"} <= {r["stem"] for r in train}                    # the clean pages are still drawn
    assert any(r["task"] == "qa_date" and "{" in r["answer"] for r in train)        # JSON answers are not transcriptions
    dropped = result["mixture"]["unclean_transcriptions_dropped"]
    assert dropped["stems"] == {"pgp_editions": {"train": ["e13_page", "e14_page"], "val": ["e15_page"]}}
    assert result["mixture"]["components"]["pgp_editions"]["pool"] == 3              # two pages + one line row


def test_drop_unclean_transcriptions_keeps_the_table_when_nothing_is_dropped() -> None:
    table = pa.table({"task": ["fragment_transcribe", "locate"], "answer": ["שורה [...]", '{"bbox_2d": [1, 2, 3, 4]}'],
                      "stem": ["a", "b"]})
    kept, dropped = mix.drop_unclean_transcriptions(table)
    assert kept is table and dropped == []
    kept, dropped = mix.drop_unclean_transcriptions(table, chars="[")
    assert kept["stem"].to_pylist() == ["b"] and dropped == ["a"]


def test_cli_neither_exports_nor_loads_a_pilot_dataset_the_plan_does_not_use(full_sources, tmp_path) -> None:
    """A plan of KTIV + extra sources runs with pilot DatasetDict dirs that do not exist."""
    nowhere = str(tmp_path / "nowhere")
    out = tmp_path / "cli"
    shares = {"ktiv_transcription": 0.5, "talmud_replay": 0.3, "synthetic": 0.2}
    mix.main(["--out", str(out), "--train-rows", "10", "--workers", "2", "--ktiv-dir", str(full_sources["ktiv"]),
              "--editions-dir", nowhere, "--grounding-dir", nowhere, "--qa-dir", nowhere,
              "--source", f"talmud={full_sources['talmud']}", "--source", f"synthetic={full_sources['synthetic']}",
              "--shares", json.dumps(shares), "--val-rows", json.dumps({"ktiv": 1, "talmud": 2, "synthetic": 1})])
    mixture = json.loads((out / "mixture.json").read_text())
    assert set(mixture["config"]["dirs"]) == {"ktiv", "talmud", "synthetic"} and not (tmp_path / "nowhere").exists()
    assert {c: i["taken"] for c, i in mixture["components"].items()} == {"ktiv_transcription": 5, "talmud_replay": 3, "synthetic": 2}
    assert mix.plan_datasets({"ktiv_transcription": 0.5, "pgp_qa": 0.0}, {"pgp_editions": 3, "talmud": 0}) == {"ktiv", "pgp_editions"}


def test_train_variant_drops_components_and_keeps_order_meta_and_card(full_sources, tmp_path) -> None:
    out = tmp_path / "mix"
    _build(full_sources, out, shares=SIX_SHARES, val_rows=SIX_VAL_ROWS)
    card_before = (out / "README.md").read_text()
    info = mix.write_train_variant(out, "ctl", ["talmud_replay", "synthetic"])
    full, kept = _rows(out, "train"), _rows(out, "train_ctl")
    assert kept == [r for r in full if r["source"] not in ("talmud_replay", "synthetic")] and len(kept) == 16
    assert info["rows"] == 16 and info["file"] == "rows/train_ctl.parquet" and "talmud_replay" not in info["by_source"]
    variant = pq.read_table(out / "rows" / "train_ctl.parquet")
    assert info["unique_images"] == len(set(variant["image_sha1"].to_pylist()))
    meta = variant.schema.metadata
    assert meta == pq.read_table(out / "rows" / "train.parquet").schema.metadata           # loads like the full file
    ds = images_once.ImagesOnceDataset(out / "rows" / "train_ctl.parquet", out / "images")
    assert len(ds) == 16 and ds[0]["image"].size[1] == 30
    assert json.loads((out / "mixture.json").read_text())["train_variants"]["ctl"] == info
    card = (out / "README.md").read_text()
    assert "rows/train_ctl.parquet   16 rows: rows/train.parquet without synthetic, talmud_replay" in card
    assert card.replace(card[card.index("rows/train_ctl.parquet"):card.index("images/<sha1>.jpg")], "") == card_before
    with pytest.raises(ValueError, match="have no train rows"):
        mix.write_train_variant(out, "bad", ["arabic_agapet"])
    with pytest.raises(ValueError, match="drop every train row"):
        mix.write_train_variant(out, "none", sorted({r["source"] for r in full}))


def test_card_mentions_model_derived_boxes_only_with_documentary_grounding(full_sources, tmp_path) -> None:
    _build(full_sources, tmp_path / "with", shares=SIX_SHARES, val_rows=SIX_VAL_ROWS)
    assert "boxes are model-derived" in (tmp_path / "with" / "README.md").read_text()
    sources = {d: p for d, p in full_sources.items() if d in ("ktiv", "talmud", "synthetic")}
    _build(sources, tmp_path / "without", shares={"ktiv_transcription": 0.5, "talmud_replay": 0.3, "synthetic": 0.2},
           val_rows={"ktiv": 1, "talmud": 2, "synthetic": 1})
    assert "model-derived" not in (tmp_path / "without" / "README.md").read_text()


def test_val_rows_come_from_trained_components_only(full_sources, tmp_path) -> None:
    """A plan that trains KTIV transcription only draws no KTIV box rows into val; a full plan is unchanged."""
    sources = {d: p for d, p in full_sources.items() if d in ("ktiv", "talmud")}
    out = tmp_path / "transcription_only"
    _build(sources, out, shares={"ktiv_transcription": 0.6, "talmud_replay": 0.4}, val_rows={"ktiv": 2, "talmud": 1})
    val = _rows(out, "val")
    assert [r["stem"] for r in val if r["source"].startswith("ktiv")] == ["k4_page"]          # k4_loc is a box row
    assert json.loads((out / "mixture.json").read_text())["val"]["ktiv"] == {
        "pool": 1, "eligible": 1, "requested": 2, "taken": 1, "dropped": 0, "refilled": 0}
    both = tmp_path / "both"
    _build(sources, both, shares={"ktiv_transcription": 0.3, "ktiv_grounding": 0.3, "talmud_replay": 0.4},
           val_rows={"ktiv": 2, "talmud": 1})
    assert {r["stem"] for r in _rows(both, "val") if r["source"].startswith("ktiv")} == {"k4_page", "k4_loc"}
    table = pa.table({"task": ["locate", "fragment_transcribe"], "stem": ["a", "b"]})
    assert mix.trained_component_rows("ktiv", table, {"talmud_replay"}) is table              # val-only dataset: untouched
    assert mix.trained_component_rows("ktiv", table, {"ktiv_grounding"})["stem"].to_pylist() == ["a"]


def _vqa_rows(tmp_path: Path) -> Dict[str, List[Dict]]:
    """split -> rows of a tiny page-parse set: one train split per task family, like build_vqa_parse.py writes."""
    parse = '[{"n": 1, "text": "שורה"}]'
    q = "Line-by-line reading of this page (JSON, in reading order):\n" + parse + "\n\nWhich line gives the date?"
    return {
        "train_parse_lines": [_row(_img(tmp_path, 60 + i), "parse_lines", f"v{i}_parse", "Parse the page into JSON.", parse) for i in range(3)],
        "train_question_from_parse": [_row(_img(tmp_path, 60 + i), "question_from_parse", f"v{i}_q", q, '{"answer": "שורה", "line": 1}') for i in range(3)],
        "train_lookup_from_parse": [_row(_img(tmp_path, 60), "lookup_from_parse", "v0_lookup", q, '{"line": 1, "text": "שורה"}')],
        "val": [_row(_img(tmp_path, 70), "parse_lines", "v70_parse", "Parse the page into JSON.", parse),
                _row(_img(tmp_path, 70), "question_from_parse", "v70_q", q, '{"answer": null}'),
                _row(_img(tmp_path, 71), "question_from_parse", "v71_q", q, '{"answer": null}'),
                _row(_img(tmp_path, 71), "lookup_from_parse", "v71_lookup", q, '{"line": 1, "text": "שורה"}')]}


def test_one_export_feeds_a_component_per_task_family_and_val_quotas_may_name_components(sources, tmp_path) -> None:
    """The page-parse set comes in as one export; each task family is its own component with its own share and val quota."""
    ds_dir = tmp_path / "vqa"
    DatasetDict({s: Dataset.from_list(rows, features=FEATURES) for s, rows in _vqa_rows(tmp_path).items()}).save_to_disk(str(ds_dir))
    export = mix.ensure_export(ds_dir, mix.export_dir_for(ds_dir))
    srcs = {"ktiv": sources["ktiv"], "vqa": export}
    shares = {"ktiv_transcription": 0.4, "vqa_parse_lines": 0.3, "vqa_question": 0.3}
    out = tmp_path / "mix"
    result = _build(srcs, out, train_rows=10, shares=shares, val_rows={"ktiv": 1, "vqa_question": 2, "vqa_parse_lines": 1})
    train, val = _rows(out, "train"), _rows(out, "val")
    assert Counter(r["source"] for r in train) == {"ktiv_transcription": 4, "vqa_parse_lines": 3, "vqa_question": 3}
    assert {r["stem"] for r in train if r["source"] == "vqa_question"} == {"v0_q", "v1_q", "v2_q"}
    assert "v0_lookup" not in {r["stem"] for r in train}                                   # an unrequested family stays out
    assert sorted(r["stem"] for r in val if r["source"].startswith("vqa")) == ["v70_parse", "v70_q", "v71_q"]
    info = result["mixture"]["val"]
    assert info["vqa_question"]["taken"] == 2 and info["vqa_question"]["pool"] == 2 and info["vqa_parse_lines"]["pool"] == 1
    assert result["mixture"]["components"]["vqa_question"]["bucket"] == "question_from_parse"
    assert mix.val_dataset("vqa_question") == "vqa" and mix.val_dataset("vqa") == "vqa" and mix.val_dataset("pgp_qa") == "pgp_qa"
    assert mix.plan_datasets(shares, {"vqa_question": 2}) == {"ktiv", "vqa"}
    dataset_wide = tmp_path / "mix2"                                                      # a dataset key still draws across its families
    _build(srcs, dataset_wide, train_rows=10, shares=shares, val_rows={"ktiv": 1, "vqa": 3})
    assert sorted(r["stem"] for r in _rows(dataset_wide, "val") if r["source"].startswith("vqa")) == ["v70_parse", "v70_q", "v71_q"]
    with pytest.raises(ValueError, match=r"val rows ask for dataset 'vqa'"):
        _build({"ktiv": sources["ktiv"]}, tmp_path / "bad", train_rows=10, shares={"ktiv_transcription": 1.0}, val_rows={"vqa_question": 1})
    with pytest.raises(ValueError, match="without a component"):
        mix.row_components("vqa", pa.table({"task": ["parse_lines", "essay"]}))


def test_a_page_parse_with_editors_marks_takes_its_questions_with_it() -> None:
    """One unclean JSON parse line drops every row of that dataset on the same image; clean pages are untouched."""
    q = '{"answer": "דויד", "line": 1}'
    table = pa.table({
        "task": ["parse_lines", "question_from_parse", "parse_lines", "question_from_parse", "locate"],
        "answer": ['[{"n": 1, "text": "שמואל $דויד$"}]', q, '[{"n": 1, "text": "שורה [...]"}]', q, '{"bbox_2d": [1, 2, 3, 4]}'],
        "stem": ["p1_parse", "p1_q", "p2_parse", "p2_q", "p3_loc"],
        images_once.SHA_COLUMN: ["img1", "img1", "img2", "img2", "img3"]})
    kept, dropped = mix.drop_unclean_transcriptions(table)
    assert kept["stem"].to_pylist() == ["p2_parse", "p2_q", "p3_loc"] and dropped == ["p1_parse", "p1_q"]
    assert mix.unclean_answer("parse_lines", '[{"n": 1, "text": "ab"}]') is False           # JSON syntax is not page text
    assert mix.unclean_answer("question_from_parse", '{"answer": "a_b"}') is False and mix.unclean_answer("line_transcribe", "a_b")
