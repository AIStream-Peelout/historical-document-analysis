# File name: test_colab_notebook_novel_v2.py
# Date: 11/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the novel_v2 notebook derivation and training-rows tooling (``colab/derive_novel_v2.py``).

novel_v2 re-uses every function of ``derive_novel_v1`` re-bound to its own names. These tests check the names and the
schedule, the pass plan of the 1,739-page export, that the re-binding reads novel_v2's namespace and leaves novel_v1
alone, the dataset's own conventions (section ``ja_ar_page``, ``label_source`` ``legacy_store_edition``, the
``_build`` manifest with ``train_page`` splits and novelty tiers, stats without a name and with internal source keys),
the notebook (pinned warm start novel-v1 step 326, the placeholder refusals, only the planned cells changed against
novel_v1's), the boot-disk guard and the push's checks and staging. No NAS, Hub, GPU or network is needed.
"""
import copy
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.finetuning.qwen_hebrew import build_v22_mixture as bvm
from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
from src.finetuning.qwen_hebrew.colab import derive_novel_v1 as v1
from src.finetuning.qwen_hebrew.colab import derive_novel_v2 as d2
from src.finetuning.qwen_hebrew.images_once import export_images_once
from src.finetuning.qwen_hebrew.novel_build.targets import measure_script
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT, SCRIPT_NEUTRAL_PROMPT

SHA = "d" * 40
EXPORT_PAGES = 1739                     # train_page of the genizah_novel_v2 export
EXPORT_PAGES_LESS_RNL = 1688            # less its 51 Russian National Library train rows (novel_v1's default)
LABEL = "legacy_store_edition"
#: two ledger sources of this round whose rows carry LABEL (read from the build's map, never spelled here)
LEDGER = sorted(k for k, v in d2.BUILD.LABEL_SOURCES.items() if v == LABEL)[:2]
HEBREW = "כתאבי אליך אטאל אללה בקאך ואדאם עזך וסלאמתך [...] שלום רב לאהובי ולכל ביתו ולכל אשר לו אמן\nואעלם אן אלכתב וצלת"
ARABIC = "كتابي اليك اطال الله بقاك وادام عزك وتاييدك وسعادتك وسلامتك من مصر لخمس خلون من شهر رمضان [...]\nوالحمد لله وحده"
TRACKED_FORBIDDEN = (*bvm.FORBIDDEN_CARD_STRINGS, "/" + "volumes" + "/")
RNL_STEM = "jaar_lse_t_001_r"


def _src(nb: dict, i: int) -> str:
    """Source of cell ``i``.

    :param nb: Notebook.
    :param i: Cell index.
    :return: The joined source.
    """
    return "".join(nb["cells"][i]["source"])


# ----------------------------------------------------------------------------- names, schedule, pass plan


def test_names_warm_start_and_schedule_are_the_planned_ones() -> None:
    """New private names for every repository; novel_v1's schedule unchanged; warm start novel-v1 step 326 by commit."""
    assert (d2.ROUND, d2.TAG, d2.RUN_NAME) == ("novel_v2", "novel-v2", "genizah_novel_v2")
    assert d2.CKPT_REPO == "isaacmg/qwen3-vl-8b-hebrew-novel-v2-ckpt" and d2.MERGED_REPO == "isaacmg/qwen3-vl-8b-hebrew-novel-v2"
    assert d2.DATA_REPO == "isaacmg/genizah_novel_v2_train" != d2.SOURCE_REPO == "isaacmg/genizah_novel_v2"
    assert d2.NOTEBOOK == "genizah_novel_v2_lambda.ipynb" and d2.EXTRAS_DRAW == "novel_v2/extras"
    assert (d2.WARM_CKPT_REPO, d2.WARM_REVISION, d2.WARM_STEP) == (
        "isaacmg/qwen3-vl-8b-hebrew-novel-v1-ckpt", "944377c9d2626805202ad11679257b928d77d910", 326)
    assert d2.warm_start_pinned() and not d2.warm_start_pinned(d2.WARM_PLACEHOLDER)
    for name in ("LR", "WARMUP_RATIO", "SEED", "ROWS_PER_STEP", "SAVE_STEPS", "EVAL_EVERY", "AUTO_PASSES", "MIN_PASSES",
                 "MAX_PASSES", "MILESTONE_GRAIN", "MILESTONE_MIN", "MILESTONE_PARTS", "LOSS_ALARM_REFERENCE",
                 "LOSS_ALARM_FACTOR", "SCRIPT_SHARE_TOLERANCE", "MEASURED_S_PER_STEP", "BUDGET_FACTOR", "PLACEHOLDER"):
        assert getattr(d2, name) == getattr(v1, name), name
    assert (d2.LR, d2.WARMUP_RATIO, d2.SEED, d2.ROWS_PER_STEP, d2.SAVE_STEPS, d2.EVAL_EVERY) == ("5e-5", "0.03", 3407, 8, 50, 100)


@pytest.mark.parametrize("pages, rows, steps, milestones, warmup", [
    (EXPORT_PAGES, 3824, 478, (150, 250, 400, 478), 15),
    (EXPORT_PAGES_LESS_RNL, 3712, 464, (150, 250, 350, 464), 14)])
def test_pass_plan_sizes_of_the_export(pages: int, rows: int, steps: int, milestones: Tuple[int, ...], warmup: int) -> None:
    """2.2 passes in whole steps over the one label source: every page 2 or 3 times, the extras drawn as novel_v2's."""
    plan = {LABEL: pages}
    assert d2.auto_rows(plan) == d2.resolve_rows("auto", plan) == rows and d2.max_steps_for(rows) == steps
    assert d2.step_rows(plan) == (-(-2 * pages // 8) * 8, (2 * pages + pages // 2) // 8 * 8)
    assert d2.milestone_steps(steps) == milestones and d2.eval_steps(steps) == milestones[1:] and d2.warmup_steps(rows) == warmup
    stems = {LABEL: [f"p{i:04d}" for i in range(pages)]}
    counts = d2.page_counts(stems, rows)
    assert sum(counts.values()) == rows and set(counts.values()) == {2, 3} and Counter(counts.values())[3] == rows - 2 * pages
    draw = bvm.component_rng(d2.SEED, "novel_v2/extras").permutation(pages)[:rows - 2 * pages]
    assert {stems[LABEL][k] for k in draw} == {s for s, n in counts.items() if n == 3}
    assert counts != v1.page_counts(stems, rows)                       # novel_v1's draw is its own


# ----------------------------------------------------------------------------- re-binding


def test_novel_v1_functions_read_novel_v2_names_and_novel_v1_is_untouched() -> None:
    """Every novel_v1 function is re-bound to this module; defaults naming novel_v1 are swapped; novel_v1 keeps its own."""
    assert len(d2.REBOUND) > 40 and {"pass_plan", "prepare_rows", "derive", "notebook_problems", "main"} <= set(d2.REBOUND)
    swapped = {repr(getattr(v1, n)) for n in d2.OVERRIDES if hasattr(v1, n) and isinstance(getattr(v1, n), (str, Path))
               and getattr(v1, n) != getattr(d2, n)}
    for name, obj in vars(d2).items():
        if callable(obj) and getattr(obj, "__module__", "") == d2.__name__ and hasattr(obj, "__globals__"):
            assert obj.__globals__ is vars(d2), name
            for default in (obj.__defaults__ or ()):
                assert repr(default) not in swapped, (name, default)
    assert d2.to_lambda.__defaults__ == (d2.NOTEBOOK,)
    assert v1.ROUND == "novel_v1" and v1.SECTION == "novel_page" and v1.pass_plan.__globals__ is vars(v1)
    assert v1.NOTEBOOK == "genizah_novel_v1_lambda.ipynb" and v1.to_lambda.__defaults__ == ("genizah_novel_v1_lambda.ipynb",)


# ----------------------------------------------------------------------------- the dataset's conventions


def _synthetic_export(root: Path, section: str = d2.SECTION, layout: str = "build", splits: Tuple[str, str] = ("train_page", "val"),
                      tier: str = "novel_strict") -> Path:
    """A tiny dataset in this round's shape: its rows, its ``_build`` manifest and stats, and its images-once export.

    :param root: Folder.
    :param section: Section of every row.
    :param layout: ``build`` (manifest in ``<dataset>_build``) or ``dataset`` (beside the saved dataset).
    :param splits: The manifest's (train, val) split names.
    :param tier: Novelty tier of every record.
    :return: The export.
    """
    plan = [(LEDGER[0], HEBREW, 9, "train_page"), (LEDGER[1], HEBREW, 4, "train_page"), (LEDGER[0], ARABIC, 3, "train_page"),
            (LEDGER[0], HEBREW, 2, "val")]
    rows: Dict[str, List[Dict[str, Any]]] = {"train_page": [], "val": []}
    manifest = []
    (root / "src").mkdir(parents=True)
    for n, (source, text, split) in enumerate([(s, t, sp) for s, t, c, sp in plan for _ in range(c)], start=1):
        stem = RNL_STEM if n == 1 else f"jaar_lse_t_{n:03d}_r"
        path = root / "src" / f"{n}.jpg"
        Image.fromarray(np.random.default_rng(n).integers(0, 255, (100, 82, 3), dtype=np.uint8)).resize((820, 1000)).save(path, "JPEG", quality=60)
        answer = f"{text}\n{n}"
        rows[split].append({"image": str(path), "question": SCRIPT_NEUTRAL_PROMPT, "answer": answer, "task": d2.TASK, "section": section,
                            "stem": stem, "label_source": d2.BUILD.LABEL_SOURCES[source], "target_chars": len(answer),
                            "target_tokens": 0, "image_width": 820, "image_height": 1000})
        manifest.append({"stem": stem, "split": splits[0] if split == "train_page" else splits[1], "row_id": f"t:{n}",
                         "source": source, "document_group": f"canonical_id:G{n}",
                         "canonical_ids": ["StPetersburg_NLR_Test_1" if n == 1 else f"Test_Collection_{n}"], "pgpids": [],
                         "shelfmark": f"Test Coll. {n}", "rnl": n == 1, "image_sha256": f"{n:064x}", "tier": tier,
                         "script": measure_script(answer), "image_path": str(path)})
    stats = {"rows": len(manifest), "rows_by_split": {k: len(v) for k, v in rows.items()},
             "rows_by_script": dict(Counter(m["script"] for m in manifest)), "rows_by_tier": {tier: len(manifest)},
             "rows_by_source": dict(Counter(m["source"] for m in manifest)), "rnl_rows": 1, "built": "2026-11-19T00:00:00Z"}
    dataset = root / "genizah_novel_v2"
    save_dataset(rows, dataset, {"README.md": "# test\n", "stats.json": json.dumps(stats), "PRIVATE.md": "private\n"})
    where = dataset if layout == "dataset" else root / "genizah_novel_v2_build"
    where.mkdir(exist_ok=True)
    (where / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
    export_images_once(dataset, root / "genizah_novel_v2_images_once")
    return root / "genizah_novel_v2_images_once"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    """(export, rows folder) of a tiny dataset of this round's shape, its one Russian National Library row left out."""
    root = tmp_path_factory.mktemp("novel_v2")
    export = _synthetic_export(root)
    out = root / "train_out"
    d2.build_rows(export, out, workers=2)
    return export, out


def test_rows_step_handles_the_section_label_source_manifest_and_stats(synthetic: Tuple[Path, Path]) -> None:
    """Section ja_ar_page and label_source legacy_store_edition pass as they are; the _build manifest, its train_page
    split and its tiers are read; the stats on the Hub copy carry no ledger source, date or holder count."""
    export, out = synthetic
    mixture = json.loads((out / "mixture.json").read_text(encoding="utf-8"))
    assert mixture["name"] == d2.LOCAL_LEAF
    config = mixture["config"]
    assert config["section"] == "ja_ar_page" and config["task"] == "fragment_transcribe" and config["dataset"] == "genizah_novel_v2"
    assert config["pages_by_source"] == {LABEL: 15} and config["rnl_dropped"] == {"train": 1, "val": 0} and config["val_rows"] == 2
    assert config["train_rows"] == d2.auto_rows({LABEL: 15}) and config["warm_start"].startswith(d2.WARM_CKPT_REPO + "@944377c9")
    assert "novel_v2/extras" in config["rule"]
    train = pq.read_table(out / "rows" / "train.parquet").to_pylist()
    assert {(r["section"], r["label_source"], r["source"]) for r in train} == {("ja_ar_page", LABEL, LABEL)}
    assert RNL_STEM not in {r["stem"] for r in train}
    counts = Counter(r["stem"] for r in train)
    assert dict(counts) == d2.page_counts({LABEL: sorted(counts)}, len(train))
    documents = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(d["split"], d["tier"], d["verdict"], d["label_source"]) for d in documents} == {
        ("train", "novel_strict", "novel", LABEL), ("val", "novel_strict", "novel", LABEL)}
    record = json.loads((out / "build_record.json").read_text(encoding="utf-8"))
    assert record["manifest_path"].endswith("genizah_novel_v2_build/manifest.jsonl")
    assert [(d["stem"], d["split"]) for d in record["rnl_dropped"]] == [(RNL_STEM, "train")]
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["rows_by_source"] == {LABEL: 18} and "built" not in stats and "rnl_rows" not in stats
    public = "".join((out / f).read_text(encoding="utf-8") for f in d2.SIDECARS if (out / f).is_file()).lower()
    assert not [w for w in d2.FORBIDDEN_CARD if w in public] and not [s for s in LEDGER if s in public]
    assert "**PRIVATE." in (out / "README.md").read_text(encoding="utf-8") and "derive_novel_v2.py rows" in public


def test_distinct_private_training_export_name(synthetic: Tuple[Path, Path], tmp_path: Path) -> None:
    """The private build can give its mixture and card the same fresh trainer identity.

    :param synthetic: Synthetic export and original trainer rows.
    :param tmp_path: Folder for the new trainer export.
    """
    export, original = synthetic
    manifest = Path(json.loads((original / "build_record.json").read_text())["manifest_path"])
    fresh = tmp_path / "genizah_novel_v2_redo_test_train"
    plan = d2.prepare_rows(export, manifest_path=manifest)
    d2.write_rows(plan, export, fresh, manifest, d2.AUTO, d2.SEED, workers=1, mixture_name=fresh.name)
    assert json.loads((fresh / "mixture.json").read_text())["name"] == fresh.name
    assert (fresh / "README.md").read_text().startswith(f"# {fresh.name} (")


def test_rows_step_refuses_novel_v1_conventions_it_does_not_know(tmp_path: Path) -> None:
    """A novel_v1 section, an unknown manifest split, a non-novel tier and a mislabelled row are refused or flagged."""
    with pytest.raises(ValueError, match="rows other than novel page transcriptions"):
        d2.prepare_rows(_synthetic_export(tmp_path / "a", section=v1.SECTION))
    with pytest.raises(ValueError, match="splits"):
        d2.prepare_rows(_synthetic_export(tmp_path / "b", splits=("training", "val")))
    plan = d2.prepare_rows(_synthetic_export(tmp_path / "c", tier="novel_page_level", layout="dataset"))
    assert {d["verdict"] for d in plan.documents} == {"novel_page_level"}       # not novel: the round's gate 8 refuses it
    export = tmp_path / "b" / "genizah_novel_v2_images_once"
    manifest = [json.loads(l) for l in d2.dataset_manifest_path(export).read_text(encoding="utf-8").splitlines()]
    other = next(k for k, v in d2.BUILD.LABEL_SOURCES.items() if v != LABEL)
    bad = tmp_path / "manifest.jsonl"
    bad.write_text("".join(json.dumps(dict(m, split=m["split"].replace("training", "train_page"),
                                           source=other if i == 2 else m["source"])) + "\n" for i, m in enumerate(manifest)))
    with pytest.raises(ValueError, match="another label_source"):
        d2.prepare_rows(export, manifest_path=bad)
    kept = d2.prepare_rows(_synthetic_export(tmp_path / "d"), keep_rnl=True)
    assert kept.record["train_pages"] == 16 and kept.record["rnl_dropped"] == {"train": 0, "val": 0}


def test_manifest_location_and_split_names(tmp_path: Path) -> None:
    """The manifest is found beside the dataset or in its _build folder; splits map to train / val, the original kept."""
    export = _synthetic_export(tmp_path / "x", layout="build")
    path = d2.dataset_manifest_path(export)
    assert path.parent.name == "genizah_novel_v2_build"
    records = d2.read_manifest(path)
    assert {(r["split"], r["manifest_split"]) for r in records.values()} == {("train", "train_page"), ("val", "val")}
    path.unlink()
    with pytest.raises(FileNotFoundError, match="--manifest"):
        d2.dataset_manifest_path(export)


def test_hub_stats_sum_ledger_sources_under_their_label_source() -> None:
    """Per-source counts go under the label source; date and holder count go; an unknown source is refused."""
    stats = {"rows": 5, "rows_by_source": {LEDGER[0]: 3, LEDGER[1]: 2}, "letters_by_source": {LEDGER[0]: 30, LEDGER[1]: 20},
             "built": "x", "rnl_rows": 1, "rows_by_script": {"hebrew": 5}}
    assert d2.hub_stats(stats) == {"rows": 5, "rows_by_source": {LABEL: 5}, "letters_by_source": {LABEL: 50},
                                   "rows_by_script": {"hebrew": 5}}
    with pytest.raises(ValueError, match="without a label source"):
        d2.hub_stats({"rows_by_source": {"unknown_source": 1}})


# ----------------------------------------------------------------------------- the notebook


@pytest.fixture(scope="module")
def editions() -> Tuple[dict, dict]:
    """(novel_v1 Lambda edition, novel_v2 Lambda edition) at the export's plan, pinned to a fake revision."""
    rows = d2.auto_rows({LABEL: EXPORT_PAGES})
    return v1.lambda_edition(SHA, rows, EXPORT_PAGES), d2.lambda_edition(SHA, rows, EXPORT_PAGES)


def test_only_the_planned_cells_differ_from_novel_v1(editions: Tuple[dict, dict]) -> None:
    """Install, collator, grad gate, bench and helpers are novel_v1's; names, warm start and texts change."""
    old, new = editions
    assert len(new["cells"]) == len(old["cells"]) == 11 and new["metadata"] == old["metadata"]
    assert [i for i in range(11) if old["cells"][i] != new["cells"][i]] == [0, 2, 3, 8, 9, 10]
    text = "\n".join(_src(new, i) for i in range(len(new["cells"])))
    for gone in ("isaacmg/genizah_novel_v1", "genizah_novel_v1\"", "novel-v1-merged", "outputs_novel-v1", "v25a-ckpt", "push_novel_v1.sh",
                 '"novel_page"', "on top of v2.5a step"):
        assert gone not in text, gone
    for name in (d2.DATA_REPO, d2.CKPT_REPO, d2.MERGED_REPO, d2.RUN_NAME, "outputs_novel-v2", '{"ja_ar_page"}', d2.LR_COMMENT):
        assert name in text, name
    assert d2.lambda_edition(SHA, 3824, EXPORT_PAGES) == new and d2.derive(copy.deepcopy(json.load(open(d2.SRC_NOTEBOOK))), SHA, 3824,
                                                                           EXPORT_PAGES) == d2.derive(json.load(open(d2.SRC_NOTEBOOK)), SHA, 3824, EXPORT_PAGES)
    plan = d2.notebook_plan(new)
    assert (plan["train_rows"], plan["pages"], plan["max_steps"], plan["milestones"]) == (3824, 1739, 478, [150, 250, 400, 478])
    assert d2.notebook_problems(new) == []


def test_warm_start_cell_pins_novel_v1_step_326(editions: Tuple[dict, dict], tmp_path: Path) -> None:
    """The model cell pins the novel-v1 checkpoint by commit and checks its trainer state is step 326."""
    c3 = _src(editions[1], 3)
    assert f'WARM_CKPT_REPO = "{d2.WARM_CKPT_REPO}"' in c3 and f'WARM_REVISION = "{d2.WARM_REVISION}"' in c3 and "\nWARM_STEP = 326\n" in c3
    assert d2.NEW_WARM_DOWNLOAD in c3 and 'MERGER_MODE = "frozen"' in c3
    (tmp_path / "last-checkpoint").mkdir()
    check = d2.NEW_WARM_DOWNLOAD.split("\n", 2)[2]
    ns = {"json": json, "local": str(tmp_path), "WARM_STEP": 326, "WARM_CKPT_REPO": "r", "WARM_REVISION": "944377c9"}
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 326}))
    exec(compile(check, "<warm step>", "exec"), dict(ns))
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 300}))
    with pytest.raises(AssertionError, match="is step 300, expected 326"):
        exec(compile(check, "<warm step>", "exec"), dict(ns))


def test_placeholders_refuse_to_run(editions: Tuple[dict, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    """An unpinned dataset stops the data cell (naming the user-run push); an unpinned warm start is reported and stopped."""
    lam = d2.lambda_edition(d2.PLACEHOLDER, 3824, EXPORT_PAGES)
    c2 = _src(lam, 2)
    block = c2[c2.index("V22_REVISION = "):c2.index("LOCAL = ")]
    with pytest.raises(RuntimeError, match="not a pinned 40-hex sha: run the user-run push"):
        exec(compile(block, "<pin guard>", "exec"), {})
    exec(compile(block.replace(d2.PLACEHOLDER, SHA, 1), "<pin guard>", "exec"), {})
    assert d2.notebook_problems(lam) == [f"dataset revision {d2.PLACEHOLDER!r} is not pinned: run {d2.PUSH_SCRIPT}"]
    monkeypatch.setattr(d2, "WARM_REVISION", d2.WARM_PLACEHOLDER)
    assert not d2.warm_start_pinned()
    unpinned = d2.lambda_edition(SHA, 3824, EXPORT_PAGES)
    assert any("warm start" in p and "not pinned" in p for p in d2.notebook_problems(unpinned))
    c3 = _src(unpinned, 3)
    guard = next(l for l in c3.splitlines() if l.strip().startswith("assert len(WARM_REVISION) == 40"))
    with pytest.raises(AssertionError, match="invalid revision"):
        exec(compile(guard.strip(), "<warm guard>", "exec"), {"WARM_REVISION": d2.WARM_PLACEHOLDER})


class _Fake:
    """A dataset with ``column`` and ``len``, as the notebook's dataset gives them."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        """Hold the rows.

        :param rows: Rows.
        """
        self.rows = rows

    def __len__(self) -> int:
        """Number of rows.

        :return: Rows.
        """
        return len(self.rows)

    def column(self, name: str) -> List[Any]:
        """One column.

        :param name: Column.
        :return: Values.
        """
        return [r[name] for r in self.rows]


def test_data_cell_gates_accept_ja_ar_pages_and_refuse_novel_v1_rows(capsys: pytest.CaptureFixture) -> None:
    """The data cell's plan, pass and hygiene gates run on this round's rows and refuse a novel_page or prompted row."""
    stems = [f"s{i}" for i in range(8)]
    counts = d2.page_counts({LABEL: stems}, 16)
    answers = {s: ARABIC if s == "s7" else HEBREW for s in stems}
    train = [{"stem": s, "source": LABEL, "task": d2.TASK, "section": d2.SECTION, "question": SCRIPT_NEUTRAL_PROMPT,
              "answer": answers[s]} for s, n in counts.items() for _ in range(n)]
    val = [dict(train[0], stem="v0")]
    mix = {"config": {"train_rows": 16, "train_pages": 8, "val_rows": 1, "pages_by_source": {LABEL: 8}, "rows_by_source": {LABEL: 16}}}
    c2 = _src(d2.lambda_edition(SHA, 16, 8), 2)
    tail = c2[c2.index("# novel_v2 plan gate"):]

    def run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the gates.

        :param rows: Train rows.
        :return: Namespace.
        """
        ns = {"train_ds": _Fake(rows), "eval_ds": _Fake(val), "MIX": mix, "Counter": Counter, "json": json}
        exec(compile(tail, "<data gates>", "exec"), ns)
        return ns

    assert run(train)["_scripts"]["arabic"] > 0 and "hygiene OK" in capsys.readouterr().out
    for rows, message in (([dict(train[0], section=v1.SECTION)] + train[1:], "novel page"),
                          ([dict(train[0], question=FRAGMENT_TRANSCRIBE_PROMPT)] + train[1:], "script-neutral")):
        with pytest.raises(AssertionError, match=message):
            run(rows)


# ----------------------------------------------------------------------------- gates, pre-push, staging, boot disk


@pytest.mark.skipif(not d2.V1_GATES.is_file(), reason="novel_v1's gates runner is git-ignored and not on this machine")
def test_gates_prepush_and_stage_on_the_synthetic_rows(synthetic: Tuple[Path, Path], tmp_path: Path,
                                                       capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """novel_v1's gates re-bound to this round pass the rows and write novel_v2's marker; the push's checks pass; the
    staging holds the row files, the sidecars and the images as tars only; small images and changed rows are refused."""
    import shutil
    export, built = synthetic
    out = tmp_path / "rows"
    shutil.copytree(built, out)
    empty = lambda cairo_dir=None: d2.load_v1_gates().d5.HeldOut()
    marker = d2.run_gates(out, export=export, tolerance=0.2, workers=2, held_out=empty)
    text = capsys.readouterr().out
    assert "ALL GATES PASSED" in text and "hygiene OK" in text and (out / "notebook_gates_novel_v2.json").is_file()
    assert marker["gates"] == d2.GATES_VERSION and marker["train_rows"] == json.loads((out / "mixture.json").read_text())["config"]["train_rows"]
    with pytest.raises(SystemExit):
        d2.run_gates(out, export=export, tolerance=0.2, workers=2, held_out=empty, min_long_side=2000)
    assert "gate 7 images FAILED" in capsys.readouterr().out
    assert d2.prepush_problems(out) == []
    with pytest.raises(SystemExit, match="no release audit"):
        d2.stage_for_hub(out, tmp_path / "stage")
    ra = d2.release_audit
    binding = {"export_files": ra.file_digests(ra.export_row_files(out)),
               "image_inventory": ra.image_inventory(out, ra.load_export(out))}
    monkeypatch.setattr(d2, "require_released", lambda src: binding)
    staged = d2.stage_for_hub(out, tmp_path / "stage", limit=4000)
    assert staged["files"][:2] == ["rows/train.parquet", "rows/val.parquet"] and "documents.jsonl" not in staged["files"]
    assert len(staged["archives"]) > 1 and not (tmp_path / "stage" / "images").exists()
    names = [n for a in staged["archives"] for n in tarfile.open(tmp_path / "stage" / a).getnames()]
    assert sorted(names) == sorted(f"images/{p.name}" for p in (out / "images").iterdir())
    assert staged["archive_sha256"] == {a: ra.file_sha256(tmp_path / "stage" / a) for a in staged["archives"]}
    archive_path = tmp_path / "stage" / staged["archives"][0]
    with tarfile.open(archive_path) as archive:
        entry = archive.next()
    with tarfile.open(archive_path, "w") as archive:
        archive.addfile(entry, io.BytesIO(bytes(entry.size)))
    with pytest.raises(ValueError, match="changed archived image"):
        ra.verify_staged_export(tmp_path / "stage", staged["archives"], binding)
    with pytest.raises(FileExistsError):
        d2.stage_for_hub(out, tmp_path / "stage")
    (out / "mixture.json").write_text((out / "mixture.json").read_text() + " ")
    assert any("mixture.json changed" in p for p in d2.prepush_problems(out))


def test_prepush_needs_the_gates_marker(synthetic: Tuple[Path, Path]) -> None:
    """Rows the gates have not passed are refused before anything is staged."""
    _, built = synthetic
    assert any(d2.MARKER in p for p in d2.prepush_problems(built))


def test_boot_disk_outputs_are_refused(tmp_path: Path) -> None:
    """Rows and staging folders on the home folder's volume are refused; another device is accepted."""
    assert d2.on_boot_disk(tmp_path / "new" / "deeper") == d2.on_boot_disk(Path.home())
    assert d2.on_boot_disk(Path.home() / "never_made_folder")
    with pytest.raises(SystemExit, match="boot disk"):
        d2.refuse_boot_disk(Path.home() / "never_made_folder", "training rows")
    with pytest.raises(SystemExit, match="boot disk"):
        d2.main(["rows", "--export", str(tmp_path), "--out", str(Path.home() / "never_made_folder")])
    assert not (Path.home() / "never_made_folder").exists()


def test_cli_notebook_and_show(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The notebook subcommand writes novel_v2's Lambda edition; show reads its plan; an unpinned one is refused."""
    plan = tmp_path / "mixture.json"
    plan.write_text(json.dumps({"config": {"train_rows": 3824, "train_pages": 1739}}))
    d2.main(["notebook", "--revision", SHA, "--plan", str(plan), "--out-dir", str(tmp_path)])
    assert json.loads((tmp_path / d2.NOTEBOOK).read_text(encoding="utf-8")) == d2.lambda_edition(SHA, 3824, 1739)
    capsys.readouterr()
    d2.main(["show", "--notebook", str(tmp_path / d2.NOTEBOOK), "--field", "eval_steps", "--require-pinned"])
    assert capsys.readouterr().out.split() == ["250", "400", "478"]
    d2.main(["notebook", "--revision", d2.PLACEHOLDER, "--pages", str(EXPORT_PAGES), "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not pinned"):
        d2.main(["show", "--notebook", str(tmp_path / d2.NOTEBOOK), "--require-pinned"])


def test_no_internal_names_in_the_tracked_files(editions: Tuple[dict, dict]) -> None:
    """The module, this test file and the notebook carry no restricted attribution, storage path or token."""
    for path in (Path(d2.__file__), Path(__file__)):
        text = path.read_text(encoding="utf-8").lower()
        assert not [w for w in TRACKED_FORBIDDEN if w in text], path
    text = json.dumps(editions[1], ensure_ascii=False).lower()
    assert not [w for w in TRACKED_FORBIDDEN if w in text] and not re.search(r"hf_[a-z0-9]{20,}", text)
