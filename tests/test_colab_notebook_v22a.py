# File name: test_colab_notebook_v22a.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""CPU-only validation of the v2.2a genizah Colab notebook.

v2.2a = v2.1b warm-started from its step-1200 flagship, trained on ONE materialised,
MAP-STYLE images-once mixture (``isaacmg/genizah_v22_pilot``: KTIV v4 transcription +
grounding, line-broken PGP editions, documentary grounding, extractive PGP QA). These
tests pin that contract:

- every code cell parses; the audited install triplet is unchanged
- the data cell loads the pilot repo through an ``ImagesOnceDataset`` mirror, resolves or
  pins a 40-hex revision, and asserts the QA / editions / documentary shares
- NO streaming: no ``streaming=True``, ``interleave_datasets`` or ``to_iterable_dataset``
- warm start pinned to v2.1b step 1200 (6724c32c…), MERGER_MODE frozen
- v22a repo/run names everywhere; no v21b checkpoint repo reused for pushing
- the prepared-dataloader gate (dispatch_batches False, patch rows vs grid, prepared ==
  direct loss on the recorded row) precedes ``trainer.train``
- map-style resume sets ``ignore_data_skip``; the eval set is the val parquet
"""
import ast
import json
import re
from pathlib import Path
from typing import List

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/genizah_v22a.ipynb"
V21B_STEP1200_SHA = "6724c32cd0f6296223978ae69635fc782d2dbb03"
PLACEHOLDER = "PIN-AFTER-PUSH"


def _cell_source(cell: dict) -> str:
    """Return a cell's source with notebook magics stripped.

    :param cell: Raw notebook cell dict.
    :returns: Compilable Python source.
    """
    src = "".join(cell["source"])
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith(("%", "!")))


@pytest.fixture(scope="module")
def code_cells() -> List[str]:
    """All code-cell sources, magics stripped.

    :returns: One string per code cell, in notebook order.
    """
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return [_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def all_code(code_cells: List[str]) -> str:
    """Every code cell joined.

    :param code_cells: Code-cell sources.
    :returns: Joined source.
    """
    return "\n".join(code_cells)


def test_every_cell_parses(code_cells: List[str]) -> None:
    """Each code cell is valid Python once magics are removed."""
    for i, src in enumerate(code_cells):
        ast.parse(src, filename=f"cell{i}")


def test_install_triplet_unchanged(code_cells: List[str]) -> None:
    """The audited unsloth / unsloth_zoo / transformers pins survive."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    raw = "".join(nb["cells"][1]["source"])
    assert '"unsloth[colab-new]==2026.8.9"' in raw
    assert '"unsloth_zoo==2026.8.6"' in raw and '"transformers==4.57.6"' in raw


def test_data_cell_is_map_style_images_once(code_cells: List[str], all_code: str) -> None:
    """The mixture is loaded map-style from the pilot repo; nothing is streamed."""
    data = code_cells[1]
    assert 'V22_REPO = "isaacmg/genizah_v22_pilot"' in data
    m = re.search(r'V22_REVISION = "([^"]+)"', data)
    assert m and (m.group(1) == PLACEHOLDER or re.fullmatch(r"[0-9a-f]{40}", m.group(1)))
    assert 'repo_type="dataset"' in data and "class ImagesOnceDataset" in data
    assert 'rows/train.parquet' in data and 'rows/val.parquet' in data
    for banned in ("streaming=True", "interleave_datasets", "to_iterable_dataset", ".take("):
        assert banned not in all_code, banned


def test_data_cell_asserts_shares_and_hygiene(code_cells: List[str]) -> None:
    """QA, editions and documentary grounding shares are asserted; hygiene covers all three task kinds."""
    data = code_cells[1]
    for src in ("pgp_qa", "pgp_editions", "documentary_grounding"):
        assert f'("{src}", ' in data
    assert "not stated" in data and '"bbox_2d"' in data and "[...]" in data


def test_warm_start_pinned_to_v21b_step1200(code_cells: List[str]) -> None:
    """Warm start = v2.1b step 1200 with the merger frozen."""
    model = code_cells[2]
    assert 'WARM_CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt"' in model
    assert f'WARM_REVISION = "{V21B_STEP1200_SHA}"' in model
    assert 'MERGER_MODE = "frozen"' in model


def test_v22a_names_everywhere(all_code: str) -> None:
    """Checkpoint, output, run and merged-model names are v22a; v21b's checkpoint repo is only the warm start."""
    assert 'CKPT_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22a-ckpt"' in all_code
    assert 'OUT_DIR = "outputs_v22a"' in all_code and 'run_name="genizah_v22a"' in all_code
    assert 'MERGED_REPO = "isaacmg/qwen3-vl-8b-hebrew-v22a-merged"' in all_code
    assert all_code.count("v21b-ckpt") == 1, "v21b checkpoint repo must appear only as the warm start"
    for stale in ("outputs_v21b", 'run_name="genizah_v21b"', "v20a-ckpt", "ktiv_pages", "synth3", "genizah_val"):
        assert stale not in all_code, stale


def test_collator_guardrail_covers_page_and_qa(code_cells: List[str]) -> None:
    """The collator check batches a KTIV page with a QA row and verifies masking on both."""
    coll = code_cells[3]
    assert "_i_page" in coll and "_i_qa" in coll and "> 40000" in coll
    assert "prompt leaked into the loss" in coll


def test_dataloader_gate_precedes_training(code_cells: List[str]) -> None:
    """The prepared-dataloader gate runs before trainer.train and compares against the recorded row."""
    train = code_cells[6]
    assert 'accelerator_config={"dispatch_batches": False}' in train
    gate = train.index("DATALOADER GATE")
    assert gate < train.index("trainer.train(")
    assert "class _Recording" in train and "_rec.seen[0]" in train
    assert "image_grid_thw" in train and "prepared-dataloader loss" in train
    assert "trainer.train_dataset = train_ds" in train, "the recording wrapper must be removed before training"
    assert "dataloader_num_workers = _saved_workers" in train


def test_map_style_resume_and_eval(code_cells: List[str]) -> None:
    """Resume avoids the batch replay; eval uses the val parquet dataset."""
    train = code_cells[6]
    assert "trainer.args.ignore_data_skip = True" in train
    assert "eval_dataset=eval_ds" in train and "train_dataset=train_ds" in train
    assert "max_steps=2000" in train and 'hub_strategy="checkpoint"' in train
