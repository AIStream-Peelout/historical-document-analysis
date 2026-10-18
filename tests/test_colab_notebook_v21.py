# File name: test_colab_notebook_v21.py
# Date: 9/10/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""CPU-only validation of the v2.1 genizah Colab notebook.

v2.1 = v2.0a + the direct-grounding data lever (genizah_ktiv_v3, six new
families, grounding share 20%), warm-started from v2.0a step 1800, merger
FROZEN. The train families are STREAMED per hub split because the dataset is
~70 GB. These tests pin that contract:

- every code cell parses; names are defined before use across cells
- the audited install triplet is unchanged
- genizah_ktiv_v3 pin present (40-hex once pushed; a loud placeholder before),
  control pins (genizah_clean_v2, synthetic_hebrew_v3) identical to v2.0a
- per-family streaming (`streaming=True`), small sources converted with
  `to_iterable_dataset`, `all_exhausted` interleave, no direct indexing of a
  streamed dataset, hygiene samples via `.take(`
- ten grounding families with weights summing to 1, GROUNDING_SHARE 0.20,
  mixture probabilities asserted to sum to 1
- warm start pinned to v2.0a-1800 (af9df6a0…), MERGER_MODE frozen
- v21b repo/run names everywhere (v21-ckpt keeps the image-blind run as evidence); no v20a/v20b
  identifier leakage in code
- eval set covers the new families
- streamed train set ⇒ accelerator_config={"dispatch_batches": False} (Accelerate's default
  DataLoaderDispatcher truncated Qwen3-VL's packed pixel_values to one patch in run js3ku3tw) and a
  gate on the PREPARED train dataloader before trainer.train()
"""
import ast
import json
import re
from pathlib import Path
from typing import List

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/genizah_v21b_dispatchfix.ipynb"
V20A_WARM_START_SHA = "af9df6a0ad4743bc8493a7cf0c14bc3b4b796dd8"
GENIZAH_SHA = "57366ad378946918731ad0012d699acc7d9ed31c"
SYNTH3_SHA = "59abcf7c30fb6753b9df89f0a099b07a68059013"
PLACEHOLDER = "PIN-AFTER-PUSH"
GROUNDING_FAMILIES = {"locate", "read_box", "layout_qa", "grounded_page", "locate_word",
                      "read_box_word", "line_index", "line_of_phrase", "grounded_detect",
                      "grounded_crop"}


def _cell_source(cell: dict) -> str:
    """Return a cell's source with notebook magics stripped.

    :param cell: Raw notebook cell dict.
    :returns: Compilable Python source.
    """
    src = "".join(cell["source"])
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith(("%", "!")))


@pytest.fixture(scope="module")
def code_cells() -> List[str]:
    """All code-cell sources, magics stripped."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    return [_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def raw_cells() -> List[str]:
    """All code-cell sources including magics."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _find_cell(cells: List[str], needle: str) -> str:
    """First cell containing ``needle``."""
    for c in cells:
        if needle in c:
            return c
    raise AssertionError(f"no cell contains {needle!r}")


def _assigned(cells: List[str], name: str) -> str:
    """The single string literal assigned to ``name`` across cells."""
    vals = []
    for c in cells:
        for node in ast.walk(ast.parse(c)):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    vals.append(node.value.value)
    assert len(vals) == 1, f"{name} assigned {len(vals)} times"
    return vals[0]


def test_every_cell_parses(code_cells: List[str]) -> None:
    for i, src in enumerate(code_cells):
        ast.parse(src)


def test_cross_cell_name_order(code_cells: List[str]) -> None:
    """Every name a cell uses must be defined by that cell or an earlier one."""
    import builtins
    defined = set(dir(builtins)) | {"userdata", "torch", "np"}
    for i, src in enumerate(code_cells):
        tree = ast.parse(src)
        used, local = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    local.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                local.add(node.name)
                local.update(a.arg for a in node.args.args + node.args.kwonlyargs)
                if node.args.vararg: local.add(node.args.vararg.arg)
                if node.args.kwarg: local.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
            elif isinstance(node, ast.arg):
                local.add(node.arg)
            elif isinstance(node, (ast.For, ast.comprehension)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name): local.add(n.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, ast.Lambda):
                local.update(a.arg for a in node.args.args)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
        missing = {u for u in used if u not in defined and u not in local}
        assert not missing, f"cell {i + 1} uses undefined names: {sorted(missing)}"
        defined |= local


def test_install_triplet_pinned(raw_cells: List[str]) -> None:
    install = raw_cells[0]
    assert '"unsloth[colab-new]==2026.8.9"' in install
    assert '"unsloth_zoo==2026.8.6"' in install
    assert '"transformers==4.57.6"' in install


def test_data_pins(code_cells: List[str]) -> None:
    assert _assigned(code_cells, "GENIZAH_REVISION") == GENIZAH_SHA
    assert _assigned(code_cells, "SYNTH3_REVISION") == SYNTH3_SHA
    assert _assigned(code_cells, "KTIV3_REPO") == "isaacmg/genizah_ktiv_v3"
    sha = _assigned(code_cells, "KTIV3_REVISION")
    assert re.fullmatch(r"[0-9a-f]{40}", sha) or sha == PLACEHOLDER, f"bad KTIV3 pin {sha!r}"
    cell = _find_cell(code_cells, "KTIV3_REVISION =")
    assert "assert len(_sha) == 40" in cell, "runtime pin gate missing (the placeholder must fail loudly)"
    assert "genizah_ktiv_v2" not in cell and "KTIV2" not in cell, "v2 dataset leaked into v2.1"


def test_ktiv3_pin_filled() -> None:
    """Fails-safe reminder: the notebook is not runnable until the push SHA is pinned."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    code = "\n".join(_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    if PLACEHOLDER in code:
        pytest.skip("KTIV3_REVISION still PIN-AFTER-PUSH — fill it from push_ktiv_dataset.py output")


def test_streaming_contract(code_cells: List[str]) -> None:
    data = _find_cell(code_cells, "def stream(")
    assert "streaming=True" in data
    assert 'split=f"train_{name}"' in data, "per-family hub splits expected"
    assert "to_iterable_dataset(" in data, "small map-style sources must become iterables"
    assert "load_dataset_builder(" in data and "num_examples" in data, "row counts must come from split metadata"
    assert ".take(" in data, "hygiene samples must use take() on streamed sources"
    assert re.search(r"ktiv_pages\[\d", "\n".join(code_cells)) is None, "direct indexing of a streamed dataset"
    mix = _find_cell(code_cells, "interleave_datasets(")
    assert 'stopping_strategy="all_exhausted"' in mix
    assert "as_stream(genizah)" in mix and "as_stream(synth3)" in mix


def test_grounding_families_and_mixture(code_cells: List[str]) -> None:
    data = _find_cell(code_cells, "GROUNDING_V21 =")
    fam = set()
    for node in ast.walk(ast.parse(data)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("GROUNDING_V20", "GROUNDING_V21") for t in node.targets):
            fam |= {e.value for e in node.value.elts}
    assert fam == GROUNDING_FAMILIES
    mix = _find_cell(code_cells, "GROUNDING_WEIGHTS =")
    weights = None
    for node in ast.walk(ast.parse(mix)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GROUNDING_WEIGHTS" for t in node.targets):
            weights = ast.literal_eval(node.value)
    assert weights and set(weights) == GROUNDING_FAMILIES and abs(sum(weights.values()) - 1.0) < 1e-9
    assert "GROUNDING_SHARE = 0.20" in mix
    assert "assert abs(sum(_probs) - 1.0) < 1e-9" in mix
    for fam_name in ("locate_word", "line_index", "grounded_detect", "grounded_crop"):
        assert f'_val("{fam_name}"' in mix, f"eval set lacks {fam_name}"


def test_warm_start_and_knob(code_cells: List[str]) -> None:
    assert _assigned(code_cells, "WARM_REVISION") == V20A_WARM_START_SHA
    assert _assigned(code_cells, "WARM_CKPT_REPO") == "isaacmg/qwen3-vl-8b-hebrew-v20a-ckpt"
    cell = _find_cell(code_cells, "MERGER_MODE =")
    assert 'MERGER_MODE = "frozen"' in cell, "v2.1 keeps the merger frozen (v20b/v19c: merger is not the box lever)"
    assert "last-checkpoint/adapter_model.safetensors" in cell
    assert "assert len(_wv) == 108" in cell, "tower-adapter warm-load gate missing"


def test_v21_names_no_leakage(code_cells: List[str]) -> None:
    assert _assigned(code_cells, "CKPT_REPO") == "isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt", \
        "v21-ckpt holds the image-blind run js3ku3tw; a fresh run needs a fresh destination"
    assert _assigned(code_cells, "OUT_DIR") == "outputs_v21b"
    assert _assigned(code_cells, "MERGED_REPO") == "isaacmg/qwen3-vl-8b-hebrew-v21-merged"
    assert 'run_name="genizah_v21b"' in _find_cell(code_cells, "run_name=")
    for c in code_cells:
        for line in c.splitlines():
            if "WARM_CKPT_REPO" in line or "v20a/b/c" in line or "v2.0a" in line:
                continue
            assert "v20a" not in line and "v20b" not in line and "v20c" not in line, line


def test_schedule_and_trainer_flags_identical_to_v20a(code_cells: List[str]) -> None:
    cell = _find_cell(code_cells, "make_trainer(")
    for flag in ("max_steps=2000", "learning_rate=5e-5", 'lr_scheduler_type="cosine"',
                 "gradient_accumulation_steps=8", 'hub_strategy="checkpoint"',
                 'dataset_kwargs={"skip_prepare_dataset": True}', "remove_unused_columns=False",
                 "resume_from_checkpoint=resume_dir"):
        assert flag in cell, flag


def test_dispatch_batches_disabled_and_gated(code_cells: List[str]) -> None:
    """The streamed train set must not go through Accelerate's DataLoaderDispatcher, and the notebook
    must prove it on the prepared dataloader (patch rows == grid product) before the first step."""
    cell = _find_cell(code_cells, "make_trainer(")
    assert 'accelerator_config={"dispatch_batches": False}' in cell
    assert "trainer.args.accelerator_config.dispatch_batches is False" in cell, "config must be asserted on the trainer"
    gate, train = cell.index("trainer.get_train_dataloader()"), cell.index("trainer.train(")
    assert gate < train, "dataloader gate must run before trainer.train()"
    body = cell[gate:train]
    assert '"Dispatcher" not in type(_dl).__name__' in body
    assert '_b["pixel_values"].shape[0] == _need' in body and 'image_grid_thw"].prod(dim=-1).sum()' in body
    assert "collator([_row])" in body and "abs(_lp - _ld)" in body, "prepared vs direct-collate loss check missing"
    assert "get_eval_dataloader()" in body, "eval batch must be checked too"


def test_resume_skips_the_stream_instead_of_replaying_the_collator(code_cells: List[str]) -> None:
    """A resumed streamed run must not replay 9,600 collations (Colab kills the silent hours)."""
    src = next(c for c in code_cells if "trainer.train(resume_from_checkpoint=resume_dir)" in c)
    assert "trainer.args.ignore_data_skip = True" in src
    assert "trainer.train_dataset = mixture.skip(_seen)" in src
    assert '_json.load(open(f"{resume_dir}/trainer_state.json"))' in src
    # The skip is derived from the checkpoint, never hard-coded.
    assert '_seen = int(_state["global_step"]) * trainer.args.gradient_accumulation_steps' in src
    assert src.index("trainer.train_dataset = mixture.skip(_seen)") < src.index(
        "trainer.train(resume_from_checkpoint=resume_dir)")
