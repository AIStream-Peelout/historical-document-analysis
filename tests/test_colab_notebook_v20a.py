# File name: test_colab_notebook_v20a.py
# Date: 8/30/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""CPU-only validation of the v2.0a genizah Colab notebook.

Same harness pattern as ``test_colab_notebook_v19b.py``, adapted to the v1.9c
protocol: ONE variable vs v19b = the 8 merger linears train FULL-WEIGHT via
PEFT ``modules_to_save`` (motivated by the r16 saturation spectra and the
2026-08-30 ablation probe: the merger update is potent but rank-starved).

- pinned install triplet and ALL dataset revisions identical to v19a/v19b
- warm start pinned to v18b step-700 (c80313f8…), weights-only; tower adapter
  live (108/108 nonzero); each merger CLONE must START EQUAL to its frozen
  original (identity) so v19c begins from v19a's function
- creation gates: 108 tower LoRA, ZERO merger LoRA (that would be v19b), 8
  trainable full-weight clones, and an un-quantized-merger gate (4-bit clones
  are untrainable)
- cell-5 grad gate per-tensor over 108 tower LoRA AND 8 merger full weights;
  export gated on tower moved + clones DIFFERING from base
- v19c repo/run names everywhere; no v19a/v19b identifier leakage in code
"""

import ast
import builtins
import json
from pathlib import Path
from typing import List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/genizah_v20a.ipynb"

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
V19A_WARM_START_SHA = "43e21bd7a6fedd323879fc5ea2c298df90a92784"
KTIV2_SHA = "1bb20e209a3b3095fc84dad7c928da4310269fd3"
GENIZAH_SHA = "57366ad378946918731ad0012d699acc7d9ed31c"
SYNTH3_SHA = "59abcf7c30fb6753b9df89f0a099b07a68059013"
MIXTURE = "_probs = [0.30, 0.19, 0.15, 0.07, 0.07, 0.09, 0.05, GROUNDING_SHARE]"
N_TOWER = 108
N_MERGER = 8


def _cell_source(cell: dict) -> str:
    """Return a cell's source with notebook magics stripped.

    :param cell: Raw notebook cell dict.
    :returns: Compilable Python source.
    """
    src = "".join(cell["source"])
    return "\n".join(
        l for l in src.split("\n") if not l.strip().startswith(("%", "!"))
    )


@pytest.fixture(scope="module")
def code_cells() -> List[str]:
    """All code-cell sources, magics stripped."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    return [_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def raw_cells() -> List[str]:
    """All code-cell sources INCLUDING magic lines (for install checks)."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _find_cell(code_cells: List[str], needle: str) -> str:
    """Return the first cell containing ``needle``.

    :param code_cells: Cell sources.
    :param needle: Substring to locate.
    :returns: The matching cell's source.
    """
    for src in code_cells:
        if needle in src:
            return src
    raise AssertionError(f"no notebook cell contains {needle!r}")


def _pinned_sha(code_cells: List[str], var_name: str) -> str:
    """Extract a single pinned revision SHA assigned to ``var_name``.

    :param code_cells: Cell sources.
    :param var_name: Variable the SHA must be assigned to.
    :returns: The assigned string.
    """
    cell = _find_cell(code_cells, var_name)
    tree = ast.parse(cell)
    shas = [n.value.value for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == var_name for t in n.targets)]
    assert len(shas) == 1, f"{var_name} assigned {len(shas)} times"
    return shas[0]


def test_every_cell_parses(code_cells: List[str]) -> None:
    """All code cells must be valid Python."""
    for src in code_cells:
        ast.parse(src)


def test_cross_cell_name_order(code_cells: List[str]) -> None:
    """No cell may use a module-level name defined only in a later cell."""
    known_runtime = {"get_ipython", "userdata", "cuda", "__name__"}
    defined = set(dir(builtins)) | known_runtime
    for i, src in enumerate(code_cells):
        tree = ast.parse(src)
        loaded, local = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (local if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    local.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, ast.comprehension):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        local.add(t.id)
            elif isinstance(node, ast.arguments):
                args = (node.args + node.posonlyargs + node.kwonlyargs
                        + ([node.vararg] if node.vararg else [])
                        + ([node.kwarg] if node.kwarg else []))
                for a in args:
                    local.add(a.arg)
        missing = loaded - defined - local
        assert not missing, f"cell {i + 1} uses undefined names: {sorted(missing)}"
        defined |= local


def test_install_triplet_pinned(raw_cells: List[str]) -> None:
    """The install cell must pin the audited triplet and assert versions."""
    install = next(c for c in raw_cells if "%pip install" in c)
    assert '"unsloth[colab-new]==2026.8.9"' in install
    assert '"unsloth_zoo==2026.8.6"' in install
    assert '"transformers==4.57.6"' in install
    assert 'version("unsloth_zoo") == "2026.8.6"' in install
    assert 'version("transformers") == "4.57.6"' in install


def test_v20_data_pins(code_cells: List[str]) -> None:
    """v2.0a pins the mash-fixed KTIV v2 build + the v1.9 control sources."""
    assert _pinned_sha(code_cells, "KTIV2_REVISION") == KTIV2_SHA
    assert _pinned_sha(code_cells, "GENIZAH_REVISION") == GENIZAH_SHA
    assert _pinned_sha(code_cells, "SYNTH3_REVISION") == SYNTH3_SHA
    data_cell = _find_cell(code_cells, "KTIV2_REVISION")
    assert "assert len(_sha) == 40" in data_cell, "runtime pin gate missing"
    assert '"isaacmg/genizah_ktiv_v2"' in data_cell
    assert '"isaacmg/synthetic_hebrew_v3"' in data_cell
    assert '"isaacmg/genizah_ktiv_v1"' not in data_cell, "v1 dataset leaked into v2.0"


def test_grounding_tasks_loaded_and_validated(code_cells: List[str]) -> None:
    """The four grounding families load, and their hygiene gates exist."""
    cell = _find_cell(code_cells, "GROUNDING_TASKS")
    assert '("locate", "read_box", "layout_qa", "grounded_page")' in cell
    assert "bad locate box" in cell, "locate coordinate validation missing"
    assert "grounded_page payload" in cell
    assert '"bbox_2d = [" in q' in cell, "read_box prompt validation missing"
    assert "0 <= v <= 1000" in cell


def test_merger_knob_tri_state(code_cells: List[str]) -> None:
    """MERGER_MODE must default to frozen and drive all three configs."""
    cell = _find_cell(code_cells, "MERGER_MODE")
    assert 'MERGER_MODE = "frozen"' in cell, "default must be the safe v1.9a config"
    assert 'assert MERGER_MODE in ("frozen", "lora", "full")' in cell
    assert 'if MERGER_MODE == "lora" else base_regex' in cell
    assert 'modules_to_save=MERGER_MODULES if MERGER_MODE == "full" else None' in cell
    assert "deepstack_merger_list" in cell
    assert f"len(_tower_lora) == {N_TOWER}" in cell
    for gate in ('assert not _merger_lora and not _merger_full',
                 'len(_merger_lora) == 8 and not _merger_full',
                 'len(_merger_full) == 8 and not _merger_lora'):
        assert gate in cell, f"per-mode creation gate missing: {gate}"
    assert cell.index("len(_merger_full) == 8") < cell.index("set_peft_model_state_dict")


def test_merger_quantization_gate(code_cells: List[str]) -> None:
    """modules_to_save cannot clone 4-bit weights — the gate must precede it."""
    cell = _find_cell(code_cells, "MERGER_MODE")
    assert 'if MERGER_MODE == "full":' in cell
    assert "Linear4bit" in cell, "un-quantized merger gate missing"
    assert "assert not _quantized" in cell
    assert cell.index("assert not _quantized") < cell.index("get_peft_model")


def test_warm_start_flagship_and_knob_compatibility(code_cells: List[str]) -> None:
    """Warm start = v1.9a flagship; knob must match the checkpoint contents."""
    cell = _find_cell(code_cells, "set_peft_model_state_dict")
    assert _pinned_sha(code_cells, "WARM_REVISION") == V19A_WARM_START_SHA
    assert '"isaacmg/qwen3-vl-8b-hebrew-v19a-ckpt"' in cell
    assert f"len(_wv) == {N_TOWER}" in cell, "warm-start tower check missing"
    assert "_loaded_merger_full" in cell and "_loaded_merger_lora" in cell, \
        "knob-vs-checkpoint compatibility guard missing"
    assert "MERGER_MODE == 'full'" in cell.replace('"', "'"), \
        "full-merger warm ckpt must force the knob"
    assert "torch.allclose" in cell, "identity check for fresh clones missing"
    assert "_sd.setdefault(" in cell, \
        "missing-merger-key injection absent — PEFT modules_to_save load KeyErrors"
    trainer_cell = _find_cell(code_cells, "SFTTrainer")
    assert '"isaacmg/qwen3-vl-8b-hebrew-v20a-ckpt"' in trainer_cell
    assert 'run_name="genizah_v20a"' in trainer_cell


def test_no_prior_version_leakage_in_code(code_cells: List[str]) -> None:
    """No v19a/v19b repo/output/run identifiers may survive in code cells."""
    joined = "\n".join(code_cells)
    for bad in ("hebrew-v19b-ckpt", "hebrew-v19b-merged", "outputs_v19a",
                "outputs_v19b", "outputs_v19c", "genizah_focus_v19a",
                "genizah_focus_v19b", "genizah_focus_v19c",
                '"v19b-merged"', '"v19c-merged"'):
        assert bad not in joined, f"prior-version identifier leaked: {bad}"
    # v19a-ckpt (warm start) and v19c-ckpt (documented alternative) are allowed


def test_collator_uses_safe_arguments(code_cells: List[str]) -> None:
    """Every collator construction must avoid the hostile defaults."""
    joined = "\n".join(code_cells)
    n_collators = joined.count("UnslothVisionDataCollator(")
    assert n_collators >= 1
    assert joined.count('resize="max"') == n_collators
    assert joined.count("train_on_responses_only=True") == n_collators


def test_label_hygiene_asserts_present(code_cells: List[str]) -> None:
    """The data cell must refuse leaked gap tokens / mojibake in BOTH sources."""
    cell = _find_cell(code_cells, "internal gap token leaked")
    assert "\\u2423" in cell or "␣" in cell, "internal gap-token guard missing"
    assert '"[...]" in a' in cell, "damage-marker presence guard missing"
    assert "&#$" in cell, "mojibake-symbol guard missing"
    assert '("genizah", genizah), ("ktiv", ktiv_pages)' in cell, \
        "hygiene must cover both real-manuscript sources"


def test_region_rows_keep_restriction(code_cells: List[str]) -> None:
    """KTIV region rows must still carry the 'Transcribe ONLY' instruction."""
    cell = _find_cell(code_cells, "ktiv_regions")
    assert '"Transcribe ONLY" in q' in cell


def test_resolution_policy_defined_and_shipped(code_cells: List[str]) -> None:
    """6.5MP floor must reach the processor AND the exported model."""
    model_cell = _find_cell(code_cells, "MIN_PIX")
    assert "MIN_PIX = 6_500_000" in model_cell
    assert "min_pixels=MIN_PIX" in model_cell and "max_pixels=MAX_PIX" in model_cell
    export_cell = _find_cell(code_cells, "push_to_hub_merged")
    assert "image_processor.save_pretrained" in export_cell


def test_vision_fix_hook(code_cells: List[str]) -> None:
    """The patch-embed fix must be guarded and ordered after the warm start."""
    cell = _find_cell(code_cells, "visual.patch_embed")
    assert 'n.endswith("visual.patch_embed")' in cell
    assert "register_forward_hook" in cell
    assert "requires_grad_(True)" in cell
    assert "torch.is_grad_enabled()" in cell, \
        "inference-mode guard missing — generate() would crash with the hook armed"
    assert cell.index("set_peft_model_state_dict") < cell.index("register_forward_hook")


def test_gradient_flow_gate_covers_tower_and_merger(code_cells: List[str]) -> None:
    """Per-tensor grad gate over 108 tower + the knob's merger family."""
    cell = _find_cell(code_cells, "loss.backward()")
    assert '".visual.blocks." in n' in cell
    assert f"len(tower_g) == {N_TOWER}" in cell and "min(tower_g) > 0" in cell
    assert "mrg_lora_g" in cell and "mrg_full_g" in cell
    assert 'MERGER_MODE == "lora"' in cell and 'MERGER_MODE == "full"' in cell
    assert "not mrg_lora_g and not mrg_full_g" in cell, "frozen-mode guard missing"
    assert "max(lang_g) > 0" in cell
    assert "zero_grad" in cell
    cells = list(code_cells)
    assert cells.index(cell) < cells.index(_find_cell(cells, "SFTTrainer"))


def test_throughput_cell_gated_and_before_training(code_cells: List[str]) -> None:
    """The batch-geometry benchmark defaults OFF and precedes the trainer."""
    cell = _find_cell(code_cells, "RUN_THROUGHPUT_BENCH")
    assert "RUN_THROUGHPUT_BENCH = False" in cell, "must default OFF"
    assert "BATCH_GEOMETRIES" in cell and "(1, 8)" in cell
    assert "reset_peak_memory_stats" in cell and "OutOfMemoryError" in cell
    cells = list(code_cells)
    assert cells.index(cell) < cells.index(_find_cell(cells, "SFTTrainer"))


def test_schedule_and_tripwire(code_cells: List[str]) -> None:
    """v1.9-lineage schedule + the full-merger LR tripwire note."""
    cell = _find_cell(code_cells, "SFTTrainer")
    assert "max_steps=2000" in cell
    assert "learning_rate=5e-5" in cell
    assert "2e-5" in cell, "eval@100 destabilization tripwire note missing"


def test_ckpt_repo_public_from_start(code_cells: List[str]) -> None:
    """v1.9c publishes its checkpoints openly (user decision 2026-08-30)."""
    cell = _find_cell(code_cells, "SFTTrainer")
    assert "hub_private_repo=False" in cell
    assert "hub_private_repo=True" not in cell


def test_export_gated_per_knob(code_cells: List[str]) -> None:
    """Export gates must follow the knob; v20a repos named."""
    cell = _find_cell(code_cells, "push_to_hub_merged")
    assert "if TRAIN_VISION_LORA:" in cell
    assert 'MERGER_MODE == "lora"' in cell and 'MERGER_MODE == "full"' in cell
    assert f"len(_moved) == {N_MERGER}" in cell
    assert '"isaacmg/qwen3-vl-8b-hebrew-v20a-merged"' in cell
    assert cell.index(f"len(_moved) == {N_MERGER}") < cell.index("push_to_hub_merged")


def test_mixture_ktiv_led_with_grounding_share(code_cells: List[str]) -> None:
    """KTIV pages lead; grounding rides the GROUNDING_SHARE knob; sums to 1."""
    cell = _find_cell(code_cells, "interleave_datasets(")
    head = cell.split("interleave_datasets(")[1][:260]
    assert "ktiv_pages" in head.split(",")[0], "ktiv_pages must be first"
    assert "grounding]" in head.replace("\n", "").replace(" ", ""), \
        "grounding dataset missing from the mixture"
    assert "GROUNDING_SHARE = 0.08" in cell
    assert MIXTURE in cell
    assert "abs(sum(_probs) - 1.0) < 1e-9" in cell
    assert 'stopping_strategy="all_exhausted"' in cell
    eval_cell = _find_cell(code_cells, "concatenate_datasets")
    for name in ("ktiv2_val", "genizah_val", "talmud_val", "synth3_eval"):
        assert name in eval_cell, f"eval must cover {name}"
    for task in ('"locate"', '"read_box"', '"layout_qa"'):
        assert task in eval_cell, f"grounding eval slice missing: {task}"


def test_masking_markers_match_real_chat_template(code_cells: List[str]) -> None:
    """The instruction/response markers must split the real rendered template."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    except OSError:
        pytest.skip("Qwen3-VL tokenizer unavailable (offline?)")

    prompt, answer = "Transcribe the text exactly.", "שלום עולם " * 10
    msgs = [
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert INSTRUCTION_PART in rendered and RESPONSE_PART in rendered
    assert rendered.count(RESPONSE_PART) == 1
    i_pos, r_pos = rendered.index(INSTRUCTION_PART), rendered.index(RESPONSE_PART)
    assert i_pos < rendered.index(prompt) < r_pos < rendered.index(answer[:20])
    joined = "\n".join(code_cells)
    assert 'instruction_part="<|im_start|>user\\n"' in joined
    assert 'response_part="<|im_start|>assistant\\n"' in joined


class _OldSFTConfig:
    """TRL generation with ``max_seq_length``."""

    def __init__(self, max_seq_length=None, learning_rate=None, output_dir=None,
                 push_to_hub=False, hub_model_id=None, seed=0):
        self.max_seq_length = max_seq_length


class _NewSFTConfig:
    """TRL generation after the ``max_length`` rename."""

    def __init__(self, max_length=None, learning_rate=None, output_dir=None,
                 push_to_hub=False, hub_model_id=None, seed=0):
        self.max_length = max_length


class _OldTrainer:
    """Accepts ``tokenizer=``."""

    def __init__(self, model=None, tokenizer=None, data_collator=None,
                 train_dataset=None, args=None):
        self.tokenizer = tokenizer


class _NewTrainer:
    """Rejects ``tokenizer=``; wants ``processing_class=``."""

    def __init__(self, model=None, processing_class=None, data_collator=None,
                 train_dataset=None, args=None):
        self.processing_class = processing_class


@pytest.mark.parametrize("cfg_cls,trainer_cls", [(_OldSFTConfig, _OldTrainer),
                                                 (_NewSFTConfig, _NewTrainer)])
def test_trl_compat_shim(code_cells: List[str], cfg_cls, trainer_cls) -> None:
    """The shim must adapt kwargs to both TRL API generations."""
    cell = _find_cell(code_cells, "def make_sft_config")
    shim = cell[cell.index("def make_sft_config"):cell.index("resume_dir = None")]
    ns = {"SFTConfig": cfg_cls, "SFTTrainer": trainer_cls,
          "print": lambda *a, **k: None}
    exec("import inspect\n" + shim, ns)

    cfg = ns["make_sft_config"](max_seq_length=12288, learning_rate=1e-4,
                                output_dir="x", seed=1)
    assert (getattr(cfg, "max_seq_length", None)
            or getattr(cfg, "max_length", None)) == 12288
    tr = ns["make_trainer"](model=1, tokenizer="TOK", data_collator=2,
                            train_dataset=3, args=cfg)
    assert (getattr(tr, "tokenizer", None)
            or getattr(tr, "processing_class", None)) == "TOK"


def _run_resume(code_cells: List[str], files: List[str],
                raise_exc: Optional[Exception] = None) -> Optional[str]:
    """Execute the notebook's resume block against a fake hub.

    :param code_cells: Cell sources.
    :param files: Fake repo file listing.
    :param raise_exc: Exception the fake ``list_repo_files`` should raise.
    :returns: The computed ``resume_dir``.
    """
    cell = _find_cell(code_cells, "resume_dir = None")
    block = cell[cell.index("resume_dir = None"):cell.index("FastVisionModel")]

    def fake_list(repo):
        if raise_exc:
            raise raise_exc
        return files

    ns = {"list_repo_files": fake_list,
          "snapshot_download": lambda *a, **k: "outputs_v20a",
          "CKPT_REPO": "x/y", "OUT_DIR": "outputs_v20a",
          "print": lambda *a, **k: None}
    exec(block, ns)
    return ns["resume_dir"]


def test_resume_last_checkpoint_contract(code_cells: List[str]) -> None:
    """Resume must use TRL's rolling ``last-checkpoint`` folder name."""
    assert _run_resume(code_cells, [], raise_exc=RuntimeError("404")) is None
    assert _run_resume(code_cells, ["README.md"]) is None
    assert (_run_resume(code_cells, ["last-checkpoint/adapter_model.safetensors"])
            == "outputs_v20a/last-checkpoint")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
