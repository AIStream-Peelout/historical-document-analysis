# File name: test_colab_notebook_v18.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""CPU-only validation of the v1.8 genizah-focus Colab notebook.

Same harness pattern as ``test_colab_notebook_v17.py``, adapted to the v1.8
A/B protocol and its hard gates:

- pinned install triplet (unsloth 2026.8.9 / unsloth_zoo 2026.8.6 /
  transformers 4.57.6 — the exact versions the 2026-08-09 audit verified the
  bug and the fix against) with import-time version asserts
- ``ARM`` drives everything: A = control (repaired data, vision frozen),
  B = candidate (vision LoRA training); default must be "A" (run order A→B)
- warm start pinned to v1.7 step-800 (068a2cc1…), weights-only
- cell-5 grad gate hardened to per-tensor ``min(vis_g) > 0`` over all 108
- patch_embed hook carries the ``torch.is_grad_enabled()`` inference guard
- cell-7 export gated on a shipped-adapter vision check
- ``max_steps=700`` (completes the 800/1500 schedule), per-arm ckpt repos
"""

import ast
import builtins
import json
from pathlib import Path
from typing import List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/genizah_focus_v18.ipynb"

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
V17_WARM_START_SHA = "068a2cc1d083ea2fe27b9d09ec801e0cb8474d59"


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
    """The install cell must pin the audited triplet and assert versions.

    Floating installs are how the audited vision-path analysis (and the
    108-tensor count) silently rot: the zoo permits transformers up to 5.5.0.
    """
    install = next(c for c in raw_cells if "%pip install" in c)
    assert '"unsloth[colab-new]==2026.8.9"' in install
    assert '"unsloth_zoo==2026.8.6"' in install
    assert '"transformers==4.57.6"' in install
    assert 'version("unsloth_zoo") == "2026.8.6"' in install
    assert 'version("transformers") == "4.57.6"' in install


def test_arm_protocol(code_cells: List[str]) -> None:
    """ARM must default to the control arm and drive the vision flag."""
    cell = _find_cell(code_cells, "TRAIN_VISION_LORA")
    assert 'ARM = "A"' in cell, "default arm must be A (control) — run A→B"
    assert 'TRAIN_VISION_LORA = ARM == "B"' in cell
    trainer_cell = _find_cell(code_cells, "SFTTrainer")
    assert 'f"isaacmg/qwen3-vl-8b-hebrew-v18{ARM.lower()}-ckpt"' in trainer_cell
    assert 'run_name=f"genizah_focus_v18{ARM.lower()}"' in trainer_cell


def test_collator_uses_safe_arguments(code_cells: List[str]) -> None:
    """Every collator construction must avoid the hostile defaults."""
    joined = "\n".join(code_cells)
    n_collators = joined.count("UnslothVisionDataCollator(")
    assert n_collators >= 1
    assert joined.count('resize="max"') == n_collators
    assert joined.count("train_on_responses_only=True") == n_collators


def test_label_hygiene_asserts_present(code_cells: List[str]) -> None:
    """The data cell must refuse leaked gap tokens / mojibake symbols."""
    cell = _find_cell(code_cells, "_sample_answers")
    assert "\\u2423" in cell or "␣" in cell, "internal gap-token guard missing"
    assert '"[...]" in a' in cell, "damage-marker presence guard missing"
    assert "&#$" in cell, "mojibake-symbol guard missing"


def test_dataset_and_warm_start_pinned(code_cells: List[str]) -> None:
    """v1.7 warm start + synth must pin SHAs; genizah v2 gated at runtime.

    ``GENIZAH_REVISION`` may hold the ``PENDING_V2_PUSH`` sentinel until the
    repaired dataset is pushed, but then the notebook itself must refuse to
    run (the runtime length assert) — launch stays impossible unpinned.
    """
    assert _pinned_sha(code_cells, "V17_REVISION") == V17_WARM_START_SHA
    sha = _pinned_sha(code_cells, "SYNTH_REVISION")
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    gen = _pinned_sha(code_cells, "GENIZAH_REVISION")
    assert gen == "PENDING_V2_PUSH" or (
        len(gen) == 40 and all(c in "0123456789abcdef" for c in gen))
    data_cell = _find_cell(code_cells, "GENIZAH_REVISION")
    assert "assert len(GENIZAH_REVISION) == 40" in data_cell, \
        "runtime pin gate missing — placeholder could reach training"
    assert '"isaacmg/genizah_clean_v2"' in data_cell, "must train on the repaired v2"


def test_resolution_policy_defined_and_shipped(code_cells: List[str]) -> None:
    """6.5MP floor must reach the processor AND the exported model."""
    model_cell = _find_cell(code_cells, "MIN_PIX")
    assert "MIN_PIX = 6_500_000" in model_cell
    assert "min_pixels=MIN_PIX" in model_cell and "max_pixels=MAX_PIX" in model_cell
    export_cell = _find_cell(code_cells, "push_to_hub_merged")
    assert "image_processor.save_pretrained" in export_cell


def test_vision_fix_hook(code_cells: List[str]) -> None:
    """The patch-embed fix must be arm-gated, guarded, and ordered correctly."""
    cell = _find_cell(code_cells, "visual.patch_embed")
    assert 'n.endswith("visual.patch_embed")' in cell
    assert "register_forward_hook" in cell
    assert "requires_grad_(True)" in cell
    assert "torch.is_grad_enabled()" in cell, \
        "inference-mode guard missing — generate() would crash with the hook armed"
    # the hook must be registered after the PEFT wrap + warm start
    assert cell.index("set_peft_model_state_dict") < cell.index("register_forward_hook")


def test_gradient_flow_verification_cell(code_cells: List[str]) -> None:
    """The pre-training gate must be per-tensor and mode-exact."""
    cell = _find_cell(code_cells, "loss.backward()")
    assert "lora_B" in cell and ".visual." in cell
    assert "len(vis_g) == 108" in cell, "must require all 108 vision tensors when ON"
    assert "min(vis_g) > 0" in cell, "per-tensor nonzero (min), not aggregate max"
    assert "max(lang_g) > 0" in cell, "language grads must always be verified"
    assert "zero_grad" in cell
    cells = list(code_cells)
    assert cells.index(cell) < cells.index(_find_cell(cells, "SFTTrainer"))


def test_schedule_completion(code_cells: List[str]) -> None:
    """700 steps completes v1.7's 800/1500 schedule from the warm start."""
    cell = _find_cell(code_cells, "SFTTrainer")
    assert "max_steps=700" in cell
    assert "learning_rate=5e-5" in cell


def test_export_gated_on_vision_check(code_cells: List[str]) -> None:
    """Arm B may not export unless the shipped adapter's vision LoRA moved."""
    cell = _find_cell(code_cells, "push_to_hub_merged")
    assert "if TRAIN_VISION_LORA:" in cell
    assert "lora_B" in cell and ".visual." in cell
    assert "== 108" in cell
    assert cell.index("== 108") < cell.index("push_to_hub_merged")


def test_mixture_includes_all_domains(code_cells: List[str]) -> None:
    """Genizah must lead the mixture with Talmud + synth retained."""
    cell = _find_cell(code_cells, "interleave_datasets(")
    assert "genizah" in cell.split("interleave_datasets(")[1][:200]
    assert "probabilities=[0.35, 0.25, 0.20, 0.15, 0.05]" in cell
    assert 'stopping_strategy="all_exhausted"' in cell


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
          "snapshot_download": lambda *a, **k: "outputs_v18a",
          "CKPT_REPO": "x/y", "OUT_DIR": "outputs_v18a",
          "print": lambda *a, **k: None}
    exec(block, ns)
    return ns["resume_dir"]


def test_resume_last_checkpoint_contract(code_cells: List[str]) -> None:
    """Resume must use TRL's rolling ``last-checkpoint`` folder name."""
    assert _run_resume(code_cells, [], raise_exc=RuntimeError("404")) is None
    assert _run_resume(code_cells, ["README.md"]) is None
    assert (_run_resume(code_cells, ["last-checkpoint/adapter_model.safetensors"])
            == "outputs_v18a/last-checkpoint")
