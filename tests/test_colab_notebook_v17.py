# File name: test_colab_notebook_v17.py
# Date: 7/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""CPU-only validation of the v1.7 genizah-focus Colab notebook.

Same harness pattern as ``test_colab_notebook_v16.py``, adapted to
``genizah_focus_v17.ipynb``'s cell layout, plus v1.7-specific guards:

- both dataset revision AND warm-start revision pinned to full 40-char SHAs
- label-hygiene asserts (no internal gap tokens / mojibake symbols; ``[...]``
  damage markers present)
- the ``TRAIN_VISION_LORA`` flag with its patch-embed gradient fix, and the
  gradient-flow verification cell that enforces whichever mode is selected
- the 6.5MP resolution policy rebuilt into the processor AND shipped with the
  exported merged model
"""

import ast
import builtins
import json
from pathlib import Path
from typing import List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/genizah_focus_v17.ipynb"

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


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
    :returns: The 40-char SHA string.
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
    """Genizah dataset and v1.6 warm start must both pin full commit SHAs."""
    for var in ("GENIZAH_REVISION", "V16_REVISION", "SYNTH_REVISION"):
        sha = _pinned_sha(code_cells, var)
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), var


def test_resolution_policy_defined_and_shipped(code_cells: List[str]) -> None:
    """6.5MP floor must reach the processor AND the exported model."""
    model_cell = _find_cell(code_cells, "MIN_PIX")
    assert "MIN_PIX = 6_500_000" in model_cell
    assert "min_pixels=MIN_PIX" in model_cell and "max_pixels=MAX_PIX" in model_cell
    export_cell = _find_cell(code_cells, "push_to_hub_merged")
    assert "image_processor.save_pretrained" in export_cell


def test_vision_lora_flag_and_fix(code_cells: List[str]) -> None:
    """The vision-LoRA gradient fix must be flag-gated and default OFF.

    All adapters through v1.6 trained language-only because Unsloth's GC hook
    never reaches the Qwen3-VL vision tower; this run must change one variable
    (data) unless the flag is deliberately flipped.
    """
    cell = _find_cell(code_cells, "TRAIN_VISION_LORA = ")
    assert "TRAIN_VISION_LORA = False" in cell, "flag must default OFF for v1.7"
    assert 'n.endswith("visual.patch_embed")' in cell
    assert "register_forward_hook" in cell
    assert "requires_grad_(True)" in cell
    # the hook must be registered after the PEFT wrap + warm start
    assert cell.index("set_peft_model_state_dict") < cell.index("register_forward_hook")


def test_gradient_flow_verification_cell(code_cells: List[str]) -> None:
    """A pre-training cell must assert grads match the configured mode."""
    cell = _find_cell(code_cells, "loss.backward()")
    assert "lora_B" in cell and ".visual." in cell
    assert "len(vis_g) == 108" in cell, "must require all 108 vision tensors when ON"
    assert "max(lang_g) > 0" in cell, "language grads must always be verified"
    assert "zero_grad" in cell
    # verification must run before the trainer cell
    cells = list(code_cells)
    assert cells.index(cell) < cells.index(_find_cell(cells, "SFTTrainer"))


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
          "snapshot_download": lambda *a, **k: "outputs_v17",
          "CKPT_REPO": "x/y", "print": lambda *a, **k: None}
    exec(block, ns)
    return ns["resume_dir"]


def test_resume_last_checkpoint_contract(code_cells: List[str]) -> None:
    """Resume must use TRL's rolling ``last-checkpoint`` folder name."""
    assert _run_resume(code_cells, [], raise_exc=RuntimeError("404")) is None
    assert _run_resume(code_cells, ["README.md"]) is None
    assert (_run_resume(code_cells, ["last-checkpoint/adapter_model.safetensors"])
            == "outputs_v17/last-checkpoint")
