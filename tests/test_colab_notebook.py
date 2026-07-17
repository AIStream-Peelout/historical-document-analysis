"""CPU-only validation of the Colab fine-tuning notebook.

Guards the notebook against the failure classes found during its first
integration runs — all checkable without CUDA/Unsloth:

- per-cell syntax and cross-cell name-definition order
- ``to_conversation`` executed against real dataset rows (when the local
  artifact exists)
- the ``train_on_responses_only`` markers rendered against the real Qwen3-VL
  chat template (prompt between the user/assistant markers, answer after)
- the TRL compat shim against fakes of both TRL API generations
- the hub-resume logic's ``last-checkpoint`` contract
"""

import ast
import builtins
import json
from pathlib import Path
from typing import List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[1]
NB_PATH = _REPO / "src/finetuning/qwen_hebrew/colab/qwen3_vl_hebrew_unsloth.ipynb"
ARTIFACT_DIR = _REPO / "src/datasets/processed/qwen_hebrew_talmud"

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


def test_every_cell_parses(code_cells: List[str]) -> None:
    """All code cells must be valid Python."""
    for i, src in enumerate(code_cells):
        ast.parse(src)  # raises SyntaxError with cell context on failure


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
    """Every collator construction must avoid the hostile defaults.

    ``resize="min"`` silently downscales to ~512px and the default leaves
    prompt tokens in the loss — both root causes of past failed runs.
    """
    joined = "\n".join(code_cells)
    n_collators = joined.count("UnslothVisionDataCollator(")
    assert n_collators >= 1
    assert joined.count('resize="max"') == n_collators
    assert joined.count("train_on_responses_only=True") == n_collators


@pytest.mark.skipif(not (ARTIFACT_DIR / "dataset_dict.json").exists(),
                    reason="dataset artifact not built")
def test_to_conversation_on_real_rows(code_cells: List[str]) -> None:
    """The conversion function must run against real smoke rows."""
    from datasets import load_from_disk

    smoke = load_from_disk(str(ARTIFACT_DIR))["smoke"]
    ns = {"smoke": smoke, "print": lambda *a, **k: None, "range": range, "len": len}
    exec(_find_cell(code_cells, "def to_conversation"), ns)
    conv = ns["converted_smoke"]
    assert len(conv) == len(smoke)
    m = conv[0]["messages"]
    assert m[0]["role"] == "user" and m[1]["role"] == "assistant"
    assert m[0]["content"][0]["type"] == "image"
    assert len(m[1]["content"][0]["text"]) > 50


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
    # and the notebook must use exactly these markers
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
    shim = cell.split("COMMON_CFG")[0]
    shim = "\n".join(l for l in shim.split("\n")
                     if not l.strip().startswith(("from trl", "from unsloth")))
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
          "snapshot_download": lambda *a, **k: "outputs_full",
          "CKPT_REPO": "x/y", "print": lambda *a, **k: None}
    exec(block, ns)
    return ns["resume_dir"]


def test_resume_last_checkpoint_contract(code_cells: List[str]) -> None:
    """Resume must use TRL's rolling ``last-checkpoint`` folder name."""
    assert _run_resume(code_cells, [], raise_exc=RuntimeError("404")) is None
    assert _run_resume(code_cells, ["README.md"]) is None
    assert (_run_resume(code_cells, ["last-checkpoint/adapter_model.safetensors"])
            == "outputs_full/last-checkpoint")
