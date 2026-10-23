# File name: test_colab_notebook_t27.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the text27b notebook derivation (``derive_t27.py``) and the additive ``host_lambda.PIN_SETS``.

Both editions must keep v2.5a's training stack where the plan keeps it (the inlined session helpers, the trainer
factories, the resume rationale, the recording wrapper, the milestone callback with a set of steps), replace the
vision-specific cells, gate every step (G1-G10), refuse unpinned data, render with ``enable_thinking=False`` and mask
the prompt, and name only private repositories and the private W&B project. The notebook logic is run here on
synthetic rows with a stand-in tokenizer; nothing touches the network.
"""
import codecs
import ast
import copy
import difflib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from src.finetuning.qwen_hebrew.colab import derive_t27 as d
from src.finetuning.qwen_hebrew.colab import host_lambda as hl
from src.finetuning.qwen_hebrew.colab.derive_v22b import session_helper_source
from src.finetuning.text27b import rows as R

COLAB_DIR = Path("src/finetuning/qwen_hebrew/colab")
SRC = COLAB_DIR / d.SOURCE_NOTEBOOK
SHA = "c" * 40
FP = "f" * 64
TYPING = "from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple\n"
#: the render contract of the Qwen3.8 template in miniature: one empty think block before every answer, the effort
#: sentence only when thinking is on (compiled like transformers does: trim / lstrip blocks)
TEMPLATE = ("{%- for m in messages -%}"
            "{{ '<|im_start|>' + m.role + '\\n' }}"
            "{%- if m.role == 'system' and enable_thinking -%}{{ 'Reasoning effort: xhigh. ' }}{%- endif -%}"
            "{%- if m.role == 'assistant' -%}{{ '<think>\\n\\n</think>\\n\\n' + (m.content | trim) }}"
            "{%- else -%}{{ m.content }}{%- endif -%}"
            "{{ '<|im_end|>\\n' }}"
            "{%- endfor -%}")


# ----------------------------------------------------------------------------- helpers and fixtures


def _src(nb: Dict[str, Any], i: int) -> str:
    """Source of cell ``i``.

    :param nb: Notebook.
    :param i: Cell index.
    :returns: The joined source.
    """
    return "".join(nb["cells"][i]["source"])


def _changed(a: Dict[str, Any], b: Dict[str, Any], i: int) -> Tuple[List[str], List[str]]:
    """(lines only in ``a``, lines only in ``b``) of cell ``i``.

    :param a: First notebook.
    :param b: Second notebook.
    :param i: Cell index.
    :returns: Two lists of lines.
    """
    diff = [l for l in difflib.unified_diff(_src(a, i).splitlines(), _src(b, i).splitlines(), lineterm="", n=0)
            if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in diff if l.startswith("-")], [l[1:] for l in diff if l.startswith("+")]


def _parses(nb: Dict[str, Any]) -> None:
    """Every code cell parses (IPython magics left out).

    :param nb: Notebook.
    :returns: None.
    """
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))


def _block(text: str, start: str, end: Optional[str] = None) -> str:
    """The part of ``text`` from ``start`` up to ``end`` (or the end).

    :param text: Cell source.
    :param start: Opening anchor (included).
    :param end: Closing anchor (excluded), or None.
    :returns: The block.
    """
    i = text.index(start)
    return text[i:] if end is None else text[i:text.index(end, i)]


def _pins() -> d.Pins:
    """Pins with t27a pinned and t27b left as placeholders.

    :returns: Pins.
    """
    return d.Pins(revisions={"t27a": SHA, "t27b": d.PLACEHOLDER}, fingerprints={"t27a": FP, "t27b": d.PLACEHOLDER})


@pytest.fixture(scope="module")
def v25a() -> Dict[str, Any]:
    """The v2.5a Colab notebook (the derivation's source).

    :returns: Notebook JSON.
    """
    if not SRC.exists():
        pytest.skip(f"{SRC} is not in this checkout")
    return json.loads(SRC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def editions(v25a: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(v2.5a Colab edition, t27 Colab edition, t27 Lambda edition).

    :param v25a: Source notebook.
    :returns: The three notebooks.
    """
    colab = d.derive(v25a, _pins())
    return v25a, colab, d.to_lambda(colab, "genizah_t27_lambda.ipynb")


def chat_row(family: str = "translate_en", pgpid: Any = 1, condition: str = "edition", split: str = "train",
             answer: str = "The answer.", **meta: Any) -> Dict[str, Any]:
    """A synthetic v0_2 chat row.

    :param family: Row family.
    :param pgpid: Document id.
    :param condition: Input condition.
    :param split: Split.
    :param answer: Assistant target.
    :param meta: Further meta fields.
    :returns: The row.
    """
    m = {"family": family, "task": family, "condition": condition, "split": split, "pgpid": pgpid, "build": "v0_2",
         "row_id": f"{family}-{pgpid}-{condition}-{split}", "lang_class": "JA"}
    m.update(meta)
    return {"messages": [{"role": "system", "content": "You are a test."},
                         {"role": "user", "content": f"Translate.\n<document>\nכתאבי {pgpid}\n</document>"},
                         {"role": "assistant", "content": answer}], "meta": m}


class FakeTok:
    """A stand-in for the HF text tokenizer: special tokens + one token per character, exact offsets, lossless decode."""

    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    PIECE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<think>|</think>|.", re.S)

    def __init__(self, template: str = TEMPLATE, honour_thinking: bool = True) -> None:
        """Compile the template like transformers does.

        :param template: Chat template source.
        :param honour_thinking: False = a template that ignores ``enable_thinking`` (always thinking).
        :returns: None.
        """
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.chat_template = template
        self._compiled = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)
        self.honour_thinking = honour_thinking
        self.pad_token_id, self.eos_token_id = 0, 2

    def convert_tokens_to_ids(self, token: str) -> int:
        """Id of a token.

        :param token: Token text.
        :returns: Its id.
        """
        return 1 + self.SPECIAL.index(token) if token in self.SPECIAL else 1000 + ord(token)

    def _token(self, i: int) -> str:
        """Token text of an id.

        :param i: Id.
        :returns: Text.
        """
        return self.SPECIAL[i - 1] if i <= len(self.SPECIAL) else chr(i - 1000)

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> Dict[str, Any]:
        """Tokenize.

        :param text: Text.
        :param add_special_tokens: Ignored (no BOS).
        :param return_offsets_mapping: Return character offsets too.
        :returns: ``input_ids`` (and ``offset_mapping``).
        """
        pieces = list(self.PIECE.finditer(text))
        out = {"input_ids": [self.convert_tokens_to_ids(m.group(0)) for m in pieces]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(m.start(), m.end()) for m in pieces]
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False) -> str:
        """Lossless decode.

        :param ids: Ids.
        :param skip_special_tokens: Ignored.
        :param clean_up_tokenization_spaces: Ignored.
        :returns: Text.
        """
        return "".join(self._token(int(i)) for i in ids)

    def apply_chat_template(self, messages: Sequence[Dict[str, Any]], chat_template: Optional[str] = None, tokenize: bool = False,
                            enable_thinking: bool = True) -> str:
        """Render (thinking is on unless switched off, like the real template's default).

        :param messages: Chat turns.
        :param chat_template: Ignored (the compiled template is used).
        :param tokenize: Must be False.
        :param enable_thinking: The template flag.
        :returns: Rendered text.
        """
        assert not tokenize
        return self._compiled.render(messages=list(messages), enable_thinking=enable_thinking or not self.honour_thinking,
                                     add_generation_prompt=False)


# ----------------------------------------------------------------------------- structure


def test_cells_compile_and_the_layout_is_v25as(editions: Tuple) -> None:
    """10 Colab cells, 11 Lambda cells, v2.5a's metadata, every code cell parses, each cell has its title."""
    src, colab, lam = editions
    assert len(colab["cells"]) == 10 and len(lam["cells"]) == 11
    assert colab["metadata"] == src["metadata"] == lam["metadata"]
    assert [c["cell_type"] for c in colab["cells"]] == [c["cell_type"] for c in src["cells"]]
    _parses(colab)
    _parses(lam)
    titles = {1: "# Cell 1 — installs + env + run knobs + G1", 2: "# Cell 2 — data: the pinned text27b mixture",
              3: "# Cell 3 — model: Qwen3.8-27B", 4: "# Cell 4 — render (enable_thinking=False)", 5: "# Cell 5 — G6 gradient flow",
              6: "# Cell 6 — G9 throughput", 8: "# Cell 8 — text27b training", 9: "# Cell 9 — export"}
    for i, title in titles.items():
        assert _src(colab, i).startswith(title), i
    assert "# ---- v22b session helpers" in _src(colab, 7)
    assert all(not c.get("outputs") for c in colab["cells"] if c["cell_type"] == "code")


def test_source_is_not_modified_and_the_derivation_is_deterministic(v25a: Dict[str, Any]) -> None:
    """Deriving twice gives the same notebook and leaves the source untouched."""
    before = copy.deepcopy(v25a)
    assert d.derive(copy.deepcopy(v25a), _pins()) == d.derive(v25a, _pins())
    assert v25a == before


def test_refuses_anything_but_the_v25a_colab_notebook(v25a: Dict[str, Any]) -> None:
    """The Lambda edition, a shorter notebook or a renamed cell is refused."""
    if not (COLAB_DIR / "genizah_v25a_lambda.ipynb").exists():
        pytest.skip("the v2.5a Lambda edition is not in this checkout")
    lam = json.loads((COLAB_DIR / "genizah_v25a_lambda.ipynb").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="10-cell"):
        d.derive(lam)
    broken = copy.deepcopy(v25a)
    broken["cells"][2]["source"] = [l.replace("from google.colab import userdata", "") for l in broken["cells"][2]["source"]]
    with pytest.raises(ValueError, match="not a Colab edition"):
        d.derive(broken)
    broken = copy.deepcopy(v25a)
    broken["cells"][4]["source"] = ["# something else\n"]
    with pytest.raises(ValueError, match="cell 4"):
        d.derive(broken)


def test_kept_blocks_are_v25as_verbatim(editions: Tuple) -> None:
    """Trainer factories, resume rationale and recording wrapper are copied byte for byte; the milestone class only
    trades its fixed interval for a set of steps."""
    src, colab, _ = editions
    v8, t8 = _src(src, 8), _src(colab, 8)
    blocks = d.kept_blocks(v8)
    for name in ("factories", "resume_comment", "recording"):
        assert blocks[name] in v8 and blocks[name] in t8, name
    assert "def make_sft_config(**kw):" in blocks["factories"] and "def make_trainer(**kw):" in blocks["factories"]
    edited = d.milestone_class(blocks["milestone"])
    assert edited in t8 and blocks["milestone"] not in t8
    removed = [l[1:] for l in difflib.ndiff(blocks["milestone"].splitlines(), edited.splitlines()) if l.startswith("- ")]
    assert len(removed) == 3 and all("every" in l for l in removed)
    assert "if step not in self.steps or step in self.confirmed:" in edited
    with pytest.raises(ValueError):
        d.kept_blocks(v8.replace(d.RECORDING_START, ""))


def test_helper_cell_is_train_sessions_with_this_runs_length_and_budget(editions: Tuple) -> None:
    """Cell 7 = train_session.py inlined; only MAX_STEPS and TIME_BUDGET_H differ, and the Lambda edition only the budget."""
    _, colab, lam = editions
    ref = session_helper_source()
    diff = [l for l in difflib.unified_diff(ref.splitlines(), _src(colab, 7).splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("---", "+++"))]
    assert sorted(l[1:].split(" =")[0] for l in diff) == ["MAX_STEPS", "MAX_STEPS", "TIME_BUDGET_H", "TIME_BUDGET_H"]
    out, new = _changed(colab, lam, 7)
    assert [l.split(" =")[0] for l in out] == [l.split(" =")[0] for l in new] == ["TIME_BUDGET_H"]
    line = next(l for l in _src(colab, 7).splitlines() if l.startswith("MAX_STEPS = "))
    for stage, n, want in (("train", 6215, 388), ("rung1", 6215, 15), ("rung2", 100, 15)):
        ns = {"STAGE": stage, "LADDER_STEPS": 15, "ROWS_PER_STEP": 16, "train_ds": [0] * n}
        exec(line, ns)
        assert ns["MAX_STEPS"] == want, stage


def test_lambda_budget_is_per_arm_and_a_safety_stop(editions: Tuple) -> None:
    """The Lambda safety stop is 1.3 x the plan's upper hours for a run, 3 h for a rung; Colab keeps 7.5 h."""
    _, colab, lam = editions
    assert "TIME_BUDGET_H = 7.5" in _src(colab, 7)
    line = next(l for l in _src(lam, 7).splitlines() if l.startswith("TIME_BUDGET_H = "))
    for arm, stage, want in (("t27a", "train", 13.0), ("t27b", "train", 32.5), ("t27b", "rung2", 3.0)):
        ns = {"ARM": arm, "STAGE": stage}
        exec(line, ns)
        assert ns["TIME_BUDGET_H"] == want
    assert d.T27A.lambda_budget_h == 13.0 and d.T27B.lambda_budget_h == 32.5


# ----------------------------------------------------------------------------- Lambda edition


def test_lambda_edition_changes_only_the_host_lines(editions: Tuple) -> None:
    """Secrets from the environment or a prompt, the work directory, the venv note, the launch notes, the wrap-up."""
    _, colab, lam = editions
    for i in (3, 4, 5, 6, 9):
        assert _src(colab, i) == _src(lam, i), i
    out, new = _changed(colab, lam, 2)
    assert sorted(out) == sorted(['from google.colab import userdata', 'login(token=userdata.get("HF_TOKEN"))',
                                  f'LOCAL = "/content/{d.LOCAL_LEAF}"'])
    assert f'LOCAL = os.path.expanduser("~/genizah_work/{d.LOCAL_LEAF}")' in new
    assert 'login(token=os.environ.get("HF_TOKEN") or getpass("Hugging Face token (write access): "))' in new
    out, new = _changed(colab, lam, 8)
    assert sorted(out) == sorted(['wandb.login(key=userdata.get("WANDB_API_KEY"))', 'RESUME_ROOT = "/content/t27_resume"'])
    assert 'RESUME_ROOT = os.path.expanduser("~/genizah_work/t27_resume")' in new
    out, new = _changed(colab, lam, 1)
    assert all(l.startswith("%pip") or "No GPU" in l for l in out) and len(out) == 3
    assert any(d.SETUP_SCRIPT in l for l in new) and "GPU: {_p.name}" in _src(lam, 1)
    for i, c in enumerate(lam["cells"]):
        for needle in ("google.colab", "/content", "userdata", "%pip"):
            assert needle not in "".join(c["source"]), (i, needle)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(lam))


def test_lambda_launch_notes_and_wrapup(editions: Tuple) -> None:
    """The intro leads with the t27 launch notes; the wrap-up names the running arm and says terminate."""
    _, colab, lam = editions
    intro = _src(lam, 0)
    assert intro.startswith("# Genizah text27b — Qwen3.8-27B text fine-tune") and intro.endswith(_src(colab, 0))
    for needle in (f"bash {d.SETUP_SCRIPT}", d.KERNEL_NAME, "T27_STAGE=rung1", "nbconvert", "genizah_t27_lambda.ipynb",
                   "terminate the instance", "PLACEHOLDER", d.RESOLVED_PINS_FILE, "GH200"):
        assert needle in intro, needle
    assert "bash lambda_setup.sh" not in intro                                   # the Qwen3-VL venv is not this stack
    last = _src(lam, 10)
    assert 'wandb.alert(title=f"{RUN_NAME}: notebook finished"' in last and "TERMINATE THE INSTANCE" in last
    assert "genizah_t27a: notebook finished" not in last


# ----------------------------------------------------------------------------- pins: host_lambda.PIN_SETS and G1


def test_pin_sets_are_additive() -> None:
    """The VLM defaults are untouched; text27b records exact smoke-environment versions."""
    from packaging.specifiers import SpecifierSet
    assert hl.PINS["transformers"] == "4.57.6" and hl.KERNEL_GATE[0] == "unsloth"
    vlm = hl.PIN_SETS["vlm"]
    assert vlm["status"] == "exact" and vlm["specifiers"] == {k: f"=={v}" for k, v in hl.PINS.items()}
    assert vlm["kernel_gate"] == hl.KERNEL_GATE and vlm["torch"] == f"=={hl.TORCH_VERSION}"
    t = hl.PIN_SETS["text27b"]
    assert t["status"] == "exact" and set(t["kernel_gate"]) <= set(t["specifiers"])
    assert all(spec.startswith("==") and "*" not in spec for spec in t["specifiers"].values())
    assert t["torch"] == "==2.11.0+cu128" and t["torchvision"] == "==0.26.0+cu128"
    for name in ("unsloth", "transformers", "trl", "peft", "bitsandbytes", "flash-linear-attention", "causal-conv1d"):
        assert name in t["kernel_gate"], name
    specs = {k: SpecifierSet(v) for k, v in t["specifiers"].items()}
    assert "5.17.0" in specs["transformers"] and "5.17.1" not in specs["transformers"]
    assert "2026.10.7" in specs["unsloth"] and "2026.10.8" not in specs["unsloth"]
    assert "1.15.0" in specs["trl"] and "1.15.1" not in specs["trl"]
    assert "text27b" not in hl.setup_script()


def test_host_lambda_still_derives_the_committed_v25a_lambda_edition(v25a: Dict[str, Any]) -> None:
    """With PIN_SETS added, host_lambda reproduces the committed v2.5a Lambda edition and lambda_setup.sh byte for byte
    (derive_v25a's Lambda step = host_lambda.to_lambda + its own session-budget line)."""
    if not (COLAB_DIR / "genizah_v25a_lambda.ipynb").exists():
        pytest.skip("the v2.5a Lambda edition is not in this checkout")
    committed = json.loads((COLAB_DIR / "genizah_v25a_lambda.ipynb").read_text(encoding="utf-8"))
    lam = hl.to_lambda(v25a, "genizah_v25a_lambda.ipynb")
    budget = next(l for l in _src(committed, 7).splitlines() if l.startswith("TIME_BUDGET_H = "))
    lam["cells"][7]["source"] = re.sub(r"(?m)^TIME_BUDGET_H = .*$", lambda _m: budget, _src(lam, 7)).splitlines(keepends=True)
    assert lam == committed
    assert (COLAB_DIR / "lambda_setup.sh").read_text(encoding="utf-8") == hl.setup_script()


def test_install_cells_install_and_gate_the_pin_set(editions: Tuple) -> None:
    """Colab installs the ranges; Lambda installs nothing; both run G1 on the kernel-gate specifiers."""
    _, colab, lam = editions
    c1, l1 = _src(colab, 1), _src(lam, 1)
    pip = [l for l in c1.splitlines() if l.startswith("%pip")]
    assert pip[0].startswith('%pip install -q "unsloth[colab-new]==2026.10.7"') and '"transformers==5.17.0"' in pip[0]
    assert '"trl==1.15.0"' in pip[0] and '"flash-linear-attention==0.5.2"' in pip[0]
    assert pip[1].startswith('%pip install -q --no-build-isolation "causal-conv1d==1.7.0"')
    for c in (c1, l1):
        assert f"T27_KERNEL_PINS = {d.kernel_pins()!r}" in c and "T27_PIN_STATUS = 'exact'" in c
        assert "pin_problems(T27_KERNEL_PINS, version)" in c and '("fla", "causal_conv1d")' in c
        assert "write these resolved versions into host_lambda.PIN_SETS['text27b']" in c
        assert "WANDB_PROJECT = 'genizah-text27b'" in c and 'os.environ.get("T27_STAGE", "rung1")' in c
        assert 'os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"' in c


def test_pin_problems_checks_ranges_and_missing_distributions() -> None:
    """G1: a version outside its range or a missing distribution is reported; '' accepts anything."""
    versions = {"transformers": "5.1.0", "unsloth": "2026.10.2", "trl": "1.15.1", "peft": "0.18.0"}

    def get(name: str) -> str:
        """Fake ``importlib.metadata.version``.

        :param name: Distribution.
        :returns: Its version.
        """
        if name not in versions:
            raise ImportError(name)
        return versions[name]

    spec = {"transformers": ">=5,<6", "unsloth": "==2026.10.*", "trl": ">=1.14", "peft": ""}
    assert d.pin_problems(spec, get) == []
    versions["transformers"] = "4.57.6"
    assert d.pin_problems(spec, get) == ["transformers: 4.57.6 does not satisfy '>=5,<6'"]
    assert d.pin_problems({"causal-conv1d": ""}, get) == ["causal-conv1d: not installed"]


def test_setup_script_builds_the_text27b_venv(tmp_path: Path) -> None:
    """The t27 venv: its own name and kernel, the placeholder ranges, causal-conv1d built in place, the resolved
    versions written out; valid bash."""
    sh = d.setup_script()
    assert sh.startswith("#!/bin/bash") and "set -euo pipefail" in sh and "status: exact" in sh
    assert f'VENV="{d.VENV}"' in sh and f"--name {d.KERNEL_NAME}" in sh and d.RESOLVED_PINS_FILE in sh
    assert '"unsloth==2026.10.7"' in sh and '"transformers==5.17.0"' in sh and '"trl==1.15.0"' in sh
    assert '--no-build-isolation "causal-conv1d==1.7.0"' in sh and "CUDA=cu128" in sh and "x86_64" in sh and "xformers" not in sh
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in sh and "--no-deps --no-cache --no-binary causal-conv1d" in sh
    assert "genizah-venv" not in sh.replace(d.VENV, "")                          # never the Qwen3-VL venv
    script = tmp_path / d.SETUP_SCRIPT
    script.write_text(sh, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True)


# ----------------------------------------------------------------------------- data cell and G2


def test_data_cell_names_the_private_dataset_and_the_arm_files(editions: Tuple) -> None:
    """Repository, per-arm revision and fingerprint pins, the arm's files and families."""
    _, colab, _ = editions
    c2 = _src(colab, 2)
    assert f"DATA_REPO = {d.DATA_REPO!r}" in c2 and f'LOCAL = "/content/{d.LOCAL_LEAF}"' in c2
    assert f"DATA_REVISIONS = {{'t27a': '{SHA}', 't27b': '{d.PLACEHOLDER}'}}" in c2
    assert f"MIXTURE_FINGERPRINTS = {{'t27a': '{FP}', 't27b': '{d.PLACEHOLDER}'}}" in c2
    assert 'TRAIN_FILE, MANIFEST_FILE = f"{ARM}/train.jsonl", f"{ARM}/mixture_manifest.json"' in c2
    assert f"DEV_FILE = {d.DEV_FILE!r}" in c2
    ns: Dict[str, Any] = {}
    exec(_block(c2, "DATA_REPO = ", "TRAIN_FILE, MANIFEST_FILE"), ns)
    assert ns["ARM_FAMILIES"]["t27a"] == ("translate_en", "translate_he")
    assert ns["ARM_FAMILIES"]["t27b"] == tuple(R.FAMILIES) and ns["CONDITIONS"] == tuple(R.CONDITIONS) and ns["BUILD"] == "v0_2"
    assert set(ns["ARM_REQUIRED"]["t27b"]) <= set(ns["ARM_FAMILIES"]["t27b"])


def _pin_guard(c2: str) -> str:
    """The pin-guard block of the data cell (constants, helper functions, the two check_pin calls).

    :param c2: Data-cell source.
    :returns: Source.
    """
    return _block(c2, "DATA_REPO = ", 'LOCAL = "')


def test_data_cell_refuses_placeholders_and_bad_pins(editions: Tuple) -> None:
    """The cell raises before any download unless both pins of the arm are hex digests of the right length."""
    _, colab, _ = editions
    guard = _pin_guard(_src(colab, 2))
    exec(guard, {"ARM": "t27a"})
    with pytest.raises(RuntimeError, match="t27b dataset revision 'PIN-AFTER-PUSH' is not a pinned 40-hex value"):
        exec(guard, {"ARM": "t27b"})
    for bad_rev, bad_fp in ((SHA[:-1], FP), ("C" * 40, FP), (SHA, FP[:-1]), (SHA, "main")):
        with pytest.raises(RuntimeError, match="is not a pinned"):
            exec(guard.replace(SHA, bad_rev).replace(FP, bad_fp), {"ARM": "t27a"})


def _g2(colab: Dict[str, Any], train_rows: List[Dict[str, Any]], dev_rows: List[Dict[str, Any]], arm: str = "t27a",
        fingerprint: Optional[str] = None) -> Dict[str, Any]:
    """Run the data cell's constants, functions and G2 block on synthetic rows.

    :param colab: The Colab edition.
    :param train_rows: Train rows.
    :param dev_rows: Dev-monitor rows.
    :param arm: The arm.
    :param fingerprint: Pinned fingerprint (default: the rows' own).
    :returns: The namespace after the block.
    """
    c2 = _src(colab, 2)
    ns: Dict[str, Any] = {"ARM": arm, "Counter": Counter, "ROWS_PER_STEP": 16, "LADDER_STEPS": 1}
    exec(_block(c2, "DATA_REPO = ", "# never train on an unpinned"), ns)
    ns.update(train_rows=train_rows, dev_rows=dev_rows,
              MIXTURE_FINGERPRINT=fingerprint or d.text_fingerprint(train_rows))
    exec(_block(c2, "# ---- G2:"), ns)
    return ns


def test_g2_passes_a_clean_mixture_and_catches_each_defect(editions: Tuple) -> None:
    """Fingerprint, split, family, a document in train and dev, a reader-exposed dev read, a missing family."""
    _, colab, _ = editions
    train = [chat_row("translate_en", i) for i in range(20)] + [chat_row("translate_he", 100 + i) for i in range(4)]
    dev = [chat_row("translate_en", 500, split="dev"), chat_row("catalogue_json", 501, split="dev")]
    _g2(colab, train, dev)
    with pytest.raises(AssertionError, match="mixture fingerprint"):
        _g2(colab, train, dev, fingerprint=FP)
    for bad, match in ((train[:-1] + [chat_row("translate_he", 999, split="dev")], "train-row problems"),
                       (train + [chat_row("catalogue_json", 998)], "train-row problems"),
                       (train + [chat_row("translate_en", 997, held_out_vision_benchmark=True)], "train-row problems"),
                       ([r for r in train if r["meta"]["family"] == "translate_en"], "needs")):
        with pytest.raises(AssertionError, match=match):
            _g2(colab, bad, dev)
    with pytest.raises(AssertionError, match="documents in both train and the dev monitor"):
        _g2(colab, train, dev + [chat_row("translate_en", 3, split="dev", row_id="x")])
    with pytest.raises(AssertionError, match="reader-exposed"):
        _g2(colab, train, dev + [chat_row("translate_en", 502, split="dev", condition="read", reader_exposed=True)])
    with pytest.raises(AssertionError, match="t27b needs"):                       # t27b also needs the catalogue families
        _g2(colab, train, dev, arm="t27b")


def test_row_problems_lists_every_schema_defect() -> None:
    """G2's row schema."""
    fams, conds = ("translate_en",), tuple(R.CONDITIONS)
    assert d.row_problems(chat_row(), "train", fams, conds) == []
    row = chat_row()
    row["messages"] = row["messages"][1:]
    assert "roles" in d.row_problems(row, "train", fams, conds)[0]
    row = chat_row(answer="   ")
    assert d.row_problems(row, "train", fams, conds) == ["empty assistant target"]
    row = chat_row(build="v0_1", condition="weird", row_id="")
    assert len(d.row_problems(row, "dev", fams, conds)) == 4
    row = chat_row()
    row["messages"][1]["content"] = ["not", "text"]
    assert d.row_problems(row, "train", fams, conds) == ["a message content is not a string"]


def test_text_fingerprint_is_the_rows_module_fingerprint() -> None:
    """The notebook's digest equals text27b.rows.fingerprint and depends on the order."""
    rows = [chat_row("translate_en", i, answer=f"answer {i} — אמת") for i in range(5)]
    assert d.text_fingerprint(rows) == R.fingerprint(rows)
    assert d.text_fingerprint(rows[::-1]) != d.text_fingerprint(rows)
    odd = chat_row()
    odd["messages"][2]["content"] = {"b": 1, "a": "ש"}
    assert d.text_fingerprint([odd]) == R.fingerprint([odd])


# ----------------------------------------------------------------------------- model cell, G3, G4


def test_model_cell_loads_the_planned_base_and_adapter(editions: Tuple) -> None:
    """27B bnb 4-bit (2B on rung 1), the language-only regex, r16 alpha16, 400 lora_A, no vision layers."""
    _, colab, _ = editions
    c3 = _src(colab, 3)
    assert f'BASE_MODEL = {d.SMOKE_BASE_MODEL!r} if SMOKE else {d.BASE_MODEL!r}' in c3
    assert d.BASE_MODEL == "unsloth/Qwen3.8-27B-unsloth-bnb-4bit" and d.SMOKE_BASE_MODEL == "Qwen/Qwen3.5-2B"
    assert "MAX_SEQ = 6144" in c3 and "LORA_RANK, LORA_ALPHA = 16, 16" in c3 and "EXPECTED_LORA_A = None if SMOKE else 400" in c3
    regex_line = next(l for l in c3.splitlines() if l.startswith("LORA_REGEX = "))
    assert ast.literal_eval(regex_line.split(" = ", 1)[1]) == d.LORA_REGEX
    for needle in ("load_in_4bit=True", "finetune_vision_layers=False", "target_modules=LORA_REGEX", "lora_dropout=0.0",
                   "random_state=3407", "lora_problems(_lora_a, LAYER_TYPES, EXPECTED_LORA_A)", "REQUIRE_FAST_PATH = True",
                   'getattr(_modeling_module, "is_fast_path_available", None), _modeling_module)'):
        assert needle in c3, needle


def _hybrid_names(n_layers: int, prefix: str = "base_model.model.model.language_model.layers") -> List[str]:
    """Synthetic module names of a hybrid qwen3_5 stack (every 4th layer full attention) plus vision and MTP modules.

    :param n_layers: Layers.
    :param prefix: Language-layer prefix.
    :returns: Module names.
    """
    names = []
    for i in range(n_layers):
        p = f"{prefix}.{i}"
        if (i + 1) % 4:
            names += [f"{p}.linear_attn.{m}" for m in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d", "norm")]
        else:
            names += [f"{p}.self_attn.{m}" for m in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        names += [f"{p}.mlp.{m}" for m in ("gate_proj", "up_proj", "down_proj")] + [f"{p}.input_layernorm"]
    names += [f"base_model.model.model.visual.blocks.{i}.{m}" for i in range(2) for m in ("attn.qkv", "attn.proj", "mlp.linear_fc1")]
    names += ["base_model.model.mtp.layers.0.self_attn.q_proj", "base_model.model.mtp.layers.0.mlp.gate_proj", "base_model.model.mtp.fc",
              "base_model.model.lm_head", "base_model.model.model.language_model.embed_tokens"]
    return names


def _layer_types(n_layers: int) -> List[str]:
    """``config.layer_types`` of the synthetic stack.

    :param n_layers: Layers.
    :returns: Layer types.
    """
    return ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n_layers)]


def test_lora_regex_selects_exactly_the_language_projections() -> None:
    """On a 64-layer hybrid stack: 400 modules, none on vision, MTP, in_proj_a / in_proj_b, norms or conv."""
    sel = d.lora_targets(_hybrid_names(64))
    assert len(sel) == 400
    assert not [n for n in sel if re.search(r"visual|mtp|in_proj_[ab]$|norm|conv1d|lm_head|embed", n)]
    counts, problems = d.lora_problems([n + ".lora_A.default.weight" for n in sel], _layer_types(64), 400)
    assert problems == [] and counts == {"deltanet": 144, "attention": 64, "mlp": 192, "vision": 0, "mtp": 0, "other": 0}
    assert d.expected_lora_counts(_layer_types(24)) == {"deltanet": 54, "attention": 24, "mlp": 72}
    assert d.lora_targets(["model.layers.0.linear_attn.in_proj_qkvz"]) == []          # a fused naming is not silently adapted


def test_lora_problems_flags_every_misplaced_adapter() -> None:
    """Vision, MTP, gate projections, a missing module and the wrong total are all reported."""
    good = [n + ".lora_A.default.weight" for n in d.lora_targets(_hybrid_names(8))]
    assert d.lora_problems(good, _layer_types(8), None)[1] == []
    extra = good + ["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight",
                    "base_model.model.mtp.layers.0.self_attn.q_proj.lora_A.default.weight",
                    "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_A.default.weight"]
    problems = d.lora_problems(extra, _layer_types(8), None)[1]
    assert any("vision tower" in p for p in problems) and any("MTP" in p for p in problems)
    assert any("in_proj_a / in_proj_b" in p for p in problems) and any(p.startswith("deltanet:") for p in problems)
    problems = d.lora_problems(good[1:], _layer_types(8), 50)[1]
    assert len(problems) == 2
    with pytest.raises(ValueError, match="unknown layer types"):
        d.expected_lora_counts(["linear_attention", "sliding_attention"])


def test_fast_path_problems_tells_kernels_from_torch_fallbacks() -> None:
    """G4 on stand-in DeltaNet layers."""

    def chunk_gated_delta_rule() -> None:
        """Stand-in fla kernel.

        :returns: None.
        """

    def torch_chunk_gated_delta_rule() -> None:
        """Stand-in torch fallback.

        :returns: None.
        """

    fast = SimpleNamespace(causal_conv1d_fn=print, causal_conv1d_update=print, chunk_gated_delta_rule=chunk_gated_delta_rule,
                           recurrent_gated_delta_rule=chunk_gated_delta_rule)
    assert d.fast_path_problems(fast, True) == []
    slow = SimpleNamespace(causal_conv1d_fn=None, chunk_gated_delta_rule=torch_chunk_gated_delta_rule)
    problems = d.fast_path_problems(slow, False)
    assert len(problems) == 3 and any("torch fallback" in p for p in problems) and any("None" in p for p in problems)
    assert "check the fast path by hand" in d.fast_path_problems(SimpleNamespace(), None)[0]


def test_fast_path_checks_module_dispatch_implementations() -> None:
    """G4 verifies the selected kernel, rejecting fallbacks, unknown packages and missing wrappers."""

    def dispatch(package: str, enabled: bool = True) -> Any:
        """Build a stand-in for Transformers' captured kernel dispatch.

        :param package: Package providing the selected implementation.
        :param enabled: Whether the wrapper selected an external implementation.
        :returns: A dispatch wrapper.
        """
        def implementation() -> None:
            """Stand-in external kernel.

            :returns: None.
            """
        implementation.__module__ = f"{package}.kernel"
        is_new_implementation = enabled

        def wrapped() -> None:
            """Dispatch to the selected implementation.

            :returns: None.
            """
            if is_new_implementation:
                implementation()

        wrapped.__wrapped__ = implementation
        return wrapped

    module = SimpleNamespace(causal_conv1d_fn=dispatch("causal_conv1d"), causal_conv1d_update=dispatch("causal_conv1d"),
                             torch_chunk_gated_delta_rule=dispatch("fla"), torch_recurrent_gated_delta_rule=dispatch("fla"))
    assert d.fast_path_problems(SimpleNamespace(), None, module) == []
    for bad in (dispatch("transformers"), dispatch("fla", False), None):
        module.torch_chunk_gated_delta_rule = bad
        problems = d.fast_path_problems(SimpleNamespace(), None, module)
        assert len(problems) == 1 and "torch_chunk_gated_delta_rule" in problems[0]

    def rebuild(name: str, package: str) -> Any:
        """Stand-in official decorator with the correct exported kernel name.

        :param name: FLA's exported kernel name.
        :param package: Kernel package.
        :returns: Decorator selecting the external kernel.
        """
        assert (name, package) == ("fused_recurrent_gated_delta_rule", "fla")

        def decorate(original: Any) -> Any:
            """Replace the reference implementation's wrapper.

            :param original: Wrapped reference implementation.
            :returns: A fast dispatch wrapper.
            """
            return dispatch(package)

        return decorate

    module.torch_chunk_gated_delta_rule = dispatch("fla")
    module.torch_recurrent_gated_delta_rule = dispatch("transformers", False)
    assert d.repair_recurrent_kernel(module, rebuild)
    assert d.fast_path_problems(SimpleNamespace(), None, module) == []
    assert not d.repair_recurrent_kernel(module, rebuild)
    assert not d.repair_recurrent_kernel(SimpleNamespace(), rebuild)


# ----------------------------------------------------------------------------- render, tokenize, mask (cell 4, G5)


def test_render_problems_agrees_with_the_rows_contract(tmp_path: Path) -> None:
    """The notebook's render check and text27b.rows.check_render accept and reject the same renders."""
    path = tmp_path / "chat_template.jinja"
    path.write_text(TEMPLATE, encoding="utf-8")
    msgs = chat_row(answer="  The answer.  ")["messages"]
    good = R.render_chat(msgs, template_path=path)
    assert d.render_problems(good, msgs) == []
    effort = R.render_chat(msgs, enable_thinking=True, template_path=path, check=False)
    assert d.render_problems(effort, msgs) and "Reasoning effort" in effort
    for bad in (good.replace("<think>\n\n</think>\n\n", ""), good.replace("<think>", "<think><think>", 1), good + "x"):
        with pytest.raises(R.RenderError):
            R.check_render(bad, msgs)
        assert d.render_problems(bad, msgs)
    prompt, completion = d.split_rendered(good, msgs[-1]["content"])
    assert (prompt, completion) == R.split_prompt_completion(msgs, template_path=path)
    with pytest.raises(ValueError):
        d.split_rendered(good, "another answer")


def test_prompt_token_count_and_the_mask_guardrail() -> None:
    """Offsets split prompt from answer; the guardrail accepts the right mask and names each wrong one."""
    tok = FakeTok()
    msgs = chat_row(answer="אמת and truth")["messages"]
    text = tok.apply_chat_template(msgs, enable_thinking=False)
    prompt, completion = d.split_rendered(text, msgs[-1]["content"])
    enc = tok(text, return_offsets_mapping=True)
    n = d.prompt_token_count(enc["offset_mapping"], len(prompt))
    ids = enc["input_ids"]
    assert tok.decode(ids[:n]) == prompt and tok.decode(ids[n:]) == completion
    labels = [-100] * n + ids[n:]
    think = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    assert d.supervised_problems(ids, labels, n, prompt, completion, tok.decode, think) == []
    assert d.supervised_problems(ids, [-100] * (n - 1) + ids[n - 1:], n - 1, prompt, completion, tok.decode, think)  # think block supervised
    assert "labels mask" in d.supervised_problems(ids, [-100] * len(ids), n, prompt, completion, tok.decode, think)[0]
    with pytest.raises(ValueError, match="straddles"):
        d.prompt_token_count([(0, 3), (3, 8), (8, 9)], 5)
    with pytest.raises(ValueError, match="straddles"):
        d.prompt_token_count([(0, 3), (5, 6), (3, 4)], 4)


def test_prepared_row_problems_reads_a_right_padded_row() -> None:
    """G7 on one padded batch row."""
    tok = FakeTok()
    msgs = chat_row()["messages"]
    text = tok.apply_chat_template(msgs, enable_thinking=False)
    prompt, completion = d.split_rendered(text, msgs[-1]["content"])
    enc = tok(text, return_offsets_mapping=True)
    ids, n = enc["input_ids"], d.prompt_token_count(enc["offset_mapping"], len(prompt))
    labels = [-100] * n + ids[n:]
    pad = [0, 0, 0]
    assert d.prepared_row_problems(ids + pad, labels + [-100] * 3, [1] * len(ids) + pad, ids, n, prompt, completion, tok.decode) == []
    assert d.prepared_row_problems(pad + ids, [-100] * 3 + labels, pad + [1] * len(ids), ids, n, prompt, completion, tok.decode)
    assert "not the row's cell-4 tokens" in d.prepared_row_problems(ids[:-1], labels[:-1], [1] * (len(ids) - 1), ids, n, prompt,
                                                                     completion, tok.decode)[0]


def _run_cell4(colab: Dict[str, Any], train_rows: List[Dict[str, Any]], dev_rows: List[Dict[str, Any]],
               tok: Optional[FakeTok] = None, max_seq: int = 6144) -> Dict[str, Any]:
    """Execute cell 4 with a stand-in tokenizer (torch and numpy are real).

    :param colab: The Colab edition.
    :param train_rows: Train rows.
    :param dev_rows: Dev-monitor rows.
    :param tok: Tokenizer stand-in.
    :param max_seq: MAX_SEQ.
    :returns: The namespace after the cell.
    """
    import torch
    ns: Dict[str, Any] = {"tokenizer": tok or FakeTok(), "torch": torch, "Counter": Counter, "train_rows": train_rows,
                          "dev_rows": dev_rows, "MAX_SEQ": max_seq}
    exec(TYPING + _src(colab, 4), ns)
    return ns


def test_cell4_renders_tokenizes_masks_and_collates(editions: Tuple, capsys: pytest.CaptureFixture) -> None:
    """The collator cell runs end to end: G5 passes, labels mask exactly the prompt, the batch is right-padded."""
    _, colab, _ = editions
    train = [chat_row("translate_en", i, answer="truth " * (i + 1)) for i in range(5)] + [chat_row("translate_he", 9, answer="אמת")]
    dev = [chat_row("translate_en", 50, split="dev"), chat_row("catalogue_json", 51, split="dev", answer='{"a": 1}')]
    ns = _run_cell4(colab, train, dev)
    assert "G5 OK: 6 rows rendered" in capsys.readouterr().out
    ds, collate = ns["train_ds"], ns["collate"]
    assert sorted(ns["dev_sets"]) == ["catalogue_json", "translate_en"] and ds.column("family")[-1] == "translate_he"
    item = ds[0]
    n = ds.n_prompt[0]
    assert (item["labels"][:n] == -100).all() and (item["labels"][n:] == item["input_ids"][n:]).all()
    assert ns["decode"](item["input_ids"][n:].tolist()) == "truth<|im_end|>\n"
    batch = collate([ds[0], ds[4]])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    short = len(ds[0]["input_ids"])
    assert int(batch["attention_mask"][0].sum()) == short and (batch["labels"][0, short:] == -100).all()
    assert (batch["input_ids"][0, short:] == ns["PAD_ID"]).all()


def test_trl_dataset_preserves_row_order_and_response_masks(editions: Tuple) -> None:
    """TRL's supported dataset interface preserves the exact tensors used by the dataloader gates.

    :param editions: Source and derived notebook editions.
    """
    from datasets import Dataset
    _, colab, _ = editions
    ns = _run_cell4(colab, [chat_row("translate_en", 1), chat_row("translate_he", 2, answer="אמת")], [])
    ns["_HFDataset"] = Dataset
    node = next(node for node in ast.parse(_src(colab, 8)).body if isinstance(node, ast.FunctionDef) and node.name == "trl_dataset")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "trl_dataset", "exec"), ns)
    converted = ns["trl_dataset"](ns["train_ds"])
    assert isinstance(converted, Dataset) and len(converted) == len(ns["train_ds"])
    for i in range(len(converted)):
        for key in ("input_ids", "labels"):
            assert ns["torch"].equal(converted[i][key], ns["train_ds"][i][key])
    actual = ns["collate"]([converted[i] for i in range(len(converted))])
    expected = ns["collate"]([ns["train_ds"][i] for i in range(len(converted))])
    assert all(ns["torch"].equal(actual[key], expected[key]) for key in expected)


def test_training_session_plan_executes_with_the_shared_helper(editions: Tuple) -> None:
    """The generated session-plan call binds its row order once and preserves schedule metadata.

    :param editions: Source and derived notebook editions.
    """
    from src.finetuning.qwen_hebrew.train_session import build_session_plan, order_fingerprint
    _, colab, _ = editions
    node = next(node for node in ast.parse(_src(colab, 8)).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "_plan" for target in node.targets))
    ns = dict(_order=[2, 0, 1], train_ds=[None] * 3, build_session_plan=build_session_plan,
              _a=SimpleNamespace(max_steps=15, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                                 learning_rate=1e-4, lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_8bit", seed=3407),
              DATA_REPO=d.DATA_REPO, DATA_REVISION=SHA, MIXTURE_FINGERPRINT=FP, BASE_MODEL=d.BASE_MODEL,
              LORA_RANK=16, LORA_ALPHA=16, LORA_REGEX=d.LORA_REGEX, MAX_SEQ=6144)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "session_plan", "exec"), ns)
    assert ns["_plan"]["order_sha256"] == order_fingerprint([2, 0, 1])
    assert ns["_plan"]["order_description"] == "mixture file order" and ns["_plan"]["max_steps"] == 15


def test_cell4_refuses_thinking_renders_and_overlong_rows(editions: Tuple) -> None:
    """G5: a template that ignores enable_thinking, or a row over MAX_SEQ, stops the notebook."""
    _, colab, _ = editions
    train = [chat_row("translate_en", i) for i in range(3)]
    with pytest.raises(AssertionError, match="G5 render contract FAILED"):
        _run_cell4(colab, train, [], tok=FakeTok(honour_thinking=False))
    with pytest.raises(AssertionError, match="rows over MAX_SEQ"):
        _run_cell4(colab, train, [], max_seq=40)
    with pytest.raises(AssertionError, match="no enable_thinking switch"):
        _run_cell4(colab, train, [], tok=FakeTok(template=TEMPLATE.replace("enable_thinking", "False")))


# ----------------------------------------------------------------------------- G6, G9, milestones, alarms, export


def test_grad_group_problems_needs_every_language_tensor() -> None:
    """G6: all tensors of each language group carry a gradient; vision / MTP gradients are refused."""
    names = [n + ".lora_B.default.weight" for n in d.lora_targets(_hybrid_names(8))]
    expected = d.expected_lora_counts(_layer_types(8))
    summary, problems = d.grad_group_problems([(n, 0.5) for n in names], expected)
    assert problems == [] and summary["mlp"][0] == 24 and summary["deltanet"][1] > 0
    entries = [(n, 0.5) for n in names[1:]] + [(names[0], 0.0)]
    assert d.grad_group_problems(entries, expected)[1] == [f"deltanet: {expected['deltanet'] - 1} of {expected['deltanet']} "
                                                          "lora_B tensors got a non-zero gradient"]
    entries = [(n, 0.5) for n in names] + [("base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight", 0.1)]
    assert "vision lora_B tensors got a gradient" in d.grad_group_problems(entries, expected)[1][0]
    assert d.grad_group_problems([(n, None) for n in names], expected)[1]


def test_milestones_bench_and_alarm_helpers() -> None:
    """Milestones per arm and rung, the geometry pick, the GO number, the 2x alarm."""
    assert d.milestone_steps(388, 50, 100) == [100, 200, 300, 388]
    assert d.milestone_steps(1227, 50, 250, (0.5,)) == [250, 500, 600, 750, 1000, 1227]
    assert d.milestone_steps(15, 10, 10) == [10, 15]
    with pytest.raises(ValueError):
        d.milestone_steps(100, 50, 75)
    assert d.pick_geometry({(1, 16): 60.0, (2, 8): 50.0, (4, 4): None}) == (2, 8)
    assert d.pick_geometry({(1, 16): 50.0, (2, 8): 50.0}) == (1, 16)
    with pytest.raises(RuntimeError):
        d.pick_geometry({(1, 16): None})
    assert d.go_estimate(36.0, 400, 1.99, 1.5) == {"train_h": 4.0, "total_h": 5.5, "usd": 10.95}
    assert d.loss_alarm(2.1, 1.0) and not d.loss_alarm(2.0, 1.0) and not d.loss_alarm(9.0, None) and not d.loss_alarm(9.0, 0.0)
    assert d.family_quota({"translate_en": 400, "analyse_document": 7}) == {"translate_en": 32, "analyse_document": 7}


def _callbacks(colab: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Execute the training cell's callback classes (milestones, alarm, kill) with stand-ins for the Hub and W&B.

    :param colab: The Colab edition.
    :param extra: Namespace entries.
    :returns: The namespace.
    """
    from transformers import TrainerCallback
    c8 = _src(colab, 8)
    ns: Dict[str, Any] = {"TrainerCallback": TrainerCallback, "loss_alarm": d.loss_alarm, "RUN_NAME": "genizah_t27a",
                          "wandb": SimpleNamespace(run=None)}
    ns.update(extra)
    exec(_block(c8, "class MilestonePushCallback(TrainerCallback):", "def add_callbacks("), ns)
    return ns


def test_training_callbacks_confirm_milestones_alarm_and_kill(editions: Tuple, capsys: pytest.CaptureFixture) -> None:
    """Milestones are confirmed only at their steps; the alarm fires once at 2x; the kill saves and stops at its step."""
    from transformers import TrainerControl, TrainerState
    _, colab, _ = editions
    confirmed: List[int] = []
    ns = _callbacks(colab, ensure_hub_checkpoint=lambda repo, out, step: confirmed.append(step))
    cb = ns["MilestonePushCallback"](SimpleNamespace(push_in_progress=None), "repo", "out", [10, 15])
    for step in (5, 10, 10, 12, 15):
        cb.on_save(None, TrainerState(global_step=step), TrainerControl())
    assert confirmed == [10, 15]
    alarm = ns["LossAlarmCallback"]({"translate_en": 1.0, "catalogue_json": 2.0}, ("translate_en", "translate_he"), 2.0)
    assert alarm.train_reference == 1.0
    alarm.on_evaluate(None, TrainerState(global_step=100), TrainerControl(), metrics={"eval_translate_en_loss": 2.5, "eval_catalogue_json_loss": 3.0})
    alarm.on_log(None, TrainerState(global_step=110), TrainerControl(), logs={"loss": 2.2})
    alarm.on_log(None, TrainerState(global_step=120), TrainerControl(), logs={"loss": 2.3})
    out = capsys.readouterr().out
    assert out.count("LOSS ALARM") == 2 and "dev translate_en loss 2.500" in out and "train loss 2.200" in out
    kill = ns["KillAtStepCallback"](10)
    control = kill.on_step_end(None, TrainerState(global_step=9), TrainerControl())
    assert not control.should_training_stop
    control = kill.on_step_end(None, TrainerState(global_step=10), TrainerControl())
    assert control.should_save and control.should_training_stop


def test_merged_dir_problems() -> None:
    """G10's merge test on a merged folder listing."""
    files = ["config.json", "model-00001-of-00002.safetensors", "model.safetensors.index.json", "chat_template.jinja"]
    keys = ["model.language_model.layers.0.mlp.up_proj.weight", "model.visual.blocks.0.attn.qkv.weight"]
    cfg = {"model_type": "qwen3_5", "vision_config": {"depth": 27}}
    assert d.merged_dir_problems(files, keys, cfg, True) == []
    problems = d.merged_dir_problems(files[:1], keys[:1] + ["x.lora_A.weight"], {"model_type": "qwen3", "vision_config": {}}, False)
    assert len(problems) == 4
    assert d.merged_dir_problems(files, keys[:1], cfg, True) == ["the config has a vision tower but the merge has no vision weights"]


# ----------------------------------------------------------------------------- training and export cells


def test_training_cell_schedule_names_and_privacy(editions: Tuple) -> None:
    """The planned schedule, new private repositories, the private W&B project, the file order and the gates."""
    _, colab, _ = editions
    c8 = _src(colab, 8)
    for needle in ('"t27a": dict(run_name="genizah_t27a", ckpt_repo="isaacmg/qwen3.8-27b-genizah-t27a-ckpt", '
                   'merged_repo="isaacmg/qwen3.8-27b-genizah-t27a-merged",',
                   'run_name="genizah_t27b", ckpt_repo="isaacmg/qwen3.8-27b-genizah-t27b-ckpt"',
                   "push_every=100, milestone_fractions=()", "push_every=250, milestone_fractions=(0.5,)",
                   "SAVE_STEPS, EVAL_STEPS, PUSH_EVERY, MILESTONE_FRACTIONS = 50, 100,", "learning_rate=1e-4",
                   'lr_scheduler_type="cosine", weight_decay=0.01', '{"warmup_ratio": 0.03}', "optim='adamw_8bit', seed=3407, data_seed=3407",
                   "hub_private_repo=True", "hub_always_push=True", 'accelerator_config={"dispatch_batches": False}',
                   '"skip_prepare_dataset": True', "max_seq_length=MAX_SEQ", "wandb.init(project=WANDB_PROJECT, name=RUN_NAME",
                   "_order = list(range(len(train_ds))) * epochs_for(", 'order_description="mixture file order"',
                   "rows_sha256=MIXTURE_FINGERPRINT", 'render="enable_thinking=False"', "lock_session_plan(CKPT_REPO, _plan, _order",
                   "family_quota(Counter(train_ds.column(\"family\")), 32)", "prepared_row_problems(",
                   "abs(_lp - _ld) <= 0.05 * max(1.0, _ld)", "trainer.evaluate()", 'STEP0_FILE = "step0_dev_loss.json"',
                   "milestone_steps(MAX_STEPS, SAVE_STEPS, PUSH_EVERY, MILESTONE_FRACTIONS)",
                   "train_with_exact_resume(trainer, train_ds, _order, resume_dir, _gate, max_retries=MAX_INSESSION_RETRIES)",
                   "KillAtStepCallback(kill_at)", "discover_resume(CKPT_REPO, _resumed_out, _resumed_root)",
                   "ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)", "get_warmup_steps(MAX_STEPS) == warmup_steps_for(MAX_STEPS, 0.03)"):
        assert needle in c8, needle
    assert "materialize_order(" not in c8                                     # the mixture's own order, not a new shuffle
    code = "\n".join(hl.code_cells(colab))
    for stale in ("FastVisionModel", "pixel_values", "image_grid_thw", "MERGER_MODE", "genizah_v25a_arabic", "outputs_v25a",
                  "v25a_resume", "v24a", "qwen3-vl", "V22_REPO", "UnslothVisionDataCollator"):
        assert stale not in code, stale
    assert code.count("qwen-hebrew-finetune") == 2                             # the guard in cell 1 (comment + assert) only
    assert 'os.environ["WANDB_PROJECT"] = WANDB_PROJECT' in _src(colab, 1)


def test_export_cell_merges_only_a_finished_run_and_pushes_privately(editions: Tuple) -> None:
    """A stopped run exports nothing; rung 1 reloads and pushes a scratch merge, rung 2 only merges, the run pushes."""
    _, colab, _ = editions
    c9 = _src(colab, 9)
    assert 'if STAGE == "train" and _final_step < MAX_STEPS:' in c9
    assert 'model.push_to_hub_merged(MERGED_REPO, tokenizer, save_method="merged_16bit", private=True)' in c9
    assert 'if STAGE in ("rung1", "train"):' in c9 and 'if STAGE == "rung1":' in c9 and "merged_dir_problems(" in c9
    assert "lora_problems(_moved, LAYER_TYPES, EXPECTED_LORA_A)" in c9 and "disk_usage" in c9


def test_rung_stages_use_scratch_repositories(editions: Tuple) -> None:
    """Rungs name private scratch repositories from the tag and must start fresh."""
    _, colab, _ = editions
    c8 = _src(colab, 8)
    ns = {"ARM": "t27a", "STAGE": "rung2", "SMOKE": False, "SCRATCH_TAG": "10221530", "KILL_AT": 10}
    exec(_block(c8, "RUNS = {", "OUT_DIR = "), ns)
    assert ns["CKPT_REPO"] == "isaacmg/qwen3.8-27b-genizah-t27-rung2-10221530-ckpt" and ns["SAVE_STEPS"] == 10
    ns.update(STAGE="rung1", SMOKE=True)
    exec(_block(c8, "RUNS = {", "OUT_DIR = "), ns)
    assert ns["MERGED_REPO"] == "isaacmg/qwen3.5-2b-genizah-t27-rung1-10221530-merged" and ns["RUN_NAME"] == "genizah_t27_rung1_10221530"
    ns.update(STAGE="train", ARM="t27b")
    exec(_block(c8, "RUNS = {", "OUT_DIR = "), ns)
    assert ns["CKPT_REPO"] == d.T27B.ckpt_repo and ns["MILESTONE_FRACTIONS"] == (0.5,) and ns["PUSH_EVERY"] == 250
    assert 'assert STAGE == "train" or resume_dir is None' in c8


def test_intro_describes_the_run_and_says_the_pins_are_placeholders(editions: Tuple) -> None:
    """The intro names the plan's settings, the ladder, every gate, and that the pins are placeholders asserted at runtime."""
    _, colab, _ = editions
    intro = _src(colab, 0)
    for needle in ("The pins are exact", "G1 asserts the", "at runtime", "enable_thinking=False", "rung1", "rung2",
                   "genizah-text27b", "never `qwen-hebrew-finetune`", d.BASE_MODEL, d.SMOKE_BASE_MODEL, "LR 1e-4", "3% warm-up",
                   "refuses to run", "To resume after a session ends"):
        assert needle in intro, needle
    for g in range(1, 11):
        assert f"| G{g} |" in intro, g


#: Absolute directories the notebooks may name: the Colab work dir and system dirs (rule 3: no machine paths).
RUNTIME_ROOTS = {"content", "dev", "bin", "tmp", "usr"}
#: An absolute directory (``/root/…``) in lower-cased text, not a URL, ``a/b`` or ``{x}/y``.
ABS_DIR = re.compile(r"(?<![\w.~{}:/\\-])/([a-z][\w.-]*)/")


def test_no_restricted_attribution_machine_paths_or_secrets(editions: Tuple) -> None:
    """Nothing names restricted sources, machine paths, hosts or index names; no token-like strings.

    :param editions: The derived notebooks.
    """
    _, colab, lam = editions
    texts = [json.dumps(colab, ensure_ascii=False).lower(), json.dumps(lam, ensure_ascii=False).lower(),
             Path(d.__file__).read_text(encoding="utf-8").lower(), d.setup_script().lower()]
    for text in texts:
        names = [codecs.decode(x, "rot_13") for x in ("sevrqoret", "swzf", "swc", "travmnu.bet")]  # F12: never spelled
        for bad in names + ["localhost", ":9200", "genizah_merged", "bibliography_"]:
            assert bad not in text, bad
        assert set(ABS_DIR.findall(text)) <= RUNTIME_ROOTS, sorted(set(ABS_DIR.findall(text)) - RUNTIME_ROOTS)
        assert not re.search(r"hf_[a-z0-9]{20,}", text)
    # positive control: a synthetic local model folder is caught, the Colab work dir and URLs are not
    local = json.dumps({"tok": "/fake_mount/fake-model-store/tokenizer.json"}).lower()
    assert set(ABS_DIR.findall(local)) - RUNTIME_ROOTS == {"fake_mount"}
    assert set(ABS_DIR.findall('local = "/content/x/y"; url = "https://example.test/a/b"')) <= RUNTIME_ROOTS


# ----------------------------------------------------------------------------- inlined functions


def test_inlined_functions_are_self_contained_and_in_the_cells(editions: Tuple) -> None:
    """Every inlined function runs from its source alone (stdlib + typing) and appears in a cell verbatim."""
    _, colab, _ = editions
    ns: Dict[str, Any] = {}
    exec(TYPING + d.inline_source(*d.NOTEBOOK_FUNCTIONS), ns)
    rows = [chat_row("translate_en", i) for i in range(3)]
    assert ns["text_fingerprint"](rows) == d.text_fingerprint(rows)
    assert ns["lora_problems"]([n + ".lora_A" for n in d.lora_targets(_hybrid_names(4))], _layer_types(4), None)[1] == []
    assert ns["milestone_steps"](1227, 50, 250, (0.5,)) == d.milestone_steps(1227, 50, 250, (0.5,))
    code = "\n".join(hl.code_cells(colab))
    for f in d.NOTEBOOK_FUNCTIONS:
        assert d.inline_source(f).strip() in code, f.__name__


def test_fill_and_extract_refuse_unknown_and_ambiguous_anchors() -> None:
    """The template filler and the block extractor fail loudly."""
    assert d._fill("a «X» b", X="1") == "a 1 b"
    with pytest.raises(ValueError, match="not in the template"):
        d._fill("a", X="1")
    with pytest.raises(ValueError, match="left unfilled"):
        d._fill("«X» «Y»", X="1")
    assert d._extract("a [b] c", "[", "]", "t") == "[b"
    with pytest.raises(ValueError):
        d._extract("[a] [b]", "[", "]", "t")
    with pytest.raises(ValueError):
        d._raw_literal("it's")


# ----------------------------------------------------------------------------- pins, CLI and the committed files


def test_pins_round_trip_and_validation(editions: Tuple) -> None:
    """The pins written into a notebook read back; malformed pins are refused."""
    _, colab, lam = editions
    assert d.pins_of(colab) == d.pins_of(lam) == _pins()
    with_template = d.Pins(revisions={"t27a": SHA, "t27b": "a" * 40}, fingerprints={"t27a": FP, "t27b": "b" * 64},
                           template_sha256="e" * 64)
    assert d.pins_of(d.derive(json.loads(SRC.read_text(encoding="utf-8")), with_template)) == with_template
    for bad in (d.Pins(revisions={"t27a": "x"}), d.Pins(fingerprints={"t27a": SHA, "t27b": FP}), d.Pins(template_sha256="z")):
        with pytest.raises(ValueError):
            bad.validate()


def test_cli_writes_both_editions_and_keeps_existing_pins(tmp_path: Path, v25a: Dict[str, Any]) -> None:
    """--t27a-mixture pins the file's fingerprint; a later run keeps it; --host colab writes one file; bad pins exit."""
    mixture = tmp_path / "t27a_epoch1.jsonl"
    rows = [chat_row("translate_en", i) for i in range(4)]
    mixture.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert d.main(["--t27a-revision", SHA, "--t27a-mixture", str(mixture), "--no-template", "--fresh", "--out-dir", str(out)]) == 0
    colab = json.loads((out / d.NOTEBOOK).read_text(encoding="utf-8"))
    lam = json.loads((out / "genizah_t27_lambda.ipynb").read_text(encoding="utf-8"))
    assert (out / d.SETUP_SCRIPT).read_text(encoding="utf-8") == d.setup_script()
    pins = d.pins_of(colab)
    assert pins.revisions["t27a"] == SHA and pins.fingerprints["t27a"] == d.text_fingerprint(rows) == R.fingerprint(rows)
    assert pins.template_sha256 is None and lam == d.to_lambda(d.derive(v25a, pins), "genizah_t27_lambda.ipynb")
    assert d.main(["--t27b-revision", "a" * 40, "--no-template", "--out-dir", str(out)]) == 0
    kept = d.pins_of(json.loads((out / d.NOTEBOOK).read_text(encoding="utf-8")))
    assert kept.revisions == {"t27a": SHA, "t27b": "a" * 40} and kept.fingerprints["t27a"] == pins.fingerprints["t27a"]
    template = tmp_path / "chat_template.jinja"
    template.write_text(TEMPLATE, encoding="utf-8")
    only = tmp_path / "only"
    assert d.main(["--host", "colab", "--template", str(template), "--out-dir", str(only)]) == 0
    assert sorted(p.name for p in only.iterdir()) == [d.NOTEBOOK]
    assert d.current_pins(only / d.NOTEBOOK).template_sha256 == d.file_sha256(template)
    with pytest.raises(SystemExit):
        d.main(["--t27a-revision", "not-a-sha", "--no-template", "--out-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="not both"):
        d.main(["--t27a-fingerprint", FP, "--t27a-mixture", str(mixture), "--out-dir", str(tmp_path / "x")])


@pytest.mark.parametrize("name, lambda_edition", [(d.NOTEBOOK, False), ("genizah_t27_lambda.ipynb", True)])
def test_committed_notebooks_are_current(name: str, lambda_edition: bool, v25a: Dict[str, Any]) -> None:
    """The generated files equal a fresh derivation at the pins written into them."""
    path = COLAB_DIR / name
    if not path.exists():
        pytest.skip(f"{path} has not been generated")
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = d.derive(v25a, d.pins_of(committed))
    if lambda_edition:
        fresh = d.to_lambda(fresh, name)
        assert (COLAB_DIR / d.SETUP_SCRIPT).read_text(encoding="utf-8") == d.setup_script()
    assert committed == fresh


# ----------------------------------------------------------------------------- rung 0


def _bpe_tokenizer(tmp_path: Path) -> Path:
    """A small byte-level BPE tokenizer with the chat special tokens, saved as ``tokenizer.json``.

    :param tmp_path: Pytest temporary directory.
    :returns: Its path.
    """
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(["Translate the document. The answer is true. כתאבי אליך אמת", "You are a test."] * 20,
                            vocab_size=300, min_frequency=1, special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def test_rung0_runs_the_gates_locally(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Rung 0 on synthetic rows: clean rows pass; a long row, a schema defect and a thinking render are counted; the
    CLI exits 0 / 1."""
    template = tmp_path / "chat_template.jinja"
    template.write_text(TEMPLATE, encoding="utf-8")
    tok_path = _bpe_tokenizer(tmp_path)
    rows = [chat_row("translate_en", i, answer="The answer is true. אמת") for i in range(3)] + [chat_row("translate_he", 7)]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))

    def encode(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Encode with offsets.

        :param text: Text.
        :returns: ``(ids, offsets)``.
        """
        enc = tok.encode(text, add_special_tokens=False)
        return enc.ids, enc.offsets

    def render(msgs: Sequence[Dict[str, Any]]) -> str:
        """Render with thinking off.

        :param msgs: Messages.
        :returns: Text.
        """
        return R.render_chat(msgs, template_path=template, check=False)

    decode: Callable[[Sequence[int]], str] = lambda ids: tok.decode(list(ids), skip_special_tokens=False)
    think = [tok.token_to_id("<think>"), tok.token_to_id("</think>")]
    rep = d.rung0(rows, render, encode, decode, think, schema=("train", d.T27A.families))
    assert rep["ok"] and rep["rows"] == 4 and set(rep["tokens"]) == {"translate_en", "translate_he"}
    assert rep["supervised"]["translate_en"] < rep["tokens"]["translate_en"]
    rep = d.rung0(rows + [chat_row("catalogue_json", 8)], render, encode, decode, think, max_seq=60, schema=("train", d.T27A.families))
    assert not rep["ok"] and rep["over_max_seq"] and rep["schema_problems"]
    thinking = lambda msgs: R.render_chat(msgs, enable_thinking=True, template_path=template, check=False)
    assert d.rung0(rows, thinking, encode, decode, think)["render_problems"]
    mixture = tmp_path / "mix.jsonl"
    mixture.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    args = ["--rung0", str(mixture), "--rung0-tokenizer", str(tok_path), "--rung0-template", str(template)]
    assert d.main(args + ["--rung0-arm", "t27a"]) == 0
    assert d.main(args + ["--rung0-arm", "t27b", "--rung0-split", "dev"]) == 1          # train rows are not dev rows
    assert '"ok": true' in capsys.readouterr().out


# ----------------------------------------------------------------------------- NFC before tokenising (fix pass F3)
#: A decomposed answer (t + U+0323 ...): the real tokenizer normalises it to NFC.
DECOMPOSED = "Fusṭāṭ letter"


class NfcTok(FakeTok):
    """:class:`FakeTok` with the real tokenizer's NFC normaliser in front."""

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> Dict[str, Any]:
        """Tokenize the NFC form of the text.

        :param text: Text.
        :param add_special_tokens: Ignored.
        :param return_offsets_mapping: Return character offsets too.
        :returns: ``input_ids`` (and ``offset_mapping``).
        """
        import unicodedata
        return super().__call__(unicodedata.normalize("NFC", text), add_special_tokens, return_offsets_mapping)


def test_nfc_messages_copies_and_normalises() -> None:
    """Contents become NFC; the input turns are not modified; non-string contents pass."""
    import unicodedata
    msgs = chat_row(answer=DECOMPOSED)["messages"] + [{"role": "tool", "content": None}]
    out = d.nfc_messages(msgs)
    assert out[2]["content"] == unicodedata.normalize("NFC", DECOMPOSED) != DECOMPOSED and msgs[2]["content"] == DECOMPOSED
    assert out[3] == {"role": "tool", "content": None} and out[0] is not msgs[0]


def test_cell4_nfc_normalises_before_tokenising(editions: Tuple, capsys: pytest.CaptureFixture) -> None:
    """A decomposed answer fails the mask guardrail when the render is not normalised (the pre-fix failure) and
    passes cell 4, whose render is NFC on both sides."""
    import unicodedata
    _, colab, _ = editions
    tok = NfcTok()
    msgs = chat_row(answer=DECOMPOSED)["messages"]
    raw = tok.apply_chat_template(msgs, enable_thinking=False)
    prompt, completion = d.split_rendered(raw, msgs[-1]["content"])
    enc = tok(raw, return_offsets_mapping=True)
    n = d.prompt_token_count(enc["offset_mapping"], len(prompt))
    think = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    assert d.supervised_problems(enc["input_ids"], [-100] * n + enc["input_ids"][n:], n, prompt, completion, tok.decode, think)
    train = [chat_row("translate_en", i, answer=DECOMPOSED) for i in range(3)]
    ns = _run_cell4(colab, train, [], tok=NfcTok())
    assert "G5 OK: 3 rows rendered" in capsys.readouterr().out
    ds = ns["train_ds"]
    k = ds.n_prompt[0]
    assert ns["decode"](ds[0]["input_ids"][k:].tolist()) == unicodedata.normalize("NFC", DECOMPOSED) + "<|im_end|>\n"
    assert train[0]["messages"][2]["content"] == DECOMPOSED


def test_rung0_nfc_normalises_before_tokenising(tmp_path: Path) -> None:
    """Rung 0 renders NFC text, so a decomposed answer passes the guardrail with an NFC-normalising tokenizer."""
    import unicodedata
    template = tmp_path / "chat_template.jinja"
    template.write_text(TEMPLATE, encoding="utf-8")
    tok = NfcTok()

    def encode(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Encode the NFC text with offsets.

        :param text: Text.
        :returns: ``(ids, offsets)``.
        """
        enc = tok(text, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def render(msgs: Sequence[Dict[str, Any]]) -> str:
        """Render with thinking off.

        :param msgs: Messages.
        :returns: Text.
        """
        return R.render_chat(msgs, template_path=template, check=False)

    rows = [chat_row("translate_en", i, answer=DECOMPOSED) for i in range(3)]
    think = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    rep = d.rung0(rows, render, encode, tok.decode, think, schema=("train", d.T27A.families))
    assert rep["ok"] and not rep["guard_problems"] and rows[0]["messages"][2]["content"] == DECOMPOSED
    assert unicodedata.is_normalized("NFC", d.nfc_messages(rows[0]["messages"])[2]["content"])
