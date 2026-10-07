# File name: test_host_lambda.py
# Date: 10/6/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""The Lambda Cloud edition of a Colab notebook: the five Colab-specific lines are replaced, nothing else moves."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew.colab import derive_v23a as dv
from src.finetuning.qwen_hebrew.colab import host_lambda as hl

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
SHA = "3b762524067c77141c121ce84a3a724991618757"


@pytest.fixture(scope="module")
def pair():
    nb = json.load(open(SRC, encoding="utf-8"))
    colab = dv.derive(nb, SHA, dv.MAIN)
    return colab, hl.to_lambda(colab, "genizah_v23a_lambda.ipynb")


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_no_colab_specific_lines_remain(pair):
    _, lam = pair
    for i, c in enumerate(lam["cells"]):
        s = "".join(c["source"])
        for needle in ("google.colab", "/content", "userdata", "%pip"):
            assert needle not in s, (i, needle)


def test_secrets_come_from_env_or_prompt_and_are_never_stored(pair):
    _, lam = pair
    c2, c8 = _src(lam, 2), _src(lam, 8)
    assert 'login(token=os.environ.get("HF_TOKEN") or getpass("Hugging Face token (write access): "))' in c2
    assert "from getpass import getpass" in c2 and "import os" in c2
    assert 'wandb.login(key=os.environ.get("WANDB_API_KEY") or getpass("W&B API key: "))' in c8
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(lam))


def test_paths_live_under_the_work_dir(pair):
    _, lam = pair
    assert 'LOCAL = os.path.expanduser("~/genizah_work/genizah_v23a_arabic")' in _src(lam, 2)
    assert 'RESUME_ROOT = os.path.expanduser("~/genizah_work/v23a_resume")' in _src(lam, 8)
    assert 'OUT_DIR = "outputs_v23a"' in _src(lam, 8)                      # resume root stays outside output_dir


def test_time_budget_is_a_safety_stop(pair):
    colab, lam = pair
    assert "TIME_BUDGET_H = 7.5" in _src(colab, 7)
    assert f"TIME_BUDGET_H = {hl.LAMBDA_TIME_BUDGET_H}" in _src(lam, 7) and hl.LAMBDA_TIME_BUDGET_H >= 30
    assert "TIME_BUDGET_H = 7.5" not in _src(lam, 7)


def test_install_cell_gates_the_kernel_pins(pair):
    colab, lam = pair
    c1 = _src(lam, 1)
    assert "LAMBDA_KERNEL_PINS = " in c1 and "lambda_setup.sh" in c1
    for name in hl.KERNEL_GATE:
        assert f"'{name}': '{hl.PINS[name]}'" in c1
    assert 'assert version("transformers") == "4.57.6"' in c1               # the Colab gates stay
    assert "GPU: {_p.name}" in c1
    assert "%pip install" in _src(colab, 1) and "%pip" not in c1


def test_everything_else_is_untouched(pair):
    colab, lam = pair
    for i in (3, 4, 5, 6, 9):
        assert _src(colab, i) == _src(lam, i), i
    diff7 = [l for l in _src(colab, 7).splitlines() if l not in _src(lam, 7).splitlines()]
    assert len(diff7) == 1 and diff7[0].startswith("TIME_BUDGET_H = 7.5")
    diff8 = [l for l in _src(colab, 8).splitlines() if l not in _src(lam, 8).splitlines()]
    assert sorted(diff8) == sorted(['wandb.login(key=userdata.get("WANDB_API_KEY"))', 'RESUME_ROOT = "/content/v23a_resume"'])
    assert len(lam["cells"]) == len(colab["cells"]) + 1


def test_wrapup_cell_finishes_wandb_and_says_terminate(pair):
    _, lam = pair
    last = _src(lam, len(lam["cells"]) - 1)
    assert "wandb.finish()" in last and "TERMINATE THE INSTANCE" in last
    assert 'wandb.alert(title="genizah_v23a: notebook finished"' in last


def test_every_code_cell_parses(pair):
    _, lam = pair
    for s in hl.code_cells(lam):
        ast.parse(s)


def test_launch_notes_lead_the_intro(pair):
    colab, lam = pair
    intro = _src(lam, 0)
    assert intro.startswith("# Genizah v2.3a — first Arabic-script run — **Lambda Cloud edition**")
    for needle in ("bash lambda_setup.sh", "nbconvert", "genizah_v23a_lambda.ipynb", "terminate the instance", "H100 PCIe", "GH200"):
        assert needle in intro, needle
    assert intro.endswith(_src(colab, 0))                                   # the Colab intro follows, unchanged


def test_setup_script_pins_the_colab_stack():
    sh = hl.setup_script()
    assert sh.startswith("#!/bin/bash") and "set -euo pipefail" in sh
    for name, ver in hl.PINS.items():
        assert f'"{name}=={ver}"' in sh, name
    assert f'"torch=={hl.TORCH_VERSION}"' in sh and f'"torchvision=={hl.TORCHVISION_VERSION}"' in sh
    assert "cu130" in sh and "cu128" in sh and "-ge 580" in sh              # CUDA build follows the driver
    assert "ipykernel install --user --name genizah" in sh
    assert f"uv venv --python {hl.PYTHON}" in sh and hl.PYTHON == "3.13" and "astral.sh/uv/install.sh" in sh
    assert "x86_64" in sh and "xformers" not in sh
    assert set(hl.KERNEL_GATE) <= set(hl.PINS)


def test_refuses_a_notebook_without_the_colab_lines(pair):
    colab, _ = pair
    broken = json.loads(json.dumps(colab))
    broken["cells"][2]["source"] = [l for l in broken["cells"][2]["source"] if "userdata.get" not in l]
    with pytest.raises(ValueError):
        hl.to_lambda(broken, "x.ipynb")


def test_lambda_files_naming():
    assert hl.lambda_files("genizah_v23a.ipynb") == ("genizah_v23a_lambda.ipynb", "lambda_setup.sh")
    assert hl.lambda_files("genizah_v23a_r64.ipynb")[0] == "genizah_v23a_r64_lambda.ipynb"


def test_cli_writes_the_edition_and_the_script(tmp_path):
    out = tmp_path / "genizah_v23a.ipynb"
    subprocess.run([sys.executable, "-m", "src.finetuning.qwen_hebrew.colab.derive_v23a", "--revision", SHA, "--host", "lambda",
                    "--out", str(out)], check=True, capture_output=True)
    nb = json.loads((tmp_path / "genizah_v23a_lambda.ipynb").read_text(encoding="utf-8"))
    assert not out.exists() and (tmp_path / "lambda_setup.sh").exists()
    assert "google.colab" not in json.dumps(nb) and "/content" not in json.dumps(nb)
    assert "TIME_BUDGET_H = 48.0" in _src(nb, 7)
