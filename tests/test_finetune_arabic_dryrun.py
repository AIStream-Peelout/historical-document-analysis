# File name: test_finetune_arabic_dryrun.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Dry runs of ``src/finetuning/kraken/finetune_arabic.sh`` with fake ``docker``, ``lms`` and venv ``python``.

Nothing real runs: the fake executables live under ``tmp_path`` and only echo their arguments (the fake python also
ticks for a few seconds so the LM Studio guard has something to pause), ``DATA`` and ``AR_DIR`` are temporary
directories and ``EXTRA_RO`` is empty, so neither Docker, LM Studio nor the NAS is touched.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "src" / "finetuning" / "kraken" / "finetune_arabic.sh"
KRAKEN_DIR = DRIVER.parent

pytestmark = pytest.mark.skipif(not Path("/bin/bash").exists() or shutil.which("perl") is None,
                                reason="needs /bin/bash and perl")

FAKE_DOCKER = """#!/bin/bash
# Fake docker: never contacts a daemon.
case $1 in
  version|image|ps) exit 0 ;;
  run) echo "FAKE-RUN $*"
       case "$*" in *" test "*) echo "  91.25%  Character Accuracy"; echo "  60.00%  Word Accuracy" ;; esac
       exit 0 ;;
  *) echo "fake docker: unexpected $*" >&2; exit 99 ;;
esac
"""

FAKE_PYTHON = """#!/bin/bash
# Fake venv python: echoes its arguments, working directory and two exported variables, then ticks.
echo "FAKE-PY cwd=$(pwd) KETOS_FROZEN_BN=${KETOS_FROZEN_BN-unset} pgid=$(ps -o pgid= -p $$ | tr -d ' ') pid=$$ args: $*"
case "$*" in *" test "*) echo "  92.50%  Character Accuracy"; echo "  61.00%  Word Accuracy" ;; esac
for ((i = 1; i <= ${FAKE_TICKS:-0}; i++)); do sleep 1; echo "tick $i"; done
exit 0
"""

FAKE_LMS = """#!/bin/bash
# Fake lms: pops the next state (idle | busy | fail) from $FAKE_LMS_SEQ; the last one repeats.
seq=${FAKE_LMS_SEQ:?}
st=$(head -n 1 "$seq")
if (( $(wc -l < "$seq") > 1 )); then tail -n +2 "$seq" > "$seq.t" && mv "$seq.t" "$seq"; fi
[[ $st == fail ]] && exit 1
s2=IDLE; [[ $st == busy ]] && s2=PROCESSINGPROMPT
echo
echo "IDENTIFIER          MODEL               STATUS    SIZE        CONTEXT    PARALLEL    DEVICE    TTL"
echo "qwen/qwen3-4b-2507  qwen/qwen3-4b-2507  IDLE      2.28 GB     32768      4           Local"
echo "qwen/qwen3.8-27b    qwen/qwen3.8-27b    $s2      16.08 GB    32768      4           Local"
echo
"""


def write_exe(path: Path, text: str) -> None:
    """Write an executable script.

    :param path: Destination file.
    :type path: Path
    :param text: Script text.
    :type text: str
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path: Path) -> Dict[str, Path]:
    """Fake executables plus a compiled-looking dataset (arrows, manifests, gate dumps newer than the arrows).

    :param tmp_path: pytest temporary directory.
    :type tmp_path: Path
    :return: Paths: ``bin``, ``python``, ``lms``, ``seq``, ``data``, ``ar``, ``tmp``.
    :rtype: Dict[str, Path]
    """
    bin_dir, venv_bin = tmp_path / "bin", tmp_path / "venv" / "bin"
    write_exe(bin_dir / "docker", FAKE_DOCKER)
    write_exe(bin_dir / "lms", FAKE_LMS)
    write_exe(venv_bin / "python", FAKE_PYTHON)
    (venv_bin / "ketos").write_text("# fake console script\n", encoding="utf-8")
    data, ar, tmp = tmp_path / "data", tmp_path / "ar", tmp_path / "tmp"
    for d in (data / "arrow", data / "models", ar, tmp):
        d.mkdir(parents=True)
    (ar / "ppocrv6_medium.safetensors").touch()
    for split in ("train", "val"):
        lines: List[str] = []
        for name in ("agapet", "baybars"):
            arrow = data / "arrow" / f"{name}_{split}.arrow"
            arrow.touch()
            os.utime(arrow, (1_000_000_000, 1_000_000_000))
            lines.append(f"/data/arrow/{name}_{split}.arrow")
            if split == "train":
                (data / "gate" / f"{name}_train").mkdir(parents=True)
                (data / "gate" / f"{name}_train" / "index.tsv").touch()
        (data / f"{split}.manifest").write_text("\n".join(lines) + "\n", encoding="utf-8")
    seq = tmp_path / "lms_seq"
    seq.write_text("idle\n", encoding="utf-8")
    return {"bin": bin_dir, "python": venv_bin / "python", "lms": bin_dir / "lms", "seq": seq, "data": data,
            "ar": ar, "tmp": tmp}


def run_driver(box: Dict[str, Path], stage: str, **env: str) -> subprocess.CompletedProcess:
    """Run the driver under ``/bin/bash`` (3.2 on macOS) with the fakes first on PATH.

    :param box: The ``sandbox`` fixture.
    :type box: Dict[str, Path]
    :param stage: Driver stage.
    :type stage: str
    :param env: Extra environment settings (knobs).
    :return: The finished process (stdout and stderr merged into ``stdout``).
    :rtype: subprocess.CompletedProcess
    """
    base = {k: v for k, v in os.environ.items() if k not in {
        "ENGINE", "DEVICE", "FROZEN_BN", "NO_GUARD", "GUARD_POLL", "GUARD_IDLE", "MEM", "NAME", "MODEL", "RESUME",
        "BASE", "QUIT", "KETOS_FROZEN_BN"}}
    base.update({"PATH": f"{box['bin']}:{os.environ.get('PATH', '/usr/bin:/bin')}", "DATA": str(box["data"]),
                 "AR_DIR": str(box["ar"]), "EXTRA_RO": "", "LMS": str(box["lms"]), "VENV_PYTHON": str(box["python"]),
                 "TMPDIR": str(box["tmp"]), "FAKE_LMS_SEQ": str(box["seq"])})
    base.update(env)
    return subprocess.run(["/bin/bash", str(DRIVER), stage], env=base, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=120)


def test_docker_train_runs_the_frozen_bn_wrapper_by_default(sandbox: Dict[str, Path]) -> None:
    """FROZEN_BN defaults to 1: the container runs the wrapper from the read-only /work mount."""
    out = run_driver(sandbox, "train", NAME="d1").stdout
    assert f"-v {KRAKEN_DIR}:/work:ro" in out
    assert "kraken-train:k711 python /work/ketos_frozen_bn.py -d cpu --threads 4 --workers 0 -v train" in out
    assert "engine=docker device=cpu" in out and "frozen_bn=1" in out


def test_docker_train_plain_ketos_with_frozen_bn_0(sandbox: Dict[str, Path]) -> None:
    """FROZEN_BN=0 keeps the old container command line."""
    out = run_driver(sandbox, "train", NAME="d0", FROZEN_BN="0").stdout
    assert ("kraken-train:k711 ketos -d cpu --threads 4 --workers 0 -v train -f binary "
            "-i /ar/ppocrv6_medium.safetensors --resize union -t /data/train.manifest -e /data/val.manifest "
            "-o /data/models/d0 ") in out


@pytest.mark.parametrize("mem", ["8g", "256m", "0", "abc"])
def test_mem_refused_in_docker_but_not_checked_natively(sandbox: Dict[str, Path], mem: str) -> None:
    """The docker MEM cap stays; native mode has no --memory cap and skips the check."""
    docker = run_driver(sandbox, "baseline", MEM=mem)
    assert docker.returncode == 1 and f"MEM={mem} refused" in docker.stdout
    native = run_driver(sandbox, "baseline", MEM=mem, ENGINE="native")
    assert native.returncode == 0 and f"MEM={mem} refused" not in native.stdout


@pytest.mark.parametrize("stage", ["compile", "gate"])
def test_native_refuses_compile_and_gate(sandbox: Dict[str, Path], stage: str) -> None:
    """compile and gate stay docker-only."""
    res = run_driver(sandbox, stage, ENGINE="native")
    assert res.returncode == 1 and "ENGINE=native runs baseline, train and test only" in res.stdout


def test_native_train_translates_paths_runs_the_module_and_guards(sandbox: Dict[str, Path]) -> None:
    """Host paths, the wrapper module from the repo root, its own process group, a guard that starts and ends."""
    data, ar = sandbox["data"], sandbox["ar"]
    res = run_driver(sandbox, "train", ENGINE="native", NAME="n1", FAKE_TICKS="2", GUARD_POLL="1")
    out = res.stdout
    assert res.returncode == 0, out
    assert (f"-m src.finetuning.kraken.ketos_frozen_bn -d mps --threads 4 --workers 0 -v train -f binary "
            f"-i {ar}/ppocrv6_medium.safetensors --resize union -t {data}/native/train.manifest "
            f"-e {data}/native/val.manifest -o {data}/models/n1 ") in out
    assert (data / "native" / "train.manifest").read_text(encoding="utf-8").split() == [
        f"{data}/arrow/agapet_train.arrow", f"{data}/arrow/baybars_train.arrow"]
    fake = next(line for line in out.splitlines() if line.startswith("FAKE-PY"))
    assert f"cwd={REPO} " in fake and "KETOS_FROZEN_BN=1" in fake
    fields = dict(f.split("=", 1) for f in fake.split(" args:")[0].split()[1:])
    assert fields["pgid"] == fields["pid"]  # its own process group
    log = (data / "models" / "train_n1.log").read_text(encoding="utf-8")
    assert "guard on: process group" in log and "guard off: process group" in log
    assert not list((data / "models").glob("native_*.pid")) and not list(sandbox["tmp"].iterdir())


def test_native_guard_pauses_while_lm_studio_is_busy(sandbox: Dict[str, Path]) -> None:
    """A busy model stops the process group; idle for GUARD_IDLE s resumes it; both are logged."""
    sandbox["seq"].write_text("idle\nbusy\nbusy\nidle\n", encoding="utf-8")
    res = run_driver(sandbox, "train", ENGINE="native", NAME="n2", FAKE_TICKS="2", GUARD_POLL="1", GUARD_IDLE="1")
    assert res.returncode == 0, res.stdout
    log = (sandbox["data"] / "models" / "train_n2.log").read_text(encoding="utf-8")
    assert "qwen/qwen3.8-27b=PROCESSINGPROMPT -> SIGSTOP process group" in log
    assert "-> SIGCONT process group" in log and "after 1 pause(s)" in log


def test_native_refuses_without_readable_lms_unless_no_guard(sandbox: Dict[str, Path]) -> None:
    """lms ps must be readable before a guarded start; NO_GUARD=1 runs unguarded with a warning in the log."""
    sandbox["seq"].write_text("fail\n", encoding="utf-8")
    res = run_driver(sandbox, "baseline", ENGINE="native")
    assert res.returncode == 1 and "lms ps unreadable before the start" in res.stdout
    res = run_driver(sandbox, "baseline", ENGINE="native", NO_GUARD="1")
    assert res.returncode == 0
    log = (sandbox["data"] / "models" / "baseline_ar_ppocrv6m_v1.log").read_text(encoding="utf-8")
    assert "NO_GUARD=1 - LM Studio guard OFF" in log and "guard on" not in log


def test_native_refuses_while_another_native_run_is_live(sandbox: Dict[str, Path]) -> None:
    """A pid file with a live pid blocks native stages and a docker compile; a stale one is removed."""
    pidfile = sandbox["data"] / "models" / "native_train_other.pid"
    with subprocess.Popen(["sleep", "30"]) as live:
        pidfile.write_text(f"{live.pid}\n", encoding="utf-8")
        res = run_driver(sandbox, "baseline", ENGINE="native")
        assert res.returncode == 1 and "a native run is active" in res.stdout
        live.kill()
    res = run_driver(sandbox, "baseline", ENGINE="native")
    assert res.returncode == 0 and "removing stale" in res.stdout and not pidfile.exists()


def test_results_rows_record_engine_and_frozen_bn(sandbox: Dict[str, Path]) -> None:
    """results.tsv rows end with ENGINE:DEVICE and the FROZEN_BN the model's run recorded ('-' for baseline)."""
    data = sandbox["data"]
    assert run_driver(sandbox, "train", ENGINE="native", NAME="r1").returncode == 0
    (data / "models" / "r1").mkdir(exist_ok=True)
    (data / "models" / "r1" / "best_0.95.safetensors").touch()
    assert run_driver(sandbox, "test", ENGINE="native", NAME="r1").returncode == 0
    assert run_driver(sandbox, "baseline").returncode == 0
    rows = [line.split("\t") for line in (data / "models" / "results.tsv").read_text(encoding="utf-8").splitlines()]
    assert rows[0][1:] == ["test", "/data/models/r1/best_0.95.safetensors", "val.manifest", "92.50%", "61.00%",
                           "native:mps", "1"]
    assert rows[1][1:] == ["baseline", "/ar/ppocrv6_medium.safetensors", "val.manifest", "91.25%", "60.00%",
                           "docker:cpu", "-"]
