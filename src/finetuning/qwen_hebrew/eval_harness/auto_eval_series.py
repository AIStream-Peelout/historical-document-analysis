# File name: auto_eval_series.py
# Date: 9/11/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Unattended checkpoint evaluation for one fine-tune series (built on hard_eval_ckpt.sh).

Polls the series' hub checkpoint repo, maps each pushed step to its commit SHA (the rolling
``last-checkpoint/`` folder is overwritten, the commits are not), and for every target step runs,
strictly one checkpoint at a time:

    stage (merge on NAS -> MLX 8-bit -> LM Studio dir)
    box evals (grounding trio results+preds, box_quality vs v2.0a, layout-QA, 10 site pages)
    lite CER (flip slice) — or the FULL hard evals (religious-140 + PGP-131 + compare) on full steps
    cleanup (unload our candidate, delete its local copies; flagship copies are never touched)

Every stage is gated on local disk, RAM headroom, the NAS mount and LM Studio reachability; a failed
gate waits and retries instead of proceeding. Nothing else in LM Studio or Docker is touched.
If the run stalls (no new push for ``--stale-hours``) and no full eval exists yet, the newest
evaluated step >= ``--fallback-min-step`` gets the full evals so a table exists on return.

Usage (repo root, under nohup; logs in logs/):
    .venv/bin/python -m src.finetuning.qwen_hebrew.eval_harness.auto_eval_series \\
        --ver v21b --box-steps 600 900 1200 1500 1800 --full-steps 1800 2000
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
H = REPO / "src/finetuning/qwen_hebrew/eval_harness"
G = REPO / "src/datasets/evaluations/grounding_eval"
OUT = REPO / "src/datasets/raw_data/cairo_genizah/ai_reads"
LOGS = REPO / "logs"
NAS = Path("/Volumes/home/studio_offload")
PY = str(REPO / ".venv/bin/python")
REFERENCE = "qwen3-vl-8b-heb-v20a-step1800"
KEEP_LOCAL = {"qwen3-vl-8b-heb-v19a-step1300", REFERENCE}   # flagship rule: never delete these copies


def log(msg: str) -> None:
    """Timestamped line to stdout (nohup redirects it to the log file).

    :param msg: Text to log.
    """
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def sh(cmd: List[str], timeout_s: int, cwd: Path = REPO) -> int:
    """Run a command, streaming its output into our log; never raises on failure.

    :param cmd: Argument vector.
    :param timeout_s: Kill after this many seconds.
    :param cwd: Working directory.
    :returns: Exit code (124 on timeout, 125 on launch error).
    """
    log("$ " + " ".join(cmd))
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout_s, stdout=sys.stdout, stderr=subprocess.STDOUT)
        return p.returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT after {timeout_s}s: {cmd[0]}")
        return 124
    except OSError as e:
        log(f"launch error: {e}")
        return 125


def free_disk_gb() -> int:
    """Free space on the local data volume in GiB."""
    st = os.statvfs("/System/Volumes/Data")
    return int(st.f_bavail * st.f_frsize / 2**30)


def free_ram_pct() -> Optional[int]:
    """System-wide free memory percentage from ``memory_pressure`` (None if unavailable)."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def lm_studio_up() -> bool:
    """True when the LM Studio server answers on :1234."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — any failure means "not reachable" for a gate
        return False


def gates_ok(min_disk_gb: int, min_ram_pct: int) -> Tuple[bool, str]:
    """Check every safety gate; returns (ok, reason).

    :param min_disk_gb: Required free local disk.
    :param min_ram_pct: Required free RAM percentage.
    :returns: Tuple of pass flag and human-readable state.
    """
    disk, ram = free_disk_gb(), free_ram_pct()
    state = f"disk {disk}Gi free, RAM {ram}% free, NAS {'up' if NAS.exists() else 'DOWN'}, LM Studio {'up' if lm_studio_up() else 'DOWN'}"
    busy = subprocess.run(["pgrep", "-f", "hard_eval_ckpt.sh"], capture_output=True).returncode == 0
    ok = (disk >= min_disk_gb and (ram is None or ram >= min_ram_pct) and NAS.exists() and lm_studio_up()
          and not busy)
    return ok, state + (", another hard_eval running" if busy else "")


def checkpoint_shas(repo_id: str, token: str) -> Dict[int, str]:
    """Map pushed step -> commit SHA of the ``…, step N, checkpoint`` commit.

    :param repo_id: Hub checkpoint repo.
    :param token: HF token.
    :returns: Dict of step to 40-hex SHA.
    """
    from huggingface_hub import HfApi
    out: Dict[int, str] = {}
    for c in HfApi(token=token).list_repo_commits(repo_id):
        m = re.search(r"step (\d+), checkpoint", c.title)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = c.commit_id
    return out


def cleanup(name: str) -> None:
    """Unload our candidate and delete its local copies (never a flagship copy).

    :param name: Model name (also the LM Studio identifier).
    """
    sh([str(Path.home() / ".lmstudio/bin/lms"), "unload", name], 120)
    if name in KEEP_LOCAL:
        log(f"{name} is on the flagship list — local copies kept")
        return
    for p in (REPO / "models" / name, Path.home() / ".lmstudio/models/isaacmg" / name, H / f"{name}-bf16"):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            log(f"removed {p}")


def box_evals(name: str) -> bool:
    """Grounding trio (results + preds), box geometry vs v2.0a, layout-QA and the 10 site pages.

    :param name: Candidate model name.
    :returns: True when every step exited 0.
    """
    ok = True
    for mode in ("locate", "read_box", "grounded"):
        ok &= sh([PY, str(G / "grounding_eval.py"), "--run", mode, "--model", name], 3 * 3600) == 0
    ok &= sh([PY, str(G / "box_quality.py"), "--preds", name, REFERENCE], 600) == 0
    sh([PY, str(H / "layout_qa_eval.py"), "--model", name], 2 * 3600)
    sh([PY, "-m", "src.datasets.consensus.two_reader_lines", "--ids", str(OUT / "jobs_v20b_sample10.jsonl"),
        "--vlm-model", name, "--out", str(OUT / f"ai_reads_{name}_sample10.jsonl"),
        "--work-dir", str(OUT / f"images_{name}")], 2 * 3600)
    return ok


def process(step: int, sha: str, ver: str, full: bool, a: argparse.Namespace) -> bool:
    """Stage, evaluate and clean up one checkpoint.

    :param step: Global step.
    :param sha: Commit SHA carrying that step's ``last-checkpoint/``.
    :param ver: Series tag (e.g. ``v21b``).
    :param full: Run the full hard evals instead of the lite slice.
    :param a: Parsed CLI args (gates).
    :returns: True on success.
    """
    name = f"qwen3-vl-8b-heb-{ver}-step{step}"
    log(f"===== {name} ({'FULL' if full else 'box+lite'}) sha {sha[:10]} =====")
    try:
        rc = sh(["/bin/zsh", str(H / "hard_eval_ckpt.sh"), str(step), sha, "stage", ver], 4 * 3600)
        if rc != 0:
            log(f"stage FAILED rc={rc}")
            return False
        box_ok = box_evals(name)
        log(f"box evals {'OK' if box_ok else 'had failures'} for {name}")
        if free_disk_gb() < a.min_disk_gb - 1:
            log(f"disk {free_disk_gb()}Gi — skipping the CER eval for {name} (box results kept)")
            return box_ok
        mode = "full" if full else "lite"
        rc = sh(["/bin/zsh", str(H / "hard_eval_ckpt.sh"), str(step), sha, mode, ver], (9 if full else 3) * 3600)
        log(f"{mode} eval rc={rc} for {name}")
        return box_ok and rc == 0
    finally:
        cleanup(name)
        log(f"cleanup done for {name}: disk {free_disk_gb()}Gi free")


def main() -> int:
    """Poll the checkpoint repo and evaluate target steps in order until the run finishes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ver", required=True, help="series tag, e.g. v21b")
    ap.add_argument("--repo", default=None, help="hub checkpoint repo (default isaacmg/qwen3-vl-8b-hebrew-<ver>-ckpt)")
    ap.add_argument("--box-steps", type=int, nargs="+", default=[600, 900, 1200, 1500, 1800])
    ap.add_argument("--full-steps", type=int, nargs="+", default=[1800, 2000])
    ap.add_argument("--max-step", type=int, default=2000)
    ap.add_argument("--poll", type=int, default=900, help="seconds between polls")
    ap.add_argument("--min-disk-gb", type=int, default=20)
    ap.add_argument("--min-ram-pct", type=int, default=35)
    ap.add_argument("--stale-hours", type=float, default=6.0)
    ap.add_argument("--fallback-min-step", type=int, default=900)
    ap.add_argument("--hours", type=float, default=60.0, help="give up after this long")
    a = ap.parse_args()
    load_dotenv(REPO / ".env")
    token = os.environ["HF1_TOKEN"]
    repo_id = a.repo or f"isaacmg/qwen3-vl-8b-hebrew-{a.ver}-ckpt"
    LOGS.mkdir(exist_ok=True)
    state_path = LOGS / f"auto_eval_{a.ver}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "failed": {}}
    targets = sorted(set(a.box_steps) | set(a.full_steps))
    log(f"auto-eval {a.ver}: repo {repo_id}, box steps {a.box_steps}, full steps {a.full_steps}")
    deadline = time.time() + a.hours * 3600
    last_seen_step, last_new_at = 0, time.time()
    while time.time() < deadline:
        try:
            shas = checkpoint_shas(repo_id, token)
        except Exception as e:  # noqa: BLE001 — network blips must not kill a two-day watch
            log(f"hub poll error: {type(e).__name__}: {e}")
            time.sleep(a.poll)
            continue
        latest = max(shas) if shas else 0
        if latest > last_seen_step:
            last_seen_step, last_new_at = latest, time.time()
            log(f"newest pushed step {latest}")
        pending = [s for s in targets if s in shas and str(s) not in state["done"] and state["failed"].get(str(s), 0) < 2]
        stale_h = (time.time() - last_new_at) / 3600
        if not pending and stale_h > a.stale_hours and not any(v.get("full") for v in state["done"].values()):
            done_steps = [int(s) for s, v in state["done"].items() if int(s) >= a.fallback_min_step and s in map(str, shas)]
            if done_steps:
                s = max(done_steps)
                log(f"run stale {stale_h:.1f}h with no full eval — running FULL on the newest evaluated step {s}")
                pending = [s]
                a.full_steps = sorted(set(a.full_steps) | {s})
        for step in pending:
            ok, why = gates_ok(a.min_disk_gb, a.min_ram_pct)
            if not ok:
                log(f"gate blocked ({why}) — retry in {a.poll}s")
                break
            log(f"gates OK ({why})")
            full = step in a.full_steps
            success = process(step, shas[step], a.ver, full, a)
            key = str(step)
            if success:
                state["done"][key] = {"full": full, "at": datetime.now().isoformat(timespec="minutes")}
            else:
                state["failed"][key] = state["failed"].get(key, 0) + 1
            state_path.write_text(json.dumps(state, indent=1))
            break   # one checkpoint per poll cycle; re-list SHAs before the next
        if last_seen_step >= a.max_step and all(str(s) in state["done"] for s in targets if s in shas):
            log("all target steps evaluated — done")
            return 0
        time.sleep(a.poll)
    log("deadline reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
