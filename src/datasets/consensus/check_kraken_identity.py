"""Check that a candidate Kraken service reproduces the production fragments exactly (pre-swap gate).

For ``--n`` pages the pipeline already read (successful job lines of ``--log``),
re-download the exact image (sha256 must equal the raw cache's), prepare it
with the pipeline's own :func:`two_reader_lines.prepare_image`, read it through
the candidate service (``--url``) with the pipeline's own
:func:`two_reader_lines.run_kraken`, and compare every fragment with the raw
cache's top-level ``frags``: same count, same text, same box, same confidence
(within ``--conf-tol``).  Used before a "same-config" swap of :8002 whose reads
keep the current HTR cache key — that claim is about outputs, so it is checked on
outputs.  Nothing is written except the report.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

_JOB = re.compile(r"^\s+(\d+)/\d+ (\S+)#(\d+): lines \d+ agreed \d+ frags (\d+) ")


def sample_jobs(log: Path, n: int) -> list:
    """Evenly spaced successful jobs of a pipeline log.

    :param log: Pipeline log.
    :type log: Path
    :param n: Number of jobs.
    :type n: int
    :return: ``(doc_id, image_index, logged fragment count)``.
    :rtype: list
    """
    jobs = [(m.group(2), int(m.group(3)), int(m.group(4)))
            for m in map(_JOB.match, log.read_text(errors="replace").splitlines()) if m]
    step = max(1, len(jobs) // n)
    return jobs[::step][:n]


def compare(cached: list, new: list, conf_tol: float) -> str:
    """First difference between two fragment lists, or '' when identical.

    :param cached: Raw-cache fragments ``{text, conf, box}``.
    :type cached: list
    :param new: Candidate fragments.
    :type new: list
    :param conf_tol: Allowed confidence difference.
    :type conf_tol: float
    :return: Description of the first difference ('' = identical).
    :rtype: str
    """
    if len(cached) != len(new):
        return f"fragment count {len(cached)} vs {len(new)}"
    for i, (a, b) in enumerate(zip(cached, new)):
        if a["text"] != b["text"]:
            return f"frag {i} text {a['text'][:30]!r} vs {b['text'][:30]!r}"
        if any(abs(x - y) > 1e-6 for x, y in zip(a["box"], b["box"])):
            return f"frag {i} box {a['box']} vs {b['box']}"
        ca, cb = a.get("conf"), b.get("conf")
        if (ca is None) != (cb is None) or (ca is not None and abs(ca - cb) > conf_tol):
            return f"frag {i} conf {ca} vs {cb}"
    return ""


async def main_async(a: argparse.Namespace) -> int:
    """Run the check and print the verdict.

    :param a: Parsed arguments.
    :type a: argparse.Namespace
    :return: Process exit code (0 = all identical).
    :rtype: int
    """
    from src.datasets.consensus.two_reader_lines import (
        KRAKEN_MODEL, download, prepare_image, raw_cache_path, run_kraken)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    same = checked = 0
    report = []
    for doc_id, idx, n_frags in sample_jobs(a.log, a.n):
        job = {"doc_id": doc_id, "image_index": idx}
        c = json.loads(raw_cache_path(job, a.vlm_model).read_text())
        data = download(c["image_url"])
        if data is None or hashlib.sha256(data).hexdigest() != c["sha256"]:
            report.append({"doc_id": doc_id, "image_index": idx, "result": "skipped: image changed or unavailable"})
            continue
        path = a.work_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', doc_id)}__{idx}.jpg"
        if prepare_image(data, path) != (c["width"], c["height"]):
            report.append({"doc_id": doc_id, "image_index": idx, "result": "skipped: oriented size differs"})
            continue
        new = await run_kraken(path, KRAKEN_MODEL, c["width"], c["height"], timeout=600.0)
        path.unlink(missing_ok=True)
        diff = "candidate service failed" if new is None else compare(c["frags"], new, a.conf_tol)
        checked += 1
        same += not diff
        report.append({"doc_id": doc_id, "image_index": idx, "frags": len(c["frags"]), "result": diff or "identical"})
        print(f"  {doc_id}#{idx}: {diff or 'identical'} ({len(c['frags'])} frags)", flush=True)
    Path(a.out).write_text(json.dumps({"checked": checked, "identical": same, "pages": report}, indent=1))
    print(f"VERDICT: {same}/{checked} pages identical")
    return 0 if checked and same == checked else 1


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="candidate service (never :8002)")
    ap.add_argument("--log", type=Path, required=True, help="pipeline log whose jobs were read by the current prod")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--conf-tol", type=float, default=1e-6)
    ap.add_argument("--vlm-model", default="qwen3-vl-8b-heb-v21b-step1200")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.url.rstrip("/").endswith(":8002"):
        sys.exit("refusing :8002 (production)")
    os.environ["KRAKEN_MICROSERVICE_URL"] = a.url
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
