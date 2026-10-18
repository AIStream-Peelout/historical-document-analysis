"""Memory soak test for a Kraken test container: repeated /transcribe_lines calls, memory after each.

Loops over the religious + PGP benchmark images (``--passes`` times), calls the
service on ``--url`` and records, per request, wall time, returned lines and the
container's anonymous memory (cgroup ``memory.stat``). Nothing is written to
the benchmark caches. Pass criteria are printed at the end: no failed request,
and the last pass's anonymous-memory maximum within ``--tolerance-gib`` of the
first pass's.

Usage (repo root): .venv/bin/python src/services/kraken_microservice/k7/soak_test.py --container kraken-k7 --out soak.jsonl
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))


def anon_gib(container: str) -> float:
    """Anonymous memory of a container, GiB (cgroup v2 memory.stat).

    :param container: Container name.
    :type container: str
    :return: GiB, or -1 when unreadable (container gone).
    :rtype: float
    """
    r = subprocess.run(["docker", "exec", container, "grep", "^anon ", "/sys/fs/cgroup/memory.stat"],
                       capture_output=True, text=True)
    return int(r.stdout.split()[1]) / 2 ** 30 if r.returncode == 0 and r.stdout else -1.0


def images() -> list:
    """Religious + PGP benchmark image paths.

    :return: Paths.
    :rtype: list
    """
    from src.datasets.evaluations.helper_eval_scripts.kraken_segmenter_ab import _BENCHES, doc_image
    out = []
    for bench, (spec, _) in _BENCHES.items():
        out += [str(doc_image(bench, d)) for d in json.load(open(spec))["docs"]]
    return out


async def main_async(a: argparse.Namespace) -> None:
    """Run the soak and print the verdict.

    :param a: Parsed arguments.
    :type a: argparse.Namespace
    """
    from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken_lines
    paths = images()
    preload_kraken_model(a.model)
    per_pass = []
    fails = 0
    with open(a.out, "w") as fh:
        for p in range(a.passes):
            mx = 0.0
            for i, path in enumerate(paths):
                t0 = time.time()
                res = await transcribe_with_kraken_lines(a.model, path, timeout=900.0)
                mem = anon_gib(a.container)
                ok = res is not None
                fails += not ok
                mx = max(mx, mem)
                fh.write(json.dumps({"pass": p, "i": i, "ok": ok, "s": round(time.time() - t0, 1), "anon_gib": round(mem, 3),
                                     "lines": len(res["lines"]) if ok else None}) + "\n")
                fh.flush()
                if mem < 0:
                    print(f"container gone at pass {p} request {i}", flush=True)
                    return
            per_pass.append(mx)
            print(f"pass {p}: {len(paths)} requests, anon max {mx:.2f} GiB, failures so far {fails}", flush=True)
    grew = per_pass[-1] - per_pass[0]
    print(f"VERDICT: {'PASS' if fails == 0 and grew <= a.tolerance_gib else 'FAIL'} "
          f"(failures {fails}, anon max first {per_pass[0]:.2f} / last {per_pass[-1]:.2f} GiB)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8003")
    ap.add_argument("--container", default="kraken-k7")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--tolerance-gib", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel"))
    a = ap.parse_args()
    if a.url.rstrip("/").endswith(":8002"):
        sys.exit("refusing :8002 (production)")
    os.environ["KRAKEN_MICROSERVICE_URL"] = a.url
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
