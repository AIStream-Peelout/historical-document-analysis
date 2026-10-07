"""Stamp the consensus records read after the 2026-09-23 Kraken cut-over with their own HTR cache key.

After ``logs/kraken_cutover_0923.sh`` swapped :8002 to kraken 7.0.3 + blla2026
under the running pipeline, new fragments were still written as the legacy
reader's (top-level raw-cache ``frags``, no ``htr_model`` field → read back as
``MiDRASH_Gen_01``), because the running code had no stamp option.  This pass
gives them a distinct key so the existing code treats the two populations
correctly without any other change:

* raw-cache entry: ``htr_model = KEY``  → ``cached_frags(entry, KEY)`` returns its
  top-level ``frags``; ``cached_frags(entry, "MiDRASH_Gen_01")`` no longer does
  (so k4-baseline probes and default-key ``--rekraken`` skip them);
* record: ``ai_read.htr_cache_key = KEY`` (``htr_model`` stays ``MiDRASH_Gen_01``,
  the recognition model, as ``--rekraken --kraken-cache-suffix`` does) → ``--rematch``
  finds the fragments again.

The k7 set comes from the pipeline logs, which are append-only (the out file can
be rewritten by ``--rematch``): every successful job line of the cut-over log from
the first k7 job on (``boundary.first_k7_job`` in the cut-over record), plus every
successful job line of ``--all-k7-logs`` (runs started after the cut-over).  Each
raw entry's fragment count must equal the count logged for that job, else the
entry is reported and left alone.  The records file is rewritten only while the
pipeline is not running (``--pid-file``); ``--raw-only`` stamps just the raw cache.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.datasets.consensus.two_reader_lines import (  # noqa: E402
    LEGACY_HTR_MODEL, raw_cache_path, replace_records, write_json_atomic)

KEY = "MiDRASH_Gen_01@k7.0.3-blla2026"
_JOB = re.compile(r"^\s+(\d+)/\d+ (\S+)#(\d+): lines \d+ agreed \d+ frags (\d+) ")


def logged_jobs(log: Path) -> List[Tuple[int, str, int, int]]:
    """Successful job lines of one pipeline log.

    :param log: Pipeline log.
    :type log: Path
    :return: ``(n, doc_id, image_index, n_frags)`` in log order.
    :rtype: List[Tuple[int, str, int, int]]
    """
    out = []
    for line in log.read_text(errors="replace").splitlines():
        m = _JOB.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))))
    return out


def k7_jobs(cutover: Dict, cutover_log: Path, all_k7_logs: List[Path]) -> Dict[Tuple[str, int], int]:
    """``(doc_id, image_index) -> logged fragment count`` for every job read by the k7 service.

    :param cutover: Parsed cut-over record.
    :type cutover: Dict
    :param cutover_log: The log the cut-over happened in.
    :type cutover_log: Path
    :param all_k7_logs: Logs of runs started after the cut-over.
    :type all_k7_logs: List[Path]
    :return: The k7 set.
    :rtype: Dict[Tuple[str, int], int]
    :raises SystemExit: When the boundary job is not found in ``cutover_log``.
    """
    first = cutover["boundary"]["first_k7_job"]
    jobs = logged_jobs(cutover_log)
    start = next((i for i, j in enumerate(jobs)
                  if (j[0], j[1], j[2]) == (first["n"], first["doc_id"], first["image_index"])), None)
    if start is None:
        sys.exit(f"boundary job {first} not in {cutover_log}")
    k7 = {(d, i): f for _, d, i, f in jobs[start:]}
    for log in all_k7_logs:
        k7.update({(d, i): f for _, d, i, f in logged_jobs(log)})
    return k7


def pid_alive(pid_file: Path) -> bool:
    """Whether the process named in a pid file is running.

    :param pid_file: Pid file.
    :type pid_file: Path
    :return: True when it names a live process.
    :rtype: bool
    """
    import os
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (OSError, ValueError, FileNotFoundError):
        return False


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cutover", type=Path, default=_REPO / "logs/kraken_cutover_0923.json")
    p.add_argument("--cutover-log", type=Path, default=_REPO / "logs/two_reader_v21b_v22b_0923.log",
                   help="the pipeline log the cut-over happened in (a .crashN.log if the watchdog resumed)")
    p.add_argument("--all-k7-logs", type=Path, nargs="*", default=[],
                   help="logs of pipeline runs started after the cut-over (every job is k7)")
    p.add_argument("--out", type=Path, default=_REPO / "src/datasets/raw_data/cairo_genizah/ai_reads/"
                                                      "ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl")
    p.add_argument("--vlm-model", default="qwen3-vl-8b-heb-v21b-step1200")
    p.add_argument("--pid-file", type=Path, default=_REPO / "logs/two_reader_v21b_v22b_0923.pid")
    p.add_argument("--key", default=KEY)
    p.add_argument("--raw-only", action="store_true", help="stamp raw-cache entries only (safe while running)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.raw_only and not a.dry_run and pid_alive(a.pid_file):
        sys.exit("pipeline still running: re-run after it exits, or use --raw-only")
    cutover = json.loads(a.cutover.read_text())
    k7 = k7_jobs(cutover, a.cutover_log, a.all_k7_logs)
    (a.cutover.parent / (a.cutover.stem + "_k7.jsonl")).write_text(
        "".join(json.dumps({"doc_id": d, "image_index": i, "frags": f}) + "\n" for (d, i), f in sorted(k7.items())))
    stats: Counter = Counter(k7_jobs=len(k7))
    stamped_raw: Set[Tuple[str, int]] = set()
    for (doc_id, idx), n_frags in sorted(k7.items()):
        path = raw_cache_path({"doc_id": doc_id, "image_index": idx}, a.vlm_model)
        if not path.exists():
            stats["raw_missing"] += 1
            continue
        entry = json.loads(path.read_text())
        current = entry.get("htr_model", LEGACY_HTR_MODEL)
        if current == a.key:
            stats["raw_already"] += 1
            stamped_raw.add((doc_id, idx))
            continue
        if current != LEGACY_HTR_MODEL or len(entry.get("frags") or []) != n_frags:
            stats["raw_mismatch"] += 1       # re-read or rekrakened since: leave it alone, report
            continue
        if not a.dry_run:
            entry["htr_model"] = a.key
            write_json_atomic(path, entry)
        stats["raw_stamped"] += 1
        stamped_raw.add((doc_id, idx))
    if not a.raw_only:
        text = a.out.read_text()
        lines = [l for l in text.splitlines() if l.strip()]
        new = []
        for l in lines:
            r = json.loads(l)
            if (r["doc_id"], r["image_index"]) in stamped_raw and r["ai_read"].get("htr_cache_key") != a.key:
                r["ai_read"]["htr_cache_key"] = a.key
                stats["records_stamped"] += 1
                l = json.dumps(r, ensure_ascii=False)
            new.append(l)
        if not a.dry_run and stats["records_stamped"]:
            replace_records(a.out, new, len(lines))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
