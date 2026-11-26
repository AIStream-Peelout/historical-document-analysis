"""New Kraken reads for gate v2: photographs read rotated 180 degrees (stored upside down), selected and prioritised.

The rotation is applied client-side to a copy of the photograph; the record keeps the ORIGINAL photograph's sha256 and
records ``meta.rotation_client`` (so :func:`align.stream_from_read` knows the read's frame and drops the boxes).
Records go to ``<out work dir>/reads/reads.jsonl`` (never into the frozen novel_v1 reads file). Kraken :8002 is this
repository's own container: one request at a time; on a connection error the job waits for ``/health`` (the container
restarts itself after an occasional out-of-memory kill) and retries once; it never restarts any container.

Selection: photographs of document groups with a Hebrew-script side left unplaced (reject / review) by a gate run,
sides of at least 40 letters; JA-signal sides first, then by the mean confidence of the photograph's 0-degree Kraken
read (lowest first: an upside-down page reads with low confidence).

Example::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.orient \\
        --pairs W2/gate/run03/pairs.jsonl --candidates W1/stages/stage1_images.jsonl --reads-dir W1 --out-work-dir W2
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import textnorm
from src.finetuning.qwen_hebrew.novel_build import reads as nb_reads
from src.finetuning.qwen_hebrew.novel_build.schema import ReadRecord

ON_DISK = frozenset({"present", "fetched", "duplicate"})
HEALTH_URL = "http://localhost:8002/health"


def mean_kraken_conf(read_dirs: Sequence[Path]) -> Dict[str, Tuple[float, int, int]]:
    """Mean line confidence, lines and letters of every photograph's 0-degree Kraken read.

    :param read_dirs: Work dirs with ``reads/reads.jsonl``.
    :return: ``{sha256: (mean confidence, read lines, read letters)}``.
    """
    out: Dict[str, Tuple[float, int, int]] = {}
    for wd in read_dirs:
        if not (Path(wd) / "reads" / "reads.jsonl").exists():
            continue
        for rec in nb_reads.load_reads(Path(wd)):
            if not rec.reader.startswith("kraken") or rec.meta.get("rotation_client"):
                continue
            confs = [c for c in (rec.meta.get("confidences") or []) if isinstance(c, (int, float))]
            if confs:
                out[rec.sha256] = (sum(confs) / len(confs), rec.n_lines, rec.letters_hebrew + rec.letters_arabic)
    return out


def already_read(work_dir: Path, rotation: int) -> Set[str]:
    """Photographs already read at a client rotation in the output work dir.

    :param work_dir: Output work dir.
    :param rotation: Client rotation.
    :return: sha256 set.
    """
    path = Path(work_dir) / "reads" / "reads.jsonl"
    if not path.exists():
        return set()
    return {r.sha256 for r in nb_reads.load_reads(Path(work_dir))
            if int(r.meta.get("rotation_client") or 0) == rotation and r.finish != "error"}


def select_photos(pairs_path: Path, cands_path: Path, conf: Dict[str, Tuple[float, int, int]], min_letters: int = 40,
                  min_read_lines: int = 5, min_read_letters: int = 100, max_conf: float = 0.80
                  ) -> List[Tuple[str, str, int, float]]:
    """Photographs to read rotated, in priority order.

    Only photographs whose 0-degree Kraken read holds text (at least ``min_read_lines`` lines and
    ``min_read_letters`` letters) read with low confidence (below ``max_conf``) qualify: the profile of an upside-down
    page. Photographs with almost no text, or read with good confidence, are not upside down.

    :param pairs_path: Gate run ``pairs.jsonl``.
    :param cands_path: Candidate file of that run.
    :param conf: ``(mean confidence, lines, letters)`` of the 0-degree Kraken read per photograph.
    :param min_letters: Letters a side needs to count.
    :param min_read_lines: Read lines the 0-degree read needs.
    :param min_read_letters: Read letters the 0-degree read needs.
    :param max_conf: Mean confidence below which a page may be upside down.
    :return: ``[(sha256, path, priority class, confidence)]``; class 0 = JA-signal side, 1 = other.
    """
    cands: Dict[str, Dict[str, Any]] = {}
    with open(cands_path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            cands[c["row_id"]] = c
    pairs = [json.loads(line) for line in open(pairs_path, encoding="utf-8")]
    placed = {(p["row_id"], sk) for p in pairs if p["decision"] == "accept" for sk in p["side_keys"]}
    best: Dict[str, Tuple[int, float, str]] = {}
    for p in pairs:
        if p["decision"] == "accept":
            continue
        c = cands.get(p["row_id"])
        if c is None or c.get("script") == "arabic" or str(p["row_id"]).startswith("planted_"):
            continue
        if any((p["row_id"], sk) in placed for sk in p["side_keys"]):
            continue
        if (p["evidence"] or {}).get("target_letters", 0) < min_letters:
            continue
        text = "\n".join(ln for t in c.get("targets") or [] if (t.get("side_key") or "") in p["side_keys"]
                         for ln in t.get("lines") or [])
        cls = 0 if textnorm.ja_signal(text) else 1
        for im in c.get("images") or []:
            sha, path = im.get("sha256"), im.get("path")
            if not sha or not path or im.get("status") not in ON_DISK or sha not in conf:
                continue
            c_mean, c_lines, c_letters = conf[sha]
            if c_lines < min_read_lines or c_letters < min_read_letters or c_mean >= max_conf:
                continue
            key = (cls, c_mean, path)
            if sha not in best or key < best[sha]:
                best[sha] = key
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    return [(sha, v[2], v[0], v[1]) for sha, v in ordered]


def buildable(pair: Dict[str, Any], verdicts: Optional[Dict[Tuple[str, str], Dict[str, Any]]]) -> bool:
    """Whether a pair could become a dataset row: every side novel on the pair's photograph and not held out.

    :param pair: Pair record.
    :param verdicts: Hold-out v2 verdicts by (row_id, side_key), or None (no filter).
    :return: True when the pair may be built (or no verdicts are given).
    """
    if verdicts is None:
        return True
    for sk in pair.get("side_keys") or [pair.get("side_key")]:
        v = verdicts.get((pair["row_id"], sk))
        if v is None or v.get("eval_holdout"):
            return False
        tier = (v.get("pair_tiers") or {}).get(pair.get("image_sha256")) or v.get("tier")
        if tier not in ("novel_strict", "novel_page_level"):
            return False
    return True


def select_trimmed_photos(pairs_path: Path, cands_path: Path, min_lines: int = 2,
                          verdicts: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
                          ) -> List[Tuple[str, str, int, float]]:
    """Photographs of pairs that lost unread edge lines (or went to review for unexplained ink): a rotated read may
    find those lines written sideways or upside down.

    :param pairs_path: Gate run ``pairs.jsonl``.
    :param cands_path: Candidate file of that run.
    :param min_lines: Trimmed (non-boundary) lines a pair needs.
    :param verdicts: Hold-out v2 verdicts; when given, only pairs that could become rows are selected.
    :return: ``[(sha256, path, priority class, trimmed letters)]``; classes 0-1 JA signal, 2-3 other Hebrew script.
    """
    cands: Dict[str, Dict[str, Any]] = {}
    with open(cands_path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            cands[c["row_id"]] = c
    best: Dict[str, Tuple[int, float, str]] = {}
    for line in open(pairs_path, encoding="utf-8"):
        p = json.loads(line)
        if p["decision"] == "reject" or p["script"] == "arabic" or str(p["row_id"]).startswith("planted_"):
            continue
        if not buildable(p, verdicts):
            continue
        ev = p.get("evidence") or {}
        boundary = {(sid, i) for sid, idx in (ev.get("boundary_trimmed") or {}).items() for i in idx}
        trimmed = [(sid, i) for sid, idx in (p.get("trimmed_lines") or {}).items() for i in idx
                   if (sid, i) not in boundary]
        review_ink = "unread_ink_may_hold_trimmed_lines" in (p.get("reasons") or [])
        kept_unread = bool({"margin_block_kept", "edge_kept"} & set(p.get("flags") or []))
        if len(trimmed) < min_lines and not review_ink and not kept_unread:
            continue
        ja = textnorm.ja_signal("\n".join(p["lines"]))
        # JA first; within a script, pairs that lost lines before pairs whose unread edge lines were kept unconfirmed
        cls = (0 if ja else 2) + (0 if (len(trimmed) >= min_lines or review_ink) else 1)
        key = (cls, -float(ev.get("edge_letters") or len(trimmed)), p["image_path"])
        sha = p["image_sha256"]
        if sha not in best or key < best[sha]:
            best[sha] = key
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    return [(sha, v[2], v[0], -v[1]) for sha, v in ordered]


def wait_for_health(timeout_s: float = 180.0) -> bool:
    """Wait until the Kraken service answers ``/health``.

    :param timeout_s: Seconds to wait at most.
    :return: True when it answered.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(HEALTH_URL, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def read_plain(sha256: str, path: Path) -> ReadRecord:
    """Kraken read of a photograph as stored (no client rotation), keyed by its sha256.

    :param sha256: Digest of the stored photograph.
    :param path: The photograph.
    :return: The read record (``finish == "error"`` on failure, never raised).
    """
    rec = nb_reads.kraken_read(path)
    rec.sha256 = sha256
    rec.meta = dict(rec.meta or {})
    rec.meta["source_path"] = str(path)
    return rec


def read_rotated(sha256: str, path: Path, rotation: int) -> ReadRecord:
    """Kraken read of a rotated copy of a photograph, keyed by the original photograph.

    :param sha256: Digest of the stored photograph (kept as the record key).
    :param path: The photograph.
    :param rotation: Degrees clockwise (90 / 180 / 270).
    :return: The read record (``finish == "error"`` on failure, never raised).
    """
    with tempfile.TemporaryDirectory() as tmp:
        rotated = Path(tmp) / f"{sha256[:16]}_r{rotation}.jpg"
        with Image.open(path) as im:
            im.convert("RGB").rotate(-rotation, expand=True).save(rotated, format="JPEG", quality=92)
        rec = nb_reads.kraken_read(rotated)
    rec.sha256 = sha256
    rec.meta = dict(rec.meta or {})
    rec.meta["rotation_client"] = rotation
    rec.meta["source_path"] = str(path)
    return rec


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    :return: The parser.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--reads-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--out-work-dir", type=Path, required=True)
    ap.add_argument("--rotation", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the selection only")
    ap.add_argument("--verdicts", type=Path, default=None, help="hold-out v2 verdicts (trimmed mode: buildable pairs only)")
    ap.add_argument("--photos-file", type=Path, default=None,
                    help="mode photos: JSON lines {sha256, path, priority} to read (e.g. intake photographs at 0)")
    ap.add_argument("--mode", choices=("unplaced", "trimmed", "photos"), default="unplaced",
                    help="unplaced: photos of unplaced sides read with low confidence; trimmed: photos of pairs that "
                         "lost unread edge lines")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Select photographs and read them rotated, appending to the output work dir's reads file.

    :param argv: Arguments (None: sys.argv).
    """
    args = build_parser().parse_args(argv)
    if args.mode == "photos":
        todo = [(r["sha256"], r["path"], int(r.get("priority", 1)), 0.0)
                for r in (json.loads(line) for line in open(args.photos_file, encoding="utf-8"))]
        todo.sort(key=lambda t: (t[2], t[0]))
    elif args.mode == "trimmed":
        verdicts = None
        if args.verdicts:
            verdicts = {}
            for line in open(args.verdicts, encoding="utf-8"):
                v = json.loads(line)
                verdicts[(v["row_id"], v["side_key"])] = v
        todo = select_trimmed_photos(args.pairs, args.candidates, verdicts=verdicts)
    else:
        conf = mean_kraken_conf(args.reads_dir)
        todo = select_photos(args.pairs, args.candidates, conf)
    done = already_read(args.out_work_dir, args.rotation)
    todo = [t for t in todo if t[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"selected {len(todo)} photographs (JA-signal first: {sum(1 for t in todo if t[2] == 0)}), "
          f"already read {len(done)}", flush=True)
    if args.dry_run:
        return
    out = Path(args.out_work_dir) / "reads" / "reads.jsonl"
    stop = Path(args.out_work_dir) / "reads" / "STOP"
    errors = 0
    t0 = time.time()
    for k, (sha, path, cls, c) in enumerate(todo, 1):
        if stop.exists():
            print("STOP file found, exiting", flush=True)
            break
        reader_fn = (lambda: read_plain(sha, Path(path))) if args.rotation == 0 else \
            (lambda: read_rotated(sha, Path(path), args.rotation))
        rec = reader_fn()
        if rec.finish == "error":
            errors += 1
            print(f"error on {sha[:12]}: {rec.meta.get('error')}; waiting for /health", flush=True)
            if not wait_for_health():
                print("Kraken did not come back; exiting", flush=True)
                break
            rec = reader_fn()
        nb_reads.append_jsonl(out, rec)
        if k % 25 == 0:
            rate = (time.time() - t0) / k
            print(f"{k}/{len(todo)} read, {errors} errors, {rate:.1f} s/photo", flush=True)
    print(f"=== orient exit: {len(todo)} selected, {errors} errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
