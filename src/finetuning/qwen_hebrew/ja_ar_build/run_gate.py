"""Run gate v2 over candidate files and cached reads; write pairs, line evidence, a run record and a funnel.

Inputs are novel_build stage-1 candidate files (``schema.Candidate`` records: targets with lines, images with sha256,
path, label, pHash) and read caches (``<work dir>/reads/reads.jsonl``). Each document group is evaluated on its own
(:func:`assign.evaluate_group`); groups run in a few worker processes (fork), the null pool is shared.

Example (frozen novel_v1 pool, cached reads only)::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.run_gate \\
        --candidates /Volumes/home/studio_offload/datasets/genizah_novel_v1_work/stages/stage1_images.jsonl \\
        --reads-dir /Volumes/home/studio_offload/datasets/genizah_novel_v1_work \\
        --old-sides /Volumes/home/studio_offload/datasets/genizah_novel_v1_work/stages/stage2_sides.jsonl \\
        --out /Volumes/home/studio_offload/datasets/genizah_ja_ar_v1_work/gate/run01 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.finetuning.qwen_hebrew.ja_ar_build import align, assign, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import NullPool, ReadStream
from src.finetuning.qwen_hebrew.ja_ar_build.assign import GateParams, Group, Photo, Side
from src.finetuning.qwen_hebrew.novel_build import reads as nb_reads

#: Image statuses of stage 1 that mean "a stored file exists".
ON_DISK = frozenset({"present", "fetched", "duplicate"})

_POOL: Optional[NullPool] = None
_PARAMS: Optional[GateParams] = None


def load_candidates(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Candidate records of every file, in file order.

    :param paths: JSONL files of ``schema.Candidate`` records.
    :return: The records (dicts).
    """
    out: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def load_streams(read_dirs: Sequence[Path]) -> Dict[str, List[ReadStream]]:
    """Usable reads of every read cache, as letter streams, by photograph sha256.

    :param read_dirs: Work dirs holding ``reads/reads.jsonl``.
    :return: ``{sha256: [ReadStream]}`` (one stream per distinct read key).
    """
    out: Dict[str, List[ReadStream]] = defaultdict(list)
    seen: set = set()
    for wd in read_dirs:
        if not (Path(wd) / "reads" / "reads.jsonl").exists():
            continue
        for rec in nb_reads.load_reads(Path(wd)):
            if not nb_reads.is_usable(rec):
                continue
            st = align.stream_from_read(rec.sha256, rec.reader, rec.lines, rec.meta or {})
            if st.key in seen:
                continue
            seen.add(st.key)
            out[rec.sha256].append(st)
    return dict(out)


def load_old_choice(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """Photograph chosen per side by the novel_v1 stage-2 decision (accepted or review), for re-pair flags.

    :param path: ``stage2_sides.jsonl`` or None.
    :return: ``{(row_id, side_key): sha256}``.
    """
    out: Dict[Tuple[str, str], str] = {}
    if path is None or not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("image_sha256"):
                out[(r["row_id"], r["side_key"])] = r["image_sha256"]
    return out


def build_groups(cands: Iterable[Dict[str, Any]], streams: Dict[str, List[ReadStream]],
                 old: Dict[Tuple[str, str], str], check_files: bool = True) -> List[Group]:
    """Group candidates by document group, with photographs on disk and their reads.

    :param cands: Candidate records.
    :param streams: Reads by sha256.
    :param old: novel_v1 photograph choice per side.
    :param check_files: Drop photographs whose file is missing.
    :return: Groups sorted by id.
    """
    by_group: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        g = by_group.setdefault(c["document_group"], {"sides": [], "photos": {}, "null_exclude": set()})
        planted = (c.get("notes") or {}).get("planted") or {}
        g["null_exclude"].update(planted.get("null_exclude_sha256") or [])
        for t in c.get("targets") or []:
            lines = [ln for ln in (t.get("lines") or []) if ln.strip()]
            if not lines:
                continue
            g["sides"].append(Side(row_id=c["row_id"], side_key=t.get("side_key") or "", source=c["source"],
                                   script=c.get("script") or textnorm.script_family(" ".join(lines)), lines=lines,
                                   skels=textnorm.line_skeletons(lines),
                                   old_sha=old.get((c["row_id"], t.get("side_key") or ""))))
        for im in c.get("images") or []:
            sha = im.get("sha256")
            if not sha or im.get("status") not in ON_DISK or not im.get("path"):
                continue
            if sha in g["photos"]:
                continue
            if check_files and not os.path.exists(im["path"]):
                continue
            g["photos"][sha] = Photo(sha256=sha, path=im["path"], label=im.get("label"), phash=im.get("phash"),
                                     width=im.get("width"), height=im.get("height"))
    groups: List[Group] = []
    for gid in sorted(by_group):
        g = by_group[gid]
        reads = {sha: streams[sha] for sha in g["photos"] if sha in streams}
        groups.append(Group(document_group=gid, sides=g["sides"], photos=g["photos"], reads=reads,
                            null_exclude=set(g["null_exclude"])))
    return groups


def _init(pool: NullPool, params: GateParams) -> None:
    """Worker initialiser (fork: the pool is inherited; this only sets globals).

    :param pool: Null pool.
    :param params: Thresholds.
    """
    global _POOL, _PARAMS
    _POOL = pool
    _PARAMS = params


def _run_one(group: Group) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate one group in a worker.

    :param group: The group.
    :return: Pair records and line records.
    """
    assert _POOL is not None and _PARAMS is not None
    if not group.sides:
        return [], []
    if not group.reads:
        pairs = []
        for side in group.sides:
            rec = assign._reject_record(side, group, {})
            pairs.append(rec)
        return pairs, []
    return assign.evaluate_group(group, _POOL, _PARAMS)


def funnel(pairs: Sequence[Dict[str, Any]], old_outcomes: Dict[Tuple[str, str], str]) -> str:
    """Markdown funnel of a run.

    :param pairs: Pair records.
    :param old_outcomes: novel_v1 stage-2 outcome per side.
    :return: Markdown text.
    """
    lines: List[str] = ["# gate v2 funnel (generated)", ""]
    by_side: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        for sk in p.get("side_keys") or [p["side_key"]]:
            by_side[(p["row_id"], sk)].append(p)
    def side_status(ps: List[Dict[str, Any]]) -> str:
        decs = {p["decision"] for p in ps}
        if "accept" in decs:
            return "accept"
        if "review" in decs:
            return "review"
        return "reject"
    tab = Counter()
    for key, ps in by_side.items():
        tab[(old_outcomes.get(key, "n/a"), side_status(ps))] += 1
    lines += ["## Sides: novel_v1 stage-2 outcome x gate v2 best decision", "",
              "| novel_v1 | accept | review | reject |", "|---|---:|---:|---:|"]
    for old in sorted({k[0] for k in tab}):
        lines.append(f"| {old} | {tab[(old, 'accept')]} | {tab[(old, 'review')]} | {tab[(old, 'reject')]} |")
    lines += ["", "## Pairs by source x script x decision", "", "| source | script | accept | review | reject | "
              "accepted letters |", "|---|---|---:|---:|---:|---:|"]
    t2 = Counter()
    letters = Counter()
    for p in pairs:
        t2[(p["source"], p["script"], p["decision"])] += 1
        if p["decision"] == "accept" and not p.get("dup_rank"):
            letters[(p["source"], p["script"])] += p["evidence"].get("target_letters", 0)
    for src, scr in sorted({(k[0], k[1]) for k in t2}):
        lines.append(f"| {src} | {scr} | {t2[(src, scr, 'accept')]} | {t2[(src, scr, 'review')]} | "
                     f"{t2[(src, scr, 'reject')]} | {letters[(src, scr)]:,} |")
    rem = Counter()
    reasons = Counter()
    for p in pairs:
        if p["decision"] == "accept":
            rem[",".join(p["remedies"]) or "as_is"] += 1
        for r in p["reasons"]:
            reasons[(p["decision"], r)] += 1
    lines += ["", "## Accepted pairs by remedy", "", "| remedies | pairs |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in rem.most_common()]
    lines += ["", "## Reasons", "", "| decision | reason | pairs |", "|---|---|---:|"]
    lines += [f"| {k[0]} | {k[1]} | {v} |" for k, v in reasons.most_common()]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    :return: The parser.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", type=Path, nargs="+", required=True, help="candidate JSONL files")
    ap.add_argument("--reads-dir", type=Path, nargs="+", required=True, help="work dirs with reads/reads.jsonl")
    ap.add_argument("--null-candidates", type=Path, nargs="*", default=None,
                    help="candidate files whose photographs feed the null (default: --candidates)")
    ap.add_argument("--old-sides", type=Path, default=None, help="novel_v1 stage2_sides.jsonl (re-pair flags)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-groups", type=int, default=None, help="first N groups only (smoke runs)")
    ap.add_argument("--params", type=Path, default=None, help="JSON file overriding GateParams fields")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Run the gate.

    :param argv: Arguments (None: sys.argv).
    """
    args = build_parser().parse_args(argv)
    t0 = time.time()
    params = GateParams()
    if args.params:
        for k, v in json.loads(args.params.read_text()).items():
            setattr(params, k, v)
    cands = load_candidates(args.candidates)
    streams = load_streams(args.reads_dir)
    null_cands = load_candidates(args.null_candidates) if args.null_candidates else cands
    null_shas = {im.get("sha256") for c in null_cands for im in (c.get("images") or []) if im.get("sha256")}
    pool = NullPool.from_streams(st for sha, sts in streams.items() if sha in null_shas for st in sts)
    old = load_old_choice(args.old_sides)
    old_outcomes: Dict[Tuple[str, str], str] = {}
    if args.old_sides and args.old_sides.exists():
        for line in open(args.old_sides, encoding="utf-8"):
            r = json.loads(line)
            old_outcomes[(r["row_id"], r["side_key"])] = r["outcome"]
    groups = build_groups(cands, streams, old)
    null_groups = build_groups(null_cands, streams, {}, check_files=False) if args.null_candidates else groups
    pool.add_targets((g.document_group, "arabic" if s.script == "arabic" else "hebrew", s.unit_skels)
                     for g in null_groups for s in g.sides)
    if args.limit_groups:
        groups = groups[: args.limit_groups]
    print(f"candidates {len(cands)}, groups {len(groups)}, photos with reads {len(streams)}, null keys "
          f"{len(pool.by_key)} ({sum(len(v) for v in pool.by_key.values())} reads), load {time.time() - t0:.0f}s",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out / "pairs.jsonl"
    lines_path = args.out / "line_evidence.jsonl"
    all_pairs: List[Dict[str, Any]] = []
    done = 0
    ctx = mp.get_context("fork")
    with open(pairs_path, "w", encoding="utf-8") as pf, open(lines_path, "w", encoding="utf-8") as lf:
        if args.workers <= 1:
            _init(pool, params)
            results = map(_run_one, groups)
            iterator: Iterable = results
            for pairs, lines in iterator:
                done += 1
                for p in pairs:
                    pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                for ln in lines:
                    lf.write(json.dumps(ln, ensure_ascii=False) + "\n")
                all_pairs.extend(pairs)
        else:
            with ctx.Pool(args.workers, initializer=_init, initargs=(pool, params)) as wp:
                for pairs, lines in wp.imap(_run_one, groups, chunksize=4):
                    done += 1
                    for p in pairs:
                        pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                    for ln in lines:
                        lf.write(json.dumps(ln, ensure_ascii=False) + "\n")
                    all_pairs.extend(pairs)
                    if done % 250 == 0:
                        print(f"groups {done}/{len(groups)} ({time.time() - t0:.0f}s)", flush=True)
    (args.out / "FUNNEL.md").write_text(funnel(all_pairs, old_outcomes), encoding="utf-8")
    run = {"inputs": {"candidates": [str(p) for p in args.candidates], "reads_dirs": [str(p) for p in args.reads_dir],
                      "old_sides": str(args.old_sides) if args.old_sides else None},
           "params": assign.params_dict(params), "groups": len(groups), "pairs": len(all_pairs),
           "decisions": dict(Counter(p["decision"] for p in all_pairs)), "seconds": round(time.time() - t0, 1)}
    (args.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run["decisions"]), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
