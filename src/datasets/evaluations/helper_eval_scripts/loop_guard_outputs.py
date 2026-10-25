# File name: loop_guard_outputs.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Loop-guarded copies of cached Arabic-script benchmark reads (offline, no model needed).

A fine-tune that collapses into repeating a phrase is penalised twice by the raw scores: the
repeated tail lowers precision and inflates the letter error rate, although the text read before
the loop may be good. The loop guard scores every model under one more, identical condition: the
answer to an image is cut where the loop starts, and the guarded copy is scored with the official
scorer (``score_arabic_benchmark.py``) next to the raw one.

The rule, on the scored letters of one image's answer (``arabic_script.arabic_letters``):

* run ``k`` is the ``run_len``-letter run starting at letter ``k``; it *repeats* when the same run
  occurs earlier in the answer;
* the loop starts at the earliest run that repeats and opens a window of ``window`` runs of which
  at least ``min_repeat`` repeat (defaults 12 letters, 48 runs, 90 %);
* the answer is cut before the character that holds that run's first letter (trailing white
  space dropped), so the guarded text keeps exactly the letters before the loop;
* only images the scorer already calls loops (``arabic_script.repeat_share`` above
  ``score_arabic_benchmark.LOOP_REPEAT_SHARE``) are guarded, so every read the scorer does not
  flag stays byte-identical (``gate=None`` applies the rule everywhere; it then also cuts some
  formula repeats inside real text).

Loops written in Hebrew letters are left alone: they hold no scored letter. Applied to the
step-600 v2.5a reads and the v2.4a / v2.3a comparison reads (58 paired documents) this
reproduces the diagnostic of 2026-10-22 (v2.5a-600 guarded median 5-gram F1 0.178). The
production page reader (``src.models.ocr.page_reader``) applies the same rule at read time over
Hebrew and Arabic letters together, through the ``letters`` parameter of :func:`guard_text`.

Input rows are the ``run_arabic_benchmark.py`` layout, one per image::

    {"doc_id": ..., "image_index": 0, "model": ..., "prompt": ..., "secs": ..., "text": "..."}

The guarded copy ``outputs/<tag>__guarded.jsonl`` has the same rows in the same order with the
text cut, plus ``loop_guard = {"cut_at", "looped", "letters_kept", "letters_removed",
"repeat_share"}`` (``cut_at``: character offset in the raw text where the loop starts, None when
nothing was cut). Rows without text (failed requests) are copied unchanged.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.loop_guard_outputs \\
        --tags qwen3-vl-8b-heb-v25a-step600__trained [more tags] [--require-images 117] [--overwrite]
"""
import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark import DEFAULT_BENCHMARK, LOOP_REPEAT_SHARE

RUN_LEN = 12
WINDOW = 48
MIN_REPEAT = 0.9
SUFFIX = "__guarded"


def repeated_runs(letters: str, run_len: int = RUN_LEN) -> List[bool]:
    """Whether each run of a letters-only string repeats an earlier run.

    :param letters: Letters-only text.
    :type letters: str
    :param run_len: Length of a run, in letters.
    :type run_len: int
    :return: One flag per run start ``0 .. len(letters) - run_len``.
    :rtype: List[bool]
    """
    seen, flags = set(), []
    for start in range(len(letters) - run_len + 1):
        run = letters[start:start + run_len]
        flags.append(run in seen)
        seen.add(run)
    return flags


def loop_start(letters: str, run_len: int = RUN_LEN, window: int = WINDOW, min_repeat: float = MIN_REPEAT) -> Optional[int]:
    """Letter index where a letters-only answer collapses into a loop.

    :param letters: Letters-only text of one image's answer.
    :type letters: str
    :param run_len: Length of a run, in letters.
    :type run_len: int
    :param window: Number of runs that must mostly repeat from the loop start on.
    :type window: int
    :param min_repeat: Share of the window's runs that must repeat an earlier run.
    :type min_repeat: float
    :return: The earliest run start ``k`` whose run repeats and whose window of ``window`` runs holds
        at least ``min_repeat`` repeating runs; None when there is no such window (also for
        texts shorter than three runs or one window).
    :rtype: Optional[int]
    """
    if len(letters) < 3 * run_len:
        return None
    flags = repeated_runs(letters, run_len)
    if len(flags) < window:
        return None
    count = sum(flags[:window])
    for k in range(len(flags) - window + 1):
        if k:
            count += flags[k + window - 1] - flags[k - 1]
        if flags[k] and count / window >= min_repeat:
            return k
    return None


def letter_char_offset(text: str, n_letters: int, letters: Callable[[str], str] = ar.arabic_letters) -> int:
    """Character offset in a raw text before which exactly ``n_letters`` scored letters lie.

    Counts letters per cluster (a character and the combining marks after it), the unit within
    which Unicode normalisation can merge characters, so the count matches ``letters`` on the
    whole text.

    :param text: Raw answer text.
    :type text: str
    :param n_letters: Number of scored letters to keep.
    :type n_letters: int
    :param letters: The letters of a text, in order (default ``arabic_script.arabic_letters``).
    :type letters: Callable[[str], str]
    :return: Offset of the cluster that holds letter ``n_letters`` (0-based); ``len(text)`` when the
        text has no more letters than that.
    :rtype: int
    """
    count, start = 0, 0
    for index in range(1, len(text) + 1):
        if index < len(text) and unicodedata.combining(text[index]):
            continue
        cluster_letters = len(letters(text[start:index]))
        if count + cluster_letters > n_letters:
            return start
        count += cluster_letters
        start = index
    return len(text)


def guard_text(text: str, run_len: int = RUN_LEN, window: int = WINDOW, min_repeat: float = MIN_REPEAT,
               gate: Optional[float] = LOOP_REPEAT_SHARE,
               letters: Callable[[str], str] = ar.arabic_letters) -> Tuple[str, Optional[int], bool]:
    """Cut one image's answer where it collapses into a loop.

    :param text: Raw answer text of one image.
    :type text: str
    :param run_len: Length of a run, in letters.
    :type run_len: int
    :param window: Number of runs that must mostly repeat from the loop start on.
    :type window: int
    :param min_repeat: Share of the window's runs that must repeat an earlier run.
    :type min_repeat: float
    :param gate: Guard only answers whose ``arabic_script.repeat_share`` exceeds this (the scorer's
        loop flag); None guards every answer.
    :type gate: Optional[float]
    :param letters: The letters the rule runs on, in order (default ``arabic_script.arabic_letters``:
        the benchmark's scored letters, so Hebrew-letter loops are left alone; the production reader
        passes Hebrew and Arabic letters together).
    :type letters: Callable[[str], str]
    :return: ``(guarded text, cut_at, looped)``: the text before the loop without trailing white
        space, the character offset in ``text`` where the loop starts and True; or the unchanged
        text, None and False.
    :rtype: Tuple[str, Optional[int], bool]
    """
    if gate is not None and ar.repeat_share(text) <= gate:
        return text, None, False
    start = loop_start(letters(text), run_len, window, min_repeat)
    if start is None:
        return text, None, False
    cut_at = letter_char_offset(text, start, letters)
    return text[:cut_at].rstrip(), cut_at, True


def guard_rows(rows: List[Dict[str, Any]], run_len: int = RUN_LEN, window: int = WINDOW, min_repeat: float = MIN_REPEAT,
               gate: Optional[float] = LOOP_REPEAT_SHARE) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Guard every row of a model's answer file.

    :param rows: Rows of ``outputs/<tag>.jsonl`` in file order.
    :type rows: List[Dict[str, Any]]
    :param run_len: See :func:`guard_text`.
    :type run_len: int
    :param window: See :func:`guard_text`.
    :type window: int
    :param min_repeat: See :func:`guard_text`.
    :type min_repeat: float
    :param gate: See :func:`guard_text`.
    :type gate: Optional[float]
    :return: The guarded rows (same order and keys, plus ``loop_guard`` on rows with text) and a
        summary over the answers the scorer uses (the last row of each image): images answered,
        flagged as loops by the scorer, cut, documents with a cut image, scored letters before and
        after, and ``inexact_cuts`` (cuts whose kept letters differ from the letters before the
        loop; 0 expected).
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    """
    guarded, last = [], {}
    for row in rows:
        if not isinstance(row.get("text"), str):
            guarded.append(dict(row))
            continue
        text = row["text"]
        new_text, cut_at, looped = guard_text(text, run_len, window, min_repeat, gate)
        before, after = ar.arabic_letters(text), ar.arabic_letters(new_text)
        guarded.append(dict(row, text=new_text, loop_guard={"cut_at": cut_at, "looped": looped, "letters_kept": len(after),
                                                             "letters_removed": len(before) - len(after),
                                                             "repeat_share": round(ar.repeat_share(text), 4)}))
        exact = not looped or after == before[:loop_start(before, run_len, window, min_repeat)]
        last[(row["doc_id"], int(row["image_index"]))] = (guarded[-1]["loop_guard"], len(before), exact)
    infos = list(last.values())
    summary = {"rows": len(rows), "images_answered": len(infos),
               "images_flagged": sum(info["repeat_share"] > LOOP_REPEAT_SHARE for info, _, _ in infos),
               "images_cut": sum(info["looped"] for info, _, _ in infos),
               "docs_cut": len({doc for (doc, _), (info, _, _) in last.items() if info["looped"]}),
               "letters_before": sum(n for _, n, _ in infos), "letters_after": sum(info["letters_kept"] for info, _, _ in infos),
               "letters_removed": sum(info["letters_removed"] for info, _, _ in infos),
               "inexact_cuts": sum(not exact for _, _, exact in infos),
               "params": {"run_len": run_len, "window": window, "min_repeat": min_repeat, "gate": gate}}
    return guarded, summary


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Rows of a JSONL file (blank lines skipped).

    :param path: File.
    :type path: Path
    :return: Parsed rows in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_rows_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL rows through a temporary file in the same directory, then rename.

    :param path: Destination.
    :type path: Path
    :param rows: Rows to write.
    :type rows: List[Dict[str, Any]]
    """
    handle, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def guard_file(outputs_dir: Path, tag: str, require_images: int = 0, overwrite: bool = False, **params: Any) -> Dict[str, Any]:
    """Write ``<outputs_dir>/<tag>__guarded.jsonl`` from ``<outputs_dir>/<tag>.jsonl``.

    :param outputs_dir: The benchmark's answer directory.
    :type outputs_dir: Path
    :param tag: Answer file stem (``<model>__<prompt>``).
    :type tag: str
    :param require_images: Refuse (write nothing) when fewer images have an answer.
    :type require_images: int
    :param overwrite: Replace an existing guarded file.
    :type overwrite: bool
    :param params: ``run_len``, ``window``, ``min_repeat``, ``gate`` for :func:`guard_rows`.
    :type params: Any
    :return: The :func:`guard_rows` summary with ``tag``, ``guarded_tag`` and ``written``.
    :rtype: Dict[str, Any]
    :raises ValueError: For a tag that is already a guarded copy, a missing source, a source with
        too few answered images, or an existing destination without ``overwrite``.
    """
    if tag.endswith(SUFFIX):
        raise ValueError(f"{tag} is already a guarded copy")
    source, dest = outputs_dir / f"{tag}.jsonl", outputs_dir / f"{tag}{SUFFIX}.jsonl"
    if not source.exists():
        raise ValueError(f"{source} not found")
    if dest.exists() and not overwrite:
        raise ValueError(f"{dest} exists (pass --overwrite to replace it)")
    rows, summary = guard_rows(read_rows(source), **params)
    if summary["images_answered"] < require_images:
        raise ValueError(f"{source} answers {summary['images_answered']} images, fewer than the required {require_images}")
    write_rows_atomic(dest, rows)
    return dict(tag=tag, guarded_tag=f"{tag}{SUFFIX}", written=str(dest), **summary)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: guard each tag's answer file and print one JSON summary line per tag.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    :return: Exit code (2 when a tag was refused).
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", required=True, help="answer file stems under <benchmark>/<outputs>/")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--outputs", default="outputs", help="subdirectory of the benchmark that holds the answer files")
    parser.add_argument("--require-images", type=int, default=0, help="refuse a file with fewer answered images (117 = first 60 documents)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing guarded file")
    parser.add_argument("--run-len", type=int, default=RUN_LEN)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-repeat", type=float, default=MIN_REPEAT)
    parser.add_argument("--no-gate", action="store_true", help="guard every answer, not only those the scorer flags as loops")
    args = parser.parse_args(argv)
    params = {"run_len": args.run_len, "window": args.window, "min_repeat": args.min_repeat,
              "gate": None if args.no_gate else LOOP_REPEAT_SHARE}
    code = 0
    for tag in args.tags:
        try:                                     # one refused tag must not stop the others
            print(json.dumps(guard_file(args.benchmark / args.outputs, tag, args.require_images, args.overwrite, **params)))
        except ValueError as error:
            print(json.dumps({"tag": tag, "refused": str(error)}))
            code = 2
    return code


if __name__ == "__main__":
    sys.exit(main())
