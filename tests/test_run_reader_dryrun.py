# File name: test_run_reader_dryrun.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests of the page reader's dry run on the ablation panel: rows, resume, decisions and the scored report, with fakes."""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_prompt_ablation as rpa
from src.datasets.evaluations.helper_eval_scripts import run_reader_dryrun as rrd
from src.models.ocr import page_reader as pr

MODEL = "qwen3-vl-8b-heb-v25a-step1200"
RECTO = ("بسم الله الرحمن الرحيم هذا ما اشترى الشيخ ابو الفضل بن يوسف من القاضي ابي علي دارا في سوق الوراقين "
         "بمدينة مصر بثمن مبلغه عشرون دينارا")
VERSO = ("وصل كتابك ايها الاخ العزيز وفهمت ما ذكرته من امر القماش والحرير والفلفل الذي حملته الى الاسكندرية "
         "في المركب الكبير")
HEBREW = "שלום עליכם אחי היקר קיבלתי את מכתבך ושמחתי מאד לשמוע ממך ומשלומך ומשלום כל בני ביתך"
PHRASE = "وصلواته على سيدنا محمد النبي "
PANEL = [{"doc_id": "a", "image_index": 0, "image_file": "a__0.jpg", "image_class": "covered"},
         {"doc_id": "a", "image_index": 1, "image_file": "a__1.jpg", "image_class": "hebrew_side"},
         {"doc_id": "b", "image_index": 0, "image_file": "b__0.jpg", "image_class": "blank_faint"},
         {"doc_id": "c", "image_index": 0, "image_file": "c__0.jpg", "image_class": "covered"}]
ROW_FIELDS = {"doc_id", "image_index", "image_file", "image_class", "text", "finish_reason", "prompt_tokens", "completion_tokens", "secs",
              "model", "prompt", "max_tokens", "temperature", *rrd.READER_FIELDS}


def _read(text: Optional[str], finish: Optional[str] = "stop", **over: Any) -> Dict[str, Any]:
    """A :func:`page_reader.read_page` result.

    :param text: Stored text.
    :type text: Optional[str]
    :param finish: Finish reason.
    :type finish: Optional[str]
    :param over: Field overrides.
    :type over: Any
    :return: The read.
    :rtype: Dict[str, Any]
    """
    read = {"text": text, "raw_text": None, "reason": None, "blank": False, "looped": False, "cut_at": None, "letters_removed": 0,
            "repeat_share": 0.0, "finish_reason": finish, "prompt_tokens": 6500, "completion_tokens": 200, "total_tokens": 6700,
            "attempts": 1, "retried": False, "prompt": "script_neutral", "max_tokens": 1200, "temperature": 0.1, "model": MODEL,
            "ink": {"mask_ok": True, "ink_permille": 90.0, "class": "text"}, "secs": 25.0}
    read.update(over)
    assert set(read) == set(pr.READ_FIELDS)
    return read


BLANK_READ = _read("", None, reason="blank", blank=True, prompt_tokens=None, completion_tokens=None, total_tokens=None, attempts=0,
                   ink={"mask_ok": True, "ink_permille": 1.0, "class": "blank_or_faint"}, secs=0.4)
LOOPED_READ = _read(RECTO, "length", raw_text=RECTO + "\n" + PHRASE * 25, looped=True, cut_at=len(RECTO) + 1 + len(PHRASE),
                    letters_removed=24 * 23, repeat_share=0.9, attempts=2, retried=True, temperature=0.0, completion_tokens=1200)
FAILED_READ = _read(None, None, reason="failed", prompt_tokens=None, completion_tokens=None, total_tokens=None)


class FakeReader:
    """A fake ``read_page`` answering by image file name and recording its calls."""

    def __init__(self, reads: Dict[str, Dict[str, Any]]) -> None:
        """
        :param reads: Read per image file name (default: a clean Arabic read).
        :type reads: Dict[str, Dict[str, Any]]
        """
        self.reads = reads
        self.calls: List[Tuple[str, str, pr.ReaderSettings]] = []

    async def __call__(self, model: str, image_path: str, settings: pr.ReaderSettings) -> Dict[str, Any]:
        """Answer one image.

        :return: The read.
        :rtype: Dict[str, Any]
        """
        self.calls.append((model, image_path, settings))
        return dict(self.reads.get(Path(image_path).name, _read(VERSO)))


def _rows(path: Path) -> List[Dict[str, Any]]:
    """JSON lines of a file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(tmp_path: Path, reader: FakeReader, served: bool = True, **kwargs: Any) -> Dict[str, int]:
    """Run the dry run with fakes."""
    return asyncio.run(rrd.run(PANEL, MODEL, tmp_path / "out", tmp_path, reader=reader, served_check=lambda m: served,
                               free_check=lambda: 99.0, status_check=lambda: [], wait_s=0.0, **kwargs))


# --- rows ---------------------------------------------------------------------------------------------------------------

def test_row_for_has_the_ablation_layout_plus_the_reader_fields() -> None:
    row = rrd.row_for(PANEL[0], LOOPED_READ, MODEL)
    assert set(row) == ROW_FIELDS
    assert {k: row[k] for k in ("doc_id", "image_index", "image_file", "image_class", "text", "finish_reason", "prompt_tokens",
                                "completion_tokens", "secs", "model", "prompt", "max_tokens", "temperature")} == {
        "doc_id": "a", "image_index": 0, "image_file": "a__0.jpg", "image_class": "covered", "text": RECTO, "finish_reason": "length",
        "prompt_tokens": 6500, "completion_tokens": 1200, "secs": 25.0, "model": MODEL, "prompt": "reader", "max_tokens": 1200,
        "temperature": 0.0}
    assert row["looped"] and row["retried"] and row["raw_text"].startswith(RECTO) and row["cut_at"] == LOOPED_READ["cut_at"]
    assert rrd.row_for(PANEL[2], BLANK_READ, MODEL)["text"] == "" and rrd.row_for(PANEL[2], BLANK_READ, MODEL)["reason"] == "blank"


def test_raw_rows_undo_the_cut_only_where_one_was_made() -> None:
    rows = {("a", 0): rrd.row_for(PANEL[0], LOOPED_READ, MODEL), ("b", 0): rrd.row_for(PANEL[2], BLANK_READ, MODEL),
            ("c", 0): rrd.row_for(PANEL[3], _read(VERSO), MODEL)}
    raw = rrd.raw_rows(rows)
    assert set(raw) == set(rows) and all(r["prompt"] == "reader-raw" for r in raw.values())
    assert raw[("a", 0)]["text"] == RECTO + "\n" + PHRASE * 25 and raw[("b", 0)]["text"] == "" and raw[("c", 0)]["text"] == VERSO
    assert rows[("a", 0)]["text"] == RECTO                                                        # the input is untouched


# --- run ------------------------------------------------------------------------------------------------------------------

def test_run_reads_every_image_once_writes_rows_counts_decisions_and_resumes(tmp_path: Path) -> None:
    reader = FakeReader({"a__0.jpg": LOOPED_READ, "b__0.jpg": BLANK_READ})
    settings = pr.ReaderSettings(max_tokens=1200)
    stats = _run(tmp_path, reader, settings=settings)
    assert stats == {"answered": 4, "failed": 0, "skipped": 0, "stopped": 0, "yield_s": 0, "blank": 1, "looped": 1, "retried": 1}
    assert [Path(c[1]) for c in reader.calls] == [tmp_path / "images" / p["image_file"] for p in PANEL]
    assert all(c[0] == MODEL and c[2] is settings for c in reader.calls)
    out = rpa.output_path(tmp_path / "out", MODEL, "reader", 1200)
    rows = _rows(out)
    assert [set(r) for r in rows] == [ROW_FIELDS] * 4
    assert [(r["doc_id"], r["image_index"], r["prompt"], r["max_tokens"]) for r in rows] == [("a", 0, "reader", 1200), ("a", 1, "reader", 1200),
                                                                                             ("b", 0, "reader", 1200), ("c", 0, "reader", 1200)]
    assert rows[0]["looped"] and rows[0]["raw_text"] and rows[1]["text"] == VERSO and rows[2]["reason"] == "blank" and rows[2]["text"] == ""
    again = _run(tmp_path, reader)                                       # everything stored, nothing re-read
    assert again["skipped"] == 4 and again["answered"] == 0 and len(reader.calls) == 4


def test_run_retries_failed_reads_next_time_and_stops_when_the_model_vanishes(tmp_path: Path) -> None:
    reader = FakeReader({"a__1.jpg": FAILED_READ})
    stats = _run(tmp_path, reader)
    assert (stats["answered"], stats["failed"], stats["stopped"]) == (3, 1, 0)
    rows = _rows(rpa.output_path(tmp_path / "out", MODEL, "reader", 1200))
    assert rows[1]["text"] is None and rows[1]["reason"] == "failed"
    again = _run(tmp_path, FakeReader({}))
    assert (again["answered"], again["skipped"]) == (1, 3)                 # only the failed image is read again
    stopped = _run(tmp_path / "second", FakeReader({"a__1.jpg": FAILED_READ}), served=False)
    assert stopped["stopped"] == 1 and stopped["answered"] == 1 and len(_rows(rpa.output_path(tmp_path / "second" / "out", MODEL, "reader", 1200))) == 1


def test_run_yields_while_another_model_is_busy(tmp_path: Path) -> None:
    states = iter([[{"identifier": "qwen/prod", "status": "generating"}], [{"identifier": "qwen/prod", "status": "idle"}]])
    seen: List[Any] = []

    def status() -> List[Dict[str, Any]]:
        state = next(states, [])
        seen.append(state)
        return state

    stats = asyncio.run(rrd.run(PANEL[:1], MODEL, tmp_path / "out", tmp_path, reader=FakeReader({}), served_check=lambda m: True,
                                free_check=lambda: 99.0, status_check=status, wait_s=0.0))
    assert stats["answered"] == 1 and stats["yield_s"] == 0 and len(seen) == 2


# --- decisions --------------------------------------------------------------------------------------------------------------

def test_decisions_and_gate_check() -> None:
    rows = {("a", 0): rrd.row_for(PANEL[0], LOOPED_READ, MODEL), ("a", 1): rrd.row_for(PANEL[1], _read(HEBREW), MODEL),
            ("b", 0): rrd.row_for(PANEL[2], BLANK_READ, MODEL), ("c", 0): rrd.row_for(PANEL[3], FAILED_READ, MODEL)}
    d = rrd.decisions(PANEL, rows)
    assert set(d) == {"covered", "hebrew_side", "blank_faint", "arabic_unedited", "other_uncovered", "all"}
    assert d["covered"] == {"images": 2, "answered": 1, "blank_gated": 0, "failed": 1, "looped": 1, "cut": 1, "retried": 1, "length": 1,
                            "tokens_mean": 1200.0, "secs_mean": 25.0}
    assert d["blank_faint"] == {"images": 1, "answered": 1, "blank_gated": 1, "failed": 0, "looped": 0, "cut": 0, "retried": 0, "length": 0,
                                "tokens_mean": None, "secs_mean": 0.4}
    assert d["arabic_unedited"]["images"] == 0 and d["all"]["images"] == 4 and d["all"]["answered"] == 3
    line = rrd.gate_check(d)
    assert line.startswith("Blank gate: 1 of 1 blank / faint images gated; 0 of 2 covered images gated (OK)")
    bad = dict(d, covered=dict(d["covered"], blank_gated=1))
    assert "NOT OK" in rrd.gate_check(bad)
    table = rrd.decisions_table(d)
    assert table.splitlines()[0].startswith("| class | images |") and "| blank_faint | 1 | 1 | 1 |" in table and "| all | 4 | 3 |" in table


# --- score ---------------------------------------------------------------------------------------------------------------------

def _record(doc_id: str, sections: Sequence[Tuple[str, str]], images: int) -> Dict[str, Any]:
    """A benchmark record (as in ``test_score_prompt_ablation``)."""
    return {"id": doc_id, "pgp_side": "",
            "images": [{"file": f"{doc_id}__{i}.jpg", "image_index": i, "label": "", "side": ""} for i in range(images)],
            "gt": {"sections": [{"side": side, "lines": [text]} for side, text in sections]}}


def _coverage(doc_id: str, per_image: Sequence[Tuple[str, Sequence[int], str]]) -> List[Dict[str, Any]]:
    """v2 coverage image rows of a document."""
    return [{"kind": "image", "doc_id": doc_id, "image_index": i, "role": role, "status": role, "sections": list(sections),
             "method": "manual", "copy_of": None, "duplicate_of": None, "joined_with": [], "script": script}
            for i, (role, sections, script) in enumerate(per_image)]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _benchmark(tmp_path: Path) -> Tuple[Path, Path]:
    """A synthetic benchmark directory with two ablation arms, and its panel file.

    :param tmp_path: Root.
    :type tmp_path: Path
    :return: ``(benchmark dir, panel file)``.
    :rtype: Tuple[Path, Path]
    """
    bench = tmp_path / "bench"
    _write_jsonl(bench / "benchmark.jsonl", [_record("a", [("", RECTO), ("", VERSO)], 2), _record("b", [("", RECTO)], 1),
                                             _record("c", [("", VERSO)], 1)])
    _write_jsonl(bench / "coverage_v2.jsonl", _coverage("a", [("covered", [0], "arabic"), ("uncovered", [], "hebrew")])
                 + _coverage("b", [("blank_or_faint", [], "none")]) + _coverage("c", [("covered", [0], "arabic")]))
    panel = bench / "panels" / "p.jsonl"
    _write_jsonl(panel, PANEL)
    texts = {("a", 0): RECTO[:60], ("a", 1): VERSO, ("b", 0): PHRASE * 3, ("c", 0): VERSO[:40]}
    for prompt, budget in rrd.REFERENCES[:1] + rrd.REFERENCES[2:]:
        _write_jsonl(rpa.output_path(bench / rpa.DEFAULT_OUT_DIR, MODEL, prompt, budget),
                     [{"doc_id": d, "image_index": i, "text": t, "finish_reason": "stop", "prompt_tokens": 800, "completion_tokens": 20,
                       "secs": 1.0, "model": MODEL, "prompt": prompt, "max_tokens": budget} for (d, i), t in texts.items()])
    return bench, panel


def test_score_writes_the_report_next_to_the_ablation_arms(tmp_path: Path) -> None:
    bench, panel = _benchmark(tmp_path)
    reads = {"a__0.jpg": LOOPED_READ, "a__1.jpg": _read(HEBREW), "b__0.jpg": BLANK_READ, "c__0.jpg": _read(VERSO)}
    out = bench / "dry"
    asyncio.run(rrd.run(PANEL, MODEL, out, bench, reader=FakeReader(reads), served_check=lambda m: True, free_check=lambda: 99.0,
                        status_check=lambda: [], wait_s=0.0))
    report, summary = rrd.score(panel, bench, out, MODEL, 1200)
    tags = [f"{MODEL}__trained__mt2500", f"{MODEL}__noscript__mt1200", f"{MODEL}__reader__mt1200", f"{MODEL}__reader-raw__mt1200"]
    assert list(summary["arms"]) == tags                                   # the missing trained / 1200 arm is left out, in order
    assert summary["reader_tag"] == tags[2] and summary["decisions"]["all"]["blank_gated"] == 1
    reader, raw = summary["arms"][tags[2]], summary["arms"][tags[3]]
    assert reader["covered"]["documents"] == 2 and reader["blank_faint"]["abstain"] == 1.0 and reader["hebrew_side"]["hebrew_output"] == 1.0
    assert reader["covered"]["images"]["finish"] == {"length": 1, "stop": 1} and reader["covered"]["images"]["loop"] == 0.0
    assert raw["covered"]["images"]["loop"] == 0.5                          # the un-cut text loops; the stored read does not
    for heading in ("# Production page reader: dry run", "## Reader decisions", "Blank gate: 1 of 1 blank / faint images gated; 0 of 2 covered",
                    "## Reader vs every other arm", "| v2.5a-1200 / trained / 2500 |", "| v2.5a-1200 / reader-raw / 1200 |",
                    "## Table 1: prompt × model × image class", "## Table 3: paired comparisons"):
        assert heading in report
    assert (out / "report.md").read_text(encoding="utf-8") == report
    assert {p.name for p in out.iterdir()} >= {"report.md", "summary.json", "images.csv", "documents.csv",
                                               f"{MODEL}__reader__mt1200.jsonl", f"{MODEL}__reader-raw__mt1200.jsonl"}
    raw_rows = _rows(out / f"{MODEL}__reader-raw__mt1200.jsonl")
    assert raw_rows[0]["text"] == LOOPED_READ["raw_text"] and raw_rows[0]["prompt"] == "reader-raw"


def test_score_needs_the_reader_rows(tmp_path: Path) -> None:
    bench, panel = _benchmark(tmp_path)
    with pytest.raises(FileNotFoundError):
        rrd.score(panel, bench, bench / "dry", MODEL, 1200)
