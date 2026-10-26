# File name: test_kraken_arabic_runners.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the Kraken Arabic readers' pure helpers (benchmark runner + edition-page segmenter); no kraken, no NAS."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_arabic_benchmark_kraken as rk
from src.finetuning.kraken import segment_edition_pages as sep

REPO = Path(__file__).resolve().parents[1]


def fake_record(text: str, confidences: Optional[List[float]] = None, baseline: Any = None, boundary: Any = None) -> SimpleNamespace:
    """A stand-in for a kraken recognition record.

    :param text: Prediction.
    :type text: str
    :param confidences: Per-character confidences.
    :type confidences: Optional[List[float]]
    :param baseline: Baseline points.
    :type baseline: Any
    :param boundary: Boundary points.
    :type boundary: Any
    :return: Object with ``prediction``, ``confidences``, ``baseline`` and ``boundary``.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(prediction=text, confidences=confidences if confidences is not None else [0.9] * len(text),
                           baseline=baseline if baseline is not None else [(10.4, 20.6), (110.5, 21.0)],
                           boundary=boundary if boundary is not None else [(10, 5), (110, 5), (110, 30), (10, 30)])


def fake_page(texts: List[str], n_dropped: int = 0) -> rk.PageRead:
    """A page read built from fake records.

    :param texts: Line predictions in order.
    :type texts: List[str]
    :param n_dropped: Lines dropped before recognition.
    :type n_dropped: int
    :return: Page read.
    :rtype: rk.PageRead
    """
    lines = [rk.line_read(fake_record(text)) for text in texts]
    return rk.PageRead(lines=lines, n_lines=len(lines) + n_dropped, n_dropped=n_dropped, image_size=(400, 600))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as JSON lines.

    :param path: Target file.
    :type path: Path
    :param rows: Rows.
    :type rows: List[Dict[str, Any]]
    :return: The path.
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def bench(tmp_path: Path) -> Path:
    """A three-document benchmark directory (no images needed: reads are faked).

    :param tmp_path: pytest temporary directory.
    :type tmp_path: Path
    :return: The benchmark directory.
    :rtype: Path
    """
    records = [{"id": "doc_a", "images": [{"file": "a__0.jpg", "image_index": 0}, {"file": "a__1.jpg", "image_index": 1}],
                "gt": {"sections": []}},
               {"id": "doc_b", "images": [{"file": "b__0.jpg", "image_index": 0}], "gt": {"sections": []}},
               {"id": "doc_c", "images": [{"file": "c__0.jpg", "image_index": "0"}], "gt": {"sections": []}}]
    write_jsonl(tmp_path / "benchmark.jsonl", records)
    return tmp_path


# ---------------------------------------------------------------- benchmark runner


def test_modules_import_without_kraken_or_torch() -> None:
    """Both modules import in a fresh interpreter without pulling in kraken or torch."""
    code = ("import sys\n"
            "import src.datasets.evaluations.helper_eval_scripts.run_arabic_benchmark_kraken\n"
            "import src.finetuning.kraken.segment_edition_pages\n"
            "assert 'kraken' not in sys.modules and 'torch' not in sys.modules, sorted(sys.modules)\n")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_doc_ids_from_keeps_first_seen_order(tmp_path: Path) -> None:
    """doc_ids_from returns unique ids in first-seen order, None without a file, [] for an empty one."""
    answers = write_jsonl(tmp_path / "answers.jsonl", [{"doc_id": "doc_b", "image_index": 0}, {"doc_id": "doc_a", "image_index": 0},
                                                       {"doc_id": "doc_b", "image_index": 1}])
    with open(answers, "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert rk.doc_ids_from(answers) == ["doc_b", "doc_a"]
    assert rk.doc_ids_from(None) is None
    assert rk.doc_ids_from(write_jsonl(tmp_path / "empty.jsonl", [])) == []


def test_select_ids_limit_and_unknown(bench: Path) -> None:
    """select_ids keeps the requested order, applies the limit and reports unknown ids."""
    records = rk.load_records(bench)
    assert list(records) == ["doc_a", "doc_b", "doc_c"]
    assert rk.select_ids(records, None) == (["doc_a", "doc_b", "doc_c"], [])
    assert rk.select_ids(records, None, limit=2) == (["doc_a", "doc_b"], [])
    assert rk.select_ids(records, ["doc_c", "nope", "doc_a"], limit=1) == (["doc_c"], ["nope"])
    assert rk.select_ids(records, []) == ([], [])          # an empty --ids-from file selects nothing, not everything


def test_done_pairs_counts_only_text_answers(tmp_path: Path) -> None:
    """done_pairs counts rows with a text answer (even empty) and not failed rows."""
    out = tmp_path / "outputs_other" / "t.jsonl"
    assert rk.done_pairs(out) == set()
    write_jsonl(out, [{"doc_id": "doc_a", "image_index": 0, "text": ""},
                      {"doc_id": "doc_a", "image_index": "1", "text": None, "error": "boom"},
                      {"doc_id": "doc_b", "image_index": 0, "text": "نص"}])
    assert rk.done_pairs(out) == {("doc_a", 0), ("doc_b", 0)}


@pytest.mark.parametrize("cut", [2, 3])        # 2: JSON cut inside the string; 3: also half of an Arabic character
def test_resume_survives_a_torn_last_row(bench: Path, cut: int) -> None:
    """A last row torn by a kill is ignored with a warning, cut off before appending, and its page is read again."""
    out = rk.output_path(bench, "torn")
    write_jsonl(out, [{"doc_id": "doc_a", "image_index": 0, "text": "نص"}])
    torn = json.dumps({"doc_id": "doc_a", "image_index": 1, "text": "سطر"}, ensure_ascii=False).encode("utf-8")
    with open(out, "ab") as fh:
        fh.write(torn[:-cut])
    warnings: List[str] = []
    assert rk.done_pairs(out, log=warnings.append) == {("doc_a", 0)}
    assert len(warnings) == 1 and "torn" in warnings[0]
    records = rk.load_records(bench)
    jobs = rk.build_jobs(records, ["doc_a"], rk.done_pairs(out, log=lambda _: None))
    assert jobs == [rk.ImageJob("doc_a", 1, "a__1.jpg")]
    assert rk.run_jobs(jobs, lambda path: fake_page(["ثان"]), bench / "images", out, "s", log=lambda _: None) == \
        {"answered": 1, "failed": 0}
    rows = rk.read_jsonl(out)                     # every line decodes: the new row did not glue onto the torn one
    assert [(r["doc_id"], r["image_index"], r["text"]) for r in rows] == [("doc_a", 0, "نص"), ("doc_a", 1, "ثان")]
    assert rk.done_pairs(out) == {("doc_a", 0), ("doc_a", 1)}


def test_done_pairs_raises_on_a_bad_row_before_the_last(tmp_path: Path) -> None:
    """Only the last row may be torn: an undecodable row elsewhere stops the resume."""
    out = tmp_path / "t.jsonl"
    out.write_text('{"doc_id": "doc_a", "image_index": 0, "text": "x"}\n{"doc_id": "doc_a", "ima\n'
                   '{"doc_id": "doc_b", "image_index": 0, "text": "y"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        rk.done_pairs(out, log=lambda _: None)


def test_end_on_line_boundary_keeps_a_complete_row_without_newline(tmp_path: Path) -> None:
    """A last row that decodes but lacks its newline keeps its content and gets the newline; no warning."""
    out = tmp_path / "t.jsonl"
    rk.end_on_line_boundary(out)                   # missing file: nothing to do
    out.write_text('{"doc_id": "doc_a", "image_index": 0, "text": "نص"}', encoding="utf-8")
    warnings: List[str] = []
    rk.end_on_line_boundary(out, log=warnings.append)
    assert out.read_text(encoding="utf-8") == '{"doc_id": "doc_a", "image_index": 0, "text": "نص"}\n' and not warnings
    assert rk.done_pairs(out) == {("doc_a", 0)}


def test_build_jobs_order_and_resume(bench: Path) -> None:
    """build_jobs follows the id order, skips done pairs and unknown ids, and casts image indices to int."""
    records = rk.load_records(bench)
    jobs = rk.build_jobs(records, ["doc_b", "doc_a", "missing", "doc_c"], {("doc_a", 0)})
    assert jobs == [rk.ImageJob("doc_b", 0, "b__0.jpg"), rk.ImageJob("doc_a", 1, "a__1.jpg"), rk.ImageJob("doc_c", 0, "c__0.jpg")]
    assert isinstance(jobs[-1].image_index, int)


def test_output_path_and_tag_check(tmp_path: Path) -> None:
    """output_path builds <bench>/outputs_other/<tag>.jsonl and rejects unsafe tags."""
    assert rk.output_path(tmp_path, "ppocrv6m_zs") == tmp_path / "outputs_other" / "ppocrv6m_zs.jsonl"
    for bad in ("", "a/b", "..", ".hidden"):
        with pytest.raises(ValueError):
            rk.output_path(tmp_path, bad)


def test_default_source_and_args() -> None:
    """The provenance string and the shared reader options have the documented defaults."""
    assert rk.default_source("/segmodels/blla_2026/blla.mlmodel", "/ar/ppocrv6_medium.safetensors", "rgb", "binarized", "7.1.1") == \
        "blla.mlmodel + ppocrv6_medium.safetensors, seg rgb, rec binarized, kraken 7.1.1"
    parser = rk.argparse.ArgumentParser()
    rk.add_reader_arguments(parser)
    args = parser.parse_args(["--tag", "t", "--segmenter", "default", "--recogniser", "/ar/r.safetensors"])
    assert (args.seg_input, args.rec_input, args.limit, args.threads, args.source) == ("rgb", "rgb", 0, 4, "")
    assert rk.source_from_args(args, version="7.1.1") == "default + r.safetensors, seg rgb, rec rgb, kraken 7.1.1"
    args.source = "custom"
    assert rk.source_from_args(args) == "custom"
    with pytest.raises(SystemExit):
        parser.parse_args(["--tag", "t", "--segmenter", "default", "--recogniser", "r", "--seg-input", "gray"])


def test_load_reader_rejects_unknown_mode_before_importing_kraken() -> None:
    """load_reader rejects an unknown input mode before any kraken import."""
    with pytest.raises(ValueError):
        rk.load_reader("default", "r.safetensors", seg_input="gray")


def test_line_read_converts_geometry_and_confidence() -> None:
    """line_read rounds geometry to integers and averages confidences (None when empty)."""
    line = rk.line_read(fake_record("سلام", confidences=[0.5, 1.0, 0.75, 0.25]))
    assert line.baseline == [[10, 21], [110, 21]]
    assert line.boundary == [[10, 5], [110, 5], [110, 30], [10, 30]]
    assert line.text == "سلام" and line.mean_confidence == 0.625
    empty = rk.line_read(SimpleNamespace(prediction="", confidences=[], baseline=None, boundary=None))
    assert (empty.baseline, empty.boundary, empty.mean_confidence) == ([], [], None)


def test_keep_bounded_lines_drops_and_casts() -> None:
    """keep_bounded_lines drops lines without a usable boundary and casts the rest to integer tuples."""
    lines = [SimpleNamespace(boundary=None), SimpleNamespace(boundary=[(0.4, 0.6), (5.5, 1.2), (5.0, 9.9)]),
             SimpleNamespace(boundary=[(1, 1), (2, 2)]), SimpleNamespace(boundary=[])]
    kept = rk.keep_bounded_lines(lines)
    assert kept == [lines[1]]
    assert kept[0].boundary == [(0, 1), (6, 1), (5, 10)] and all(isinstance(v, int) for p in kept[0].boundary for v in p)


def test_page_text_skips_blank_lines() -> None:
    """page_text joins the non-blank line predictions with newlines."""
    assert rk.page_text(fake_page(["الاول", " ", "الثاني"])) == "الاول\nالثاني"


def test_answer_row_shape() -> None:
    """answer_row has the scorer's fields, plus error and null text for a failed page."""
    job = rk.ImageJob("doc_a", 1, "a__1.jpg")
    row = rk.answer_row(job, fake_page(["نص"], n_dropped=2), 3.2, "src")
    assert row == {"doc_id": "doc_a", "image_index": 1, "text": "نص", "secs": 3.2, "source": "src", "n_lines": 3, "n_dropped": 2}
    failed = rk.answer_row(job, None, 0.1, "src", "RuntimeError('x')")
    assert failed["text"] is None and failed["error"] == "RuntimeError('x')" and (failed["n_lines"], failed["n_dropped"]) == (0, 0)


def test_run_jobs_writes_rows_and_failed_pages_are_retried(bench: Path) -> None:
    """run_jobs appends one row per image and a failed image is retried on the next run."""
    records = rk.load_records(bench)
    out = rk.output_path(bench, "fake")
    seen: List[Path] = []

    def flaky(path: Path) -> rk.PageRead:
        """Fake reader that fails on one image.

        :param path: Image path.
        :type path: Path
        :return: Fake page.
        :rtype: rk.PageRead
        """
        seen.append(path)
        if path.name == "b__0.jpg":
            raise RuntimeError("segmenter crashed")
        return fake_page([f"سطر {path.stem}"])

    jobs = rk.build_jobs(records, list(records), rk.done_pairs(out))
    stats = rk.run_jobs(jobs, flaky, bench / "images", out, "fake source", log=lambda _: None)
    assert stats == {"answered": 3, "failed": 1}
    assert seen[0] == bench / "images" / "a__0.jpg"
    rows = rk.read_jsonl(out)
    assert [(r["doc_id"], r["image_index"]) for r in rows] == [("doc_a", 0), ("doc_a", 1), ("doc_b", 0), ("doc_c", 0)]
    assert rows[2]["text"] is None and "segmenter crashed" in rows[2]["error"]
    assert set(rows[0]) == {"doc_id", "image_index", "text", "secs", "source", "n_lines", "n_dropped"}

    retry = rk.build_jobs(records, list(records), rk.done_pairs(out))
    assert retry == [rk.ImageJob("doc_b", 0, "b__0.jpg")]
    assert rk.run_jobs(retry, lambda path: fake_page(["ثانية"]), bench / "images", out, "s", log=lambda _: None) == {"answered": 1, "failed": 0}
    assert rk.build_jobs(records, list(records), rk.done_pairs(out)) == []


def test_runner_main_without_pending_jobs_never_loads_models(bench: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner's main returns before loading models when every image is answered."""
    write_jsonl(rk.output_path(bench, "t"), [{"doc_id": "doc_b", "image_index": 0, "text": "x"}])
    ids_from = write_jsonl(bench / "ids.jsonl", [{"doc_id": "doc_b"}, {"doc_id": "unknown"}])

    def boom(_: Any) -> None:
        """Fail if the models would be loaded.

        :param _: Parsed arguments.
        :type _: Any
        :return: None.
        :rtype: None
        """
        raise AssertionError("models must not load when nothing is pending")

    monkeypatch.setattr(rk, "reader_from_args", boom)
    rk.main(["--bench", str(bench), "--tag", "t", "--segmenter", "default", "--recogniser", "r", "--ids-from", str(ids_from)])


# ---------------------------------------------------------------- edition-page segmenter


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    """A four-page edition manifest; the image of pgpid 9000 is missing.

    :param tmp_path: pytest temporary directory.
    :type tmp_path: Path
    :return: The manifest path.
    :rtype: Path
    """
    images = tmp_path / "pgp_arabic_editions_v2_images"
    images.mkdir()
    rows = []
    for pgpid, index, lines in (("8317", 0, 34), ("8317", 1, 19), ("8315", 0, 0), ("9000", 0, 12)):
        name = f"{pgpid}_{index}.jpg"
        if pgpid != "9000":
            (images / name).write_bytes(b"jpeg")
        rows.append({"stem": f"pgpar_{pgpid}_{index}_page", "pgpid": pgpid, "canonical_id": f"Cambridge_{pgpid}", "image_index": index,
                     "image_path": f"/Volumes/home/studio_offload/datasets/pgp_arabic_editions_v2_images/{name}", "lines": lines})
    return write_jsonl(tmp_path / "pgp_arabic_editions_v2" / "manifest.jsonl", rows)


def test_read_ids_and_select_pages(tmp_path: Path, manifest: Path) -> None:
    """read_ids ignores comments and blanks; select_pages matches stem, pgpid or canonical id in manifest order."""
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# keep these\npgpar_8315_0_page\n\n8317   # whole document\n", encoding="utf-8")
    ids = sep.read_ids(ids_file)
    assert ids == {"pgpar_8315_0_page", "8317"}
    assert sep.read_ids(None) is None
    rows = rk.read_jsonl(manifest)
    assert [r["stem"] for r in sep.select_pages(rows, ids)] == ["pgpar_8317_0_page", "pgpar_8317_1_page", "pgpar_8315_0_page"]
    assert [r["stem"] for r in sep.select_pages(rows, {"Cambridge_9000"})] == ["pgpar_9000_0_page"]
    assert [r["stem"] for r in sep.select_pages(rows, None, limit=2)] == ["pgpar_8317_0_page", "pgpar_8317_1_page"]


def test_resolve_image_and_split_missing(tmp_path: Path, manifest: Path) -> None:
    """resolve_image honours --image-root and split_missing separates rows without an image."""
    rows = rk.read_jsonl(manifest)
    root = tmp_path / "pgp_arabic_editions_v2_images"
    assert sep.resolve_image(rows[0]) == Path(rows[0]["image_path"])
    assert sep.resolve_image(rows[0], root) == root / "8317_0.jpg"
    jobs, missing = sep.split_missing(rows, root)
    assert [(row["stem"], path.name) for row, path in jobs] == [("pgpar_8317_0_page", "8317_0.jpg"), ("pgpar_8317_1_page", "8317_1.jpg"),
                                                                ("pgpar_8315_0_page", "8315_0.jpg")]
    assert [row["stem"] for row in missing] == ["pgpar_9000_0_page"]


def test_page_json_shape_and_summary(manifest: Path) -> None:
    """page_json and summary_row have the documented fields and plain JSON types."""
    row = rk.read_jsonl(manifest)[0]
    page = rk.PageRead(lines=[rk.line_read(fake_record("سطر اول", confidences=[1.0, 0.5])), rk.line_read(fake_record("", confidences=[]))],
                       n_lines=3, n_dropped=1, image_size=(4000, 5200))
    payload = sep.page_json(row, page, 12.3, "seg + rec")
    assert set(payload) == {"stem", "pgpid", "canonical_id", "image_path", "image_size", "expected_lines", "n_lines", "n_dropped", "secs",
                            "source", "lines"}
    assert (payload["stem"], payload["pgpid"], payload["canonical_id"]) == ("pgpar_8317_0_page", "8317", "Cambridge_8317")
    assert payload["image_path"] == row["image_path"] and payload["image_size"] == [4000, 5200]
    assert (payload["expected_lines"], payload["n_lines"], payload["n_dropped"], payload["secs"]) == (34, 3, 1, 12.3)
    assert payload["lines"][0] == {"index": 0, "baseline": [[10, 21], [110, 21]], "boundary": [[10, 5], [110, 5], [110, 30], [10, 30]],
                                   "text": "سطر اول", "mean_confidence": 0.75}
    assert payload["lines"][1]["index"] == 1 and payload["lines"][1]["text"] == "" and payload["lines"][1]["mean_confidence"] is None
    json.dumps(payload)                                                       # plain JSON types only
    assert sep.summary_row(row, page, 12.3) == {"stem": "pgpar_8317_0_page", "expected_lines": 34, "n_lines": 3, "n_dropped": 1,
                                                "ratio": 0.088, "secs": 12.3}
    assert sep.summary_row({"stem": "s", "lines": 0}, page, 1.0)["ratio"] is None


def test_page_json_path_rejects_unsafe_stems(tmp_path: Path) -> None:
    """page_json_path and tag_dir reject stems and tags that are not plain file names."""
    assert sep.page_json_path(tmp_path, "pgpar_1_0_page") == tmp_path / "pgpar_1_0_page.json"
    for bad in ("", "../x", "a/b", ".x"):
        with pytest.raises(ValueError):
            sep.page_json_path(tmp_path, bad)
    with pytest.raises(ValueError):
        sep.tag_dir(tmp_path, "a/b")


def test_run_pages_resumes_and_retries_failures(tmp_path: Path, manifest: Path) -> None:
    """run_pages writes page JSONs and summary rows, skips done pages and retries failed ones."""
    rows = rk.read_jsonl(manifest)
    run_dir = sep.tag_dir(tmp_path / "out", "blla_ppocr")

    def flaky(row: Dict[str, Any], path: Path) -> rk.PageRead:
        """Fake reader that fails on one page.

        :param row: Manifest row.
        :type row: Dict[str, Any]
        :param path: Image path.
        :type path: Path
        :return: Fake page.
        :rtype: rk.PageRead
        """
        if path.name == "8317_1.jpg":
            raise ValueError("no lines")
        return fake_page(["سطر", "آخر"], n_dropped=1)

    jobs, missing = sep.split_missing(sep.pending_pages(rows, run_dir), tmp_path / "pgp_arabic_editions_v2_images")
    assert len(jobs) == 3 and len(missing) == 1
    assert sep.run_pages(jobs, flaky, run_dir, "src", log=lambda _: None) == {"written": 2, "failed": 1}
    assert sorted(p.name for p in run_dir.iterdir()) == ["pgpar_8315_0_page.json", "pgpar_8317_0_page.json", "summary.jsonl"]
    written = json.loads((run_dir / "pgpar_8317_0_page.json").read_text(encoding="utf-8"))
    assert [line["text"] for line in written["lines"]] == ["سطر", "آخر"] and written["n_lines"] == 3
    summary = rk.read_jsonl(run_dir / sep.SUMMARY_FILE)
    assert [(s["stem"], s["ratio"]) for s in summary] == [("pgpar_8317_0_page", 0.088), ("pgpar_8315_0_page", None)]

    pending = sep.pending_pages(rows + rows[:1], run_dir)                   # duplicates collapse; done pages skipped
    assert [row["stem"] for row in pending] == ["pgpar_8317_1_page", "pgpar_9000_0_page"]
    retry, _ = sep.split_missing(pending, tmp_path / "pgp_arabic_editions_v2_images")
    assert sep.run_pages(retry, lambda row, path: fake_page(["ثالث"]), run_dir, "src", log=lambda _: None) == {"written": 1, "failed": 0}
    assert len(rk.read_jsonl(run_dir / sep.SUMMARY_FILE)) == 3
    assert not list(run_dir.glob("*.tmp"))


def test_summary_is_rebuilt_from_page_jsons(tmp_path: Path, manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary row lost or doubled by an interrupted run is rebuilt from the page JSONs, in manifest order."""
    rows = rk.read_jsonl(manifest)
    run_dir = sep.tag_dir(tmp_path / "out", "t")
    run_dir.mkdir(parents=True)
    page = fake_page(["سطر"], n_dropped=1)
    for row in (rows[2], rows[0]):                                       # written out of manifest order
        sep.write_json_atomic(sep.page_json_path(run_dir, row["stem"]), sep.page_json(row, page, 2.5, "src"))
    expected = [sep.summary_row(rows[0], page, 2.5), sep.summary_row(rows[2], page, 2.5)]
    write_jsonl(run_dir / sep.SUMMARY_FILE, [expected[1], expected[1]])  # pgpar_8317_0_page's row lost, the other doubled
    assert sep.rebuild_summary(run_dir, rows) == 2
    assert rk.read_jsonl(run_dir / sep.SUMMARY_FILE) == expected
    assert not list(run_dir.glob("*.tmp"))

    write_jsonl(run_dir / sep.SUMMARY_FILE, expected[1:])               # main rebuilds it too, even with nothing to read
    monkeypatch.setattr(sep, "reader_from_args", lambda _: pytest.fail("models must not load when nothing is pending"))
    sep.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--tag", "t", "--segmenter", "default",
              "--recogniser", "r", "--image-root", str(tmp_path / "pgp_arabic_editions_v2_images"),
              "--ids", str(_ids(tmp_path, rows[0]["stem"], rows[2]["stem"]))])
    assert rk.read_jsonl(run_dir / sep.SUMMARY_FILE) == expected


def test_segmenter_main_without_pending_pages_never_loads_models(tmp_path: Path, manifest: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The segmenter's main returns before loading models when no page is readable."""

    def boom(_: Any) -> None:
        """Fail if the models would be loaded.

        :param _: Parsed arguments.
        :type _: Any
        :return: None.
        :rtype: None
        """
        raise AssertionError("models must not load when nothing is pending")

    monkeypatch.setattr(sep, "reader_from_args", boom)
    sep.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--tag", "t", "--segmenter", "default",
              "--recogniser", "r", "--image-root", str(tmp_path / "pgp_arabic_editions_v2_images"), "--limit", "1",
              "--ids", str(_ids(tmp_path, "9000"))])


def _ids(tmp_path: Path, *ids: str) -> Path:
    """Write an ids file.

    :param tmp_path: Directory.
    :type tmp_path: Path
    :param ids: Ids, one per line.
    :type ids: str
    :return: The file.
    :rtype: Path
    """
    path = tmp_path / "ids_main.txt"
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return path


class FakeSegmenter:
    """Segmenter stand-in: one usable line and one without a boundary; remembers the image mode it saw."""

    def __init__(self) -> None:
        """Start with no recorded calls.

        :return: None.
        :rtype: None
        """
        self.modes: List[str] = []

    def predict(self, image: Any, config: Any) -> SimpleNamespace:
        """Return a fixed segmentation.

        :param image: PIL image.
        :type image: Any
        :param config: Ignored.
        :type config: Any
        :return: Object with ``lines``.
        :rtype: SimpleNamespace
        """
        self.modes.append(image.mode)
        return SimpleNamespace(lines=[SimpleNamespace(baseline=[(2.0, 8.0), (30.0, 8.0)], boundary=[(1.6, 1), (31, 1), (31, 12), (1, 12)]),
                                      SimpleNamespace(baseline=[(2, 20), (30, 20)], boundary=None)])


class FakeRecogniser:
    """Recogniser stand-in: one record per segmentation line; remembers the image mode it saw."""

    def __init__(self, text: str = "كلمة") -> None:
        """Start with no recorded calls.

        :param text: Prediction returned for every line.
        :type text: str
        :return: None.
        :rtype: None
        """
        self.text = text
        self.modes: List[str] = []

    def predict(self, image: Any, segmentation: Any, config: Any) -> Any:
        """Yield one record per remaining line.

        :param image: PIL image.
        :type image: Any
        :param segmentation: Segmentation with ``lines``.
        :type segmentation: Any
        :param config: Ignored.
        :type config: Any
        :return: Generator of fake records.
        :rtype: Any
        """
        self.modes.append(image.mode)
        for line in segmentation.lines:
            yield fake_record(self.text, baseline=line.baseline, boundary=line.boundary)


@pytest.mark.parametrize("seg_input,rec_input,expected", [("rgb", "rgb", ("RGB", "RGB", 0)), ("binarized", "rgb", ("1", "RGB", 1)),
                                                          ("rgb", "binarized", ("RGB", "1", 1))])
def test_read_page_with_fake_kraken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seg_input: str, rec_input: str, expected: tuple) -> None:
    """read_page routes RGB or binarized images to segmenter and recogniser and drops boundary-less lines."""
    from PIL import Image

    calls: List[str] = []

    def nlbin(image: Any) -> Any:
        """Fake binarisation.

        :param image: PIL image.
        :type image: Any
        :return: Bilevel image.
        :rtype: Any
        """
        calls.append(image.mode)
        return image.convert("1")

    monkeypatch.setitem(sys.modules, "kraken", SimpleNamespace(binarization=SimpleNamespace(nlbin=nlbin)))
    path = tmp_path / "page.png"
    Image.new("L", (40, 30), 255).save(path)
    reader = rk.KrakenReader(FakeSegmenter(), None, FakeRecogniser(), None, seg_input, rec_input)
    page = rk.read_page(reader, path)
    assert (reader.segmenter.modes[0], reader.recogniser.modes[0], len(calls)) == expected
    assert (page.n_lines, page.n_dropped, page.image_size) == (2, 1, (40, 30))
    assert len(page.lines) == 1 and page.lines[0].text == "كلمة"
    assert page.lines[0].boundary == [[2, 1], [31, 1], [31, 12], [1, 12]] and page.lines[0].baseline == [[2, 8], [30, 8]]


def test_read_page_skips_recognition_without_usable_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """read_page does not call the recogniser when no line has a usable boundary."""
    from PIL import Image

    monkeypatch.setitem(sys.modules, "kraken", SimpleNamespace(binarization=None))
    path = tmp_path / "page.jpg"
    Image.new("RGB", (20, 10), (250, 240, 220)).save(path)
    segmenter = SimpleNamespace(predict=lambda image, config: SimpleNamespace(lines=[SimpleNamespace(boundary=None)]))
    recogniser = SimpleNamespace(predict=lambda *args: pytest.fail("recogniser called without lines"))
    page = rk.read_page(rk.KrakenReader(segmenter, None, recogniser, None), path)
    assert (page.lines, page.n_lines, page.n_dropped) == ([], 1, 1)


# ---------------------------------------------------------------- edition-page re-read mode (--geometry-from)


SOURCE_LINES = [{"index": 0, "baseline": [[2, 8], [30, 8]], "boundary": [[1, 1], [31, 1], [31, 12], [1, 12]], "text": "قديم",
                 "mean_confidence": 0.5},
                {"index": 1, "baseline": [[2, 22], [30, 23]], "boundary": [[1, 15], [31, 15], [31, 27], [1, 27]], "text": "",
                 "mean_confidence": None}]


def install_fake_kraken(monkeypatch: pytest.MonkeyPatch, nlbin_calls: Optional[List[str]] = None) -> None:
    """Put a fake ``kraken`` (binarization) and ``kraken.containers`` (keyword-recording classes) in ``sys.modules``.

    :param monkeypatch: pytest monkeypatch.
    :type monkeypatch: pytest.MonkeyPatch
    :param nlbin_calls: Receives the mode of every image binarised, when given.
    :type nlbin_calls: Optional[List[str]]
    :return: None.
    :rtype: None
    """

    def nlbin(image: Any) -> Any:
        """Fake binarisation.

        :param image: PIL image.
        :type image: Any
        :return: Bilevel image.
        :rtype: Any
        """
        if nlbin_calls is not None:
            nlbin_calls.append(image.mode)
        return image.convert("1")

    monkeypatch.setitem(sys.modules, "kraken", SimpleNamespace(binarization=SimpleNamespace(nlbin=nlbin)))
    monkeypatch.setitem(sys.modules, "kraken.containers", SimpleNamespace(BaselineLine=lambda **kw: SimpleNamespace(**kw),
                                                                          Segmentation=lambda **kw: SimpleNamespace(**kw)))


def source_page(row: Dict[str, Any], lines: List[Dict[str, Any]], image_size: tuple = (40, 30), n_dropped: int = 1) -> Dict[str, Any]:
    """An earlier run's page JSON for a manifest row.

    :param row: Manifest row.
    :type row: Dict[str, Any]
    :param lines: Page-JSON lines (``index`` is reassigned by position).
    :type lines: List[Dict[str, Any]]
    :param image_size: ``(width, height)``.
    :type image_size: tuple
    :param n_dropped: Lines the segmenter dropped.
    :type n_dropped: int
    :return: Page JSON as :func:`sep.page_json` writes it.
    :rtype: Dict[str, Any]
    """
    page = rk.PageRead(lines=[rk.LineRead(line["baseline"], line["boundary"], line["text"], line["mean_confidence"]) for line in lines],
                       n_lines=len(lines) + n_dropped, n_dropped=n_dropped, image_size=image_size)
    return sep.page_json(row, page, 9.9, "blla.mlmodel + ppocrv6_medium.safetensors, seg rgb, rec rgb, kraken 7.1.1")


def geometry_of(payload: Dict[str, Any]) -> List[tuple]:
    """``(index, baseline, boundary)`` of every line of a page JSON.

    :param payload: Page JSON.
    :type payload: Dict[str, Any]
    :return: One tuple per line.
    :rtype: List[tuple]
    """
    return [(line["index"], line["baseline"], line["boundary"]) for line in payload["lines"]]


def test_segmentation_from_lines_with_fake_kraken(monkeypatch: pytest.MonkeyPatch) -> None:
    """segmentation_from_lines builds a baselines Segmentation of BaselineLines in page-JSON order, with integer tuples."""
    install_fake_kraken(monkeypatch)
    lines = [dict(SOURCE_LINES[0], boundary=[[1.6, 1], [31, 1], [31, 12], [1, 12]]), SOURCE_LINES[1]]
    seg = sep.segmentation_from_lines(lines, "/data/8317_0.jpg")
    assert (seg.type, seg.imagename, seg.text_direction, seg.script_detection) == ("baselines", "/data/8317_0.jpg", "horizontal-rl", False)
    assert [(line.id, line.type) for line in seg.lines] == [("line_0", "baselines"), ("line_1", "baselines")]
    assert seg.lines[0].baseline == [(2, 8), (30, 8)] and seg.lines[0].boundary == [(2, 1), (31, 1), (31, 12), (1, 12)]
    assert seg.lines[1].baseline == [(2, 22), (30, 23)] and seg.lines[1].boundary == [(1, 15), (31, 15), (31, 27), (1, 27)]
    assert all(isinstance(v, int) for line in seg.lines for p in line.baseline + line.boundary for v in p)
    assert sep.segmentation_from_lines([]).lines == []
    with pytest.raises(ValueError):
        sep.segmentation_from_lines([dict(SOURCE_LINES[0], boundary=[[1, 1], [2, 2]])])


def test_page_from_geometry_keeps_geometry_and_counts(manifest: Path) -> None:
    """page_from_geometry keeps the source geometry, indices and counts, takes the new text, and rejects mismatches."""
    row = rk.read_jsonl(manifest)[0]
    source = source_page(row, SOURCE_LINES)
    page = sep.page_from_geometry(source, [fake_record("جديد", confidences=[0.5, 1.0, 0.75, 0.25]), fake_record("", confidences=[])], (40, 30))
    assert (page.n_lines, page.n_dropped, page.image_size) == (3, 1, (40, 30))
    assert [(line.text, line.mean_confidence) for line in page.lines] == [("جديد", 0.625), ("", None)]
    payload = sep.page_json(row, page, 1.5, "geometry from t + r")
    assert geometry_of(payload) == geometry_of(source) and set(payload) == set(source)
    with pytest.raises(ValueError):
        sep.page_from_geometry(source, [fake_record("x")], (40, 30))                 # one record for two lines
    with pytest.raises(ValueError):
        sep.page_from_geometry(dict(source, lines=source["lines"][::-1]), [fake_record("x")] * 2, (40, 30))


@pytest.mark.parametrize("rec_input,expected", [("rgb", ("RGB", 0)), ("binarized", ("1", 1))])
def test_reread_page_with_fake_kraken(tmp_path: Path, manifest: Path, monkeypatch: pytest.MonkeyPatch, rec_input: str, expected: tuple) -> None:
    """reread_page hands the stored lines to the recogniser alone and checks the image size first."""
    from PIL import Image

    nlbin_calls: List[str] = []
    install_fake_kraken(monkeypatch, nlbin_calls)
    path = tmp_path / "page.jpg"
    Image.new("RGB", (40, 30), (250, 240, 220)).save(path)
    source = source_page(rk.read_jsonl(manifest)[0], SOURCE_LINES)
    recogniser = FakeRecogniser("جديد")
    page = sep.reread_page(rk.KrakenReader(None, None, recogniser, None, rec_input=rec_input), source, path)
    assert (recogniser.modes[0], len(nlbin_calls)) == expected
    assert [line.text for line in page.lines] == ["جديد", "جديد"] and (page.n_lines, page.n_dropped) == (3, 1)
    assert [(line.baseline, line.boundary) for line in page.lines] == [(line["baseline"], line["boundary"]) for line in SOURCE_LINES]

    never = SimpleNamespace(predict=lambda *args: pytest.fail("recogniser must not run"))
    with pytest.raises(ValueError):                                                  # geometry found on another image size
        sep.reread_page(rk.KrakenReader(None, None, never, None), dict(source, image_size=[41, 30]), path)
    empty = sep.reread_page(rk.KrakenReader(None, None, never, None), dict(source, lines=[], n_lines=2, n_dropped=2), path)
    assert (empty.lines, empty.n_lines, empty.n_dropped) == ([], 2, 2)


def test_geometry_source_names_the_geometry_run(tmp_path: Path) -> None:
    """geometry_source names the geometry run, recogniser, input and kraken version; --source overrides it."""
    args = SimpleNamespace(geometry_from=tmp_path / "out" / "blla2026_ppocrv6m", recogniser="/ar/ar_ft.safetensors", rec_input="binarized",
                           source="")
    assert sep.geometry_source(args, version="7.1.1") == "geometry from blla2026_ppocrv6m + ar_ft.safetensors, rec binarized, kraken 7.1.1"
    args.source = "custom"
    assert sep.geometry_source(args) == "custom"


def test_geometry_from_rereads_with_recogniser_only(tmp_path: Path, manifest: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture) -> None:
    """A segment run, then a --geometry-from run without --segmenter: same geometry, new text, missing source JSON skipped, resumable."""
    from PIL import Image

    install_fake_kraken(monkeypatch)
    images = tmp_path / "pgp_arabic_editions_v2_images"
    for name in ("8317_0.jpg", "8315_0.jpg"):                                       # 8317_1.jpg stays fake bytes: it must never be opened
        Image.new("RGB", (40, 30), (250, 240, 220)).save(images / name)
    rows = rk.read_jsonl(manifest)
    out = tmp_path / "out"
    common = ["--manifest", str(manifest), "--out-dir", str(out), "--image-root", str(images)]

    monkeypatch.setattr(sep, "reader_from_args", lambda args: rk.KrakenReader(FakeSegmenter(), None, FakeRecogniser("قديم"), None))
    sep.main(common + ["--tag", "blla_ppocr", "--segmenter", "default", "--recogniser", "/ar/ppocr.safetensors", "--source", "seg run",
                       "--ids", str(_ids(tmp_path, rows[0]["stem"], rows[2]["stem"]))])
    source_dir = out / "blla_ppocr"
    sources = {row["stem"]: json.loads(sep.page_json_path(source_dir, row["stem"]).read_text(encoding="utf-8")) for row in (rows[0], rows[2])}
    assert [line["text"] for line in sources[rows[0]["stem"]]["lines"]] == ["قديم"]

    recogniser = FakeRecogniser("جديد")
    monkeypatch.setattr(sep, "reader_from_args", lambda _: pytest.fail("the segmenter must not load with --geometry-from"))
    monkeypatch.setattr(sep, "recogniser_from_args", lambda args: rk.KrakenReader(None, None, recogniser, None, rec_input=args.rec_input))
    monkeypatch.setattr(sep, "kraken_version", lambda: "7.1.1")
    reread = common + ["--tag", "reread", "--recogniser", "/ar/ar_ft.safetensors", "--geometry-from", str(source_dir),
                       "--ids", str(_ids(tmp_path, "8317", "8315"))]
    capsys.readouterr()
    sep.main(reread)
    log = capsys.readouterr().out
    assert "3 pages selected, 0 done before, 2 to read, 0 without an image, 1 without a geometry JSON" in log
    assert "no geometry: pgpar_8317_1_page" in log

    run_dir = out / "reread"
    assert sorted(p.name for p in run_dir.iterdir()) == ["pgpar_8315_0_page.json", "pgpar_8317_0_page.json", "summary.jsonl"]
    for stem, source in sources.items():
        payload = json.loads(sep.page_json_path(run_dir, stem).read_text(encoding="utf-8"))
        assert set(payload) == set(source) and geometry_of(payload) == geometry_of(source)
        assert [line["text"] for line in payload["lines"]] == ["جديد"]
        assert (payload["n_lines"], payload["n_dropped"], payload["image_size"]) == (source["n_lines"], source["n_dropped"], [40, 30])
        assert payload["source"] == "geometry from blla_ppocr + ar_ft.safetensors, rec rgb, kraken 7.1.1"
    assert recogniser.modes == ["RGB", "RGB"]
    summary = rk.read_jsonl(run_dir / sep.SUMMARY_FILE)
    assert [(s["stem"], s["n_lines"], s["n_dropped"]) for s in summary] == [("pgpar_8317_0_page", 2, 1), ("pgpar_8315_0_page", 2, 1)]
    assert {stem: json.loads(sep.page_json_path(source_dir, stem).read_text(encoding="utf-8")) for stem in sources} == sources

    monkeypatch.setattr(sep, "recogniser_from_args", lambda _: pytest.fail("models must not load when nothing is pending"))
    sep.main(reread)                                                                 # resumed: done pages skipped, summary rebuilt
    assert rk.read_jsonl(run_dir / sep.SUMMARY_FILE) == summary


def test_geometry_from_skips_pages_without_a_source_json(tmp_path: Path, manifest: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    """split_no_geometry separates rows without a source JSON; main lists and skips them without loading a model."""
    rows = rk.read_jsonl(manifest)
    source_dir = tmp_path / "out" / "blla_ppocr"
    source_dir.mkdir(parents=True)
    sep.write_json_atomic(sep.page_json_path(source_dir, rows[1]["stem"]), source_page(rows[1], SOURCE_LINES))
    found, absent = sep.split_no_geometry(rows, source_dir)
    assert [r["stem"] for r in found] == ["pgpar_8317_1_page"]
    assert [r["stem"] for r in absent] == ["pgpar_8317_0_page", "pgpar_8315_0_page", "pgpar_9000_0_page"]

    monkeypatch.setattr(sep, "recogniser_from_args", lambda _: pytest.fail("models must not load when nothing is pending"))
    capsys.readouterr()
    sep.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--tag", "reread", "--recogniser", "r",
              "--image-root", str(tmp_path / "pgp_arabic_editions_v2_images"), "--geometry-from", str(source_dir),
              "--ids", str(_ids(tmp_path, "8315"))])
    log = capsys.readouterr().out
    assert "0 to read, 0 without an image, 1 without a geometry JSON" in log and "no geometry: pgpar_8315_0_page" in log
    assert not (tmp_path / "out" / "reread").exists()


def test_geometry_from_argument_checks(tmp_path: Path, manifest: Path, capsys: pytest.CaptureFixture) -> None:
    """--segmenter stays required without --geometry-from; --geometry-from must be an existing directory other than the run's."""
    base = ["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--tag", "t", "--recogniser", "r"]
    (tmp_path / "out" / "t").mkdir(parents=True)
    for extra, message in (([], "--segmenter is required unless --geometry-from is given"),
                           (["--geometry-from", str(tmp_path / "nope")], "is not a directory"),
                           (["--geometry-from", str(tmp_path / "out" / "t")], "not this run's")):
        with pytest.raises(SystemExit):
            sep.main(base + extra)
        assert message in capsys.readouterr().err
