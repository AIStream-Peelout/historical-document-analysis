# File name: test_side_match.py
# Date: 10/15/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the side matcher (rules, file formats, budget, cache resume, calibration) with fake reads: no network, no API."""
import asyncio
import csv
import hashlib
import http.client
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import httpx
import pytest
from PIL import Image as PILImage

from src.finetuning.qwen_hebrew import side_match as sm

ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
MODEL = "claude-opus-5-5"
SPEC_ROW_KEYS = ["pgpid", "edition_fingerprint", "decision", "reader", "pages", "evidence"]
SPEC_PAGE_KEYS = ["canvas_index", "image_url", "assign"]
USAGE = {"input_tokens": 5000, "output_tokens": 2000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
COST_PER_READ = (5000 * 4.0 + 2000 * 20.0) / 1e6          # 0.06 USD at the Opus 5.5 prices


def fake_lines(seed: int, lines: int = 8, words: int = 8) -> List[str]:
    """Pseudo-Arabic edition lines whose letter 5-grams do not collide across seeds.

    :param seed: Random seed.
    :param lines: Number of lines.
    :param words: Words per line.
    :return: Lines.
    """
    rng = random.Random(seed)
    return [" ".join("".join(rng.choice(ALPHABET) for _ in range(rng.randint(3, 6))) for _ in range(words)) for _ in range(lines)]


def make_record(sections: Dict[str, List[str]], labels: List[str], pgpid: str = "1001", fingerprint: str = "f" * 40,
                indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """An ``unassigned.jsonl`` record (SPEC format 1).

    :param sections: Edition lines by section label.
    :param labels: Canvas labels.
    :param pgpid: PGP id.
    :param fingerprint: Edition fingerprint.
    :param indices: Canvas indices (default 0..n-1).
    :return: Record.
    """
    indices = indices if indices is not None else list(range(len(labels)))
    return {"pgpid": pgpid, "shelfmark": f"T-S Ar.{pgpid}", "canonical_id": f"Cambridge_CUL_T_S_Ar_{pgpid}",
            "library": "Cambridge University Library", "edition_fingerprint": fingerprint, "sections": sections,
            "canvases": [{"index": index, "label": label, "image_url": f"https://example.org/{pgpid}/{index}.jpg"}
                         for index, label in zip(indices, labels)],
            "reason": "edition without side labels and several images"}


def text(lines: List[str]) -> str:
    """Join lines as a read would.

    :param lines: Lines.
    :return: Text.
    """
    return "\n".join(lines)


def stem(url: str) -> str:
    """The image cache stem of a URL (as ``ImageFetcher.stem``).

    :param url: Image URL.
    :return: Stem.
    """
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


class FakeReader:
    """Returns a fixed read per image (looked up from the cached file name) and records its calls."""

    def __init__(self, texts: Dict[str, Optional[str]], usage: Optional[Dict[str, int]] = None, stop_reason: str = "end_turn") -> None:
        """Set the reads.

        :param texts: ``{image_url: text}``.
        :param usage: Token usage returned with every read.
        :param stop_reason: Stop reason returned with every read.
        """
        self.by_stem = {stem(url): value for url, value in texts.items()}
        self.usage = usage or USAGE
        self.stop_reason = stop_reason
        self.calls: List[Path] = []

    async def __call__(self, image_path: Path) -> Dict[str, Any]:
        """Read one image.

        :param image_path: Reader copy of the image.
        :return: Fake reader result.
        """
        self.calls.append(image_path)
        return {"text": self.by_stem[image_path.name.split("_")[0]], "stop_reason": self.stop_reason, "usage": dict(self.usage),
                "served_model": MODEL}


async def refusing_reader(image_path: Path) -> Dict[str, Any]:
    """A reader that must never be called.

    :param image_path: Image.
    :return: Never returns.
    """
    raise AssertionError(f"unexpected API call for {image_path}")


def fake_download(url: str, dst: Path) -> Dict[str, Any]:
    """Write a small JPEG instead of downloading.

    :param url: Image URL (unused).
    :param dst: Destination.
    :return: Like ``save_image``.
    """
    PILImage.new("RGB", (60, 40), "white").save(dst, format="JPEG")
    return {"file": dst.name}


async def no_sleep(seconds: float) -> None:
    """Skip a backoff pause.

    :param seconds: Requested pause.
    """
    return None


def write_unassigned(path: Path, records: List[Dict[str, Any]]) -> Path:
    """Write ``unassigned.jsonl``.

    :param path: Destination.
    :param records: Records.
    :return: The path.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File.
    :return: Objects.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(tmp_path: Path, records: List[Dict[str, Any]], reader: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the side matcher on records with fake I/O inside ``tmp_path``.

    :param tmp_path: Pytest temporary directory.
    :param records: ``unassigned.jsonl`` records.
    :param reader: Fake reader.
    :param kwargs: Extra ``run_side_match`` arguments.
    :return: Run summary.
    """
    unassigned = write_unassigned(tmp_path / "unassigned.jsonl", records)
    return sm.run_side_match(unassigned, tmp_path / "side_assignments.jsonl", reads_cache=tmp_path / "reads.jsonl",
                             image_cache=tmp_path / "images", read=reader, download=fake_download, delay_s=0.0,
                             log=lambda message: None, sleep=no_sleep, **kwargs)


# ---- rules ------------------------------------------------------------------------------------------------------

def test_unlabelled_accept_puts_the_whole_edition_on_one_canvas() -> None:
    """The canvas that holds the edition gets ``all``; an unrelated text on the other canvas does not veto."""
    lines = fake_lines(1)
    record = make_record({"": lines}, ["00000001.tif", "00000002.tif"], indices=[3, 7])
    row = sm.decide_record(record, [text(fake_lines(2)), text(lines[:6])], MODEL)
    assert row["decision"] == "accept"
    assert row["pages"] == [{"canvas_index": 7, "image_url": "https://example.org/1001/7.jpg", "assign": "all"}]
    assert row["evidence"]["rule"] == "unlabelled" and row["evidence"]["best_recall"] >= 0.6


@pytest.mark.parametrize("reads, reason", [
    (["a", "b"], "no canvas recalls"),                                               # nothing read
    (["half1", "half2"], "ambiguous"),                                               # text split over two canvases: never split
    (["all", "line0"], "edition text also read on another canvas"),                  # other canvas holds edition text
    (["loop", "NONE"], "looping"),                                                   # best canvas's read collapsed into a loop
])
def test_unlabelled_drops(reads: List[str], reason: str) -> None:
    """Every veto of the unlabelled rule drops the document without pages.

    :param reads: Which read each canvas gets (names resolved below).
    :param reason: Expected start of the drop reason.
    """
    lines = fake_lines(1)
    texts = {"a": "NONE", "b": "", "half1": text(lines[:4]), "half2": text(lines[4:]), "all": text(lines), "line0": lines[0],
             "loop": lines[0] + "\n" + (lines[1] + " ") * 30, "NONE": "NONE"}
    row = sm.decide_record(make_record({"": lines}, ["a.tif", "b.tif"]), [texts[name] for name in reads], MODEL)
    assert row["decision"] == "drop" and row["pages"] == []
    assert reason in row["evidence"]["reason"]


def test_unlabelled_loop_is_the_only_veto_in_the_loop_case() -> None:
    """The looping read clears recall, ratio and precision, so the loop guard is what drops it."""
    lines = fake_lines(1)
    loop = lines[0] + "\n" + (lines[1] + " ") * 30
    scores = sm.score_canvases({"": lines}, [0, 1], [loop, "NONE"])
    assert scores[0]["recall"][""] >= sm.MIN_RECALL and scores[1]["precision"] == 0.0
    assert scores[0]["repeat_share"] >= sm.MAX_REPEAT_SHARE


def test_labelled_joint_assignment_accept() -> None:
    """Recto and verso go to the canvases that hold them, whatever the canvas order."""
    recto, verso = fake_lines(3), fake_lines(4)
    record = make_record({"recto": recto, "verso": verso}, ["00000001.tif", "00000002.tif"])
    row = sm.decide_record(record, [text(verso[:5]), text(recto)], MODEL)
    assert row["decision"] == "accept" and row["evidence"]["rule"] == "labelled"
    assert sorted((page["canvas_index"], page["assign"]) for page in row["pages"]) == [(0, "verso"), (1, "recto")]
    assert row["evidence"]["margin"] >= sm.MIN_MARGIN


def test_labelled_drop_when_assignments_tie() -> None:
    """Two photographs of the same page (both reads hold both sides): no assignment wins by the margin."""
    recto, verso = fake_lines(3), fake_lines(4)
    both = text(recto + verso)
    row = sm.decide_record(make_record({"recto": recto, "verso": verso}, ["x.tif", "y.tif"]), [both, both], MODEL)
    assert row["decision"] == "drop" and "ambiguous" in row["evidence"]["reason"]


def test_labelled_drop_when_a_side_is_not_found() -> None:
    """The best assignment wins clearly but the verso is on no canvas."""
    recto, verso = fake_lines(3), fake_lines(4)
    row = sm.decide_record(make_record({"recto": recto, "verso": verso}, ["x.tif", "y.tif"]), [text(recto), "NONE"], MODEL)
    assert row["decision"] == "drop" and row["evidence"]["reason"] == "verso text not found on its canvas"


def test_labelled_respects_canvas_labels_that_name_a_side() -> None:
    """A canvas labelled ``1v`` never receives the recto; an unnamed canvas may take either side."""
    recto, verso = fake_lines(3), fake_lines(4)
    sections = {"recto": recto, "verso": verso}
    contradicted = sm.decide_record(make_record(sections, ["1v", "x.tif"]), [text(recto), text(verso)], MODEL)
    assert contradicted["decision"] == "drop"
    row = sm.decide_record(make_record(sections, ["1r", "a.tif", "b.tif"]), [text(recto), "NONE", text(verso)], MODEL)
    assert sorted((page["canvas_index"], page["assign"]) for page in row["pages"]) == [(0, "recto"), (2, "verso")]


def test_prechecks_drop_without_reading() -> None:
    """Whole volumes and refused reads are dropped; text outside the side labels gets no row at all."""
    lines = fake_lines(1)
    volume = sm.decide_record(make_record({"": lines}, [f"{n:08d}.tif" for n in range(9)]), [], MODEL)
    assert volume["decision"] == "drop" and volume["evidence"]["reason"] == "whole-volume manifest (9 canvases)"
    for sections in ({"": lines[:2], "recto": lines[2:]}, {"": lines[:2], "recto": lines[2:5], "verso": lines[5:]}):
        mixed = make_record(sections, ["a", "b"])
        assert sm.precheck(mixed) == "" and sm.unsupported(mixed).startswith("edition has text outside its side labels")
        assert not sm.needs_reading(mixed)
        with pytest.raises(ValueError):
            sm.decide_record(mixed, [text(lines), "NONE"], MODEL)
    refused = sm.decide_record(make_record({"": lines}, ["a", "b"]), [None, text(lines)], MODEL)
    assert refused["decision"] == "drop" and refused["evidence"]["refused"] == [0]


def test_text_outside_side_labels_is_left_open(tmp_path: Path) -> None:
    """Such a document is neither read nor written (no final drop), and the run and the dry run count it."""
    lines = fake_lines(1)
    mixed = make_record({"": lines[:2], "recto": lines[2:]}, ["00000001.tif", "00000002.tif"], pgpid="7")
    reason = sm.unsupported(mixed)
    dry = run(tmp_path, [mixed], refusing_reader, dry_run=True)
    assert dry["left_out_unsupported"] == {reason: 1} and dry["documents_to_read"] == 0 and dry["images_to_read"] == 0
    summary = run(tmp_path, [mixed], refusing_reader)
    assert summary["left_out_unsupported"] == {reason: 1} and summary["decided"] == 0 and summary["pending"] == 0
    assert read_jsonl(tmp_path / "side_assignments.jsonl") == []


def test_builder_places_the_matchers_rows() -> None:
    """Rows of both rules go through the builder's apply_side_assignment unchanged (the shared spec)."""
    from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as bpe

    lines, recto, verso = fake_lines(1), fake_lines(3), fake_lines(4)
    for sides, reads, expected in (({"": lines}, ["NONE", text(lines)], {1: lines}),
                                   ({"recto": recto, "verso": verso}, [text(verso), text(recto)], {0: verso, 1: recto})):
        record = make_record(sides, ["00000001.tif", "00000002.tif"], fingerprint=bpe.edition_fingerprint(sides))
        row = sm.decide_record(record, reads, MODEL)
        canvases = [{"url": canvas["image_url"], "side": ""} for canvas in record["canvases"]]
        assert bpe.apply_side_assignment(sides, canvases, row) == (expected, "")


def test_labelled_one_side_spread_over_two_images_drops() -> None:
    """A one-side edition read partly on each of two canvases is not put whole on the larger part."""
    lines = fake_lines(1)
    reads = [text(lines[:5]), text(lines[5:])]
    labelled = sm.decide_record(make_record({"recto": lines}, ["00000001.tif", "00000002.tif"]), reads, MODEL)
    assert labelled["decision"] == "drop" and labelled["evidence"]["reason"] == "edition text also read on a canvas outside the assignment"
    assert labelled["evidence"]["other_canvases_with_edition_text"] == [1]
    unlabelled = sm.decide_record(make_record({"": lines}, ["00000001.tif", "00000002.tif"]), reads, MODEL)
    assert unlabelled["decision"] == "drop"
    unrelated = sm.decide_record(make_record({"recto": lines}, ["00000001.tif", "00000002.tif"]),
                                 [text(lines[:5]), text(fake_lines(9))], MODEL)
    assert unrelated["decision"] == "accept" and unrelated["pages"][0]["assign"] == "recto"      # other text does not veto


# ---- files, budget, cache ---------------------------------------------------------------------------------------

def four_documents() -> List[Dict[str, Any]]:
    """An unlabelled accept, a labelled accept, a whole volume and an ambiguous unlabelled document.

    :return: Records.
    """
    a, recto, verso, d = fake_lines(1), fake_lines(3), fake_lines(4), fake_lines(5)
    return [make_record({"": a}, ["00000001.tif", "00000002.tif"], pgpid="1", fingerprint="1" * 40),
            make_record({"recto": recto, "verso": verso}, ["00000001.tif", "00000002.tif"], pgpid="2", fingerprint="2" * 40),
            make_record({"": a}, [f"{n:08d}.tif" for n in range(9)], pgpid="3", fingerprint="3" * 40),
            make_record({"": d}, ["a.tif", "b.tif"], pgpid="4", fingerprint="4" * 40)]


def four_reads(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Reads that make :func:`four_documents` accept, accept, (unread), drop.

    :param records: :func:`four_documents` result.
    :return: ``{image_url: text}``.
    """
    one, two, _, four = records
    url = lambda record, n: record["canvases"][n]["image_url"]       # noqa: E731
    return {url(one, 0): text(one["sections"][""]), url(one, 1): "NONE",
            url(two, 0): text(two["sections"]["verso"]), url(two, 1): text(two["sections"]["recto"]),
            url(four, 0): text(four["sections"][""][:4]), url(four, 1): text(four["sections"][""][4:])}


def test_output_formats_follow_the_spec_and_pass_the_fingerprint_through(tmp_path: Path) -> None:
    """side_assignments.jsonl has exactly the spec's keys; fingerprints come back unchanged; whole volumes are not read."""
    records = four_documents()
    reader = FakeReader(four_reads(records))
    summary = run(tmp_path, records, reader)
    rows = read_jsonl(tmp_path / "side_assignments.jsonl")
    assert [row["pgpid"] for row in rows] == ["1", "2", "3", "4"]
    for row, record in zip(rows, records):
        assert list(row) == SPEC_ROW_KEYS
        assert row["edition_fingerprint"] == record["edition_fingerprint"] and row["reader"] == MODEL
        assert row["decision"] in ("accept", "drop") and isinstance(row["evidence"], dict)
        for page in row["pages"]:
            assert list(page) == SPEC_PAGE_KEYS and isinstance(page["canvas_index"], int)
            assert page["assign"] in ("all", "recto", "verso")
            assert page["image_url"] == next(c["image_url"] for c in record["canvases"] if c["index"] == page["canvas_index"])
    assert [row["decision"] for row in rows] == ["accept", "accept", "drop", "drop"]
    assert rows[0]["pages"][0]["assign"] == "all" and {p["assign"] for p in rows[1]["pages"]} == {"recto", "verso"}
    assert len(reader.calls) == 6                                    # the 9-canvas volume was never read
    assert summary["pending"] == 0 and summary["accepted"] == 2 and summary["spent_usd"] == pytest.approx(6 * COST_PER_READ, abs=1e-4)
    assert summary["tokens"]["input_tokens"] == 6 * 5000
    with open(tmp_path / "side_assignments_review.csv", encoding="utf-8") as fh:
        review = list(csv.DictReader(fh))
    assert len(review) == 3 and list(review[0]) == sm.REVIEW_FIELDS
    assert all(Path(line["image_path"]).exists() for line in review)
    cached = read_jsonl(tmp_path / "reads.jsonl")
    assert len(cached) == 6 and {row["prompt_hash"] for row in cached} == {sm.prompt_hash(sm.READER_PROMPT)}
    assert all(row["model"] == MODEL and row["cost_usd"] == pytest.approx(COST_PER_READ) for row in cached)
    with PILImage.open(review[0]["image_path"]) as image:
        assert max(image.size) <= sm.MAX_SIDE


def test_cache_resume_makes_no_new_call(tmp_path: Path) -> None:
    """A second run reads nothing, survives a cut-off cache line, and writes the same assignments."""
    records = four_documents()
    run(tmp_path, records, FakeReader(four_reads(records)))
    first = (tmp_path / "side_assignments.jsonl").read_text(encoding="utf-8")
    with open(tmp_path / "reads.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"image_url": "https://exa')                         # a killed run's last line
    summary = run(tmp_path, records, refusing_reader)
    assert summary["reads"] == {} and summary["spent_usd"] == 0.0
    assert (tmp_path / "side_assignments.jsonl").read_text(encoding="utf-8") == first


def test_read_after_a_cut_off_line_is_kept(tmp_path: Path) -> None:
    """A read appended after a killed run's partial line starts on its own line, so a resume does not pay for it again."""
    records = four_documents()
    run(tmp_path, records, FakeReader(four_reads(records)))
    with open(tmp_path / "reads.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"image_url": "https://exa')                         # a killed run's last line, no newline
    extra = make_record({"": fake_lines(7)}, ["a.tif", "b.tif"], pgpid="5", fingerprint="5" * 40)
    reader = FakeReader({extra["canvases"][0]["image_url"]: text(fake_lines(7)), extra["canvases"][1]["image_url"]: "NONE"})
    run(tmp_path, records + [extra], reader)
    assert len(reader.calls) == 2
    assert len(sm.ReadCache(tmp_path / "reads.jsonl").reads) == 8       # 6 earlier reads + the 2 new ones, none lost
    summary = run(tmp_path, records + [extra], refusing_reader)
    assert summary["reads"] == {} and summary["accepted"] == 3
    assert sm.ends_with_newline(tmp_path / "reads.jsonl")


def test_incremental_rerun_keeps_earlier_assignments(tmp_path: Path) -> None:
    """A re-run on the builder's re-emitted file (only the documents still open) keeps the earlier rows of --out."""
    records = budget_documents(2, 2)
    reads = budget_reads(records)
    first = run(tmp_path, records, FakeReader(reads), max_usd=0.25, concurrency=1)      # room for document 1 only
    assert first["decided"] == 1 and first["pending"] == 1
    assert [(row["pgpid"], row["decision"]) for row in read_jsonl(tmp_path / "side_assignments.jsonl")] == [("1", "accept")]
    second = run(tmp_path, records[1:], FakeReader(reads))                                  # what the builder re-emits
    assert second["decided"] == 1 and second["rows_kept_from_earlier_runs"] == 1 and second["rows_in_out"] == 2
    assert [(row["pgpid"], row["decision"]) for row in read_jsonl(tmp_path / "side_assignments.jsonl")] == [
        ("1", "accept"), ("2", "accept")]
    changed = dict(records[0], edition_fingerprint="9" * 40)                               # document 1's edition changed
    run(tmp_path, [changed], refusing_reader)
    rows = read_jsonl(tmp_path / "side_assignments.jsonl")
    assert [row["pgpid"] for row in rows] == ["1", "2"] and rows[0]["edition_fingerprint"] == "9" * 40   # replaced in place


def test_merge_assignments() -> None:
    """Later decisions replace earlier rows of the same pgpid in place; new ones are appended."""
    previous = [{"pgpid": "1", "decision": "drop"}, {"pgpid": "2", "decision": "accept"}]
    merged = sm.merge_assignments(previous, [{"pgpid": 1, "decision": "accept"}, {"pgpid": "3", "decision": "drop"}])
    assert [(str(row["pgpid"]), row["decision"]) for row in merged] == [("1", "accept"), ("2", "accept"), ("3", "drop")]


def budget_documents(count: int, canvases: int) -> List[Dict[str, Any]]:
    """Unlabelled documents whose edition is on canvas 0 (the other canvases read NONE).

    :param count: Documents.
    :param canvases: Canvases per document.
    :return: Records.
    """
    return [make_record({"": fake_lines(seed)}, [f"{n:08d}.tif" for n in range(canvases)], pgpid=str(seed))
            for seed in range(1, count + 1)]


def budget_reads(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Reads of :func:`budget_documents`.

    :param records: Records.
    :return: ``{image_url: text}``.
    """
    return {canvas["image_url"]: text(record["sections"][""]) if canvas["index"] == 0 else "NONE"
            for record in records for canvas in record["canvases"]}


def test_budget_stops_new_documents(tmp_path: Path) -> None:
    """With room for one document, the first is read and decided, the others stay pending and unwritten."""
    records = budget_documents(3, 2)
    reader = FakeReader(budget_reads(records))
    summary = run(tmp_path, records, reader, max_usd=0.25, concurrency=1)
    assert len(reader.calls) == 2 and summary["budget_stopped"] is True
    assert summary["decided"] == 1 and summary["pending"] == 2 and summary["reads"]["documents_left_for_budget"] == 2
    assert summary["spent_usd"] <= 0.25
    assert [row["pgpid"] for row in read_jsonl(tmp_path / "side_assignments.jsonl")] == ["1"]


def test_budget_caps_every_call_when_reads_cost_more_than_the_prior(tmp_path: Path) -> None:
    """Reads at the worst case (largest image, MAX_TOKENS out) never take the spend past --max-usd.

    Three 4-canvas documents fit the per-document reservation at the prior (~$0.07 a read); before the per-call
    gate, all 12 reads ran (~4x the cap). Each call in flight now holds the worst case, so 5 reads fit $1.
    """
    worst_usage = {"input_tokens": sm.MAX_IMAGE_TOKENS + sm.PROMPT_TOKENS_PRIOR, "output_tokens": sm.MAX_TOKENS}
    worst = sm.worst_read_cost(sm.prices_for(MODEL))
    assert sm.call_cost(worst_usage, sm.prices_for(MODEL)) == pytest.approx(worst) and worst > 2 * sm.prior_image_cost(sm.prices_for(MODEL))
    records = budget_documents(3, 4)
    reader = FakeReader(budget_reads(records), usage=worst_usage, stop_reason="max_tokens")
    summary = run(tmp_path, records, reader, max_usd=1.0, concurrency=3)
    assert summary["spent_usd"] <= 1.0 and len(reader.calls) == int(1.0 // worst) == 5
    assert summary["budget_stopped"] is True and summary["reads"]["documents_left_for_budget"] == 3
    assert summary["decided"] == 0 and summary["pending"] == 3 and summary["worst_case_usd_per_call"] == pytest.approx(worst, abs=1e-4)
    assert len(read_jsonl(tmp_path / "reads.jsonl")) == 5                 # every paid read is cached for the next run


@pytest.mark.parametrize("max_usd, canvases, concurrency, before", [(0.25, 2, 1, 2), (1.0, 4, 3, 12)])
def test_budget_overshoot_is_at_most_one_read_per_worker(tmp_path: Path, max_usd: float, canvases: int, concurrency: int,
                                                         before: int) -> None:
    """Reads billed past the output cap (16k tokens out, $0.34; a real read stops at MAX_TOKENS) overshoot by one per worker.

    These are the reviewers' probes: the smoke setting ($0.25, a 2-canvas document) spent $0.67 on 2 reads, and
    $1 with three 4-canvas documents spent $4.04 on 12 reads, before every call was gated.

    :param max_usd: Budget (0.25 is the smoke setting).
    :param canvases: Canvases per document.
    :param concurrency: Workers.
    :param before: Reads made before the per-call gate.
    """
    usage = {"input_tokens": 4250, "output_tokens": 16_000}
    cost = sm.call_cost(usage, sm.prices_for(MODEL))
    records = budget_documents(3, canvases)
    reader = FakeReader(budget_reads(records), usage=usage, stop_reason="max_tokens")
    summary = run(tmp_path, records, reader, max_usd=max_usd, concurrency=concurrency)
    assert summary["spent_usd"] <= max_usd + concurrency * cost + 1e-9 and summary["budget_stopped"] is True
    assert len(reader.calls) < before
    if concurrency == 1:
        assert len(reader.calls) == 1                                     # the second call is refused once $0.34 is spent


def test_budget_accounting() -> None:
    """The prior is used until the first read; reservations block when they would pass the budget."""
    budget = sm.Budget(max_usd=0.3, prices=sm.prices_for(MODEL))
    assert budget.per_image() == pytest.approx(sm.prior_image_cost(sm.prices_for(MODEL)))
    first = budget.reserve(2)
    assert first is not None and budget.reserved == pytest.approx(first)
    assert budget.record(USAGE) == pytest.approx(COST_PER_READ) and budget.per_image() == pytest.approx(COST_PER_READ)
    assert budget.reserve(3) is None and budget.stopped                 # 0.06 + 0.14 reserved + 0.18 > 0.3
    assert budget.reserve(0) is None                                    # nothing starts after a stop
    budget.release(first)
    assert budget.reserved == 0.0
    calls = sm.Budget(max_usd=0.5, prices=sm.prices_for(MODEL))
    assert calls.worst_read() == pytest.approx((5034 * 4.0 + sm.MAX_TOKENS * 20.0) / 1e6)
    assert calls.start_call() and calls.start_call() and calls.in_flight == 2      # 2 x 0.18 <= 0.5
    assert not calls.start_call() and calls.stopped and calls.in_flight == 2       # a third could pass 0.5
    calls.end_call()
    calls.record(USAGE)
    assert calls.start_call()                                                     # 0.06 + 2 x 0.18 <= 0.5
    with pytest.raises(ValueError):
        sm.prices_for("claude-unknown")


def test_failed_download_leaves_the_document_pending(tmp_path: Path) -> None:
    """A refused download fails its document only; the run goes on and a later run can finish it."""
    records = four_documents()[:2]

    def flaky(url: str, dst: Path) -> Dict[str, Any]:
        """Refuse the first document's images.

        :param url: Image URL.
        :param dst: Destination.
        :return: Like ``save_image``.
        """
        if "/1/" in url:
            raise OSError("HTTP Error 403")
        return fake_download(url, dst)

    unassigned = write_unassigned(tmp_path / "unassigned.jsonl", records)
    summary = sm.run_side_match(unassigned, tmp_path / "out.jsonl", reads_cache=tmp_path / "reads.jsonl", image_cache=tmp_path / "img",
                                read=FakeReader(four_reads(four_documents())), download=flaky, delay_s=0.0, log=lambda m: None)
    assert summary["reads"]["documents_failed"] == 1 and summary["pending"] == 1 and summary["decided"] == 1


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"partial", 1000), PILImage.DecompressionBombError("too many pixels"),
                                   ValueError("unknown url type")])
def test_download_errors_that_are_not_oserror_leave_the_document_pending(tmp_path: Path, error: Exception) -> None:
    """A body cut short, an oversized image or a malformed URL fails only its document; the run still writes its files.

    :param error: What the download raises for the first document.
    """
    records = four_documents()[:2]

    def flaky(url: str, dst: Path) -> Dict[str, Any]:
        """Fail the first document's images.

        :param url: Image URL.
        :param dst: Destination.
        :return: Like ``save_image``.
        """
        if "/1/" in url:
            raise error
        return fake_download(url, dst)

    unassigned = write_unassigned(tmp_path / "unassigned.jsonl", records)
    summary = sm.run_side_match(unassigned, tmp_path / "out.jsonl", reads_cache=tmp_path / "reads.jsonl", image_cache=tmp_path / "img",
                                read=FakeReader(four_reads(four_documents())), download=flaky, delay_s=0.0, log=lambda m: None)
    assert summary["reads"]["documents_failed"] == 1 and summary["pending"] == 1 and summary["decided"] == 1
    assert [row["pgpid"] for row in read_jsonl(tmp_path / "out.jsonl")] == ["2"]


def test_fetch_never_writes_into_local_path_directories(tmp_path: Path) -> None:
    """A missing ``local_path`` (a frozen benchmark's images/) is downloaded into the cache, never into its directory."""
    frozen = tmp_path / "bench" / "images"
    frozen.mkdir(parents=True)
    canvas = {"index": 0, "label": "", "image_url": "https://example.org/x.jpg", "local_path": str(frozen / "x.jpg")}
    fetcher = sm.ImageFetcher(tmp_path / "cache", delay_s=0.0, download=fake_download)
    prepared, size = asyncio.run(fetcher.fetch(canvas))
    assert list(frozen.iterdir()) == [] and (tmp_path / "cache" / f"{stem(canvas['image_url'])}.jpg").exists()
    assert prepared.exists() and size == (60, 40)
    PILImage.new("RGB", (30, 20), "white").save(frozen / "y.jpg", format="JPEG")

    def no_download(url: str, dst: Path) -> None:
        """Fail: the local file exists.

        :param url: Image URL.
        :param dst: Destination.
        """
        raise AssertionError(f"unexpected download of {url}")

    local = {"index": 1, "label": "", "image_url": "https://example.org/y.jpg", "local_path": str(frozen / "y.jpg")}
    assert asyncio.run(sm.ImageFetcher(tmp_path / "cache", delay_s=0.0, download=no_download).fetch(local))[1] == (30, 20)


def test_make_client_pins_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited ANTHROPIC_BASE_URL is ignored (its host is logged, never the key); requests go to api.anthropic.com."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com/v1")
    messages: List[str] = []
    client = sm.make_client(env_file=None, log=messages.append)
    assert client.base_url.host == "api.anthropic.com" and client.max_retries == 0
    assert len(messages) == 1 and "gateway.example.com" in messages[0] and "test-placeholder" not in messages[0]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    messages.clear()
    assert sm.make_client(env_file=None, log=messages.append).base_url.host == "api.anthropic.com" and messages == []


def test_default_out_is_next_to_unassigned() -> None:
    """The default --out is the round README's side_match/side_assignments.jsonl, next to unassigned.jsonl."""
    assert sm.DEFAULT_OUT == sm.REPO / "logs" / "next_round" / "arabic_v24a" / "side_match" / "side_assignments.jsonl"
    assert sm.DEFAULT_CALIBRATION_OUT.parent == sm.DEFAULT_OUT.parent


def test_authentication_error_halts_the_run(tmp_path: Path) -> None:
    """A rejected key fails the first document and starts no other: every further call would fail the same way."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    calls: List[Path] = []

    async def bad_key(image_path: Path) -> Dict[str, Any]:
        """Answer like the API does to an invalid key.

        :param image_path: Image.
        :return: Never returns.
        """
        calls.append(image_path)
        raise anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=request), body=None)

    records = [make_record({"": fake_lines(seed)}, ["a.tif", "b.tif"], pgpid=str(seed)) for seed in (1, 2, 3)]
    summary = run(tmp_path, records, bad_key, concurrency=2)
    assert summary["halted"] is True and summary["decided"] == 0 and summary["pending"] == 3
    assert len(calls) == 2 and summary["reads"]["documents_failed"] == 2      # the two documents already started
    assert read_jsonl(tmp_path / "side_assignments.jsonl") == []


def test_retries_back_off_and_give_up() -> None:
    """Retryable errors are retried with growing pauses; others are raised at once."""
    pauses: List[float] = []

    async def record_pause(seconds: float) -> None:
        """Record a pause.

        :param seconds: Pause.
        """
        pauses.append(seconds)

    calls = {"n": 0}

    async def flaky(path: Path) -> Dict[str, Any]:
        """Fail twice, then answer.

        :param path: Image.
        :return: Result.
        """
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("overloaded")
        return {"text": "ok"}

    result = asyncio.run(sm.read_with_retries(flaky, Path("x.jpg"), retryable=lambda e: isinstance(e, ValueError), sleep=record_pause,
                                              log=lambda m: None))
    assert result == {"text": "ok"} and len(pauses) == 2 and pauses[1] > pauses[0]
    calls["n"] = -10
    with pytest.raises(ValueError):
        asyncio.run(sm.read_with_retries(flaky, Path("x.jpg"), attempts=3, retryable=lambda e: True, sleep=record_pause, log=lambda m: None))
    with pytest.raises(ValueError):
        asyncio.run(sm.read_with_retries(flaky, Path("x.jpg"), retryable=lambda e: False, sleep=record_pause, log=lambda m: None))


def test_is_retryable_on_sdk_errors() -> None:
    """Rate limits, connection errors and 5xx are retried; a 400 is not."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert sm.is_retryable(anthropic.APIConnectionError(request=request))
    assert sm.is_retryable(anthropic.InternalServerError("x", response=httpx.Response(529, request=request), body=None))
    assert sm.is_retryable(anthropic.RateLimitError("x", response=httpx.Response(429, request=request), body=None))
    assert not sm.is_retryable(anthropic.BadRequestError("x", response=httpx.Response(400, request=request), body=None))


def test_dry_run_calls_nothing_and_estimates(tmp_path: Path) -> None:
    """A dry run downloads and reads nothing and writes no file; after a run it counts cached reads."""
    records = four_documents()
    summary = run(tmp_path, records, refusing_reader, dry_run=True)
    assert summary["images_to_read"] == 6 and summary["documents_to_read"] == 3
    assert summary["dropped_before_reading"] == {"whole-volume manifest (9 canvases)": 1}
    assert summary["estimated_usd"] == pytest.approx(6 * sm.prior_image_cost(sm.prices_for(MODEL)), abs=0.01)
    assert not (tmp_path / "images").exists() and not (tmp_path / "side_assignments.jsonl").exists()
    run(tmp_path, records, FakeReader(four_reads(records)))
    again = run(tmp_path, records, refusing_reader, dry_run=True)
    assert again["images_to_read"] == 0 and again["images_cached"] == 6 and again["estimate_basis"].startswith("mean of 6")


def test_prepare_image_downscales(tmp_path: Path) -> None:
    """The reader copy is a JPEG at most MAX_SIDE pixels long."""
    src = tmp_path / "big.png"
    PILImage.new("RGBA", (3000, 1000), (255, 255, 255, 255)).save(src)
    assert sm.prepare_image(src, tmp_path / "small.jpg") == (2000, 667)
    with PILImage.open(tmp_path / "small.jpg") as image:
        assert image.format == "JPEG" and image.size == (2000, 667)


def test_reader_prompt() -> None:
    """The prompt asks for Arabic script only, line by line, without commentary."""
    assert "Arabic-script text visible" in sm.READER_PROMPT and "line" in sm.READER_PROMPT
    assert sm.READER_PROMPT.endswith("Return ONLY the transcription with no commentary.")
    assert len(sm.prompt_hash(sm.READER_PROMPT)) == 12


def test_cli_requires_unassigned() -> None:
    """Without --calibrate the unassigned file is required."""
    with pytest.raises(SystemExit):
        sm.main([])


# ---- calibration ------------------------------------------------------------------------------------------------

@pytest.fixture()
def bench(tmp_path: Path) -> Dict[str, Any]:
    """A small benchmark: a recto/verso document, a verso-only document and an unlabelled one (not eligible).

    :param tmp_path: Pytest temporary directory.
    :return: ``{dir, texts}`` with ``texts`` the reads by ``(doc_id, image_index)``.
    """
    root = tmp_path / "bench"
    (root / "images").mkdir(parents=True)
    recto, verso, single = fake_lines(3), fake_lines(4), fake_lines(6)
    docs = [("Cambridge_CUL_T_S_Ar_1_1", ["1r", "1v"], [("recto", recto), ("verso", verso)]),
            ("Cambridge_CUL_T_S_Ar_1_2", ["1r", "1v"], [("verso", single[:4]), ("verso", single[4:])]),
            ("Philadelphia_CAJS_Halper_1", ["", ""], [("", recto)])]
    images = []
    with open(root / "benchmark.jsonl", "w", encoding="utf-8") as fh:
        for n, (doc_id, labels, sections) in enumerate(docs):
            record_images = []
            for index, label in enumerate(labels):
                name = f"{doc_id}__{index}.jpg"
                PILImage.new("RGB", (80, 60), "white").save(root / "images" / name, format="JPEG")
                side = {"1r": "recto", "1v": "verso"}.get(label, "")
                record_images.append({"file": name, "image_index": index, "label": label, "side": side, "source": "iiif"})
                images.append({"doc_id": doc_id, "image_index": index, "image_url": f"https://example.org/{doc_id}/{index}.jpg"})
            fh.write(json.dumps({"id": doc_id, "pgpids": [str(500 + n)], "shelfmark": doc_id, "library": "CUL", "images": record_images,
                                 "gt": {"sections": [{"side": side, "lines": lines} for side, lines in sections]}}, ensure_ascii=False) + "\n")
    with open(root / "images_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"images": images}, fh)
    texts = {("Cambridge_CUL_T_S_Ar_1_1", 0): text(recto), ("Cambridge_CUL_T_S_Ar_1_1", 1): text(verso[2:]),
             ("Cambridge_CUL_T_S_Ar_1_2", 0): "NONE", ("Cambridge_CUL_T_S_Ar_1_2", 1): text(single)}
    return {"dir": root, "texts": texts}


def test_calibration_cases_hide_the_labels(bench: Dict[str, Any]) -> None:
    """Eligible documents become labelled cases, one-sided editions also unlabelled cases, with every label hidden."""
    cases = sm.calibration_cases(bench["dir"], limit=20)
    assert [(case["doc_id"], case["mode"]) for case in cases] == [
        ("Cambridge_CUL_T_S_Ar_1_1", "labelled"), ("Cambridge_CUL_T_S_Ar_1_2", "labelled"), ("Cambridge_CUL_T_S_Ar_1_2", "unlabelled")]
    assert all(canvas["label"] == "" for case in cases for canvas in case["record"]["canvases"])
    assert cases[1]["record"]["sections"] == {"verso": fake_lines(6)}           # the two verso parts are merged
    assert cases[2]["record"]["sections"] == {"": fake_lines(6)} and cases[2]["truth"] == [(1, "all")]
    assert cases[0]["record"]["canvases"][1]["image_url"] == "https://example.org/Cambridge_CUL_T_S_Ar_1_1/1.jpg"
    assert len(sm.calibration_cases(bench["dir"], limit=1)) == 1


def test_calibration_from_reads_on_disk(bench: Dict[str, Any], tmp_path: Path) -> None:
    """Reads from a benchmark outputs file: every case is decided and scored against the hidden labels."""
    reads = tmp_path / "outputs.jsonl"
    with open(reads, "w", encoding="utf-8") as fh:
        for (doc_id, index), value in bench["texts"].items():
            fh.write(json.dumps({"doc_id": doc_id, "image_index": index, "text": value}, ensure_ascii=False) + "\n")
    report = sm.run_calibration(bench["dir"], tmp_path / "calibration.json", calibrate_reads=reads, image_cache=tmp_path / "img",
                                reads_cache=tmp_path / "reads.jsonl")
    assert report["reader"] == "file:outputs" and report["spent_usd"] == 0.0
    assert report["summary"]["labelled"] == {"cases": 2, "accepted": 2, "dropped": 0, "correct": 2, "incorrect": 0, "no_read": 0,
                                             "precision": 1.0, "drop_reasons": {}}
    assert report["summary"]["unlabelled"]["correct"] == 1
    with open(tmp_path / "calibration_review.csv", encoding="utf-8") as fh:
        review = list(csv.DictReader(fh))
    assert len(review) == 4 and {line["correct"] for line in review} == {"True"}
    assert all(line["image_path"].endswith(".jpg") and Path(line["image_path"]).exists() for line in review)


def test_calibration_through_the_reader_scores_a_wrong_pick(bench: Dict[str, Any], tmp_path: Path) -> None:
    """With the API path (a fake reader on the local benchmark images), a confident wrong pick counts as incorrect."""
    texts = {f"https://example.org/{doc_id}/{index}.jpg": value for (doc_id, index), value in bench["texts"].items()}
    swapped = dict(texts)
    swapped["https://example.org/Cambridge_CUL_T_S_Ar_1_2/0.jpg"], swapped["https://example.org/Cambridge_CUL_T_S_Ar_1_2/1.jpg"] = (
        texts["https://example.org/Cambridge_CUL_T_S_Ar_1_2/1.jpg"], texts["https://example.org/Cambridge_CUL_T_S_Ar_1_2/0.jpg"])
    reader = FakeReader(swapped)
    report = sm.run_calibration(bench["dir"], tmp_path / "calibration.json", reads_cache=tmp_path / "reads.jsonl",
                                image_cache=tmp_path / "img", read=reader, log=lambda m: None, sleep=no_sleep)
    assert len(reader.calls) == 4 and report["reads"]["reads"] == 4
    assert report["summary"]["labelled"]["correct"] == 1 and report["summary"]["labelled"]["incorrect"] == 1
    assert report["summary"]["labelled"]["precision"] == 0.5 and report["summary"]["unlabelled"]["incorrect"] == 1
    dry = sm.run_calibration(bench["dir"], tmp_path / "dry.json", reads_cache=tmp_path / "reads.jsonl", image_cache=tmp_path / "img",
                             read=refusing_reader, dry_run=True)
    assert dry["images_to_read"] == 0 and dry["images_cached"] == 4
