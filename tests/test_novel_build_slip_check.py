"""Tests of the shelf-mark slip check: outcome mapping, selection, resumability, stops, the eye queue, no moves.

Synthetic stage1 candidates with legacy-store photographs (invented shelf marks, tiny JPEGs) in ``tmp_path``; the
LM Studio call is a stub keyed by image file. No network, no LM Studio.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.novel_build import paths, slip_check
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl

Answer = Union[str, None, Exception]


def _jpeg(path: Path) -> str:
    """Write a small JPEG.

    :param path: Destination.
    :return: The path as a string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (90, 80, 70)).save(path, "JPEG")
    return str(path)


def _store_image(work: Path, name: str, origin: str = "fjp_local", slip_outcome: Optional[str] = None,
                 local: bool = True, url_name: Optional[str] = None) -> ImageRef:
    """A legacy-store photograph.

    :param work: Work dir.
    :param name: Local file name.
    :param origin: Image origin.
    :param slip_outcome: Earlier slip outcome, if any.
    :param local: Write the local file.
    :param url_name: File name in the URL (default: ``name``).
    :return: The image reference.
    """
    path = work / "images" / name
    if local:
        _jpeg(path)
    sha = hashlib.sha256(name.encode()).hexdigest()
    return ImageRef(image_id=f"{origin}:0", origin=origin, url=f"https://example.invalid/store/{url_name or name}",
                    path=str(path) if local else None, sha256=sha, slip_outcome=slip_outcome,
                    status="present" if local else "listed")


def _cand(row_id: str, shelfmark: str, cids: Sequence[str], images: Sequence[ImageRef]) -> Candidate:
    """A stage1 candidate.

    :param row_id: Candidate id.
    :param shelfmark: Shelf mark.
    :param cids: Canonical ids.
    :param images: Photographs.
    :return: The candidate.
    """
    return Candidate(row_id=row_id, source="fjp", document_group=f"canonical_id:{cids[0]}", canonical_ids=list(cids),
                     pgpids=[], ktiv_id=None, shelfmark=shelfmark, script="hebrew", languages_stated=[],
                     verdict="novel", witnesses=[], unit="side", n_letters=40, images=list(images))


def make_work(tmp_path: Path) -> Path:
    """Stage1 with five checkable photographs and four that are skipped.

    :param tmp_path: Pytest temp dir.
    :return: The work dir.
    """
    work = tmp_path / "work"
    cands = [
        _cand("Cambridge_CUL_T_S_97_4321:0", "T-S 97.4321", ["Cambridge_CUL_T_S_97_4321"],
              [_store_image(work, "T-S_97.4321_1r.jpg"),                       # match
               _store_image(work, "12345_1r.jpg"),                             # PGP-id named: skipped
               _store_image(work, "T-S_97.4321_1v.jpg", slip_outcome="confirmed"),  # already slip-checked: skipped
               _store_image(work, "T-S_97.4321_2r.jpg", origin="iiif_cudl")]),  # not the legacy store: skipped
        _cand("Cambridge_CUL_T_S_97_4325:0", "T-S 97.4325", ["Cambridge_CUL_T_S_97_4325"],
              [_store_image(work, "T-S_97.4325_1r.jpg"),                       # mismatch
               _store_image(work, "T-S_97.4325_1v.jpg"),                       # no label
               _store_image(work, "T-S_97.4325_2r.jpg"),                       # unreadable
               _store_image(work, "T-S_97.4325_2v.jpg", origin="fjp_bucket", local=False)]),  # no file: skipped
        _cand("Cambridge_CUL_Or_9999_9_99:0", "Cambridge CUL: Or.9999 9.99", ["Cambridge_CUL_Or_9999_9_99"],
              [_store_image(work, "0.jpg", origin="fjp_bucket", url_name="Or_9999_9_99_1r.jpg")]),  # match by id
        _cand("Cambridge_CUL_T_S_97_4330:0", "T-S 97.4330", ["Cambridge_CUL_T_S_97_4330"],
              [_store_image(work, "T-S_97.4330_1r.jpg")]),                     # error
    ]
    write_jsonl(paths.stage_path(work, "images"), cands)
    return work


ANSWERS: Dict[str, Answer] = {
    "T-S_97.4321_1r.jpg": "Shelf mark: TS 97.4321.", "T-S_97.4325_1r.jpg": "T-S 97.4322",
    "T-S_97.4325_1v.jpg": "NONE", "T-S_97.4325_2r.jpg": "a torn slip, letters faded",
    "0.jpg": "Or. 9999 9.99", "T-S_97.4330_1r.jpg": RuntimeError("model unloaded")}


class Stub:
    """LM Studio stand-in: answers by image file name and counts its calls."""

    def __init__(self, answers: Dict[str, Answer], finish: str = "stop") -> None:
        """Hold the answers.

        :param answers: File name -> answer text, None (failed call) or an exception to raise.
        :param finish: Finish reason of every answer.
        """
        self.answers = answers
        self.finish = finish
        self.calls: List[str] = []

    def __call__(self, model_key: str, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Answer one request.

        :param model_key: Model key.
        :param image_path: Image file.
        :param prompt: Prompt.
        :return: Completion meta, or None.
        """
        assert model_key == "test-vlm" and prompt == slip_check.SLIP_PROMPT
        name = Path(image_path).name
        self.calls.append(name)
        answer = self.answers[name]
        if isinstance(answer, Exception):
            raise answer
        return None if answer is None else {"text": answer, "finish_reason": self.finish}


def _by_file(work: Path) -> Dict[str, Dict[str, Any]]:
    """Latest check record per image file name.

    :param work: Work dir.
    :return: ``file name -> record``.
    """
    return {Path(r["image_path"]).name: r for r in slip_check.load_checks(work / slip_check.CHECKS_FILE).values()}


def test_outcomes_map_reads_to_match_mismatch_no_label_unreadable_error(tmp_path: Path) -> None:
    """Each kind of answer gets its outcome; only eligible photographs are read."""
    work = make_work(tmp_path)
    stub = Stub(ANSWERS)
    counts = slip_check.run_slip_checks(work, model="test-vlm", transcribe=stub)
    got = {name: rec["outcome"] for name, rec in _by_file(work).items()}
    assert got == {"T-S_97.4321_1r.jpg": "match", "T-S_97.4325_1r.jpg": "mismatch", "T-S_97.4325_1v.jpg": "no_label",
                   "T-S_97.4325_2r.jpg": "unreadable", "0.jpg": "match", "T-S_97.4330_1r.jpg": "error"}
    assert sorted(stub.calls) == sorted(ANSWERS)
    assert counts["skipped_not_shelfmark_named"] == 1 and counts["skipped_has_slip_outcome"] == 1
    assert counts["skipped_no_local_file"] == 1 and counts["eye_queue"] == 2
    records = _by_file(work)
    assert records["T-S_97.4321_1r.jpg"]["matched_row_ids"] == ["Cambridge_CUL_T_S_97_4321:0"]
    assert records["T-S_97.4325_1r.jpg"]["read_keys"] == ["cambridgecults97|4322"]
    assert records["0.jpg"]["file_name"] == "Or_9999_9_99_1r.jpg"
    assert records["T-S_97.4330_1r.jpg"]["reason"].startswith("RuntimeError: model unloaded")


@pytest.mark.parametrize("answer, outcome", [
    ("T-S 97.4321", "match"), ("TS 97.4321 recto", "match"), ("Cambridge_CUL_T_S_97_4321", "match"),
    ("`T-S 97.4321`", "match"), ("The shelf mark is T-S 97.4321", "match"), ("T-S 97.4322", "mismatch"),
    ("ENA 9999.12", "mismatch"), ("NONE", "no_label"), ("None.", "no_label"), ("No label visible", "no_label"),
    ("", "unreadable"), ("illegible", "unreadable"), (None, "error")])
def test_answer_outcomes(tmp_path: Path, answer: Optional[str], outcome: str) -> None:
    """Answers about one photograph filed under T-S 97.4321."""
    work = tmp_path / "work"
    image = _store_image(work, "T-S_97.4321_1r.jpg")
    item = slip_check.SlipItem(sha256=image.sha256, path=image.path, url=image.url, origin=image.origin,
                               file_name="T-S_97.4321_1r.jpg",
                               filings=[slip_check.filing(_cand("c1", "T-S 97.4321", ["Cambridge_CUL_T_S_97_4321"],
                                                                [image]))])
    record = slip_check.check_image(item, model="test-vlm", transcribe=Stub({"T-S_97.4321_1r.jpg": answer}))
    assert record["outcome"] == outcome


def test_an_answer_cut_by_the_token_budget_is_unreadable(tmp_path: Path) -> None:
    """A read that hit the token budget is not trusted, whatever it says."""
    work = make_work(tmp_path)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(ANSWERS, finish="length"))
    record = _by_file(work)["T-S_97.4321_1r.jpg"]
    assert (record["outcome"], record["reason"]) == ("unreadable", "answer_cut_by_token_budget")


def test_runs_resume_by_sha256_and_retry_errors(tmp_path: Path) -> None:
    """A second run reads nothing already checked except the photograph whose call failed."""
    work = make_work(tmp_path)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(ANSWERS))
    again = Stub({**ANSWERS, "T-S_97.4330_1r.jpg": "T-S 97.4330"})
    counts = slip_check.run_slip_checks(work, model="test-vlm", transcribe=again)
    assert again.calls == ["T-S_97.4330_1r.jpg"] and counts["match"] == 1
    assert counts["skipped_already_checked"] == 5
    assert _by_file(work)["T-S_97.4330_1r.jpg"]["outcome"] == "match"
    lines = (work / slip_check.CHECKS_FILE).read_text().splitlines()
    assert len(lines) == 7                                                     # appended, never rewritten


def test_limit_caps_the_reads_of_a_run(tmp_path: Path) -> None:
    """``limit`` reads at most N photographs."""
    work = make_work(tmp_path)
    stub = Stub(ANSWERS)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=stub, limit=2)
    assert len(stub.calls) == 2


def test_eye_queue_lists_mismatches_and_unreadables(tmp_path: Path) -> None:
    """The eye queue holds every mismatch and unreadable with the read, the expected mark and the image path."""
    work = make_work(tmp_path)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(ANSWERS))
    queue = list(read_jsonl(work / slip_check.EYE_QUEUE_JSONL))
    assert [(q["outcome"], Path(q["image_path"]).name) for q in queue] == [
        ("mismatch", "T-S_97.4325_1r.jpg"), ("unreadable", "T-S_97.4325_2r.jpg")]
    assert queue[0]["read"] == "T-S 97.4322" and queue[0]["expected"] == ["T-S 97.4325"]
    assert queue[0]["row_ids"] == ["Cambridge_CUL_T_S_97_4325:0"]
    text = (work / slip_check.EYE_QUEUE_MD).read_text()
    assert text.startswith("# Slip-check eye queue") and "6 photographs checked, 2 here" in text
    assert "| 1 | mismatch | label_names_another_mark | T-S 97.4322 | T-S 97.4325 |" in text
    assert str(work / "images" / "T-S_97.4325_2r.jpg") in text and "a torn slip, letters faded" in text


def test_five_errors_in_a_row_stop_the_run(tmp_path: Path) -> None:
    """The run stops after five consecutive failed calls."""
    work = make_work(tmp_path)
    failing = Stub({name: RuntimeError("down") for name in ANSWERS})
    counts = slip_check.run_slip_checks(work, model="test-vlm", transcribe=failing)
    assert len(failing.calls) == 5 and counts["error"] == 5 and counts["stopped_after_errors"] == 1


def test_stop_file_stops_before_the_next_read(tmp_path: Path) -> None:
    """An existing stop file stops the run before any read."""
    work = make_work(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("")
    stub = Stub(ANSWERS)
    counts = slip_check.run_slip_checks(work, model="test-vlm", transcribe=stub, stop_file=stop)
    assert stub.calls == [] and counts["stopped_by_file"] == 1


def test_nothing_is_dropped_or_moved(tmp_path: Path) -> None:
    """Every photograph stays where it was and the stage1 file is not touched."""
    work = make_work(tmp_path)
    stage1 = paths.stage_path(work, "images")
    before = stage1.read_bytes()
    files = sorted(p for p in (work / "images").rglob("*") if p.is_file())
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(ANSWERS))
    assert stage1.read_bytes() == before
    assert sorted(p for p in (work / "images").rglob("*") if p.is_file()) == files


def test_selection_rules() -> None:
    """Shelf-mark-named means the first ``_`` part of the name is not all digits; the URL names a bucket file."""
    assert slip_check.shelfmark_named("Mittwoch_4_1r.jpg") and slip_check.shelfmark_named("T-S_97.4321_1r.jpg")
    assert not slip_check.shelfmark_named("12345_1r.jpg") and not slip_check.shelfmark_named("12345.jpg")
    image = ImageRef(image_id="fjp_bucket:0", origin="fjp_bucket", url="https://example.invalid/a/Or%201_2.jpg",
                     path="/w/images/x/0.jpg")
    assert slip_check.store_file_name(image) == "Or 1_2.jpg"


def test_clean_answer_strips_prefixes_quotes_and_markup() -> None:
    """The label is the first meaningful line without a ``Shelf mark:`` prefix, quotes or a final full stop."""
    assert slip_check.clean_answer("```\nShelf mark: \"T-S 97.4321\".\n```") == "T-S 97.4321"
    assert slip_check.clean_answer("Label:\nENA 9999.12") == "ENA 9999.12"
    assert slip_check.clean_answer("  \n") == ""
    assert slip_check.is_no_label("NONE") and not slip_check.is_no_label("No. 12")


def test_default_transcribe_calls_the_lm_studio_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default call goes through the LM Studio function with the model, image, prompt and token budget."""
    seen: Dict[str, Any] = {}

    async def fake(model_id: str, image_path: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Record the request.

        :param model_id: Model key.
        :param image_path: Image file.
        :param prompt: Prompt.
        :param kwargs: Further options.
        :return: A completion meta.
        """
        seen.update(model_id=model_id, image_path=image_path, prompt=prompt, **kwargs)
        await asyncio.sleep(0)
        return {"text": "NONE", "finish_reason": "stop"}

    monkeypatch.setattr(slip_check.lms_transcriber, "transcribe_with_lm_studio_meta", fake)
    assert slip_check.lms_transcribe("m", "/x.jpg", slip_check.SLIP_PROMPT, max_tokens=64)["text"] == "NONE"
    assert seen == {"model_id": "m", "image_path": "/x.jpg", "prompt": slip_check.SLIP_PROMPT, "max_tokens": 64}


def test_main_runs_with_a_required_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reads with the given model and writes the check and eye-queue files."""
    work = make_work(tmp_path)
    stub = Stub(ANSWERS)
    monkeypatch.setattr(slip_check, "lms_transcribe", lambda m, p, q, max_tokens: stub(m, p, q))
    slip_check.main(["--work-dir", str(work), "--model", "test-vlm", "--limit", "3"])
    assert len(stub.calls) == 3
    assert (work / slip_check.CHECKS_FILE).is_file() and (work / slip_check.EYE_QUEUE_MD).is_file()
    with pytest.raises(SystemExit):
        slip_check.main(["--work-dir", str(work)])
