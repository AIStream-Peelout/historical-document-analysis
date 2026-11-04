"""Tests of the shelf-mark slip check: outcome mapping, selection, resumability, stops, the eye queue, no moves, the
OCR-normalised second comparison tier, ``--reclassify`` over stored reads and ``--rank-eye-queue`` (A / B / C over a
synthetic shelf-mark index).

Synthetic stage1 candidates with legacy-store photographs (invented shelf marks, tiny JPEGs) in ``tmp_path``; the
LM Studio call is a stub keyed by image file; the shelf-mark index is built from a handful of records. No network, no
LM Studio.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytest
from PIL import Image

from src.datasets import shelfmarks as shelf
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


# ------------------------------------------------------------------------------ tier 2: the read as OCR of the mark

OR_56 = ("Cambridge CUL: Or.1081 1.56", "Cambridge_CUL_Or_1081_1_56")
OR_63 = ("Cambridge CUL: Or.1081 1.63", "Cambridge_CUL_Or_1081_1_63")
OR_7 = ("CUL Or.1081 1.7", "Cambridge_CUL_Or_1081_1_7")
AS_245 = ("Cambridge CUL: T-S AS 160.245", "Cambridge_CUL_T_S_AS_160_245")
AS_27 = ("Cambridge CUL: T-S AS 160.27", "Cambridge_CUL_T_S_AS_160_27")
MISC_11_17 = ("Cambridge CUL: T-S Misc.11.17", "Cambridge_CUL_T_S_Misc_11_17")
K27_33A = ("Cambridge CUL: T-S K27.33a", "Cambridge_CUL_T_S_K27_33a")
TS_8J5_2 = ("T-S 8J5.2", "Cambridge_CUL_T_S_8J5_2")


def _filing(shelfmark: str, canonical_id: str) -> Dict[str, Any]:
    """The filing record of a photograph filed under one mark and id.

    :param shelfmark: Filed shelf mark.
    :param canonical_id: Its canonical id.
    :return: The :func:`slip_check.filing` record.
    """
    return slip_check.filing(_cand("c1", shelfmark, [canonical_id], []))


def _decide(read: str, mark: Tuple[str, str]) -> Dict[str, Any]:
    """Decide the outcome of a raw answer about a photograph filed under a mark.

    :param read: Raw answer.
    :param mark: ``(shelf mark, canonical id)`` the photograph is filed under.
    :return: The :func:`slip_check.classify_read` decision.
    """
    return slip_check.classify_read(read, "stop", [_filing(*mark)])


@pytest.mark.parametrize("read, mark", [
    ("Or. 108i.1-56", OR_56),             # an i after a digit is a 1
    ("Or. 108i .1 .63", OR_63),           # ... whatever the spaces and dots around it
    ("Or. 1081..163", OR_63),             # separators are dropped: a dot in the wrong place is noise
    ("Or. I081.1.56", OR_56),             # a confusable may start the digit run
    ("Or. 1O81.1.56", OR_56),             # an O inside the digits is a 0
    ("T-S AS 160.24S", AS_245),           # an S after a digit is a 5
    ("T-S AS160.24S", AS_245),            # the S of AS is the series, not a 5
    ("T-S12.30l", ("Cambridge CUL: T-S 12.301", "Cambridge_CUL_T_S_12_301")),   # T-S12: the S stays a letter
    ("T-S Misc 11.17 P2", MISC_11_17),    # the P2 piece marker is not part of the mark
    ("Misc 26:53-7", ("Cambridge CUL: T-S Misc.26.53.7", "Cambridge_CUL_T_S_Misc_26_53_7")),  # series without T-S
    ("16507", ("Philadelphia Penn U.: E 16507", "Philadelphia_CAJS_E_16507")),   # no series, five digits
    ("TSL ENA 3604 1", ("New York JTS: ENA 3604.1", "New_York_JTS_ENA_3604_1"))])  # institution letters in front
def test_ocr_noise_on_the_right_mark_is_match_fuzzy(read: str, mark: Tuple[str, str]) -> None:
    """The same digit groups after the OCR normalisation, with the series agreeing or not read: ``match_fuzzy``."""
    decision = _decide(read, mark)
    assert decision["outcome"] == "match_fuzzy", decision
    assert decision["reason"] == "label_matches_filing_after_ocr_normalisation"
    assert decision["matched_row_ids"] == ["c1"] and decision["fuzzy"]["marks"][0]["partial"] is None


def test_the_three_noisy_oxford_style_reads_of_the_brief() -> None:
    """``Or. 108i.1-56`` and ``Or. 108i .1 .63`` match ``Or. 1081.1.56`` / ``Or. 1081.1.63``; ``Or. 108 i 1b`` (the
    slip says ``Or. 1081.1.6``) stays a mismatch: a lone ``i`` between spaces and a ``b`` for a 6 are not digit-run
    confusables."""
    assert _decide("Or. 108i.1-56", OR_56)["outcome"] == "match_fuzzy"
    assert _decide("Or. 108i .1 .63", OR_63)["outcome"] == "match_fuzzy"
    decision = _decide("Or. 108 i 1b", ("CUL Or.1081 1.6", "Cambridge_CUL_Or_1081_1_6"))
    assert decision["outcome"] == "mismatch" and decision["fuzzy"]["read_groups"] == ["108", "i", "1", "b"]


@pytest.mark.parametrize("read, mark, kind", [
    ("Or 1081", OR_7, "prefix"),            # the volume only
    ("Or. 1.56", OR_56, "suffix"),          # the box and the item without the volume
    ("T-S k27.33", K27_33A, "prefix"),      # the piece letter not read
    ("T-S 8J5", TS_8J5_2, "prefix")])       # letter groups count as groups
def test_the_first_or_last_groups_of_the_mark_are_match_fuzzy_partial(read: str, mark: Tuple[str, str],
                                                                      kind: str) -> None:
    """A read whose groups are the first or the last groups of the mark's, with the series agreeing, is partial."""
    decision = _decide(read, mark)
    assert decision["outcome"] == "match_fuzzy_partial", decision
    assert decision["reason"] == "label_is_part_of_filed_mark"
    assert decision["fuzzy"]["marks"][0]["partial"] == kind


@pytest.mark.parametrize("read, mark", [
    ("T-S 8J5.1", TS_8J5_2),                                  # one digit differs
    ("T-S 8K5.2", TS_8J5_2),                                  # the letter group differs
    ("B 4348-2", ("B 4349", "Manchester_JRL_B_4349")),        # more digits than the mark
    ("T-S AS 160.26", AS_27),                                 # the neighbouring fragment
    ("T-S AS 100.27", AS_27),                                 # a digit differs, whatever the cause
    ("T-S AS 160.11", ("Cambridge CUL: T-S AS 160.111", "Cambridge_CUL_T_S_AS_160_111")),   # truncated read
    ("ENA 1081.1.56", OR_56),                                 # the same digits under another series
    ("2", ("Paris AIU: IV.A.2", "Paris_AIU_IV_A_2")),         # no series and a single digit names nothing
    ("P1", ("Cambridge CUL: T-S C5.11", "Cambridge_CUL_T_S_C5_11")),   # only the piece marker
    ("Or. 1081.5", ("Cambridge CUL: Or.1081 1.5", "Cambridge_CUL_Or_1081_1_5")),   # a middle group missing
    ("T-S K25.210-29", ("T-S K25.240.29", "Cambridge_CUL_T_S_K25_240_29"))])
def test_a_different_digit_group_or_series_stays_a_mismatch(read: str, mark: Tuple[str, str]) -> None:
    """Digits that differ, groups missing in the middle, or another series are not forgiven."""
    decision = _decide(read, mark)
    assert decision["outcome"] == "mismatch", decision
    assert decision["reason"] == "label_names_another_mark" and decision["matched_row_ids"] == []
    assert decision["fuzzy"]["marks"] == [] and decision["fuzzy"]["outcome"] == "mismatch"


def test_the_strict_comparison_goes_first_and_leaves_no_fuzzy_field() -> None:
    """A read the facade accepts is ``match`` and never reaches tier 2; the other answers keep their outcomes."""
    decision = _decide("T-S 8J5.2", TS_8J5_2)
    assert decision["outcome"] == "match" and decision["fuzzy"] is None and decision["matched_row_ids"] == ["c1"]
    record = slip_check.apply_decision({"outcome": "error", "fuzzy": {"stale": True}}, decision)
    assert "fuzzy" not in record and record["comparison"] == slip_check.COMPARISON
    assert _decide("NONE", TS_8J5_2)["outcome"] == "no_label"
    assert _decide("illegible", TS_8J5_2)["outcome"] == "unreadable"


@pytest.mark.parametrize("token, mapped", [
    ("108i", "1081"), ("24S", "245"), ("1O5", "105"), ("12o", "120"), ("1l5", "115"), ("108|", "1081"),
    ("I081", "1081"), ("l5", "15"), ("A2i", "A21"),
    ("AS160", "AS160"), ("S12", "S12"), ("Misc", "Misc"), ("IV", "IV"), ("Or", "Or"), ("1b", "1b"), ("8J5", "8J5")])
def test_map_digit_confusables_works_only_inside_digit_runs(token: str, mapped: str) -> None:
    """Letters glued to the front of a digit run and tokens without a digit are left alone."""
    assert slip_check.map_digit_confusables(token) == mapped


def test_mark_form_reduces_a_spelling_to_series_and_digit_groups() -> None:
    """Series letters before the first digit, then the digit and letter groups; separators only delimit."""
    form = slip_check.mark_form("T-S 8J5.1")
    assert (form.series, form.parts, form.squashed, form.norm, form.n_digits) == (
        "ts", ("8", "j", "5", "1"), "8j51", "ts8j51", 3)
    read = slip_check.mark_form("Or. 108i .1 .63", ocr=True)
    assert (read.series, read.parts, read.norm) == ("or", ("1081", "1", "63"), "or1081163")
    assert slip_check.mark_form("Or. 1081..163", ocr=True).norm == read.norm
    assert slip_check.mark_form("Misc 11.17 P2", ocr=True).parts == ("11", "17")
    assert slip_check.mark_form("illegible").parts == () and slip_check.mark_form("").series == ""


def test_series_agreement_allows_a_missing_front_or_an_added_institution_only() -> None:
    """Equal, a tail of the other, or no series read; any other letters disagree."""
    assert slip_check.series_agreement("", "tsas") == "read_has_none"
    assert slip_check.series_agreement("tsas", "tsas") == "same"
    assert slip_check.series_agreement("misc", "cambridgecultsmisc") == "read_is_tail_of_mark"
    assert slip_check.series_agreement("tslena", "ena") == "mark_is_tail_of_read"
    assert slip_check.series_agreement("ena", "cambridgeculor") is None
    assert slip_check.series_agreement("tska", "tsk") is None


#: ``(filed mark, canonical id, image file, the stubbed VLM answer)``; :data:`FUZZY_OUTCOMES` gives each outcome.
FUZZY_ROWS: List[Tuple[str, str, str, Answer]] = [
    ("Cambridge CUL: T-S 97.4321", "Cambridge_CUL_T_S_97_4321", "T-S_97.4321_1r.jpg", "T-S 97.4321"),
    ("Cambridge CUL: Or.9991 1.56", "Cambridge_CUL_Or_9991_1_56", "Or_9991_1_56_1r.jpg", "Or. 999i.1-56"),
    ("Cambridge CUL: T-S AS 160.245", "Cambridge_CUL_T_S_AS_160_245", "T-S_AS_160.245_1r.jpg", "T-S AS 160.24S"),
    ("Cambridge CUL: T-S Misc.11.17", "Cambridge_CUL_T_S_Misc_11_17", "T-S_Misc_11.17_2r.jpg", "T-S Misc 11.17 P2"),
    ("CUL Or.9991 1.7", "Cambridge_CUL_Or_9991_1_7", "Or_9991_1_7_1r.jpg", "Or 9991"),
    ("Cambridge CUL: T-S AS 160.27", "Cambridge_CUL_T_S_AS_160_27", "T-S_AS_160.27_1r.jpg", "T-S AS 160.26"),
    ("Cambridge CUL: T-S 97.4322", "Cambridge_CUL_T_S_97_4322", "T-S_97.4322_1r.jpg", "NONE"),
    ("Cambridge CUL: T-S 97.4323", "Cambridge_CUL_T_S_97_4323", "T-S_97.4323_1r.jpg", "illegible"),
    ("Cambridge CUL: T-S 97.4324", "Cambridge_CUL_T_S_97_4324", "T-S_97.4324_1r.jpg", RuntimeError("model unloaded"))]
FUZZY_ANSWERS: Dict[str, Answer] = {name: answer for _, _, name, answer in FUZZY_ROWS}
FUZZY_OUTCOMES: Dict[str, str] = {
    "T-S_97.4321_1r.jpg": "match", "Or_9991_1_56_1r.jpg": "match_fuzzy", "T-S_AS_160.245_1r.jpg": "match_fuzzy",
    "T-S_Misc_11.17_2r.jpg": "match_fuzzy", "Or_9991_1_7_1r.jpg": "match_fuzzy_partial",
    "T-S_AS_160.27_1r.jpg": "mismatch", "T-S_97.4322_1r.jpg": "no_label", "T-S_97.4323_1r.jpg": "unreadable",
    "T-S_97.4324_1r.jpg": "error"}


def make_fuzzy_work(tmp_path: Path) -> Path:
    """Stage1 with nine photographs, one per kind of outcome.

    :param tmp_path: Pytest temp dir.
    :return: The work dir.
    """
    work = tmp_path / "work"
    cands = [_cand(f"{cid}:0", mark, [cid], [_store_image(work, name)]) for mark, cid, name, _ in FUZZY_ROWS]
    write_jsonl(paths.stage_path(work, "images"), cands)
    return work


def _outcomes(work: Path) -> Dict[str, str]:
    """Outcome per image file name.

    :param work: Work dir.
    :return: ``file name -> outcome``.
    """
    return {name: rec["outcome"] for name, rec in _by_file(work).items()}


def _downgrade_to_strict(work: Path) -> None:
    """Rewrite the check file as the strict-only comparison wrote it: fuzzy matches are plain mismatches and no record
    has a ``fuzzy`` field or a comparison tag.

    :param work: Work dir.
    """
    checks = work / slip_check.CHECKS_FILE
    rows = list(read_jsonl(checks))
    for row in rows:
        if row["outcome"] in ("match_fuzzy", "match_fuzzy_partial"):
            row.update(outcome="mismatch", reason="label_names_another_mark", matched_row_ids=[])
        row.pop("fuzzy", None)
        row.pop("comparison", None)
    write_jsonl(checks, rows)


def test_a_fresh_run_records_fuzzy_matches_with_their_normalised_strings(tmp_path: Path) -> None:
    """Fuzzy outcomes are counted and kept in the check file with ``fuzzy`` audit fields; the eye queue omits them."""
    work = make_fuzzy_work(tmp_path)
    counts = slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(FUZZY_ANSWERS))
    assert _outcomes(work) == FUZZY_OUTCOMES
    assert counts["match_fuzzy"] == 3 and counts["match_fuzzy_partial"] == 1 and counts["eye_queue"] == 2
    assert set(slip_check.MATCH_OUTCOMES) < set(slip_check.OUTCOMES)                    # no match is ever in the queue
    assert not set(slip_check.MATCH_OUTCOMES) & set(slip_check.EYE_OUTCOMES)
    records = _by_file(work)
    fuzzy = records["Or_9991_1_56_1r.jpg"]
    assert fuzzy["matched_row_ids"] == ["Cambridge_CUL_Or_9991_1_56:0"]
    assert fuzzy["reason"] == "label_matches_filing_after_ocr_normalisation"
    assert fuzzy["comparison"] == slip_check.COMPARISON
    audit = fuzzy["fuzzy"]
    assert (audit["read_norm"], audit["read_series"], audit["read_groups"]) == ("or9991156", "or", ["9991", "1", "56"])
    assert audit["marks"][0]["row_id"] == "Cambridge_CUL_Or_9991_1_56:0" and audit["marks"][0]["partial"] is None
    assert audit["marks"][0]["norm"] == "cambridgeculor9991156" and audit["marks"][0]["groups"] == ["9991", "1", "56"]
    assert records["T-S_Misc_11.17_2r.jpg"]["fuzzy"]["read_norm"] == "tsmisc1117"        # the P2 marker is gone
    assert records["Or_9991_1_7_1r.jpg"]["fuzzy"]["marks"][0]["partial"] == "prefix"
    assert records["T-S_AS_160.27_1r.jpg"]["fuzzy"]["marks"] == []                     # tier 2 ran and found nothing
    assert "fuzzy" not in records["T-S_97.4321_1r.jpg"] and "fuzzy" not in records["T-S_97.4322_1r.jpg"]
    queue = list(read_jsonl(work / slip_check.EYE_QUEUE_JSONL))
    assert [(q["outcome"], Path(q["image_path"]).name) for q in queue] == [
        ("mismatch", "T-S_AS_160.27_1r.jpg"), ("unreadable", "T-S_97.4323_1r.jpg")]
    text = (work / slip_check.EYE_QUEUE_MD).read_text()
    assert "9 photographs checked, 2 here" in text and "match_fuzzy" in text and "Or_9991_1_56_1r" not in text


def test_reclassify_checks_rewrites_stored_outcomes_and_counts_the_moves(tmp_path: Path) -> None:
    """Outcomes of a strict-only file are redone from the stored reads: fuzzy matches come back, nothing else moves."""
    work = make_fuzzy_work(tmp_path)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(FUZZY_ANSWERS))
    fresh = _by_file(work)
    _downgrade_to_strict(work)
    slip_check.write_eye_queue(work, slip_check.load_checks(work / slip_check.CHECKS_FILE))
    assert len(list(read_jsonl(work / slip_check.EYE_QUEUE_JSONL))) == 6               # four stale + two real
    summary = slip_check.reclassify_checks(work, generated_at="2026-11-03T00:00:00Z")
    assert summary["records"] == 9 and summary["eye_queue"] == 2
    assert summary["changed"] == {"mismatch->match_fuzzy": 3, "mismatch->match_fuzzy_partial": 1}
    assert summary["outcomes"] == {"match": 1, "match_fuzzy": 3, "match_fuzzy_partial": 1, "mismatch": 1,
                                   "no_label": 1, "unreadable": 1, "error": 1}
    after = _by_file(work)
    assert _outcomes(work) == FUZZY_OUTCOMES and list(after) == list(fresh)            # same photographs, same order
    for name, record in after.items():
        assert record == fresh[name], name                                          # reads, times, audit: as fresh
    assert after["T-S_97.4324_1r.jpg"]["read"] is None                                 # the error record has no read
    text = (work / slip_check.EYE_QUEUE_MD).read_text()
    assert text.startswith("# Slip-check eye queue") and "Generated 2026-11-03T00:00:00Z" in text
    assert "9 photographs checked, 2 here" in text


def test_main_reclassify_needs_no_model_makes_no_call_and_is_idempotent(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """``--reclassify`` runs without ``--model``, never reaches the VLM and gives the same file when run again."""
    work = make_fuzzy_work(tmp_path)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(FUZZY_ANSWERS))
    _downgrade_to_strict(work)

    def refuse(*args: Any, **kwargs: Any) -> None:
        """Fail the test when the VLM is called.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        """
        raise AssertionError("--reclassify must not call the VLM")

    monkeypatch.setattr(slip_check, "lms_transcribe", refuse)
    monkeypatch.setattr(slip_check.lms_transcriber, "transcribe_with_lm_studio_meta", refuse)
    slip_check.main(["--work-dir", str(work), "--reclassify"])
    assert _outcomes(work) == FUZZY_OUTCOMES
    once = (work / slip_check.CHECKS_FILE).read_bytes()
    queue_once = (work / slip_check.EYE_QUEUE_JSONL).read_bytes()
    slip_check.main(["--work-dir", str(work), "--reclassify"])
    assert (work / slip_check.CHECKS_FILE).read_bytes() == once
    assert (work / slip_check.EYE_QUEUE_JSONL).read_bytes() == queue_once
    assert not (work / (slip_check.CHECKS_FILE + ".tmp")).exists()
    with pytest.raises(SystemExit):                                                    # a read run still needs a model
        slip_check.main(["--work-dir", str(work)])


def test_reclassify_without_a_check_file_fails(tmp_path: Path) -> None:
    """There is nothing to reclassify in a work dir without ``slip_checks.jsonl``."""
    (tmp_path / "work").mkdir()
    with pytest.raises(FileNotFoundError):
        slip_check.reclassify_checks(tmp_path / "work")


# ------------------------------------------------------------------- ranking the eye queue over a shelf-mark index

#: ``(ES id, display mark, institution)`` of the synthetic merged corpus the reads are resolved against.
INDEX_MARKS: List[Tuple[str, str, str]] = [
    ("Cambridge_CUL_T_S_AS_160_26", "T-S AS 160.26", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_AS_160_27", "T-S AS 160.27", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_AS_160_24", "T-S AS 160.24", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_AS_160_245", "T-S AS 160.245", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_AS_160_11", "T-S AS 160.11", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_AS_160_111", "T-S AS 160.111", "Cambridge CUL"),
    ("Cambridge_CUL_Or_1081_1_56", "Or.1081 1.56", "Cambridge CUL"),
    ("Cambridge_CUL_Or_1081_1_57", "Or.1081 1.57", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_Misc_28_79", "T-S Misc.28.79", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_Misc_28_79_1", "T-S Misc.28.79.1", "Cambridge CUL"),
    ("Cambridge_CUL_T_S_Misc_28_79_2", "T-S Misc.28.79.2", "Cambridge CUL"),
    ("Paris_AIU_II_B_21", "II.B.21", "Paris AIU"),
    ("Paris_AIU_II_B_22", "II.B.22", "Paris AIU"),
    ("Cambridge_CUL_II_B_21", "II.B.21", "Cambridge CUL")]       # the same bare mark elsewhere: no institution, no hit


def synthetic_index() -> shelf.ShelfmarkIndex:
    """The one shelf-mark index the ranking tests resolve against: the ids of :data:`INDEX_MARKS` and nothing else.

    :return: The index.
    """
    return shelf.ShelfmarkIndex([{"canonical_id": cid, "shelfmark_display": display, "institution": institution,
                                  "pgpids": []} for cid, display, institution in INDEX_MARKS])


#: ``(filed mark, canonical id, image file, the stubbed VLM answer)``; :data:`RANK_OUTCOMES` and :data:`RANK_PRIORITIES`
#: give what each becomes.
RANK_ROWS: List[Tuple[str, str, str, Answer]] = [
    ("Cambridge CUL: T-S AS 160.27", "Cambridge_CUL_T_S_AS_160_27", "T-S_AS_160.27_1r.jpg", "T-S AS 160.26"),
    ("Cambridge CUL: T-S AS 160.27", "Cambridge_CUL_T_S_AS_160_27", "T-S_AS_160.27_1v.jpg", "T-S AS 100.27"),
    ("Cambridge CUL: T-S AS 160.111", "Cambridge_CUL_T_S_AS_160_111", "T-S_AS_160.111_1r.jpg", "T-S AS 160.11"),
    ("Paris AIU: II.B.22", "Paris_AIU_II_B_22", "II_B_22_1r.jpg", "II.B.21"),
    ("Cambridge CUL: Or.1081 1.56", "Cambridge_CUL_Or_1081_1_56", "Or_1081_1_56_1r.jpg", "Or. 108i.1-57"),
    ("Cambridge CUL: T-S Misc.28.79", "Cambridge_CUL_T_S_Misc_28_79", "T-S_Misc_28.79_1r.jpg", "T-S Misc 28.79.14"),
    ("Cambridge CUL: T-S AS 160.24", "Cambridge_CUL_T_S_AS_160_24", "T-S_AS_160.24_1r.jpg", "illegible"),
    ("Cambridge CUL: T-S AS 160.245", "Cambridge_CUL_T_S_AS_160_245", "T-S_AS_160.245_1r.jpg", "T-S AS 160.245"),
    ("Cambridge CUL: T-S AS 160.245", "Cambridge_CUL_T_S_AS_160_245", "T-S_AS_160.245_1v.jpg", "T-S AS 160.24S")]
RANK_ANSWERS: Dict[str, Answer] = {name: answer for _, _, name, answer in RANK_ROWS}
#: Outcome after the ranking: the upgrade is the Misc.28.79 photograph, whose read names a child of the filed parent.
RANK_OUTCOMES: Dict[str, str] = {
    "T-S_AS_160.27_1r.jpg": "mismatch", "T-S_AS_160.27_1v.jpg": "mismatch", "T-S_AS_160.111_1r.jpg": "mismatch",
    "II_B_22_1r.jpg": "mismatch", "Or_1081_1_56_1r.jpg": "mismatch", "T-S_Misc_28.79_1r.jpg": "match_fuzzy",
    "T-S_AS_160.24_1r.jpg": "unreadable", "T-S_AS_160.245_1r.jpg": "match", "T-S_AS_160.245_1v.jpg": "match_fuzzy"}
RANK_PRIORITIES: Dict[str, Optional[str]] = {
    "T-S_AS_160.27_1r.jpg": "A", "T-S_AS_160.27_1v.jpg": "B", "T-S_AS_160.111_1r.jpg": "A", "II_B_22_1r.jpg": "A",
    "Or_1081_1_56_1r.jpg": "A", "T-S_Misc_28.79_1r.jpg": None, "T-S_AS_160.24_1r.jpg": "C",
    "T-S_AS_160.245_1r.jpg": None, "T-S_AS_160.245_1v.jpg": None}


def make_rank_work(tmp_path: Path) -> Path:
    """A work dir with stage1 for :data:`RANK_ROWS` and the strict-plus-fuzzy check file of a stubbed run (no ranking
    yet).

    :param tmp_path: Pytest temp dir.
    :return: The work dir.
    """
    work = tmp_path / "work"
    cands = [_cand(f"{cid}:{n}", mark, [cid], [_store_image(work, name)])
             for n, (mark, cid, name, _) in enumerate(RANK_ROWS)]
    write_jsonl(paths.stage_path(work, "images"), cands)
    slip_check.run_slip_checks(work, model="test-vlm", transcribe=Stub(RANK_ANSWERS))
    return work


def test_normalise_read_text_keeps_a_spelling_the_facade_can_parse() -> None:
    """Confusables mapped, the page marker dropped, separators between digit groups made dots (``1-56`` is a range)."""
    assert slip_check.normalise_read_text("Or. 108i.1-56") == "Or. 1081.1.56"
    assert slip_check.normalise_read_text("T-S AS 160 27") == "T-S AS 160.27"
    assert slip_check.normalise_read_text("T-S Misc 11.17 P2") == "T-S Misc 11.17"
    assert slip_check.normalise_read_text("T-S AS160.24S") == "T-S AS160.245"
    assert slip_check.normalise_read_text("T-S Misc 26:53-7") == "T-S Misc 26.53.7"
    assert slip_check.normalise_read_text("T-S 8J5.1") == "T-S 8J5.1"


def test_edit_distance_and_mark_distance() -> None:
    """The distance of a read from the filed mark is the edit distance of the tier-2 strings."""
    assert slip_check.edit_distance("kitten", "sitting") == 3 and slip_check.edit_distance("", "abc") == 3
    assert slip_check.edit_distance("tsas16026", "tsas16027") == 1 and slip_check.edit_distance("a", "a") == 0
    filings = [_filing("Cambridge CUL: T-S AS 160.27", "Cambridge_CUL_T_S_AS_160_27")]
    assert slip_check.mark_distance("T-S AS 160.26", filings) == 1
    assert slip_check.mark_distance("T-S AS 160.2", filings) == 1
    assert slip_check.mark_distance("illegible", filings) is None
    assert slip_check.is_truncated_read("T-S AS 160.2", filings)
    assert not slip_check.is_truncated_read("T-S AS 160.26", filings)


@pytest.mark.parametrize("resolved, own, kind, note", [
    ({"x_1"}, {"o_1"}, "A", ""),                                    # another existing mark
    ({"x_1", "x_2"}, {"o_1"}, "A", ""),                             # several ids, none the photograph's own
    ({"o_1"}, {"o_1", "o_2"}, "upgrade", ""),                       # only its own ids
    ({"o_1", "o_2"}, {"o_1", "o_2"}, "upgrade", ""),
    (set(), {"o_1"}, "B", ""),                                      # names no existing mark
    ({"o_1", "x_1"}, {"o_1"}, "B", "resolves_to_own_and_other_ids")])   # its own id among others
def test_rank_kind(resolved: set, own: set, kind: str, note: str) -> None:
    """A: only other existing ids; upgrade: only its own; B: nothing, or its own among others."""
    assert slip_check.rank_kind(resolved, own) == (kind, note)


def test_resolve_read_tries_the_normalised_spelling_under_the_filings_institution() -> None:
    """One index; the filing's institution resolves a bare mark; only the normalised spelling resolves ``108i``."""
    index = synthetic_index()
    assert slip_check.resolve_read("II.B.21", ["Paris_AIU"], index) == (["Paris_AIU_II_B_21"], "exact", False)
    assert slip_check.resolve_read("II.B.21", [], index) == ([], "unmatched", False)
    assert slip_check.resolve_read("Or. 108i.1-57", ["Cambridge_CUL"], index)[0] == ["Cambridge_CUL_Or_1081_1_57"]
    assert shelf.resolve("Or. 108i.1-57", index=index, institution="Cambridge_CUL").cids == []     # the raw read alone
    assert slip_check.resolve_read("T-S AS 100.27", ["Cambridge_CUL"], index) == ([], "unmatched", False)
    assert slip_check.resolve_read("T-S Misc 28.79.14", ["Cambridge_CUL"], index)[1] == "parent"


def test_rank_checks_gives_a_b_c_upgrades_and_writes_both_files(tmp_path: Path) -> None:
    """A: the read names another existing mark; B: none; C: unreadable; the child of the filed parent is upgraded."""
    work = make_rank_work(tmp_path)
    assert {n: r["outcome"] for n, r in _by_file(work).items()}["T-S_Misc_28.79_1r.jpg"] == "mismatch"   # before
    summary = slip_check.rank_checks(work, synthetic_index(), generated_at="2026-11-03T00:00:00Z")
    assert summary["priorities"] == {"A": 4, "B": 1, "C": 1} and summary["upgraded"] == 1
    assert summary["changed"] == {"mismatch->match_fuzzy": 1} and summary["eye_queue"] == 6 and summary["records"] == 9
    records = _by_file(work)
    assert _outcomes(work) == RANK_OUTCOMES
    assert {name: rec.get("eye_priority") for name, rec in records.items()} == RANK_PRIORITIES
    near = records["T-S_AS_160.27_1r.jpg"]
    assert near["resolved_ids"] == ["Cambridge_CUL_T_S_AS_160_26"] and near["eye_label"] == "likely_misfiled"
    assert (near["resolved_how"], near["resolved_ambiguous"], near["mark_distance"]) == ("exact", False, 1)
    noise = records["T-S_AS_160.27_1v.jpg"]
    assert noise["resolved_ids"] == [] and noise["eye_label"] == "likely_ocr_noise"
    assert records["II_B_22_1r.jpg"]["resolved_ids"] == ["Paris_AIU_II_B_21"]
    assert records["Or_1081_1_56_1r.jpg"]["resolved_ids"] == ["Cambridge_CUL_Or_1081_1_57"]
    assert records["T-S_AS_160.111_1r.jpg"]["rank_note"] == slip_check.TRUNCATION_NOTE
    assert "rank_note" not in near and records["T-S_AS_160.24_1r.jpg"]["eye_label"] == "unreadable"
    assert records["T-S_AS_160.24_1r.jpg"]["resolved_ids"] == []
    up = records["T-S_Misc_28.79_1r.jpg"]                                               # upgraded, off the queue
    assert up["reason"] == "label_resolves_to_filed_ids_in_index"
    assert up["matched_row_ids"] == ["Cambridge_CUL_T_S_Misc_28_79:5"]
    assert up["resolved_ids"] == ["Cambridge_CUL_T_S_Misc_28_79"]
    assert up["rank_note"].startswith(slip_check.UPGRADE_NOTE)
    assert up["fuzzy"]["upgraded_by_index"] is True and "eye_priority" not in up
    for name in ("T-S_AS_160.245_1r.jpg", "T-S_AS_160.245_1v.jpg"):                    # matches are never ranked
        assert not set(slip_check.RANK_FIELDS) & set(records[name])
    queue = list(read_jsonl(work / slip_check.EYE_QUEUE_JSONL))
    assert [(q["eye_priority"], Path(q["image_path"]).name) for q in queue] == [
        ("A", "II_B_22_1r.jpg"), ("A", "Or_1081_1_56_1r.jpg"), ("A", "T-S_AS_160.27_1r.jpg"),
        ("A", "T-S_AS_160.111_1r.jpg"), ("B", "T-S_AS_160.27_1v.jpg"), ("C", "T-S_AS_160.24_1r.jpg")]
    assert queue[2]["resolved_ids"] == ["Cambridge_CUL_T_S_AS_160_26"] and queue[2]["mark_distance"] == 1
    assert queue[5]["resolved_ids"] == [] and queue[5]["eye_label"] == "unreadable"
    text = (work / slip_check.EYE_QUEUE_MD).read_text()
    assert "9 photographs checked, 6 here: A 4, B 1, C 1; 1 read(s) the index upgraded to `match_fuzzy`" in text
    headings = ["## A: likely_misfiled (4)", "## B: likely_ocr_noise (1)", "## C: unreadable (1)"]
    assert text.index(headings[0]) < text.index(headings[1]) < text.index(headings[2])
    a_row = (f"| 3 | {work / 'images' / 'T-S_AS_160.27_1r.jpg'} | Cambridge CUL: T-S AS 160.27 | T-S AS 160.26 | "
             "Cambridge_CUL_T_S_AS_160_26 | 1 |  |")
    assert a_row in text and "| 4 |" in text and slip_check.TRUNCATION_NOTE in text
    assert "T-S_Misc_28.79_1r" not in text and "T-S_AS_160.245" not in text


def test_ranking_is_repeatable_and_a_reclassify_drops_it(tmp_path: Path) -> None:
    """A second ranking gives the same files; ``--reclassify`` clears the ranking and the upgrade."""
    work = make_rank_work(tmp_path)
    slip_check.rank_checks(work, synthetic_index())
    once = (work / slip_check.CHECKS_FILE).read_bytes()
    queue_once = (work / slip_check.EYE_QUEUE_JSONL).read_bytes()
    slip_check.rank_checks(work, synthetic_index())
    assert (work / slip_check.CHECKS_FILE).read_bytes() == once
    assert (work / slip_check.EYE_QUEUE_JSONL).read_bytes() == queue_once
    summary = slip_check.reclassify_checks(work)
    assert summary["changed"] == {"match_fuzzy->mismatch": 1}
    records = _by_file(work)
    assert not any(set(slip_check.RANK_FIELDS) & set(rec) for rec in records.values())
    assert records["T-S_Misc_28.79_1r.jpg"]["outcome"] == "mismatch" and "upgraded_by_index" not in records[
        "T-S_Misc_28.79_1r.jpg"]["fuzzy"]
    flat = (work / slip_check.EYE_QUEUE_MD).read_text()
    assert "| # | outcome | reason |" in flat and "## A:" not in flat
    queue = list(read_jsonl(work / slip_check.EYE_QUEUE_JSONL))
    assert len(queue) == 7 and not any("eye_priority" in q for q in queue)


def test_queue_order_and_a_mixed_queue(tmp_path: Path) -> None:
    """A, B, C, then unranked; within A the nearest first and, at one distance, the truncated reads last."""
    def entry(sha: str, priority: Optional[str], distance: Optional[int], note: str = "",
              outcome: str = "mismatch") -> Dict[str, Any]:
        """An eye entry.

        :param sha: sha256 (also the file name).
        :param priority: ``eye_priority`` or None.
        :param distance: ``mark_distance``.
        :param note: ``rank_note``.
        :param outcome: Outcome.
        :return: The entry.
        """
        out: Dict[str, Any] = {"sha256": sha, "file_name": sha, "outcome": outcome, "mark_distance": distance,
                               "rank_note": note or None}
        if priority:
            out["eye_priority"] = priority
        return out

    entries = [entry("u", None, None), entry("c", "C", None, outcome="unreadable"), entry("b", "B", 0),
               entry("a3", "A", 3), entry("a1t", "A", 1, slip_check.TRUNCATION_NOTE), entry("a1", "A", 1),
               entry("a2", "A", 2)]
    ordered = sorted(entries, key=slip_check.queue_order)
    assert [e["sha256"] for e in ordered] == ["a1", "a1t", "a2", "a3", "b", "c", "u"]
    text = slip_check.render_eye_queue(ordered, 9, "now", 2)
    assert "A 4, B 1, C 1, 1 unranked; 2 read(s)" in text and "## unranked (run `--rank-eye-queue`) (1)" in text


def test_main_rank_eye_queue_loads_one_index_and_makes_no_vlm_call(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """``--rank-eye-queue`` needs no model, loads the index once and never reaches the VLM."""
    work = make_rank_work(tmp_path)
    loads: List[int] = []

    def one_index() -> shelf.ShelfmarkIndex:
        """Count the loads and give the synthetic index.

        :return: The synthetic index.
        """
        loads.append(1)
        return synthetic_index()

    def refuse(*args: Any, **kwargs: Any) -> None:
        """Fail the test when the VLM is called.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        """
        raise AssertionError("--rank-eye-queue must not call the VLM")

    monkeypatch.setattr(slip_check, "load_shelfmark_index", one_index)
    monkeypatch.setattr(slip_check, "lms_transcribe", refuse)
    monkeypatch.setattr(slip_check.lms_transcriber, "transcribe_with_lm_studio_meta", refuse)
    slip_check.main(["--work-dir", str(work), "--rank-eye-queue"])
    assert loads == [1] and _outcomes(work) == RANK_OUTCOMES
    assert {name: rec.get("eye_priority") for name, rec in _by_file(work).items()} == RANK_PRIORITIES


def test_rank_checks_without_a_check_file_fails(tmp_path: Path) -> None:
    """There is nothing to rank in a work dir without ``slip_checks.jsonl``."""
    (tmp_path / "work").mkdir()
    with pytest.raises(FileNotFoundError):
        slip_check.rank_checks(tmp_path / "work", synthetic_index())


def test_the_index_is_the_merged_corpus_cached_where_rekey_caches_it(tmp_path: Path) -> None:
    """Defaults: the merged corpus and ``logs/next_round/unseen_v1/cache``; a given file is built once, then loaded."""
    assert slip_check.MERGED_CORPUS == shelf.MERGED_SHELFMARKS
    assert slip_check.INDEX_CACHE_DIR == paths.REPO_ROOT / "logs" / "next_round" / "unseen_v1" / "cache"
    merged = tmp_path / "merged_shelfmarks.jsonl"
    write_jsonl(merged, [{"canonical_id": cid, "shelfmark_display": display, "institution": institution, "pgpids": []}
                         for cid, display, institution in INDEX_MARKS])
    cache = tmp_path / "cache"
    index = slip_check.load_shelfmark_index(merged, cache)
    assert len(index) == len(INDEX_MARKS) and len(list(cache.glob("shelfmark_index_*.pkl"))) == 1
    stamp = next(cache.glob("shelfmark_index_*.pkl")).stat().st_mtime_ns
    again = slip_check.load_shelfmark_index(merged, cache)
    assert len(again) == len(index) and next(cache.glob("shelfmark_index_*.pkl")).stat().st_mtime_ns == stamp
    assert slip_check.resolve_read("T-S AS 160.26", ["Cambridge_CUL"], again)[0] == ["Cambridge_CUL_T_S_AS_160_26"]
