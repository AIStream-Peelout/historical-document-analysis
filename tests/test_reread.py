"""Tests for the re-read helpers (src/datasets/consensus/reread.py) and the LM Studio payload extension."""
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from src.datasets.consensus import reread as rr
from src.datasets.consensus.read_health import LABEL_CAPPED, LABEL_LOOP, LABEL_NEAR_EMPTY, LABEL_OK, LABEL_PARSE_LOSS, LABEL_SKIPPED
from src.models.ocr import lms_transcriber as lms

LINES = ["שלום וברכה לאדוני הנכבד", "אחר כן אודיעך כי קבלתי", "את כתבך ושמחתי בו מאד", "ודע כי המעות הגיעו",
         "ליד יעקב בן יצחק הזקן", "ושלום רב לך ולכל ביתך", "וכתבתי ביום שני בשבוע", "בחדש תשרי שנת אלף"]


def lines_of(texts: List[str]) -> List[Dict[str, Any]]:
    """Line dicts as the reader returns them (text plus a box that must be carried along).

    :param texts: Line texts.
    :type texts: List[str]
    :return: ``{text, box}`` dicts.
    :rtype: List[Dict[str, Any]]
    """
    return [{"text": t, "box": [0, i, 10, i + 1]} for i, t in enumerate(texts)]


# ----------------------------------------------------------------------------- half-page crops


def test_half_boxes_overlap_around_the_middle() -> None:
    top, bottom = rr.half_boxes(1000, 2000, overlap=0.12)
    assert top == (0, 0, 1000, 1120) and bottom == (0, 880, 1000, 2000)
    assert top[3] - bottom[1] == 240                         # 12 % of the height is in both halves


def test_half_boxes_without_overlap_meet_in_the_middle() -> None:
    top, bottom = rr.half_boxes(800, 1001, overlap=0.0)
    assert top[3] == bottom[1] == 500 or top[3] == bottom[1] == 501


@pytest.mark.parametrize("w, h, overlap", [(0, 100, 0.1), (100, 0, 0.1), (100, 100, 1.0), (100, 100, -0.1)])
def test_half_boxes_reject_bad_input(w: int, h: int, overlap: float) -> None:
    with pytest.raises(ValueError):
        rr.half_boxes(w, h, overlap)


def test_crop_halves_share_the_overlap_pixels() -> None:
    im = Image.new("L", (40, 200))
    im.putdata([y for y in range(200) for _ in range(40)])  # each row's value is its y
    top, bottom = rr.crop_halves(im, overlap=0.2)
    assert top.size == (40, 120) and bottom.size == (40, 120)
    assert [top.getpixel((5, y)) for y in range(80, 120)] == [bottom.getpixel((5, y - 80)) for y in range(80, 120)]


def test_to_page_box_maps_a_bottom_half_box_back() -> None:
    _, bottom = rr.half_boxes(1000, 2000, overlap=0.12)     # rows 880-2000
    box = rr.to_page_box([100, 0, 900, 1000], bottom, 1000, 2000)
    assert box == [100.0, 440.0, 900.0, 1000.0]


# ----------------------------------------------------------------------------- join


def test_line_match_scores() -> None:
    assert rr.line_match(LINES[0], LINES[0]) == pytest.approx(1.0)
    assert rr.line_match("וברכה לאדוני", LINES[0]) > 0.9               # a line cut by the crop edge
    assert rr.line_match(LINES[0], LINES[5]) < 0.6
    assert rr.line_match("", LINES[0]) == 0.0 and rr.line_match("123", LINES[0]) == 0.0


def test_join_drops_the_overlap_lines() -> None:
    top = lines_of(LINES[:5])
    bottom = lines_of(LINES[3:])                             # lines 3 and 4 read by both halves
    joined, dropped = rr.join_halves(top, bottom)
    assert [ln["text"] for ln in joined] == LINES and dropped == 2


def test_join_keeps_the_longer_reading_of_a_line_cut_by_the_crop() -> None:
    top = lines_of(LINES[:4] + ["ליד יעקב בן"])             # the top crop's edge cut line 4
    bottom = lines_of(["ושמחתי בו מאד", LINES[3], LINES[4], LINES[5], LINES[6]])   # the bottom crop's edge cut line 2
    joined, dropped = rr.join_halves(top, bottom)
    assert [ln["text"] for ln in joined] == LINES[:7]
    assert dropped == 3


def test_join_survives_a_line_one_half_skipped() -> None:
    top = lines_of(LINES[:3] + LINES[4:6])                   # the top read skipped line 3
    bottom = lines_of(LINES[3:])
    joined, _ = rr.join_halves(top, bottom)
    texts = [ln["text"] for ln in joined]
    assert texts == LINES[:3] + LINES[4:]                    # line 3 is lost, nothing is duplicated
    assert len(texts) == len(set(texts))


def test_join_without_overlap_concatenates() -> None:
    joined, dropped = rr.join_halves(lines_of(LINES[:4]), lines_of(LINES[4:]))
    assert [ln["text"] for ln in joined] == LINES and dropped == 0


def test_join_window_protects_repeats_far_below_the_overlap() -> None:
    body = [f"שורה מספר {k} של הדף הזה" for k in range(10)]
    top = lines_of(LINES[:5])
    bottom = lines_of(LINES[3:] + body + [LINES[0]])         # the page closes with its opening formula again
    joined, dropped = rr.join_halves(top, bottom, window=4)
    texts = [ln["text"] for ln in joined]
    assert dropped == 2 and texts == LINES + body + [LINES[0]]


def test_join_carries_boxes_and_copies_lines() -> None:
    top, bottom = lines_of(LINES[:5]), lines_of(LINES[3:])
    joined, _ = rr.join_halves(top, bottom)
    assert all("box" in ln for ln in joined)
    joined[0]["text"] = "x"
    assert top[0]["text"] == LINES[0]


# ----------------------------------------------------------------------------- jobs


def cand(job_id: str, cls: str, held_out: bool = True) -> Dict[str, Any]:
    """A re-read candidate.

    :param job_id: Job id.
    :type job_id: str
    :param cls: Failure class.
    :type cls: str
    :param held_out: Held-out flag.
    :type held_out: bool
    :return: Candidate dict.
    :rtype: Dict[str, Any]
    """
    return {"job_id": job_id, "failure_class": cls, "held_out": held_out}


def test_select_jobs_filters_and_balances() -> None:
    cands = ([cand(f"loop{i}", LABEL_LOOP) for i in range(20)] + [cand(f"cap{i}", LABEL_CAPPED) for i in range(2)]
             + [cand(f"near{i}", LABEL_NEAR_EMPTY) for i in range(3)] + [cand(f"skip{i}", LABEL_SKIPPED) for i in range(3)]
             + [cand("parse0", LABEL_PARSE_LOSS), cand("ok0", LABEL_OK), cand("trained0", LABEL_LOOP, held_out=False),
                cand("loop0", LABEL_LOOP)])                  # duplicate id
    jobs = rr.select_jobs(cands, limit=8)
    assert [j["failure_class"] for j in jobs[:4]] == [LABEL_CAPPED, LABEL_NEAR_EMPTY, LABEL_SKIPPED, LABEL_LOOP]
    assert {j["failure_class"] for j in jobs} == {LABEL_CAPPED, LABEL_NEAR_EMPTY, LABEL_SKIPPED, LABEL_LOOP}
    assert len(jobs) == 8 and len({j["job_id"] for j in jobs}) == 8
    everything = rr.select_jobs(cands, limit=0)
    assert len(everything) == 20 + 2 + 3 + 3
    assert not {"parse0", "ok0", "trained0"} & {j["job_id"] for j in everything}
    assert "trained0" in {j["job_id"] for j in rr.select_jobs(cands, limit=0, held_out_only=False)}


def test_select_jobs_keeps_blank_image_loops_apart_and_held_out_first() -> None:
    cands = ([{**cand(f"blank{i}", LABEL_LOOP), "blank_image": True} for i in range(10)]
             + [cand(f"text{i}", LABEL_LOOP) for i in range(10)]
             + [cand(f"trained{i}", LABEL_LOOP, held_out=False) for i in range(10)])
    jobs = rr.select_jobs(cands, limit=6, held_out_only=False)
    assert [rr.failure_stratum(j) for j in jobs] == ["loop", "loop/blank_image"] * 3
    assert not any(j["job_id"].startswith("trained") for j in jobs)        # held-out reads come first
    more = rr.select_jobs(cands, limit=26, held_out_only=False)
    assert sum(j["job_id"].startswith("trained") for j in more) == 6
    assert {j["job_id"] for j in rr.select_jobs(cands, limit=0)} <= {j["job_id"] for j in rr.select_jobs(cands, 0, False)}


def test_select_jobs_is_deterministic_and_fills_with_the_big_class() -> None:
    cands = [cand(f"loop{i}", LABEL_LOOP) for i in range(30)] + [cand("cap0", LABEL_CAPPED)]
    a = rr.select_jobs(cands, limit=10)
    b = rr.select_jobs(list(reversed(cands)), limit=10)
    assert [j["job_id"] for j in a] == [j["job_id"] for j in b]
    assert a[0]["job_id"] == "cap0" and sum(j["failure_class"] == LABEL_LOOP for j in a) == 9


# ----------------------------------------------------------------------------- variants


def test_variant_plan_runs_the_raised_cap_on_capped_jobs_only() -> None:
    names = list(rr.VARIANTS)
    assert [v.name for v in rr.variant_plan(LABEL_CAPPED, names)] == names
    assert "max_tokens_6000" not in [v.name for v in rr.variant_plan(LABEL_LOOP, names)]
    with pytest.raises(KeyError):
        rr.variant_plan(LABEL_LOOP, ["nonsense"])


def test_default_variants_carry_the_request_changes() -> None:
    v = rr.default_variants(seed=11, repeat_penalty=1.2, capped_max_tokens=7000)
    assert v["temp03_seed"].temperature == 0.3 and v["temp03_seed"].extra_payload == {"seed": 11}
    assert v["repeat_penalty"].extra_payload == {"repeat_penalty": 1.2}
    assert v["max_tokens_7000"].max_tokens == 7000 and v["max_tokens_7000"].only_classes == (LABEL_CAPPED,)
    assert v["same"].max_tokens is None and v["same"].temperature is None and not v["same"].extra_payload
    assert v["halves"].requests_per_job() == 2 and v["same"].requests_per_job() == 1


# ----------------------------------------------------------------------------- scoring


def test_paired_summary_counts_and_policy() -> None:
    rows = [{"cer": 0.2, "err": 20, "ref": 100, "base_cer": 0.9, "base_err": 90, "label": LABEL_OK},
            {"cer": 0.95, "err": 95, "ref": 100, "base_cer": 0.9, "base_err": 90, "label": LABEL_LOOP},
            {"cer": 0.5, "err": 50, "ref": 100, "base_cer": 0.51, "base_err": 51, "label": LABEL_OK}]
    s = rr.paired_summary(rows)
    assert (s["n"], s["better"], s["worse"], s["same"], s["healthy"]) == (3, 1, 1, 1, 2)
    assert s["pooled_cer"] == pytest.approx(165 / 300) and s["pooled_base_cer"] == pytest.approx(231 / 300)
    assert s["policy_pooled_cer"] == pytest.approx((20 + 90 + 50) / 300)
    assert rr.paired_summary([]) == {"n": 0}


# ----------------------------------------------------------------------------- LM Studio payload


def test_chat_payload_without_extras_is_the_usual_body() -> None:
    body = lms.chat_payload("m", "data:image/jpeg;base64,AA", "prompt", 0.1, 3500)
    assert list(body) == ["model", "temperature", "max_tokens", "messages"]
    assert body["messages"][0]["content"][1] == {"type": "text", "text": "prompt"}
    assert lms.chat_payload("m", "d", "p", 0.1, 3500, {}) == lms.chat_payload("m", "d", "p", 0.1, 3500)


def test_chat_payload_merges_extras_but_never_model_or_messages() -> None:
    body = lms.chat_payload("m", "d", "p", 0.3, 3500, {"seed": 7, "repeat_penalty": 1.1})
    assert body["seed"] == 7 and body["repeat_penalty"] == 1.1 and body["temperature"] == 0.3
    with pytest.raises(ValueError):
        lms.chat_payload("m", "d", "p", 0.1, 3500, {"model": "other"})


def test_transcribe_sends_the_extra_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sent: List[Dict[str, Any]] = []

    class Response:
        """Fake aiohttp response."""

        status = 200

        async def json(self) -> Dict[str, Any]:
            """Reply body.

            :return: One choice.
            :rtype: Dict[str, Any]
            """
            return {"choices": [{"message": {"content": " [] "}}]}

        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    class Session:
        """Fake aiohttp session that records the posted body."""

        def __init__(self, timeout: Any = None) -> None:
            self.timeout = timeout

        def post(self, url: str, json: Dict[str, Any]) -> Response:
            """Record the body.

            :param url: Endpoint.
            :type url: str
            :param json: Body.
            :type json: Dict[str, Any]
            :return: Fake response.
            :rtype: Response
            """
            sent.append(json)
            return Response()

        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(lms.aiohttp, "ClientSession", Session)
    img = tmp_path / "page.jpg"
    Image.new("RGB", (8, 8)).save(img)
    out = asyncio.run(lms.transcribe_with_lm_studio("m", str(img), "p", max_tokens=6000, temperature=0.3,
                                                    extra_payload={"seed": 5}))
    assert out == "[]"
    assert sent[0]["seed"] == 5 and sent[0]["max_tokens"] == 6000 and sent[0]["temperature"] == 0.3
    asyncio.run(lms.transcribe_with_lm_studio("m", str(img), "p"))
    assert "seed" not in sent[1] and sent[1]["temperature"] == lms.DEFAULT_CONFIG.temperature
