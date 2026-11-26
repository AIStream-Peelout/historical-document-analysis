# File name: test_score_reader_decode_arms.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests of the decode-arm scorer's own pieces (the covered-only scoring itself is tested in test_score_arabic_covered)."""
from typing import Any, Dict

from src.datasets.evaluations.helper_eval_scripts import score_reader_decode_arms as sd

TEXT = "بسم الله الرحمن الرحيم الى الشيخ ابو سعد اطال الله بقاه وادام عزه"


def _doc(doc: str, f1: float, ler: float) -> Dict[str, Any]:
    return {"doc_id": doc, "ngram_f1": f1, "ler": ler}


def test_median_difference_moves_with_a_change_near_the_median() -> None:
    old = [_doc(f"d{i}", v, 0.5) for i, v in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))]
    new = [_doc(f"d{i}", v, 0.5) for i, v in enumerate((0.1, 0.2, 0.35, 0.4, 0.5))]
    out = sd.median_difference(new, old, "ngram_f1", resamples=2000)
    assert out["documents"] == 5 and out["difference"] == 0.05 and out["ci"][0] <= 0.05 <= out["ci"][1]
    assert sd.median_difference(new, [], "ngram_f1") == {"documents": 0}


def test_behaviour_counts_the_reader_fields_and_tolerates_saved_reads() -> None:
    keys = [("a", 0), ("a", 1), ("b", 0)]
    rows = {("a", 0): {"text": TEXT, "first_finish_reason": "length", "finish_reason": "stop", "retried": True, "attempts": 2,
                       "reason": None, "stripped_chars": 4, "rotation": 180, "crop": {"accepted": True}, "completion_tokens": 100,
                       "orientation": {"gate": "markers", "probes": {"0": {}, "90": {}, "180": {}, "270": {}}}},
            ("a", 1): {"text": "", "first_finish_reason": None, "finish_reason": None, "retried": False, "attempts": 0,
                       "reason": "blank", "stripped_chars": 0, "rotation": 0, "crop": None},
            ("b", 0): {"text": TEXT + " [...]" * 25, "first_finish_reason": "stop", "finish_reason": "stop", "retried": False,
                       "attempts": 1, "reason": None, "stripped_chars": 0, "rotation": 0, "crop": {"accepted": False},
                       "completion_tokens": 300}}
    b = sd.behaviour(rows, keys)
    assert (b["first_length"], b["final_length"], b["retried"], b["requests"], b["blank_gated"]) == (1, 0, 1, 7, 1)
    assert (b["filtered"], b["turned"], b["cropped"], b["crops_accepted"], b["floods"], b["tokens_mean"]) == (1, 1, 2, 1, 1, 200.0)
    assert b["probed"] == 1
    saved = sd.behaviour({k: {"text": TEXT} for k in keys}, keys)
    assert saved["first_length"] is None and saved["requests"] is None and saved["answered"] == 3


def test_image_score_uses_the_copied_image_sections() -> None:
    record = {"id": "a", "images": [{"image_index": 0}, {"image_index": 1}],
              "gt": {"sections": [{"side": "", "lines": [TEXT]}, {"side": "", "lines": ["كتاب"]}]}}
    coverage = {0: {"sections": [0]}, 1: {"sections": []}}
    assert sd.image_score(record, coverage, 1, 0, TEXT)["ngram_f1"] == 1.0
    assert sd.image_score(record, coverage, 1, None, TEXT)["ngram_f1"] is None
    assert sd.image_score(record, coverage, 0, None, None)["ler"] is None
