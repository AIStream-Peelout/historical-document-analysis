# File name: test_rover_consensus.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the ROVER output-space consensus (toy hypotheses, no benchmark data).

The religious-140 adapter is tested on a toy spec written to ``tmp_path``.
"""
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from src.datasets.evaluations.helper_eval_scripts import rover_consensus as rc

GOOD = "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהו"
GOOD_VARIANT = "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהי"
LOOP = "ויאמר אלהים יהי אור " * 8
UNRELATED = "שמע ישראל יהוה אלהינו יהוה אחד ואהבת את יהוה אלהיך בכל לבבך"


def test_identical_readers_return_the_same_text():
    text = "שלום עולם\nשורה שניה כאן"
    result = rc.rover([text, text, text])
    assert result.text == text
    assert result.backbone == 0 and result.n_slots == len(rc.tokenize(text.split("\n"), "word"))


@pytest.mark.parametrize("backbone", [None, 0, 2])
def test_majority_beats_a_single_substitution_even_from_the_backbone(backbone):
    hyps = ["אבג דהו זחט", "אבג דהו זחט", "אבג דהי זחט"]
    assert rc.rover(hyps, backbone=backbone).text == "אבג דהו זחט"


def test_deleted_word_is_restored_by_the_majority():
    hyps = ["אחד שתים שלש", "אחד שלש", "אחד שתים שלש"]
    for backbone in (None, 1):
        assert rc.rover(hyps, backbone=backbone).text == "אחד שתים שלש"


def test_inserted_word_loses_to_null_even_when_it_is_in_the_backbone():
    hyps = ["אחד שתים", "אחד נוסף שתים", "אחד שתים"]
    assert rc.rover(hyps).text == "אחד שתים"
    forced = rc.rover(hyps, backbone=1)
    assert forced.backbone == 1 and forced.text == "אחד שתים"
    assert forced.n_slots == 3                   # the insertion has its own NULL-heavy slot


def test_weights_break_a_one_one_tie():
    hyps = ["אבג דהו", "אבג דהי"]
    assert rc.rover(hyps, weights=[1, 1], backbone=0).text == "אבג דהו"   # tie -> backbone
    assert rc.rover(hyps, weights=[1, 2], backbone=0).text == "אבג דהי"   # heavier model wins


def test_tie_without_the_backbone_goes_to_the_strongest_supporter():
    hyps = ["אבגד", "אבגה", "אבגו", "אבגו"]
    # tally: backbone 1.0, אבגה 1.5, אבגו 0.5 + 1.0 = 1.5 -> tie; strongest supporter 1.5 > 1.0
    assert rc.rover(hyps, weights=[1.0, 1.5, 0.5, 1.0], backbone=0).text == "אבגה"


def test_vote_slot_null_is_a_candidate():
    assert rc.vote_slot(("א", None, None), (0, 1, 2), [1, 1, 1], backbone=0) is None
    assert rc.vote_slot(("א", None), (0, 1), [1, 1], backbone=0) == "א"
    assert rc.vote_slot(("א", None), (0, 1), [1, 1], backbone=1) is None


def test_newline_tokens_survive_voting():
    two_lines = "אחד שתים\nשלש ארבע"
    hyps = [two_lines, two_lines, "אחד שתים שלש ארבע"]
    assert rc.rover(hyps).text == two_lines
    assert rc.tokenize(["אחד שתים", "שלש"], "word") == ["אחד", "שתים", rc.NEWLINE, "שלש"]
    assert rc.detokenize([rc.NEWLINE, "א", rc.NEWLINE, rc.NEWLINE, "ב", rc.NEWLINE], "word") == "א\nב"


def test_normalize_lines_keeps_line_structure_and_drops_markers():
    raw = "[...] אבג דהו\n\n**זחט** [?]\n[...]\n"
    assert rc.normalize_lines(raw) == ["אבג דהו", "זחט"]


def test_char_unit_votes_characters():
    hyps = ["שלום עולם\nטוב", "שלום עולם\nטוב", "שלוס עולם\nטוב"]
    assert rc.rover(hyps, unit="char").text == "שלום עולם\nטוב"
    assert rc.rover(["אבגד", "אבגד", "אגד"], unit="char").text == "אבגד"


def test_alignment_path_covers_every_slot_and_token_once():
    network = rc.build_network([["אחד", "שתים", "שלש"], ["אחד", "שלש", "ארבע"]], [0, 1], gap=350)
    path = rc.align_to_network(network, ["אפס", "אחד", "שלש"], gap=350)
    assert sorted(i for i, _ in path if i is not None) == list(range(len(network.slots)))
    assert [j for _, j in path if j is not None] == [0, 1, 2]
    assert (1, None) in path                     # "שתים" slot gets NULL from the new reader


def test_dissimilar_words_become_insert_delete_pairs():
    network = rc.seed_network(["אבגד"], 0)
    assert rc.align_to_network(network, ["אבגה"], gap=350) == [(0, 0)]           # d = 0.25
    split = rc.align_to_network(network, ["שתקר"], gap=350)                        # d = 1.0
    assert split in ([(0, None), (None, 0)], [(None, 0), (0, None)])


def test_failure_filter_drops_a_looping_reader():
    gt_letters = rc.letters_only(GOOD)
    raws = [GOOD, GOOD_VARIANT, LOOP]
    modes = rc.failure_modes(raws, gt_letters, "gt")
    assert modes == ["substantive", "substantive", "loop_collapse"]
    weights = [1.0, 1.0, 5.0]
    assert rc.rover(raws, weights=weights).text != GOOD          # the heavy loop dominates
    voters = rc.select_voters(modes, weights)
    assert voters == [0, 1]
    assert rc.rover([raws[i] for i in voters], [weights[i] for i in voters]).text == GOOD


def test_failure_filter_falls_back_to_the_heaviest_reader():
    assert rc.select_voters(["loop_collapse", "abstained", "hallucinated"], [1.0, 3.0, 2.0]) == [1]
    assert rc.select_voters(["abstained", "abstained"], [1.0, 1.0]) == [0]


def test_peer_filter_flags_a_reader_that_matches_nobody_without_the_gt():
    modes = rc.failure_modes([GOOD, GOOD_VARIANT, UNRELATED, LOOP], None, "peer")
    assert modes == ["substantive", "substantive", "hallucinated", "loop_collapse"]
    with pytest.raises(ValueError):
        rc.failure_modes([GOOD], None, "gt")


def test_backbone_choice_is_deterministic():
    hyps = ["אחד שתים שלש ארבע", "אחד שתים שלש חמש", "אחד שתים שש ארבע", "זה משהו אחר לגמרי"]
    first = rc.rover(hyps)
    assert all(rc.rover(hyps) == first for _ in range(3))
    sim = rc.similarity_matrix(hyps)
    assert first.backbone == rc.choose_backbone(sim, [1.0] * 4) == 0
    assert first.order[0] == 0 and sorted(first.order) == [0, 1, 2, 3]
    tied = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert rc.choose_backbone(tied, [1.0, 1.0, 1.0]) == 0      # exact tie -> lowest index
    assert rc.choose_backbone(tied, [1.0, 2.0, 2.0]) == 1      # ... unless a weight is higher


def test_single_hypothesis_passes_through_unchanged():
    raw = "[...] אבג דהו\nזחט טיכ [?]"
    assert rc.rover([raw]).text == "\n".join(rc.normalize_lines(raw))
    assert rc.rover([""]).text == ""


def test_leave_one_out_weights_exclude_the_document_itself():
    cers = {"a": {"m": 0.1}, "b": {"m": 0.2}, "c": {"m": 0.4}}
    weights = rc.leave_one_out_weights(cers, ["m"], {"a": "x", "b": "x", "c": "y"}, per_bucket=False)
    assert weights["a"]["m"] == pytest.approx(1 / 0.3)
    assert weights["c"]["m"] == pytest.approx(1 / 0.15)


def test_consensus_for_document_filters_and_falls_back_to_an_automatic_backbone():
    models, raws = ["a", "b", "c"], [GOOD, GOOD_VARIANT, LOOP]
    weights = {"a": 1.0, "b": 1.0, "c": 5.0}
    gt_modes = ["substantive", "substantive", "loop_collapse"]
    result, voters, fallback = rc.consensus_for_document(raws, models, weights, gt_modes,
                                                         backbone="c", filter_source="gt")
    assert voters == ["a", "b"] and not fallback and result.text == GOOD
    unfiltered, voters, _ = rc.consensus_for_document(raws, models, weights, gt_modes)
    assert voters == models and unfiltered.text != GOOD
    _, voters, _ = rc.consensus_for_document(raws, models, weights, ["substantive"] * 3,
                                             filter_source="peer")
    assert voters == ["a", "b"]                  # the peer test recomputes modes without the GT


# ── religious-140 benchmark adapter ───────────────────────────────────────────

def _religious_fixture(root: Path) -> Tuple[Path, Path]:
    """Write a toy religious-140 spec and ``raw_outputs`` tree.

    Two scored pages (one per column class) read by readers a, b and c, one
    page under 50 GT letters, and one page that reader c did not read.

    :param root: Folder to write into.
    :type root: Path
    :return: (spec path, outputs folder).
    :rtype: Tuple[Path, Path]
    """
    first, second = f"{GOOD}\n{UNRELATED}", f"{UNRELATED}\n{GOOD}"
    pages = [("ms1_p1", 1, first, {"a": first, "b": first, "c": f"{GOOD_VARIANT}\n{UNRELATED}"}),
             ("short", 1, "אבג דהו", {"a": "אבג", "b": "אבג", "c": "אבג"}),
             ("ms1_p2", 2, second, {"a": second, "b": second, "c": UNRELATED}),
             ("ms2_p1", 1, first, {"a": first, "b": first})]
    outputs = root / "raw_outputs"
    for doc_id, _, _, raws in pages:
        (outputs / doc_id).mkdir(parents=True)
        for model, text in raws.items():
            (outputs / doc_id / f"{model}.txt").write_text(text)
    spec = root / "genizah_religious_v1.json"
    spec.write_text(json.dumps({"docs": [{"doc_id": d, "n_columns": n, "gt": gt}
                                         for d, n, gt, _ in pages]}, ensure_ascii=False))
    return spec, outputs


def test_religious_loader_mirrors_the_runner(tmp_path: Path) -> None:
    """GT used as stored, short pages unscored, missing readers listed, column buckets."""
    spec, outputs = _religious_fixture(tmp_path)
    docs, skipped = rc.load_religious_documents(["a", "b", "c"], spec, outputs)
    assert [d.doc_id for d in docs] == ["ms1_p1", "ms1_p2"] and skipped == ["ms2_p1"]
    assert [d.bucket for d in docs] == ["single-col", "multi-col"]
    stored = json.loads(spec.read_text())["docs"][0]["gt"]
    assert docs[0].gt_ink == rc.genizah_visible_ink_gt(stored)
    assert docs[0].gt_letters == rc.letters_only(docs[0].gt_ink)
    assert docs[1].raws == {"a": f"{UNRELATED}\n{GOOD}", "b": f"{UNRELATED}\n{GOOD}",
                            "c": UNRELATED}
    docs, skipped = rc.load_religious_documents(["a", "b"], spec, outputs)
    assert [d.doc_id for d in docs] == ["ms1_p1", "ms1_p2", "ms2_p1"] and skipped == []


def test_religious_adapter_uses_the_runners_paths_and_metric_functions() -> None:
    """The adapter reads the runner's folders and scores with the runner's own functions."""
    runner = pytest.importorskip(
        "src.datasets.evaluations.helper_eval_scripts.run_religious_benchmark")
    source = rc.SOURCES[rc.RELIGIOUS]
    assert source.outputs_dir == runner._OUT
    assert source.spec == runner._BENCH / "genizah_religious_v1.json"
    for name in ("cer_pair", "genizah_visible_ink_gt", "normalize_ink_hypothesis", "letters_only",
                 "aligned_prf", "asserted_letters", "classify", "ngram_precision"):
        assert getattr(rc, name) is getattr(runner, name), name


def test_aggregate_breaks_out_the_requested_buckets() -> None:
    """PGP script buckets by default; the religious column classes when asked."""
    rows = [{"bucket": "multi-col", "cer": 0.2, "mode": "substantive"},
            {"bucket": "hebrew", "cer": 0.4, "mode": "loop_collapse"}]
    assert list(rc.aggregate(rows, "cer", "mode")["buckets"]) == ["hebrew"]
    religious = rc.aggregate(rows, "cer", "mode", rc.RELIGIOUS_BUCKETS)
    assert list(religious["buckets"]) == ["multi-col"]
    assert religious["buckets"]["multi-col"]["mean"] == pytest.approx(0.2)
    assert religious["n"] == 2 and religious["modes"]["loop_collapse"] == 1


def test_run_on_the_religious_benchmark_end_to_end(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """``--benchmark religious`` votes per page and reports the column buckets."""
    spec, outputs = _religious_fixture(tmp_path)
    monkeypatch.setitem(rc.SOURCES, rc.RELIGIOUS,
                        rc.BenchmarkSource(outputs, rc.RELIGIOUS_BUCKETS, spec))
    args = rc.parse_args(["--models", "a", "b", "c", "--benchmark", "religious",
                          "--out-dir", str(tmp_path / "out")])
    summary = rc.run(args)
    assert summary["tag"] == "religious_word_unweighted_3m"
    assert summary["config"]["benchmark"] == "religious"
    assert summary["n_docs"] == 2 and summary["skipped_missing_output"] == ["ms2_p1"]
    consensus = summary["systems"]["ROVER consensus"]
    assert list(consensus["buckets"]) == ["single-col", "multi-col"]
    assert consensus["mean"] == 0.0 < summary["systems"]["c"]["mean"]   # a + b outvote c
    assert "multi-col" in rc.summary_table(summary)
    assert (tmp_path / "out" / summary["tag"] / "consensus_ms1_p2.txt").read_text() == \
        f"{UNRELATED}\n{GOOD}\n"
