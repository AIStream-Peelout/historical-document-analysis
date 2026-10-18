# File name: test_slot_oracle.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the per-slot oracle over the ROVER network (toy pages, no benchmark data)."""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from src.datasets.evaluations.helper_eval_scripts import rover_consensus as rc
from src.datasets.evaluations.helper_eval_scripts import slot_oracle as so

GOOD = "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהו"
GOOD_VARIANT = "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהי"
UNRELATED = "שמע ישראל יהוה אלהינו יהוה אחד ואהבת את יהוה אלהיך בכל לבבך"
MISSING = "זחטכנסעפצקדזחטכנסעפצ"     # letters no toy reader writes: the alignment of it is unambiguous


@pytest.mark.parametrize("unit", rc.UNITS)
@pytest.mark.parametrize("hyps", [
    ["אבג דהו זחט", "אבג דהו זחט", "אבג דהי זחט"],
    ["אחד שתים שלש", "אחד שלש", "אחד שתים שלש"],
    ["שלום עולם\nטוב", "שלום עולם\nטוב", "שלוס עולם\nטוב"],
    [GOOD, GOOD_VARIANT, UNRELATED, f"{GOOD}\n{UNRELATED}"],
])
def test_rover_text_matches_the_tool_with_the_reference_column_added(hyps, unit):
    """The GT column never votes: voting the merged network reproduces rover()."""
    result = so.analyse_page(hyps, [GOOD], unit)
    assert result.rover_text == rc.rover(hyps, unit=unit).text
    assert result.backbone == rc.rover(hyps, unit=unit).backbone


@pytest.mark.parametrize("unit", rc.UNITS)
def test_a_minority_reading_of_the_gt_is_learnable_and_the_oracle_takes_it(unit):
    result = so.analyse_page(["אבג דהי", "אבג דהי", "אבג דהו"], ["אבג דהו"], unit)
    assert result.rover_text == "אבג דהי" and result.oracle_text == "אבג דהו"
    stats = result.stats
    assert stats["learnable_slots"] == stats["learnable_substitution"] == 1
    assert stats["learnable_single"] == 1 and stats.get("unreachable_slots", 0) == 0
    assert stats["learnable_chars"] == (3 if unit == "word" else 1)
    assert stats["gt_chars"] == (6 if unit == "word" else 7)        # char level counts the space


def test_a_gt_word_no_reader_wrote_is_unreachable_and_the_closest_word_is_emitted():
    result = so.analyse_page(["אבג דהי"] * 3, ["אבג דהו"], "word")
    assert result.oracle_text == "אבג דהי"
    assert result.stats["unreachable_substitution"] == 1 and "learnable_slots" not in result.stats
    closest = so.analyse_page(["אבזו", "אבזו", "אבגד"], ["אבגה"], "word")
    assert closest.rover_text == "אבזו" and closest.oracle_text == "אבגד"   # 0.25 < 0.5


def test_a_slot_only_the_gt_fills_emits_nothing():
    result = so.analyse_page(["אחד שלש"] * 3, ["אחד שתים שלש"], "word")
    assert result.oracle_text == "אחד שלש"
    stats = result.stats
    assert stats["gt_only_slots"] == 1 and stats["gt_only_chars"] == 4
    assert stats["unreachable_deletion"] == 1 and stats["unreachable_chars"] == 4


def test_a_reader_that_skipped_the_slot_lets_the_oracle_drop_an_insertion():
    result = so.analyse_page(["אחד נוסף שתים", "אחד נוסף שתים", "אחד שתים"], ["אחד שתים"], "word")
    assert result.rover_text == "אחד נוסף שתים" and result.oracle_text == "אחד שתים"
    stats = result.stats
    assert stats["learnable_insertion"] == 1 and stats["learnable_chars"] == 0
    unreachable = so.analyse_page(["אחד נוסף שתים"] * 2, ["אחד שתים"], "word")
    assert unreachable.oracle_text == "אחד נוסף שתים"            # nobody skipped it
    assert unreachable.stats["unreachable_insertion"] == 1


def test_ties_go_to_the_backbone_then_to_the_better_supported_entry():
    assert so.oracle_entry(["אבגד", "אבגו"], "אבגה", "word") == "אבגד"
    assert so.oracle_entry(["אבגו", "אבגד"], "אבגה", "word") == "אבגו"
    assert so.oracle_entry(["שתקר", "אבגו", "אבגד", "אבגד"], "אבגה", "word") == "אבגד"
    assert so.oracle_entry(["ב", None, "ג"], "א", "char") == "ב"    # all at distance 1
    assert so.oracle_entry([None, None], "א", "char") is None       # GT-only slot
    lost = so.analyse_page(["אבגד", "אבזד"], ["אבזד"], "char")     # 1-1 tie -> backbone ג
    assert lost.rover_text == "אבגד" and lost.oracle_text == "אבזד"
    assert lost.stats["learnable_lost_tie"] == 1


def test_whitespace_follows_the_scorer():
    assert so.token_key(rc.NEWLINE, "char") == " " and so.token_key(rc.NEWLINE, "word") is None
    assert so.token_key("א", "word") == "א" and so.token_key(None, "char") is None
    char = so.analyse_page(["אב\nגד"] * 3, ["אב גד"], "char")       # flat GT vs line breaks
    assert char.oracle_text == "אב\nגד"
    assert char.stats["right_slots"] == char.stats["slots"] == 5
    word = so.analyse_page(["אחד שתים\nשלש"] * 2 + ["אחד שתים שלש"], ["אחד שתים שלש"], "word")
    assert word.stats["right_slots"] == word.stats["slots"]       # a newline is a separator
    assert word.stats["gt_chars"] == 3 + 4 + 3


def test_the_oracle_never_scores_worse_than_rover_on_toy_pages():
    gt = f"{GOOD}\n{UNRELATED}"
    hyps = [f"{GOOD_VARIANT}\n{UNRELATED}", f"{GOOD}\nשמע ישראל יהוה", gt.replace("ובהו", "ובהי")]
    gt_ink = rc.genizah_visible_ink_gt(gt)
    for unit in rc.UNITS:
        result = so.analyse_page(hyps, so.gt_ink_lines(gt), unit)
        rover = rc.score_output(result.rover_text, gt_ink, rc.letters_only(gt_ink)).cer
        best = rc.score_output(result.oracle_text, gt_ink, rc.letters_only(gt_ink)).cer
        assert best <= rover and best == 0.0                     # every GT token was read once


def test_near_misses_look_for_the_gt_entry_in_nearby_gt_null_slots():
    targets = ["א", None, "ב", None, None, "ג"]
    keys = [set(), {"א"}, {"ד"}, set(), set(), set()]
    assert so.near_misses(targets, keys, [0, 2, 5]) == [0]
    far = ["א", None, None, None]
    assert so.near_misses(far, [set(), set(), set(), {"א"}], [0], window=2) == []
    assert so.near_misses(far, [set(), set(), set(), {"א"}], [0], window=3) == [0]


def test_the_oracle_is_wrong_exactly_on_the_unreachable_slots():
    hyps = [f"{GOOD_VARIANT}\nשמע ישראל", f"{GOOD}\nשמע ישראל יהוה", "בראשית ברא אלהים"]
    for unit in rc.UNITS:
        stats = so.analyse_page(hyps, so.gt_ink_lines(f"{GOOD}\n{UNRELATED}"), unit).stats
        assert stats["oracle_wrong_slots"] == stats["unreachable_slots"] > 0
        assert sum(stats.get(f"{c}_slots", 0) for c in so.CLASSES) == stats["slots"]
        for cls in ("learnable", "unreachable"):
            assert sum(stats.get(f"{cls}_{e}", 0) for e in so.ERROR_TYPES) == stats.get(f"{cls}_slots", 0)
        assert stats.get("dissent_lone_right", 0) == stats.get("learnable_single", 0)


def test_dissent_tallies_count_losing_candidates_and_their_hits():
    stats = Counter()
    # keys of five readers: winner א (2), lone ב (reader 7) = GT, ג twice
    so.count_dissent(stats, ["א", "ב", "א", "ג", "ג"], "א", "ב", [5, 7, 6, 8, 9])
    assert stats["dissent_lone"] == stats["dissent_lone_right"] == 1
    assert stats["dissent_multi"] == 1 and stats["dissent_multi_right"] == 0
    assert stats["lone_7"] == stats["lone_right_7"] == 1
    models = [f"m{i}" for i in range(10)]
    by_model = so.split_lone_dissent(stats, models)
    assert by_model["m7"] == {"n": 1, "right": 1} and "lone_7" not in stats


def test_block_mask_separates_dense_stretches_from_isolated_misreadings():
    flags = [False] * 5 + [True] * 6 + [False] * 5 + [True] + [False] * 5
    mask = so.block_mask(flags, window=2, share=0.7)
    assert mask[5:11] == [False, True, True, True, True, False]   # edges of the run dip below 0.7
    assert not mask[16] and sum(mask) == 4
    assert so.block_mask([True, True], window=10) == [True, True]   # windows truncate at the edges


def test_block_spans_hold_exactly_the_block_mask_flags():
    flags = [False] * 5 + [True] * 6 + [False] + [True] * 5 + [False] * 5 + [True] + [False] * 5
    mask = so.block_mask(flags, window=2, share=0.7)
    cores = so.block_spans(flags, window=2, share=0.7)
    assert cores == [(6, 16)]                                  # keeps the unflagged 11 inside
    assert [i for s, e in cores for i in range(s, e) if flags[i]] == [i for i, m in enumerate(mask) if m]
    assert so.block_spans(flags, window=2, share=0.7, extend=True) == [(5, 17)]   # whole run
    assert so.block_spans(flags, window=2, share=0.7, min_len=13, extend=True) == []
    two = [True] * 8 + [False, False] + [True] * 8
    assert so.block_spans(two, window=2, share=0.7, extend=True) == [(0, 8), (10, 18)]
    rng = random.Random(7)
    for _ in range(200):
        flags = [rng.random() < 0.6 for _ in range(60)]
        mask = so.block_mask(flags, window=3)
        spans = so.block_spans(flags, window=3, extend=True)
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))            # sorted, disjoint
        assert all(flags[s] and flags[e - 1] for s, e in spans)
        assert all(any(s <= i < e for s, e in spans) for i, m in enumerate(mask) if m)


def test_gt_entries_reproduce_the_slot_statistics():
    hyps = [f"{GOOD_VARIANT}\nשמע ישראל", f"{GOOD}\nשמע ישראל יהוה", "בראשית ברא אלהים"]
    chars = so.page_chars(hyps, so.gt_ink_lines(f"{GOOD}\n{UNRELATED}"))
    stats = so.slot_statistics(chars.network, chars.page.backbone, "char")[2]
    counts = Counter(e.cls for e in chars.entries)
    assert len(chars.entries) == len(chars.tokens) == stats["gt_chars"]
    assert all(counts.get(c, 0) == stats.get(f"{c}_chars", 0) for c in so.CLASSES)
    flags = [e.cls == "unreachable" for e in chars.entries]
    assert sum(so.block_mask(flags, so.BLOCK_WINDOW["char"])) == stats["unreachable_block_chars"] > 0


def test_reader_texts_take_the_slots_between_the_neighbouring_gt_entries():
    # columns: readers 0 and 1, then the reference; slot 2 is a reader-only insertion
    net = rc.Network((("א", "א", "א"), ("ב", None, "ג"), (None, "ד", None), ("ה", "ה", "ו")), (0, 1, 2))
    entries = so.gt_entries(net, 0, "char")
    assert [(e.slot, e.cls) for e in entries] == [(0, "right"), (1, "unreachable"), (3, "unreachable")]
    assert so.reader_texts(net, entries, 1, 3) == ["בה", "דה"]
    assert so.reader_texts(net, entries, 1, 2) == ["ב", "ד"]


def test_positions_lines_and_raw_context():
    assert so.token_lines(["אב", "גד"]) == [0, 0, 0, 1, 1] and so.token_lines([]) == []
    assert so.aligned_positions("אב גד", "אב [...] גד") == [0, 1, 2, 9, 10]
    assert so.aligned_positions("אבז", "אב") == [0, 1, None]
    assert so.raw_positions("אבז", "אב") == [0, 1, 1] and so.raw_positions("זאב", "אב") == [0, 0, 1]
    assert so.raw_context("0123456789", 3, 5, context=2) == "12⟦345⟧67"


def test_page_blocks_report_a_line_no_reader_wrote():
    gt = f"{GOOD} [ש]ם\n{MISSING}"
    lines = so.gt_ink_lines(gt)
    chars = so.page_chars([f"{GOOD} ם", f"{GOOD} ם", f"{GOOD_VARIANT} ם"], lines)
    blocks = so.page_blocks(chars, ["a", "b", "c"], lines, so.alignment_raw(gt))
    assert len(blocks) == 1
    block = blocks[0]
    assert block["gt_text"] == f"\n{MISSING}" and block["length"] == len(MISSING) + 1
    assert block["unreachable"] == block["gt_only"] == block["length"]
    assert block["readers"] == {"a": "", "b": "", "c": ""}
    assert (block["line_start"], block["line_end"], block["n_lines"]) == (0, 1, 2)
    assert "[ש]ם⟦\n" + MISSING[:5] in block["raw_context"] and block["raw_context"].endswith("⟧")
    assert so.page_blocks(chars, ["a", "b", "c"], lines, so.alignment_raw(gt), min_len=500) == []


def test_gt_lines_rebuild_the_scoring_reference():
    raw = "שורה [א]חת כאן\nשנייה ... כאן\n\n[...]\nשלישית @ סוף"
    lines = so.gt_ink_lines(raw)
    assert len(lines) == 3 and " ".join(lines) == rc.genizah_visible_ink_gt(raw)
    assert so.reference_lines(raw, rc.genizah_visible_ink_gt(raw)) == (lines, False)
    assert so.reference_lines(raw, "אחר לגמרי") == (["אחר לגמרי"], True)


def test_decomposition_shares_and_headroom():
    stats = {"gt_chars": 10, "right_chars": 7, "learnable_chars": 2, "unreachable_chars": 1,
             "learnable_slots": 3, "unreachable_slots": 1, "learnable_lost_tie": 1,
             "learnable_single": 3, "near_miss_chars": 1}
    shares = so.decomposition_shares(stats)
    assert shares["wrong_slots"] == 4 and shares["learnable_share"] == pytest.approx(0.75)
    assert shares["right_char_share"] == pytest.approx(0.7)
    assert shares["lost_tie_share"] == pytest.approx(1 / 3) and shares["near_miss_share"] == 1.0
    summary = {"systems": {"rover_char": {"mean": 0.2, "median": 0.15},
                           "slot_oracle_char": {"mean": 0.12, "median": 0.1}}}
    assert so.headroom(summary, "char") == pytest.approx({"mean": 0.08, "median": 0.05})


def _toy_religious(root: Path) -> None:
    """Write a two-page toy religious spec read by readers a, b and c.

    :param root: Folder to write into.
    :type root: Path
    """
    first, second = f"{GOOD}\n{UNRELATED}", f"{UNRELATED}\n{GOOD}"
    pages = [("p1", 1, first, {"a": first, "b": f"{GOOD_VARIANT}\n{UNRELATED}",
                               "c": f"{GOOD_VARIANT}\n{UNRELATED}"}),
             ("p2", 2, second, {"a": second, "b": UNRELATED, "c": second})]
    for doc_id, _, _, raws in pages:
        (root / "raw_outputs" / doc_id).mkdir(parents=True)
        for model, text in raws.items():
            (root / "raw_outputs" / doc_id / f"{model}.txt").write_text(text)
    (root / "spec.json").write_text(json.dumps(
        {"docs": [{"doc_id": d, "n_columns": n, "gt": gt} for d, n, gt, _ in pages]},
        ensure_ascii=False))


def test_run_configuration_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Religious adapter, both units, scores, decomposition and written outputs."""
    _toy_religious(tmp_path)
    monkeypatch.setitem(rc.SOURCES, rc.RELIGIOUS, rc.BenchmarkSource(
        tmp_path / "raw_outputs", rc.RELIGIOUS_BUCKETS, tmp_path / "spec.json"))
    summary, rows = so.run_configuration(rc.RELIGIOUS, ["a", "b", "c"])
    assert summary["n_docs"] == 2 and summary["gt_line_fallbacks"] == 0
    systems = summary["systems"]
    assert systems["rover_word"]["mean"] > 0.0           # p1: b + c outvote a's ובהו
    assert systems["slot_oracle_word"]["mean"] == systems["slot_oracle_char"]["mean"] == 0.0
    assert systems["page_oracle"]["mean"] == 0.0 and summary["best_single"]["mean"][0] == "a"
    assert summary["decomposition"]["word"]["learnable_slots"] == 1
    assert list(systems["rover_char"]["buckets"]) == ["single-col", "multi-col"]
    assert rows[0]["rover_word_cer"] > rows[0]["slot_oracle_word_cer"] == 0.0
    so.main(["--benchmark", rc.RELIGIOUS, "--models", "a", "b", "c", "--tag", "toy",
             "--out-dir", str(tmp_path / "out")])
    written = json.loads((tmp_path / "out" / "toy.json").read_text())
    assert written["tag"] == "toy" and written["n_docs"] == 2
    assert (tmp_path / "out" / "toy_per_doc.csv").read_text().startswith("doc_id,bucket,")


def test_dump_blocks_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--dump-blocks writes one JSON line per block with the page fields."""
    gt = f"[אב]ג {GOOD}\n{MISSING}"
    (tmp_path / "raw_outputs" / "p1").mkdir(parents=True)
    for model in ("a", "b", "c"):
        (tmp_path / "raw_outputs" / "p1" / f"{model}.txt").write_text(f"ג {GOOD}")
    (tmp_path / "spec.json").write_text(json.dumps(
        {"docs": [{"doc_id": "p1", "n_columns": 1, "gt": gt, "shelf_mark": "T-S X", "sys_num": "9",
                   "fl": "FL1"}]}, ensure_ascii=False))
    monkeypatch.setitem(rc.SOURCES, rc.RELIGIOUS, rc.BenchmarkSource(
        tmp_path / "raw_outputs", rc.RELIGIOUS_BUCKETS, tmp_path / "spec.json"))
    out = tmp_path / "blocks.jsonl"
    so.main(["--benchmark", rc.RELIGIOUS, "--models", "a", "b", "c", "--dump-blocks", str(out)])
    blocks = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(blocks) == 1 and blocks[0]["block_id"] == f"p1:{blocks[0]['start']}"
    assert blocks[0]["doc_id"] == "p1" and blocks[0]["bucket"] == "single-col"
    assert blocks[0]["shelf_mark"] == "T-S X" and blocks[0]["fl"] == "FL1"
    visible = len(re.findall(r"[א-ת]", f"ג {GOOD}{MISSING}"))
    assert blocks[0]["recon_frac"] == pytest.approx(2 / (2 + visible), abs=1e-4)   # [אב] restored
    assert blocks[0]["gt_text"] == f"\n{MISSING}" and set(blocks[0]["readers"]) == {"a", "b", "c"}
    with pytest.raises(SystemExit):
        so.parse_args(["--models", "a"])                    # --tag or --dump-blocks is required
