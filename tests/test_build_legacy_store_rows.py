# File name: test_build_legacy_store_rows.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the legacy-store row builder: skeletons, nulls, duplicates, ownership, assignment, coverage, leakage, build.

Everything is synthetic: random Hebrew-letter texts stand in for transcriptions, noisy copies of them for the Kraken
reads, and the end-to-end test writes small JPEGs and input files into ``tmp_path``. No NAS, no registries, no network.
"""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from datasets import load_from_disk
from PIL import Image

from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
from src.datasets.shelfmarks import ShelfmarkIndex
from src.finetuning.qwen_hebrew import build_legacy_store_rows as bl

LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
SIDE_TEXT = LETTERS * 3            # 66 letters, 63 4-grams: enough matched grams for every recall used below


def hebrew_text(rng: random.Random, words: int = 40) -> str:
    """Random Hebrew-letter words.

    :param rng: Random generator.
    :param words: Word count.
    :return: Text.
    """
    return " ".join("".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 6))) for _ in range(words))


def noisy(text: str, rng: random.Random, rate: float = 0.12) -> str:
    """A read of a text with some letters misread.

    :param text: Text.
    :param rng: Random generator.
    :param rate: Share of letters replaced.
    :return: Noisy copy.
    """
    return "".join(rng.choice(LETTERS) if ch in LETTERS and rng.random() < rate else ch for ch in text)


def image(cid: str, index: int, token: str = "") -> bl.ImageInfo:
    """A synthetic image.

    :param cid: Document.
    :param index: Image index.
    :param token: File-name token.
    :return: Image.
    """
    return bl.ImageInfo(cid, index, f"{cid}_{index}.jpg", f"/tmp/{cid}_{index}.jpg", f"u/{cid}/{index}",
                        f"{cid}-{index}", 100, 120, token)


def record(cid: str, text: str, **extra: Any) -> Dict[str, Any]:
    """A cleaned side record.

    :param cid: Document.
    :param text: Cleaned text.
    :param extra: Field overrides.
    :return: Record.
    """
    letters = sum(ch in LETTERS for ch in text)
    rec = {"canonical_id": cid, "doc_id": cid, "shelf_mark": cid.replace("_", " "), "main_language": "hebrew",
           "section_index": 0, "side": None, "column": None, "text": text, "letters": letters, "hebrew_letters": letters,
           "arabic_letters": 0, "gaps": 0, "gap_density": 0.0, "flags": {}, "images": [], "family": "documentary_plain",
           "side_decision": "undecidable", "order_prior_image": None, "shared_text_docs": [], "section_digest": cid}
    rec.update(extra)
    return rec


def pair(n4: float, p95: float = 0.1, p99: float = 0.2, n3_ok: bool = True) -> Dict[str, Any]:
    """A usable pair score with the fields the decisions read.

    :param n4: 4-gram recall.
    :param p95: Null p95.
    :param p99: Null p99.
    :param n3_ok: 3-gram recall clears its p95.
    :return: Pair score.
    """
    return {"usable": True, "n4": n4, "n3": n4, "clears_p95": n4 > p95, "clears_p99": n4 > p99,
            "n3_clears_p95": n3_ok, "read_letters": 150}


def fake_gate(clean: Sequence[str] = (), bench: Sequence[str] = ()) -> DecontamGate:
    """A decontamination gate with synthetic inventories (no files read).

    :param clean: Shelf marks whose loose keys count as clean v2.
    :param bench: Shelf marks whose loose keys count as benchmark.
    :return: Gate.
    """
    gate = DecontamGate.__new__(DecontamGate)
    gate.ktiv_train_sys = set()
    gate.clean_keys = {DecontamGate.loose_key(s) for s in clean}
    gate.bench_keys = {DecontamGate.loose_key(s) for s in bench}
    return gate


def fake_index(records: Iterable[Dict[str, Any]] = ()) -> ShelfmarkIndex:
    """A shelf-mark index over synthetic merged-corpus records (no file read).

    :param records: ``{canonical_id, shelfmark_display, pgpids}`` records.
    :return: Index.
    """
    return ShelfmarkIndex(list(records))


def fixture_index(cids: Iterable[str]) -> ShelfmarkIndex:
    """An index whose ES ids are the given documents, each displayed as its id with spaces.

    :param cids: Canonical ids.
    :return: Index.
    """
    return fake_index({"canonical_id": c, "shelfmark_display": c.replace("_", " ")} for c in cids)


# ----------------------------------------------------------------------------------------------------------- text

def test_hebrew_skeleton_folds_finals_and_drops_points() -> None:
    """Points and spaces go, finals become base letters, Latin and digits are dropped."""
    assert bl.hebrew_skeleton("שָׁלוֹם עֲלֵיכֶם 12 ab") == "שלומעליכמ"
    assert bl.hebrew_skeleton("ךםןףץ") == "כמנפצ"


def test_length_bins_edges() -> None:
    """Bins: under 100, 100-299, 300-600, over 600."""
    assert [bl.length_bin(n) for n in (0, 99, 100, 299, 300, 600, 601)] == [
        "under_100", "under_100", "100_300", "100_300", "300_600", "300_600", "over_600"]


def test_clipped_recall_counts_a_gram_at_most_as_often_as_the_read() -> None:
    """A gram repeated in the side counts only as often as the read holds it."""
    side, read = Counter({"אבגד": 3, "הוזח": 1}), Counter({"אבגד": 1})
    assert bl.clipped_recall(side, read) == pytest.approx(0.25)
    assert bl.clipped_recall(Counter(), read) == 0.0


def test_make_read_usability() -> None:
    """Zero lines, NO TEXT, loops and short reads are unusable; script is the majority."""
    rng = random.Random(1)
    assert bl.make_read("a", "kraken", "", 0).unusable == "kraken_zero_lines"
    assert bl.make_read("b", "gemini", "NO TEXT", None).unusable == "no_text"
    assert bl.make_read("c", "kraken", "אבגדהוזחטיכלמ " * 20, 20).unusable == "loop"
    assert bl.make_read("d", "kraken", "אבג דה", 1).unusable == f"under_{bl.MIN_READ_LETTERS}_letters"
    good = bl.make_read("e", "kraken", hebrew_text(rng), 5)
    assert not good.unusable and good.script == bl.HEBREW and good.usable_for(bl.HEBREW)
    arabic = bl.make_read("f", "gemini", "بسم الله الرحمن الرحيم وصلى الله على سيدنا محمد", None)
    assert arabic.script == bl.ARABIC and not arabic.usable_for(bl.HEBREW)


def test_label_agrees_with_tokens() -> None:
    """A bare side matches any leaf; a leaf-and-side label only its own token."""
    assert bl.label_agrees("r", "1r") is True
    assert bl.label_agrees("1r", "2r") is False
    assert bl.label_agrees("v", "1r") is False
    assert bl.label_agrees("", "1r") is None and bl.label_agrees("r", "") is None


# ----------------------------------------------------------------------------------------------------------- null

def test_threshold_stats_and_bin_fallback() -> None:
    """Percentiles with a bootstrap CI; a bin under MIN_BIN_NULL falls back to the unstratified null."""
    rng = np.random.default_rng(0)
    stats = bl.threshold_stats(np.linspace(0, 1, 101), rng, 50)
    assert stats["p95"] == pytest.approx(0.95) and stats["p99"] == pytest.approx(0.99)
    assert stats["p99_ci"][0] <= stats["p99"] <= stats["p99_ci"][1] + 1e-9
    assert bl.threshold_stats(np.zeros(0), rng, 10) is None
    null = {"n4": {bl.ALL_BINS: {"n": 50, "p95": 0.1, "p99": 0.2}, "under_100": {"n": 5, "p95": 0.5, "p99": 0.6},
                   "100_300": {"n": 45, "p95": 0.05, "p99": 0.07}}}
    assert bl.pick_threshold(null, 4, "under_100") == (null["n4"][bl.ALL_BINS], bl.ALL_BINS)
    assert bl.pick_threshold(null, 4, "100_300") == (null["n4"]["100_300"], "100_300")


def test_side_null_leaves_out_own_and_excluded_documents() -> None:
    """The null never holds a read of the side's own document, of an excluded document or of its own image."""
    rng = random.Random(2)
    text = hebrew_text(rng)
    reads = {f"s{i}": bl.make_read(f"s{i}", "kraken", hebrew_text(rng), 5) for i in range(30)}
    reads["own"] = bl.make_read("own", "kraken", text, 5)
    reads["partner"] = bl.make_read("partner", "kraken", noisy(text, rng), 5)
    owners = {f"s{i}": {f"D{i}"} for i in range(30)} | {"own": {"A"}, "partner": {"B"}}
    pool = bl.build_null_pool(bl.HEBREW, reads, owners)
    assert len(pool) == 32
    side = bl.make_side(record("A", text), 0, [])
    null = bl.side_null(side, pool, {"A", "B"}, {"own"}, np.random.default_rng(0), 20)
    assert null["reads"] == 30
    assert null["n4"][bl.ALL_BINS]["max"] < 0.5                  # the copies of its own text are not in the null


# ----------------------------------------------------------------------------------------------- duplicates, ownership

def test_duplicate_versions_keep_the_version_with_more_letters() -> None:
    """A duplicate version and its earlier original: the longer one stays."""
    rng = random.Random(3)
    text = hebrew_text(rng, 40)
    short = " ".join(text.split()[:30])
    sides = [bl.make_side(record("D", short), 0, []),
             bl.make_side(record("D", text, side_decision="duplicate_version", section_index=1), 1, []),
             bl.make_side(record("D", hebrew_text(rng, 30), section_index=2), 2, [])]
    assert bl.duplicate_versions(sides) == {"D#0": "D#1"}


def test_ownership_owner_tie_and_no_owner() -> None:
    """The best p99-clearing recall owns the text; equal bests tie; nothing clearing: no owner."""
    a, b = bl.make_side(record("X_2", "אבגד"), 0, []), bl.make_side(record("X_28", "אבגד"), 0, [])
    owned = bl.decide_ownership([a, b], {a.key: {0: pair(0.6)}, b.key: {0: pair(0.3), 1: pair(0.05)}})
    assert owned[a.key]["role"] == "owner" and owned[b.key]["role"] == "not_owner"
    tied = bl.decide_ownership([a, b], {a.key: {0: pair(0.6)}, b.key: {0: pair(0.6)}})
    assert {v["role"] for v in tied.values()} == {"tie"}
    none = bl.decide_ownership([a, b], {a.key: {0: pair(0.15)}, b.key: {0: pair(0.01)}})
    assert {v["role"] for v in none.values()} == {"no_owner"}


def test_ownership_contested_when_the_lead_is_under_twice_the_runner_up() -> None:
    """Two copies clearing p99 with a lead under 2x: every copy of the group is contested; a sole copy clearing p99
    owns the text whatever the others score below p99."""
    a, b = bl.make_side(record("X_2", "אבגד"), 0, []), bl.make_side(record("X_28", "אבגד"), 0, [])
    c = bl.make_side(record("X_280", "אבגד"), 0, [])
    contested = bl.decide_ownership([a, b, c], {a.key: {0: pair(0.5)}, b.key: {0: pair(0.3)}, c.key: {0: pair(0.01)}})
    assert {v["role"] for v in contested.values()} == {"contested"}
    assert all(v["owner"] is None for v in contested.values())
    sole = bl.decide_ownership([a, b], {a.key: {0: pair(0.25)}, b.key: {0: pair(0.19)}})
    assert sole[a.key]["role"] == "owner" and sole[b.key]["role"] == "not_owner"
    double = bl.decide_ownership([a, b], {a.key: {0: pair(0.7)}, b.key: {0: pair(0.35)}})
    assert double[a.key]["role"] == "owner" and double[b.key]["role"] == "not_owner"


# ---------------------------------------------------------------------------------------------------- assignment

def test_decide_document_accepts_one_to_one() -> None:
    """Two sides, two images, each clearing p99 only on its own image: both accepted on p99."""
    s1, s2 = bl.make_side(record("A", SIDE_TEXT), 0, []), bl.make_side(record("A", SIDE_TEXT[::-1]), 1, [])
    im = [image("A", 0, "1r"), image("A", 1, "1v")]
    pairs = {s1.key: {0: pair(0.5), 1: pair(0.02)}, s2.key: {0: pair(0.01), 1: pair(0.4)}}
    out = bl.decide_document([s1, s2], im, pairs, {})
    assert out[s1.key]["outcome"] == "accepted" and out[s1.key]["image_index"] == 0 and out[s1.key]["basis"] == "p99"
    assert out[s2.key]["outcome"] == "accepted" and out[s2.key]["image_index"] == 1


def test_decide_document_conflict_and_spill_go_to_review() -> None:
    """Another side clearing p95 on the image, or the side clearing p95 on a second image, sends it to review."""
    s1, s2 = bl.make_side(record("B", SIDE_TEXT), 0, []), bl.make_side(record("B", SIDE_TEXT[::-1]), 1, [])
    im = [image("B", 0), image("B", 1)]
    conflict = bl.decide_document([s1, s2], im, {s1.key: {0: pair(0.5), 1: pair(0.0)},
                                                 s2.key: {0: pair(0.4), 1: pair(0.0)}}, {})
    assert {d["outcome"] for d in conflict.values()} == {"needs_review"}
    assert any("other_side_clears_p95_on_image" in d["reasons"] for d in conflict.values())
    assert any("clears_p95_on_image_assigned_to_another_side" in d["reasons"] for d in conflict.values())
    spill = bl.decide_document([s1], im, {s1.key: {0: pair(0.5), 1: pair(0.15)}}, {})
    assert spill[s1.key]["outcome"] == "needs_review" and "side_clears_p95_on_another_image" in spill[s1.key]["reasons"]
    weak3 = bl.decide_document([s1], im, {s1.key: {0: pair(0.5, n3_ok=False), 1: pair(0.0)}}, {})
    assert weak3[s1.key]["reasons"] == ["n3_below_p95"]


def test_decide_document_p95_needs_an_agreeing_label() -> None:
    """Between p95 and p99 a side is accepted only when its label names the image; the order prior relaxes nothing."""
    labelled = bl.make_side(record("C", SIDE_TEXT, side="1v"), 0, [])
    bare = bl.make_side(record("C2", SIDE_TEXT), 0, [])
    im = [image("C", 0, "1r"), image("C", 1, "1v")]
    out = bl.decide_document([labelled], im, {labelled.key: {0: pair(0.0), 1: pair(0.15)}}, {})
    assert out[labelled.key]["outcome"] == "accepted" and out[labelled.key]["basis"] == "p95+side_label"
    out = bl.decide_document([bare], [image("C2", 0), image("C2", 1)], {bare.key: {0: pair(0.0), 1: pair(0.15)}}, {})
    assert out[bare.key]["outcome"] == "needs_review"
    assert out[bare.key]["reasons"] == ["clears_p95_not_p99_without_agreeing_prior"]
    missing = bl.decide_document([bare], [image("C2", 0)], {bare.key: {0: pair(0.01)}}, {})
    assert missing[bare.key] == {**missing[bare.key], "outcome": "unassigned", "reasons": ["not_found"]}
    ordered_images = [image("C3", 0), image("C3", 1)]
    ordered = bl.make_side(record("C3", SIDE_TEXT, order_prior_image=1, images=[{"url": "u/C3/0"}, {"url": "u/C3/1"}]),
                           0, ordered_images)
    assert ordered.order_prior_index == 1
    out = bl.decide_document([ordered], ordered_images, {ordered.key: {0: pair(0.0), 1: pair(0.15)}}, {})
    assert out[ordered.key]["outcome"] == "needs_review"
    assert out[ordered.key]["reasons"] == ["clears_p95_not_p99_without_agreeing_prior"]


def test_decide_document_under_5_matched_grams_goes_to_review() -> None:
    """A pair clearing its threshold on fewer than 5 matched 4-grams goes to review; 5 suffice."""
    text = LETTERS + LETTERS[0]                 # 23 letters: 20 4-grams
    side = bl.make_side(record("G", text), 0, [])
    im = [image("G", 0, "1r")]
    four = bl.decide_document([side], im, {side.key: {0: pair(0.2, p95=0.05, p99=0.1)}}, {})
    assert four[side.key]["outcome"] == "needs_review" and four[side.key]["matched_n4"] == 4
    assert four[side.key]["reasons"] == [bl.UNDER_MATCHED_REASON] == ["under_5_matched_4grams"]
    assert four[side.key]["basis"] == "p99" and four[side.key]["image_index"] == 0
    five = bl.decide_document([side], im, {side.key: {0: pair(0.25, p95=0.05, p99=0.1)}}, {})
    assert five[side.key]["outcome"] == "accepted" and five[side.key]["matched_n4"] == 5
    labelled = bl.make_side(record("G", text, side="1r"), 0, [])
    thin = bl.decide_document([labelled], im, {labelled.key: {0: pair(0.15, p95=0.1, p99=0.2)}}, {})
    assert thin[labelled.key]["basis"] == "p95+side_label" and thin[labelled.key]["outcome"] == "needs_review"


def test_shared_image_across_documents_goes_to_review() -> None:
    """One image accepted for sides of two documents: both to review."""
    a, b = bl.make_side(record("E_6", "אבגד"), 0, []), bl.make_side(record("E_7", "הוזח"), 0, [])
    shared = bl.ImageInfo("E_6", 0, "f.jpg", "/tmp/f.jpg", "u", "same", 10, 10, "1r")
    images = {"E_6": [shared], "E_7": [bl.ImageInfo("E_7", 0, "f.jpg", "/tmp/f.jpg", "u", "same", 10, 10, "1r")]}
    decisions = {a.key: {"outcome": "accepted", "reasons": [], "image_index": 0},
                 b.key: {"outcome": "accepted", "reasons": [], "image_index": 0}}
    moved = bl.shared_image_conflicts(decisions, {a.key: a, b.key: b}, images)
    assert sorted(moved) == sorted([a.key, b.key])
    assert decisions[a.key]["reasons"] == ["image_accepted_for_another_document"]


# ------------------------------------------------------------------------------------------------------ coverage

def test_coverage_flags_and_exclusions() -> None:
    """Ratio, script and unread-image flags are kept; apparatus, editorial terms, gaps and short texts exclude."""
    rng = random.Random(4)
    text = hebrew_text(rng, 30)
    read = bl.make_read("r", "kraken", text + " " + hebrew_text(rng, 60), 9)
    side = bl.make_side(record("F", text, family="edition_apparatus", flags={"editorial_terms": 2},
                               gap_density=45.0), 0, [])
    p = {"read_letters": len(read.skeletons[bl.HEBREW])}
    flags, exclusions, info = bl.coverage(side, read, p, True, False)
    assert "letter_ratio_out_of_range" in flags and "other_image_without_usable_read" in flags
    assert info["letter_ratio"] > 2.0
    assert exclusions == ["edition_apparatus", "editorial_terms", "gap_density_over_30pct", "read_exceeds_target_2x"]
    short = bl.make_side(record("F", "אבגדה"), 0, [])
    _, exclusions, _ = bl.coverage(short, read, {"read_letters": 5}, False, False)
    assert exclusions == [f"under_{bl.MIN_TARGET_LETTERS}_letters"]
    mixed = bl.make_read("m", "gemini", text + " بسم الله الرحمن الرحيم وصلى الله على سيدنا محمد", None)
    flags, _, _ = bl.coverage(bl.make_side(record("F", text), 0, []), mixed,
                              {"read_letters": len(mixed.skeletons[bl.HEBREW])}, False, False)
    assert "read_other_script_over_10pct" in flags


def test_coverage_excludes_a_read_over_twice_the_target() -> None:
    """Read letters over 2x the target's exclude the pair; up to 2x (and under 0.5) they are only flagged."""
    rng = random.Random(6)
    text = hebrew_text(rng, 30)
    side = bl.make_side(record("H", text), 0, [])
    read = bl.make_read("h", "kraken", text, 9)
    target = len(side.skeleton)
    for read_letters, flagged, excluded in ((int(target * 2.5), True, True), (target * 2, True, False),
                                            (int(target * 1.8), True, False), (target, False, False),
                                            (int(target * 0.3), True, False)):
        flags, exclusions, info = bl.coverage(side, read, {"read_letters": read_letters}, False, False)
        assert ("letter_ratio_out_of_range" in flags) is flagged, read_letters
        assert exclusions == (["read_exceeds_target_2x"] if excluded else []), read_letters
    assert bl.MAX_READ_OVER_TARGET == 2.0


def test_row_stem_names_the_image_token() -> None:
    """The stem's side is the image's file-name token, the record's label only when the image has none."""
    side = bl.make_side(record("Doc_S", SIDE_TEXT, side="1r"), 0, [])
    assert bl.row_stem(side, image("Doc_S", 1, "1v")) == ("1v", "legacy_Doc_S_1v_c1_page")
    assert bl.row_stem(side, image("Doc_S", 0, "")) == ("1r", "legacy_Doc_S_1r_c0_page")
    bare = bl.make_side(record("Doc_S", SIDE_TEXT), 1, [])
    assert bl.row_stem(bare, image("Doc_S", 2, "")) == ("none", "legacy_Doc_S_none_c2_page")


# ------------------------------------------------------------------------------------------------------- leakage

LEAK_INDEX_RECORDS = [
    {"canonical_id": "Cambridge_CUL_T_S_8_4", "shelfmark_display": "T-S 8.4"},
    {"canonical_id": "Cambridge_CUL_T_S_10J12_4", "shelfmark_display": "T-S 10J12.4", "pgpids": ["555"]},
    {"canonical_id": "Cambridge_CUL_T_S_99_1", "shelfmark_display": "T-S 99.1"},
    {"canonical_id": "Cambridge_CUL_T_S_2_2", "shelfmark_display": "T-S 2.2"},
    {"canonical_id": "Cambridge_CUL_T_S_1_1", "shelfmark_display": "T-S 1.1"},
    {"canonical_id": "New_York_JTS_ENA_2_26", "shelfmark_display": "ENA 2.26"},
    {"canonical_id": "New_York_JTS_ENA_22_6", "shelfmark_display": "ENA 22.6"},
]


def test_identity_leakage_rules() -> None:
    """Registered ES id, registered shelf mark (resolved to its ES id), clean v2 key, benchmark key and registered PGP id
    each hold a document out."""
    index = fake_index(LEAK_INDEX_RECORDS)
    held = bl.held_out_ids(["Cambridge_CUL_T_S_8_4", "Cambridge_CUL_T_S_10J12_4"], [], [], {}, index)
    gate = fake_gate(clean=["T-S 13J1.1"], bench=["T-S 16.100"])
    assert bl.identity_leakage("Cambridge_CUL_T_S_8_4", "T-S 8.4", [], held, index, gate)["holdout"]
    joined = bl.identity_leakage("X", "T-S 10J12.4 + T-S 99.1", [], held, index, gate)
    assert "registered benchmark id ['Cambridge_CUL_T_S_10J12_4']" in joined["holdout"]
    assert joined["reasons"][0].startswith("identity:canonical_id:")
    assert bl.identity_leakage("Y", "T-S 13J1.1", [], held, index, gate)["clean_v2_keys"]
    assert bl.identity_leakage("Z", "T-S 16.100", [], held, index, gate)["benchmark_keys"]
    pgp_held = bl.HeldOut(pgpids=frozenset({"123"}))
    assert bl.identity_leakage("W", "T-S 1.1", ["123"], pgp_held, index, gate)["reasons"]
    assert bl.identity_leakage("V", "T-S 2.2", [], held, index, gate)["reasons"] == []
    # a PGP id the index links to the document's ES id holds it out (the old pgp_index route)
    assert "PGP id ['555']" in bl.identity_leakage("U", "T-S 10J12.4", [], bl.HeldOut(pgpids=frozenset({"555"})),
                                                   index, gate)["holdout"]
    # no letter/digit squash: ENA 2.26 is not ENA 22.6
    ena = bl.held_out_ids(["New_York_JTS_ENA_22_6"], [], [], {}, index)
    assert bl.identity_leakage("New_York_JTS_ENA_2_26", "ENA 2.26", [], ena, index, gate)["reasons"] == []


def test_held_out_ids_are_es_ids() -> None:
    """Registry ids, registered PGP documents (by ``pgpid:<n>`` and by their export shelf mark) and the frozen
    benchmark all become ES canonical ids; the document side carries ES ids only."""
    index = fake_index(LEAK_INDEX_RECORDS)
    held = bl.held_out_ids(["Cambridge_CUL_T_S_8_4"], ["555", "777"], ["Cambridge_CUL_T_S_2_2"],
                           {"777": "T-S 99.1 + T-S 1.1"}, index)
    assert held.ids == {"Cambridge_CUL_T_S_8_4", "Cambridge_CUL_T_S_10J12_4", "Cambridge_CUL_T_S_99_1",
                        "Cambridge_CUL_T_S_1_1"}
    assert held.frozen_ids == {"Cambridge_CUL_T_S_2_2"} and held.pgpids == {"555", "777"}
    assert held.ids <= index.ids and held.frozen_ids <= index.ids
    gate = fake_gate()
    frozen = bl.identity_leakage("Doc_F", "T-S 2.2", [], held, index, gate)
    assert frozen["holdout"].startswith("frozen 131-fragment benchmark id")
    assert frozen["ids"] == ["Cambridge_CUL_T_S_2_2", "Doc_F"]
    assert set(frozen["ids"]) - {"Doc_F"} <= index.ids and not any("|" in i or ":" in i for i in frozen["ids"])


def test_text_leakage_hebrew_and_arabic() -> None:
    """Two shared 25-letter Hebrew shingles, or three shared rasm shingles with one GT document, hold a text out."""
    rng = random.Random(5)
    gt_text = hebrew_text(rng, 50)
    arabic_gt = "بسم الله الرحمن الرحيم كتابي اليك اطال الله بقاك وادام عزك ونعمته من الفسطاط"
    gt = bl.build_gt_index({"genizah_test_v1": [("doc1", gt_text)], "arabic_script_benchmark_v0": [("a1", arabic_gt)]})
    gate = fake_gate()
    hit = bl.text_leakage(hebrew_text(rng, 10) + " " + " ".join(gt_text.split()[:20]), gt, gate.content_contaminated)
    assert hit["hebrew"]["genizah_test_v1"] >= 2 and hit["reasons"]
    assert bl.text_leakage(hebrew_text(rng, 40), gt, gate.content_contaminated)["reasons"] == []
    arabic_hit = bl.text_leakage(arabic_gt, gt, gate.content_contaminated)
    assert "arabic_script_benchmark_v0" in arabic_hit["arabic"]


# ------------------------------------------------------------------------------------------------ whole pipeline

def world(rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Tuple[str, int]], Dict[str, str]]:
    """Synthetic documents covering every path.

    :param rng: Random generator.
    :return: ``(records, images (cid, index, token), reads {image key: (text, lines)}, named texts)``.
    """
    t = {name: hebrew_text(rng, 40) for name in ("A1", "A2", "B1", "B2", "S", "S2", "D", "E", "L", "M", "U", "R")}
    records = [
        record("Doc_A", t["A1"], side="1r", section_index=0), record("Doc_A", t["A2"], side="2r", section_index=1),
        record("Doc_B", t["B1"]), record("Doc_B", t["B2"], section_index=1),
        record("X_2", t["S"], shared_text_docs=["X_28"], section_digest="sh1"),
        record("X_28", t["S"], shared_text_docs=["X_2"], section_digest="sh1"),
        record("Y_1", t["S2"], shared_text_docs=["Y_10"], section_digest="sh2"),
        record("Y_10", t["S2"], shared_text_docs=["Y_1"], section_digest="sh2"),
        record("Doc_D", " ".join(t["D"].split()[:30])),
        record("Doc_D", t["D"], side_decision="duplicate_version", section_index=1),
        record("Doc_E", t["E"], family="edition_apparatus"),
        record("Cambridge_CUL_T_S_8_4", t["L"]), record("Doc_M", t["M"]), record("Doc_U", t["U"]),
        record("Doc_R", t["R"]), record("Doc_Z", hebrew_text(rng, 40))]
    reads: Dict[str, Tuple[str, int]] = {}
    images: List[Tuple[str, int, str]] = []

    def add(cid: str, index: int, text: str, lines: int = 8, token: str = "") -> None:
        """Register an image and its read.

        :param cid: Document.
        :param index: Image index.
        :param text: Read text.
        :param lines: Kraken line count.
        :param token: File-name token.
        """
        images.append((cid, index, token))
        reads[f"{cid}#{index}"] = (text, lines)

    add("Doc_A", 0, noisy(t["A1"], rng), token="1r")
    add("Doc_A", 1, noisy(t["A2"], rng), token="1v")
    add("Doc_B", 0, noisy(t["B1"] + " " + t["B2"], rng))
    add("Doc_B", 1, hebrew_text(rng))
    add("X_2", 0, noisy(t["S"], rng))
    add("X_28", 0, hebrew_text(rng))
    add("Y_1", 0, hebrew_text(rng))
    add("Y_10", 0, hebrew_text(rng))
    add("Doc_D", 0, noisy(t["D"], rng))
    add("Doc_D", 1, hebrew_text(rng))
    for name, cid in (("E", "Doc_E"), ("L", "Cambridge_CUL_T_S_8_4"), ("M", "Doc_M")):
        add(cid, 0, noisy(t[name], rng))
    add("Doc_U", 0, hebrew_text(rng))
    add("Doc_R", 0, noisy(t["R"], rng) + " " + hebrew_text(rng, 50))
    add("Doc_Z", 0, "", lines=0)
    for i in range(70):
        add(f"Noise_{i}", 0, hebrew_text(rng))
    t["S3"], t["R2"] = hebrew_text(rng, 40), hebrew_text(rng, 40)       # drawn last: the draws above stay as they were
    records += [record("W_3", t["S3"], shared_text_docs=["W_30"], section_digest="sh3"),
                record("W_30", t["S3"], shared_text_docs=["W_3"], section_digest="sh3"), record("Doc_R2", t["R2"])]
    add("W_3", 0, noisy(t["S3"], rng))
    add("W_30", 0, noisy(t["S3"], rng))
    add("Doc_R2", 0, noisy(t["R2"], rng) + " " + hebrew_text(rng, 25))
    return records, [{"cid": c, "index": i, "token": tok} for c, i, tok in images], reads, t


def write_inputs(src: Path, records: List[Dict[str, Any]], images: List[Dict[str, Any]],
                 reads: Dict[str, Tuple[str, int]]) -> None:
    """Write the builder's input files (with small distinct JPEGs).

    :param src: Input directory.
    :param records: Side records.
    :param images: Images.
    :param reads: Reads by ``cid#index``.
    """
    (src / "gemini_reads_arabic").mkdir(parents=True)
    (src / "gemini_reads_arabic" / "index.jsonl").write_text("", encoding="utf-8")
    manifest, kraken = [], []
    for n, im in enumerate(images):
        path = src / "images" / im["cid"] / f"{im['cid']}_{im['index']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        Image.new("RGB", (40 + n % 7, 50), (n % 255, (3 * n) % 255, (7 * n) % 255)).save(buf, "JPEG")
        path.write_bytes(buf.getvalue())
        sha = hashlib.sha256(buf.getvalue()).hexdigest()
        token = im["token"]
        manifest.append({"canonical_id": im["cid"], "image_index": im["index"], "file_name": path.name,
                         "path": str(path), "url": f"https://example.invalid/{im['cid']}/{im['index']}", "sha256": sha,
                         "width": 40 + n % 7, "height": 50, "side_token": token[-1:] if token else "none",
                         "folio": int(token[:-1]) if token else None})
        text, lines = reads[f"{im['cid']}#{im['index']}"]
        kraken.append({"sha256": sha, "canonical_id": im["cid"], "n_lines": lines,
                       "lines": [{"text": text}] if lines else []})
    for rec in records:
        rec["images"] = [{"url": m["url"], "token": ""} for m in manifest if m["canonical_id"] == rec["canonical_id"]]
    (src / "images_manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    (src / "kraken_reads.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kraken),
                                            encoding="utf-8")
    (src / "cleaned_sections.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                                encoding="utf-8")


def test_build_end_to_end(tmp_path: Path) -> None:
    """Accept, conflict, ownership, duplicate, coverage and leakage paths through ``build``, with export and report.

    :param tmp_path: pytest's temporary directory (inputs, outputs, report and queue).
    """
    rng = random.Random(7)
    records, images, reads, texts = world(rng)
    src = tmp_path / "legacy_store_diplomatic"
    write_inputs(src, records, images, reads)
    out = tmp_path / "legacy_store_diplomatic_v9"
    index = fixture_index({im["cid"] for im in images})
    leakage = bl.LeakageResources(held=bl.HeldOut(ids=frozenset({"Cambridge_CUL_T_S_8_4"})), index=index,
                                  gate=fake_gate(), gt_sets={"genizah_test_v1": [("gt_m", texts["M"])]}, pgpids={})
    report, queue_path = tmp_path / "logs" / "ROWS_REPORT.md", tmp_path / "logs" / "eye_queue_rows.jsonl"
    stats = bl.build(src, out, None, report, queue_path, True, 50, 1, leakage)

    decisions = {d["key"]: d for d in (json.loads(l) for l in (out / "decisions.jsonl").read_text().splitlines())}
    outcome = {k: d["outcome"] for k, d in decisions.items()}
    assert outcome["Doc_A#0"] == outcome["Doc_A#1"] == "accepted"
    assert decisions["Doc_A#0"]["basis"] == "p99" and decisions["Doc_A#0"]["image_index"] == 0
    assert outcome["Doc_B#0"] == outcome["Doc_B#1"] == "needs_review"
    assert outcome["X_2#0"] == "accepted" and outcome["X_28#0"] == "dropped"
    assert decisions["X_28#0"]["reasons"] == ["shared_text_not_owner"]
    assert outcome["Y_1#0"] == outcome["Y_10#0"] == "needs_review"
    assert decisions["Y_1#0"]["reasons"] == ["shared_text_no_owner"]
    assert outcome["W_3#0"] == outcome["W_30#0"] == "needs_review"
    assert decisions["W_3#0"]["reasons"] == decisions["W_30#0"]["reasons"] == ["ownership_contested"]
    assert {decisions[k]["shared_text"]["role"] for k in ("W_3#0", "W_30#0")} == {"contested"}
    assert outcome["Doc_D#0"] == "dropped" and outcome["Doc_D#1"] == "accepted"
    assert outcome["Doc_U#0"] == "unassigned" and decisions["Doc_U#0"]["reasons"] == ["not_found"]
    assert outcome["Doc_Z#0"] == "unassigned" and decisions["Doc_Z#0"]["reasons"] == ["no_usable_read"]
    assert decisions["Doc_E#0"]["row"] == "excluded:edition_apparatus"
    assert decisions["Cambridge_CUL_T_S_8_4#0"]["row"] == \
        "held_out:identity:canonical_id:registered benchmark id ['Cambridge_CUL_T_S_8_4']"
    assert decisions["Doc_M#0"]["row"].startswith("held_out:text:genizah_test_v1")
    assert decisions["Doc_R#0"]["row"] == "excluded:read_exceeds_target_2x"
    assert decisions["Doc_R#0"]["coverage"]["letter_ratio"] > 2.0
    assert "letter_ratio_out_of_range" in decisions["Doc_R2#0"]["coverage"]["flags"]
    assert 1.5 < decisions["Doc_R2#0"]["coverage"]["letter_ratio"] <= 2.0
    assert decisions["Doc_A#1"]["side_label"] == "2r"

    manifest = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    stems = {m["stem"] for m in manifest}
    assert {"legacy_Doc_A_1r_c0_page", "legacy_Doc_A_1v_c1_page", "legacy_X_2_none_c0_page",
            "legacy_Doc_D_none_c0_page", "legacy_Doc_R2_none_c0_page"} == stems
    assert all(m["matched_n4"] >= bl.MIN_MATCHED_GRAMS for m in manifest)
    assert next(m for m in manifest if m["stem"] == "legacy_Doc_A_1v_c1_page")["side_label"] == "2r"
    assert all(m["legacy_store"] and m["prompt"] == bl.PROMPT and m["leakage"] == "clean" for m in manifest)
    # rows carry ES canonical ids: the document id and the ids the hold-out checked are ids of the index
    assert all(m["canonical_id"] in index.ids and m["canonical_id"] in m["es_ids"] and set(m["es_ids"]) <= index.ids
               for m in manifest)
    dsd = load_from_disk(str(out))
    rows = [r for split in dsd for r in dsd[split]]
    assert len(rows) == len(manifest) == sum(stats["rows"].values())
    row = next(r for r in rows if r["stem"] == "legacy_Doc_A_1r_c0_page")
    assert row["question"] == bl.PROMPT and row["answer"] == texts["A1"] and row["section"] == bl.SECTION
    assert "PRIVATE" in (out / "README.md").read_text() and "NO TRAINING USE" in (out / "README.md").read_text()
    export = out.with_name(out.name + "_images_once")
    assert json.loads((export / "manifest.json").read_text())["n_rows"] == len(manifest)
    assert "PRIVATE" in (export / "README.md").read_text()
    queue = [json.loads(l) for l in queue_path.read_text().splitlines()]
    assert {q["key"] for q in queue} == {k for k, o in outcome.items() if o == "needs_review"}
    assert all(len(q["text_first_words"].split()) <= bl.QUOTE_WORDS for q in queue)
    assert "## Funnel" in report.read_text()
    assert stats["shared_text"]["group_outcomes"] == {"owner_found": 1, "no_owner": 1, "contested": 1}


def test_check_out_dir_refuses_the_inputs(tmp_path: Path) -> None:
    """The output may not be, contain or sit inside the input directory, and must be a v<N> directory.

    :param tmp_path: pytest's temporary directory.
    """
    src = tmp_path / "legacy_store_diplomatic"
    with pytest.raises(ValueError):
        bl.check_out_dir(src / "legacy_store_diplomatic_v1", src)
    with pytest.raises(ValueError):
        bl.check_out_dir(tmp_path / "other_name", src)
    bl.check_out_dir(tmp_path / "legacy_store_diplomatic_v1", src)


def test_dominance_ratio_and_triage() -> None:
    """Competing evidence far below the matched recall is background-like; a near-equal spill is a competing text."""
    pairs = {"a": {0: pair(0.8), 1: pair(0.08)}, "b": {0: pair(0.07), 1: pair(0.7)}}
    conflict = {"image_index": 0, "reasons": ["other_side_clears_p95_on_image"], "conflicts": ["b"], "other_images": []}
    ratio = bl.dominance_ratio("a", conflict, pairs)
    assert ratio == pytest.approx(0.1) and bl.triage_band(ratio) == "background_like"
    spill = {"image_index": 0, "reasons": ["side_clears_p95_on_another_image"], "conflicts": [], "other_images": [1]}
    assert bl.triage_band(bl.dominance_ratio("a", spill, {"a": {0: pair(0.3), 1: pair(0.25)}})) == "competing_text"
    assert bl.dominance_ratio("a", {"image_index": 0, "reasons": ["n3_below_p95"]}, pairs) is None
    assert bl.triage_band(None) is None


# ------------------------------------------------------------------------------------ slip-corrected images manifest

def test_decide_document_slip_side_relaxes_p95_like_a_side_label() -> None:
    """Between p95 and p99 a side is accepted when the image's slip names its side, even without a file-name token;
    a slip naming the other side relaxes nothing, and the 5-matched-grams rule still holds."""
    side = bl.make_side(record("S", SIDE_TEXT, side="1v"), 0, [])
    slipped = [bl.ImageInfo("S", 0, "S_0.jpg", "/tmp/S_0.jpg", "u/S/0", "S-0", 100, 120, "", "1r"),
               bl.ImageInfo("S", 1, "S_1.jpg", "/tmp/S_1.jpg", "u/S/1", "S-1", 100, 120, "", "1v")]
    out = bl.decide_document([side], slipped, {side.key: {0: pair(0.0), 1: pair(0.15)}}, {})
    assert out[side.key]["outcome"] == "accepted" and out[side.key]["basis"] == "p95+slip_side"
    assert bl.priors_of(side, slipped[1]) == {"side_label": None, "slip_side": True, "order_prior": None}
    wrong = bl.decide_document([side], slipped, {side.key: {0: pair(0.15), 1: pair(0.0)}}, {})
    assert wrong[side.key]["outcome"] == "needs_review"
    assert wrong[side.key]["reasons"] == ["clears_p95_not_p99_without_agreeing_prior"]
    both = [bl.ImageInfo("S", 0, "S_0.jpg", "/tmp/S_0.jpg", "u/S/0", "S-0", 100, 120, "1v", "1v")]
    out = bl.decide_document([side], both, {side.key: {0: pair(0.15)}}, {})
    assert out[side.key]["basis"] == "p95+slip_side" and out[side.key]["outcome"] == "accepted"
    short = bl.make_side(record("S", LETTERS + LETTERS[0], side="v"), 0, [])          # 20 4-grams
    thin = bl.decide_document([short], slipped[1:], {short.key: {1: pair(0.15, p95=0.1, p99=0.2)}}, {})
    assert thin[short.key]["basis"] == "p95+slip_side" and thin[short.key]["outcome"] == "needs_review"
    assert thin[short.key]["reasons"] == [bl.UNDER_MATCHED_REASON]


def test_load_images_applies_moves_detachments_and_slip_sides(tmp_path: Path) -> None:
    """A corrected manifest: a moved row joins its new document without the old file-name token, a detached row is
    nobody's image, an emptied document stays with no images, and slip sides reach the images.

    :param tmp_path: pytest's temporary directory.
    """
    base = {"file_name": "f.jpg", "path": "/tmp/f.jpg", "url": "u", "width": 10, "height": 10}
    rows = [dict(base, canonical_id="A", old_canonical_id="A", image_index=0, sha256="a0", side_token="r", folio=1,
                 slip_side="1r"),
            dict(base, canonical_id="A", old_canonical_id="B", image_index=1, sha256="b0", side_token=None, folio=None,
                 old_side_token="r", slip_side="1v"),
            dict(base, canonical_id=None, old_canonical_id="B", image_index=None, sha256="b1", side_token="v", folio=1,
                 slip_side=None),
            dict(base, canonical_id="C", image_index=0, sha256="c0", side_token="none", folio=None)]
    path = tmp_path / "images_manifest_v2.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    images = bl.load_images(path)
    assert sorted(images) == ["A", "B", "C"] and images["B"] == []
    assert [(im.index, im.token, im.slip_side, im.old_canonical_id) for im in images["A"]] == [(0, "1r", "1r", ""),
                                                                                             (1, "", "1v", "B")]
    assert images["C"][0].slip_side == "" and images["C"][0].old_canonical_id == ""
    assert bl.manifest_corrections(path) == {"rows": 4, "detached": 1, "relabelled": 1, "with_slip_side": 2}
    moved = images["A"][1]
    assert bl.row_stem(bl.make_side(record("A", SIDE_TEXT, side="v"), 0, []), moved) == ("1v", "legacy_A_1v_c1_page")


def test_build_with_a_corrected_images_manifest(tmp_path: Path) -> None:
    """A photo filed under the wrong document: not found with the original manifest, accepted after the slip pass
    moves it (``--images-manifest``); the emptied document's side becomes ``no_image``; the original manifest stays
    the default.

    :param tmp_path: pytest's temporary directory.
    """
    rng = random.Random(11)
    t = {name: hebrew_text(rng, 40) for name in ("P", "O")}
    records = [record("Doc_P", t["P"], side="1r"), record("Doc_O", t["O"])]
    images = [{"cid": "Doc_P", "index": 0, "token": "1r"}, {"cid": "Doc_O", "index": 0, "token": "1r"}]
    reads = {"Doc_P#0": (hebrew_text(rng), 8), "Doc_O#0": (noisy(t["P"], rng), 8)}     # P's photo filed under O
    for i in range(60):
        images.append({"cid": f"Noise_{i}", "index": 0, "token": ""})
        reads[f"Noise_{i}#0"] = (hebrew_text(rng), 8)
    src = tmp_path / "legacy_store_diplomatic"
    write_inputs(src, records, images, reads)
    leakage = bl.LeakageResources(held=bl.HeldOut(), index=fake_index(), gate=fake_gate(), gt_sets={}, pgpids={})

    before = bl.build(src, tmp_path / "legacy_store_diplomatic_v1", None, None, None, False, 50, 1, leakage)
    assert before["rows"] == {"train_page": 0, "val": 0} or sum(before["rows"].values()) == 0
    assert before["scope"]["images_manifest"]["relabelled"] == 0
    first = {d["key"]: d for d in (json.loads(l) for l in
                                    (tmp_path / "legacy_store_diplomatic_v1" / "decisions.jsonl").read_text().splitlines())}
    assert first["Doc_P#0"]["outcome"] == "unassigned" and first["Doc_P#0"]["reasons"] == ["not_found"]

    manifest = [json.loads(l) for l in (src / "images_manifest.jsonl").read_text().splitlines()]
    v2 = []
    for row in manifest:
        row = dict(row, old_canonical_id=row["canonical_id"], slip_side=None, relabel_source=None)
        if row["canonical_id"] == "Doc_O":         # its slip names Doc_P, photograph 1
            row.update(canonical_id="Doc_P", image_index=1, old_image_index=0, side_token=None, folio=None,
                       old_side_token="r", slip_side="1r", relabel_source="shelf_mark_slip")
        elif row["canonical_id"] == "Doc_P":       # its slip names a fragment outside the set
            row.update(canonical_id=None, image_index=None, relabel_source="shelf_mark_slip")
        v2.append(row)
    corrected = src / "images_manifest_v2.jsonl"
    corrected.write_text("".join(json.dumps(r) + "\n" for r in v2), encoding="utf-8")

    out = tmp_path / "legacy_store_diplomatic_v2"
    after = bl.build(src, out, None, tmp_path / "R.md", tmp_path / "q.jsonl", True, 50, 1, leakage,
                     images_manifest=corrected)
    decisions = {d["key"]: d for d in (json.loads(l) for l in (out / "decisions.jsonl").read_text().splitlines())}
    assert decisions["Doc_P#0"]["outcome"] == "accepted" and decisions["Doc_P#0"]["image_index"] == 1
    assert decisions["Doc_P#0"]["images"][0]["filed_under"] == "Doc_O"
    assert decisions["Doc_O#0"]["outcome"] == "unassigned" and decisions["Doc_O#0"]["reasons"] == [bl.NO_IMAGE_REASON]
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["stem"] for r in rows] == ["legacy_Doc_P_1r_c1_page"]
    assert rows[0]["image_filed_under"] == "Doc_O" and rows[0]["image_slip_side"] == "1r"
    assert rows[0]["prior_conflict"] is False and decisions["Doc_P#0"]["prior_conflict"] is False
    assert after["scope"]["images_manifest"] == {"path": str(corrected), "rows": len(v2), "detached": 1,
                                                 "relabelled": 1, "with_slip_side": 1}
    assert "moved to another document by a shelf-mark slip 1" in (tmp_path / "R.md").read_text()


# ------------------------------------------------------------------------------- slip side against file-name token

def slip_image(cid: str, index: int, token: str, slip_side: str) -> bl.ImageInfo:
    """A synthetic image with a file-name token and a slip side.

    :param cid: Document.
    :param index: Image index.
    :param token: File-name token.
    :param slip_side: Side the shelf-mark slip names.
    :return: Image.
    """
    return bl.ImageInfo(cid, index, f"{cid}_{index}.jpg", f"/tmp/{cid}_{index}.jpg", f"u/{cid}/{index}",
                        f"{cid}-{index}", 100, 120, token, slip_side)


def test_sides_conflict_and_image_side() -> None:
    """Token and slip side conflict only when both are set and disagree (leaf too when both have one); the slip side
    names the image, an agreeing token only fills in a missing leaf."""
    assert bl.sides_conflict("1v", "1r") and bl.sides_conflict("1r", "2r") and bl.sides_conflict("1r", "v")
    assert not bl.sides_conflict("r", "1r") and not bl.sides_conflict("1v", "v") and not bl.sides_conflict("1v", "1v")
    assert not bl.sides_conflict("", "1r") and not bl.sides_conflict("1r", "")
    pairs = (("1v", "1r"), ("1v", ""), ("", "2r"), ("1v", "v"), ("r", "1r"), ("", ""))
    assert [bl.image_side(slip_image("X", 0, tok, slip)) for tok, slip in pairs] == ["1r", "1v", "2r", "1v", "1r", ""]


def test_conflicting_token_and_slip_side_relax_nothing() -> None:
    """Token 1v and slip side 1r (a photograph-number slip read against the file name): a side labelled 1r, 1v, r or
    v between p95 and p99 goes to review with the conflict recorded; nothing breaks ties; p99 still accepts."""
    im = [slip_image("R", 0, "1v", "1r")]
    for label in ("1r", "1v", "r", "v"):
        side = bl.make_side(record("R", SIDE_TEXT, side=label), 0, [])
        d = bl.decide_document([side], im, {side.key: {0: pair(0.15)}}, {})[side.key]
        assert d["outcome"] == "needs_review" and d["basis"] is None and d["prior_conflict"] is True, label
        assert d["reasons"] == ["clears_p95_not_p99_without_agreeing_prior", bl.PRIOR_CONFLICT_REASON], label
        assert bl.relaxing_prior(side, im[0]) is None and bl.prior_weight(side, im[0]) == 0.0, label
    raw = bl.priors_of(bl.make_side(record("R", SIDE_TEXT, side="1v"), 0, []), im[0])
    assert raw == {"side_label": True, "slip_side": False, "order_prior": None}      # diagnostics keep both sources
    bare = bl.make_side(record("R", SIDE_TEXT), 0, [])
    d = bl.decide_document([bare], im, {bare.key: {0: pair(0.15)}}, {})[bare.key]
    assert d["reasons"] == ["clears_p95_not_p99_without_agreeing_prior"] and d["prior_conflict"] is True
    strong = bl.make_side(record("R", SIDE_TEXT, side="1r"), 0, [])
    d = bl.decide_document([strong], im, {strong.key: {0: pair(0.5)}}, {})[strong.key]
    assert d["outcome"] == "accepted" and d["basis"] == "p99" and d["prior_conflict"] is True


def test_slip_side_alone_relaxes() -> None:
    """An image with a slip side and no file-name token relaxes p95 through the slip side, without a conflict."""
    im = [slip_image("Q", 0, "", "1r")]
    for label in ("1r", "r"):
        side = bl.make_side(record("Q", SIDE_TEXT, side=label), 0, [])
        d = bl.decide_document([side], im, {side.key: {0: pair(0.15)}}, {})[side.key]
        assert d["outcome"] == "accepted" and d["basis"] == "p95+slip_side" and d["prior_conflict"] is False, label
        assert bl.prior_weight(side, im[0]) == bl.PRIOR_EPS["slip_side"]
    other = bl.make_side(record("Q", SIDE_TEXT, side="1v"), 0, [])
    d = bl.decide_document([other], im, {other.key: {0: pair(0.15)}}, {})[other.key]
    assert d["outcome"] == "needs_review" and d["reasons"] == ["clears_p95_not_p99_without_agreeing_prior"]


def test_agreeing_token_and_slip_side_relax_once() -> None:
    """A token and a slip side naming the same side relax once (basis ``p95+slip_side``, one tie-break weight); an
    agreeing token fills in a leafless slip side, and a leafless token agrees with a slip side's letter."""
    for token, slip in (("1v", "1v"), ("1v", "v"), ("v", "1v")):
        im = [slip_image("G", 0, token, slip)]
        side = bl.make_side(record("G", SIDE_TEXT, side="1v"), 0, [])
        d = bl.decide_document([side], im, {side.key: {0: pair(0.15)}}, {})[side.key]
        assert d["outcome"] == "accepted" and d["basis"] == "p95+slip_side" and d["prior_conflict"] is False, slip
        assert bl.relaxing_prior(side, im[0]) in bl.RELAXING_PRIORS
        assert bl.prior_weight(side, im[0]) == bl.PRIOR_EPS["slip_side"], slip
    token_only = bl.make_side(record("G", SIDE_TEXT, side="v"), 0, [])
    d = bl.decide_document([token_only], [image("G", 0, "1v")], {token_only.key: {0: pair(0.15)}}, {})[token_only.key]
    assert d["basis"] == "p95+side_label" and d["prior_conflict"] is False


def test_row_stem_prefers_the_slip_side() -> None:
    """The stem names the slip side over the file-name token, also when the two disagree; an agreeing token fills in a
    leafless slip side; the label only when the image names no side."""
    side = bl.make_side(record("Doc_S", SIDE_TEXT, side="2v"), 0, [])
    assert bl.row_stem(side, slip_image("Doc_S", 0, "1v", "1r")) == ("1r", "legacy_Doc_S_1r_c0_page")
    assert bl.row_stem(side, slip_image("Doc_S", 1, "1v", "1v")) == ("1v", "legacy_Doc_S_1v_c1_page")
    assert bl.row_stem(side, slip_image("Doc_S", 2, "1v", "v")) == ("1v", "legacy_Doc_S_1v_c2_page")
    assert bl.row_stem(side, slip_image("Doc_S", 3, "", "2r")) == ("2r", "legacy_Doc_S_2r_c3_page")
    assert bl.row_stem(side, slip_image("Doc_S", 4, "1r", "")) == ("1r", "legacy_Doc_S_1r_c4_page")
    assert bl.row_stem(side, slip_image("Doc_S", 5, "", "")) == ("2v", "legacy_Doc_S_2v_c5_page")
