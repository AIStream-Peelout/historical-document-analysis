# File name: test_unreachable_profile.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the unreachable-text profile and the scholar-dissent labelling (toy data only)."""
from typing import Dict, List, Sequence

import pytest

from src.datasets.evaluations.helper_eval_scripts import rover_consensus as rc
from src.datasets.evaluations.helper_eval_scripts import slot_oracle as so
from src.datasets.evaluations.helper_eval_scripts import unreachable_profile as up

TEXT_A = "שלום רב לכם אחי היקר ואני מודיע לכם כי הגיע הספר אלינו בשלום"
TEXT_B = "שלום רב לכם אחי היקר ואני מודיע לכם כי הגיע הספר אלינו בשלום".replace("מודיע", "מודיה")


def _page(doc_id: str, gt: str, raws: Sequence[str], config: up.Config) -> up.PageProfile:
    """Profile one toy page.

    :param doc_id: Page id.
    :type doc_id: str
    :param gt: Raw GT.
    :type gt: str
    :param raws: One output per reader of ``config``.
    :type raws: Sequence[str]
    :param config: Reader set (no extra readers).
    :type config: up.Config
    :return: Annotated page profile.
    :rtype: up.PageProfile
    """
    ink = rc.genizah_visible_ink_gt(gt)
    doc = rc.BenchDoc(doc_id, "single-col", ink, rc.letters_only(ink), dict(zip(config.models, raws)))
    lines = so.gt_ink_lines(gt)
    chars = so.page_chars(list(raws), lines)
    page = up.profile_page(doc, lines, so.alignment_raw(gt), chars, config, {"genre": "letter"})
    up.annotate_blocks(page, config, {"link": f"https://example.org/{doc_id}"}, up.short_name)
    return page


CONFIG = up.Config("toy", rc.RELIGIOUS, ("a", "b", "c"))


def test_damage_marks_kinds():
    raw = "אב [...] גד (!) הו? זח (השם) טי ]כל מ..נ"
    kinds = [(raw[s:e], k) for s, e, k in up.damage_marks(raw)]
    assert kinds == [("[...]", "lacuna"), ("(!)", "sic"), ("?", "uncertain"), ("(השם)", "paren"),
                     ("]", "lacuna"), ("..", "lacuna")]
    weiss = "מעכש(ו ) ( ) אב (?) (חתימת יד)"
    assert [k for _, _, k in up.damage_marks(weiss)] == ["lacuna", "lacuna", "uncertain", "lacuna"]


def test_mark_distance_and_gaps():
    raw = "אב [...] גדהוזח"
    dist = up.mark_distance(len(raw), up.damage_marks(raw), ("lacuna",))
    assert dist[:4] == [3, 2, 1, 0] and dist[8] == 1 and dist[-1] == 7
    assert up.mark_distance(3, [], ("lacuna",)) == [4, 4, 4]
    assert up.gap_count("אב [...] גד [הו] זח") == 2 and up.gap_count("אבגד") == 0


def test_letters_and_grams():
    assert up.letters("אב 12 cd גד") == "אבגד"
    assert up.grams("אבגדה", 4) == {"אבגד": 1, "בגדה": 1}
    assert up.gram_share("אבגדה", {"אבגד"}, 4) == 0.5 and up.gram_share("אבג", {"אבג"}, 4) == 0.0


@pytest.mark.parametrize("span,full,rest,expected", [
    ("שלום רב לכם אחי", "x", "", "shifted"),
    ("", "z", "", "dropped"),
    ("", "שלום רב לכם אחי היקר", "", "dropped, wrote it elsewhere"),
    ("ואני מודיע לכם", "ואני מודיע לכם", "ואני מודיע לכם כי", "other page text"),
    ("קקקק", "שלום רב לכם אחי קקקק", "", "wrote it elsewhere"),
    ("טטטטטט", "טטטטטט טטטטטט", "", "looped"),
    ("צפסנמ", "צפסנמ", "", "misread"),
])
def test_reader_behaviour(span, full, rest, expected):
    assert up.reader_behaviour("שלום רב לכם אחי", span, full, rest) == expected


def test_block_note_counts_and_shared_reading():
    note = up.block_note({"m1": "dropped", "m2": "dropped", "m3": "misread"},
                         {"m1": "", "m2": "", "m3": "גגג"}, "אבגד", str.upper)
    assert note == "2/3 dropped; M3 misread"
    shared = up.block_note({m: "misread" for m in ("a", "b", "c")}, {m: "מא אנפדת" for m in ("a", "b", "c")},
                           "דינ ינקץ", str)
    assert shared == "3/3 misread; 3 readers wrote the same letters «מאאנפדת»"


def test_edition_lines_sides_regions_and_classes():
    content = ("Recto\n1 אב גד [הו]\n2 זח טי\nכל מן\nRight margin, perpendicular\nסע פצ\n"
               "Verso\nקרש\nשקר\nתרק\nרקת")              # verso letters never occur on the recto
    lines = up.edition_lines(content)
    assert [(l.side, l.region) for l in lines] == [("recto", "main")] * 3 + [("recto", "margin")] + \
        [("verso", "main")] * 4
    assert lines[0].ink == "אב גד" and lines[0].raw == "1 אב גד [הו]"
    assert up.line_classes(lines) == ["first", "middle", "last", "margin", "first", "middle", "middle", "last"]
    per_char, coverage = up.edition_char_lines("זח טי כל מן", lines)
    assert coverage == 1.0 and per_char[0] == 1 and per_char[-1] == 2
    source, best, _, cov = up.best_edition("זח טי כל מן", {"x": "אאא", "y": content})
    assert source == "y" and best == lines and cov == 1.0
    assert up.best_edition("אב", {})[0] is None


def test_edition_kinds_word_dissent_and_passages():
    keys = [("א", "א"), ("ב", "ג"), ("ד", None)] + [("ה", None)] * 25 + [("ו", "ו")]
    kinds = up.edition_kinds(keys)
    assert kinds[0] == "agree" and kinds[1] == "dissent" and kinds[-1] == "agree"
    assert kinds[2:28] == ["passage"] * 26                     # a long stretch one edition lacks
    assert up.edition_kinds([("א", None), ("ב", "ב")]) == ["dissent", "agree"]   # one omitted letter


def test_edition_slots_label_the_gt_against_both_editions():
    labels = up.edition_slots(list(TEXT_A), TEXT_A, TEXT_B)
    assert len(labels) == len(TEXT_A)
    k = TEXT_A.index("מודיע") + 4
    assert labels[k] == up.EditionSlot("dissent", "a", "ה", True)
    assert labels[0].label == "agree" and labels[0].follows == "both"
    assert sum(lab.label == "dissent" for lab in labels) == 1
    extra = up.edition_slots(list(TEXT_A + " נוסף"), TEXT_A, TEXT_B)
    assert [lab.label for lab in extra[-5:]] == ["gt_only"] * 5


def test_pair_profile_counts():
    prof = up.pair_profile(TEXT_A, TEXT_B)
    assert prof["differ_slots"] == prof["a_differ"] == prof["a_dissent"] == 1 and prof["runs"] == [1]
    assert prof["subs"] == {"הע": 1} and prof["passage_slots"] == 0 and prof["a_chars"] == len(TEXT_A)


def test_gini_concentration_and_allocation():
    assert up.gini([1, 1, 1, 1]) == 0.0 and up.gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert up.concentration([5, 0, 3, 2]) == {"gini": 0.4, "top10_share": 1.0, "pages_with_any": 3,
                                              "pages": 4, "pages_for_50": 1, "pages_for_80": 2}
    assert up.allocate({"a": 10, "b": 5, "c": 1}, 6) == {"a": 3, "b": 2, "c": 1}
    assert up.allocate({"a": 10, "b": 0}, 3) == {"a": 3, "b": 0}
    assert sum(up.allocate({"a": 9, "b": 1, "c": 1, "d": 1}, 3).values()) == 3


def _blocks() -> List[Dict[str, object]]:
    """Forty toy blocks over eight pages, two buckets and three length classes.

    :return: Block dicts with the fields the sampler reads.
    :rtype: List[Dict[str, object]]
    """
    return [{"block_id": f"p{i % 8}:{i}", "doc_id": f"p{i % 8}", "bucket": "x" if i % 2 else "y",
             "length": (10, 30, 100)[i % 3], "min_damage_distance": 3 if i % 5 == 0 else 50}
            for i in range(40)]


def test_sampling_is_stratified_capped_and_deterministic():
    blocks = _blocks()
    sample = up.sample_blocks(blocks, far=10, near=3)
    assert len(sample) == 13 and sample == up.sample_blocks(blocks, far=10, near=3)
    assert sum(b["sample_group"] == "near damage" for b in sample) == 3
    assert all(b["min_damage_distance"] <= up.DAMAGE_WINDOW for b in sample if b["sample_group"] == "near damage")
    per_page = {}
    for b in sample:
        per_page[b["doc_id"]] = per_page.get(b["doc_id"], 0) + 1
    assert max(per_page.values()) <= up.SAMPLE_PER_PAGE
    assert {up.length_class(b["length"]) for b in sample} == {"8-20", "21-60", ">60"}
    assert len({b["block_id"] for b in sample}) == len(sample)


def test_profile_page_blocks_categories_and_confusions():
    gt = f"{TEXT_A}\nזחטכנסעפצקדזחטכנסעפצ"
    raws = [TEXT_A.replace("רב", "דב", 1)] * 3
    page = _page("p1", gt, raws, CONFIG)
    assert page.read_ratio > up.SHORT_READ
    assert len(page.blocks) == 1 and page.blocks[0]["gt_text"].strip() == "זחטכנסעפצקדזחטכנסעפצ"
    block = page.blocks[0]
    assert block["kind"] == "dropped, written by no reader" and block["note"] == "3/3 dropped"
    assert block["link"] == "https://example.org/p1" and block["min_damage_distance"] > up.DAMAGE_WINDOW
    decomposition = up.unreachable_decomposition([page])
    assert decomposition["block: dropped, written by no reader"] == block["unreachable"]
    assert decomposition["local"] == 1                         # the ר no reader wrote
    assert up.confusion_pairs([page]) == {"letter_substitutions": 1, "letter_pairs": {"דר": 1.0}}
    near = _page("p2", "שלום רב [...] לכם אחי היקר ואני מודיע לכם כי הגיע", ["שלום דב לכם אחי היקר ואני מודיע לכם כי הגיע"] * 3,
                 CONFIG)
    assert up.unreachable_decomposition([near]) == {"near damage": 1}
    summary = up.characterise([page, near], CONFIG)
    assert summary["totals"]["unreachable"] == sum(decomposition.values()) + 1
    assert summary["blocks"]["blocks"] == 1 and summary["decomposition"]["near damage"] == 1
    page.read_ratio = 0.3                                     # the page now counts as read short
    assert up.structure_pages([page]) == {up.STRUCTURE_CATEGORIES[3]: {"p1": block["unreachable"]}}


def test_char_category_puts_edition_structure_first():
    page = _page("p1", f"{TEXT_A}\nזחטכנסעפצקדזחטכנסעפצ", [TEXT_A] * 3, CONFIG)
    k = page.blocks[0]["start"] + 3
    kinds = [up.block_kind(b) for b in page.blocks]
    assert up.char_category(page, k, kinds) == "block: dropped, written by no reader"
    page.read_ratio = 0.3
    assert up.char_category(page, k, kinds) == up.STRUCTURE_CATEGORIES[3]
    page.line_class[k] = "margin"
    assert up.char_category(page, k, kinds) == up.STRUCTURE_CATEGORIES[2]
    page.edition = {"second_side": [True] * len(page.cls), "coverage": 0.99}
    assert up.char_category(page, k, kinds) == up.STRUCTURE_CATEGORIES[1]
    page.line_class[k] = "unmapped"
    assert up.char_category(page, k, kinds) == up.DIFFERS_CATEGORY      # GT and edition read differently
    page.edition["coverage"] = 0.5                                         # the GT holds a second text
    assert up.char_category(page, k, kinds) == up.STRUCTURE_CATEGORIES[0]


def test_dissent_overlap_finds_readers_siding_with_the_other_scholar():
    # GT = edition A; edition B reads מודיה; every reader wrote B's ה, so the GT's ע is unreachable
    page = _page("p1", TEXT_A, [TEXT_B] * 3, CONFIG)
    editions = {"7": {"src A": "Recto\n" + TEXT_A, "src B": TEXT_B}}
    out = up.dissent_overlap([page], editions, {"7"}, {"p1": "7"})
    assert out["documents"][0]["same_coverage"] and out["same_coverage"]["n_docs"] == 1
    assert out["same_coverage"]["labels"]["unreachable"] == {"agree": 0, "dissent": 1, "passage": 0,
                                                             "gt_differs": 0, "gt_only": 0}
    votes = out["same_coverage"]["votes"]
    assert votes["unreachable | other edition has a different reading | ROVER voted other scholar"] == 1
    assert out["same_coverage"]["by_category"]["local | dissent"] == 1
    profile = up.two_edition_profile(editions, {"7"})
    assert profile["all"]["pairs"] == 1 and profile["letter_pairs"] == {"הע": 1.0}


def test_names_and_links():
    assert up.short_name(up.V6[3]) == "v22i-a50" and up.short_name(up.V6[0]) == "v21b-1200"
    assert up.short_name(up.KSEG) == "Kraken seg"
    pgp = up.page_link("verified", {"image_url": "https://img/x.jpg"}, "123")
    assert pgp == {"link": "https://geniza.princeton.edu/en/documents/123/", "image": "https://img/x.jpg"}
    assert up.page_link("verified", {"image_url": "https://img/x.jpg"}, None) == {"link": "https://img/x.jpg"}
    ktiv = up.page_link(rc.RELIGIOUS, {"sys_num": "99", "fl": "FL1"}, None)
    assert ktiv["link"].endswith("docId=PNX_MANUSCRIPTS99") and "/FL1/" in ktiv["image"]
