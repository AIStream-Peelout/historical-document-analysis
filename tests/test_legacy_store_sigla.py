# File name: test_legacy_store_sigla.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Unit tests for the legacy-store sigla cleaner (``legacy_store_sigla``).

Every string is synthetic: common Hebrew words composed for the test, never transcription
text. Covers each action of the sigla specification (keep letters, keep spans, drop spans,
gaps, dropped marks, structure, finals, repairs), the family classifier, side and column
splitting, unmatched brackets (flagged, never erasing text) and a seeded random round trip
asserting that cleaning never invents a letter and never loses one silently.
"""
import random
import re
from typing import Dict, List, Optional, Set, Tuple

import pytest

from src.datasets.cleaning import legacy_store_sigla as L

GAP = L.GAP


def one(text: str, family: str = "documentary_plain", mapping: Optional[Dict[str, str]] = None) -> L.CleanedSection:
    """Clean a synthetic section that must yield exactly one side.

    :param text: Synthetic raw text.
    :param family: Family forced on the cleaner.
    :param mapping: Optional finals mapping.
    :return: The single cleaned side.
    """
    sides = L.clean_section(text, mapping, family=family)
    assert len(sides) == 1, [s.text for s in sides]
    inv, lost = L.letter_audit(text, sides, mapping)
    assert not inv and not lost, (inv, lost)
    return sides[0]


# --------------------------------------------------------------------------- keep letters / spans


def test_uncertain_wrap_keeps_letters_and_counts_them() -> None:
    """?x? (torn, near-certain reading) keeps its letters and counts them as uncertain."""
    s = one("כתב ?ספ?ר טוב", "ktiv_sigla")
    assert s.text == "כתב ספר טוב"
    assert s.uncertain_letters == 2


def test_stray_question_mark_dropped() -> None:
    """An unpaired ? is a mark, never a letter."""
    assert one("שלום? בית ?").text == "שלום בית"


def test_at_wrap_and_stray_at() -> None:
    """@x@ letters belong to the word; a stray @ is dropped."""
    assert one("@ה@יה בית @ ספר", "ktiv_sigla").text == "היה בית ספר"


@pytest.mark.parametrize("family", ["ktiv_sigla", "documentary_plain", "liturgical_markup"])
def test_double_bracket_is_interlinear_ink(family: str) -> None:
    """[[x]] is an interlinear addition (KTIV 'b'): its letters stay in reading position.

    :param family: Section family under test.
    """
    s = one("ש[[ל]]ום בית", family)
    assert s.text == "שלום בית"
    assert s.recon_letters == 0


def test_caret_interlinear_and_empty_caret_pair() -> None:
    """^x^ is interlinear ink; an empty ^ ^ (edition page-edge mark) disappears."""
    assert one("ש^ל^ום ^ ^ בית").text == "שלום בית"


def test_erasure_dropped_and_correction_kept() -> None:
    """((x)) scraped letters are dropped (recorded); the {x} correction that follows is ink."""
    s = one("ספר ב((ק)) {ב}ית טוב", "ktiv_sigla")
    assert s.text == "ספר בבית טוב"
    assert ("erasure", "ק") in s.dropped_spans


def test_parenthesis_is_kept_deletion_in_ktiv_sigla() -> None:
    """(x) in a KTIV-sigla section is struck but legible ink (KTIV 'm'): kept."""
    s = one("ספר (טוב) בית", "ktiv_sigla")
    assert s.text == "ספר טוב בית"
    assert s.flags["deletion_kept"] == 1


def test_correction_brace_kept_in_every_family() -> None:
    """{x} is the scribe's correction (KTIV 'o'), unlike the July supplied-letter rule."""
    for family in L.FAMILIES:
        assert one("ב{י}ת ספר שלום", family).text == "בית ספר שלום"


def test_line_filler_kept_and_empty_angle_dropped() -> None:
    """<x'> line fillers are ink; an empty < > is a stroke without letters."""
    assert one("דבר <ו'> ומים בית < > ספר", "ktiv_sigla").text == "דבר ו' ומים בית ספר"


def test_angle_addition_kept_in_editions() -> None:
    """<x> in an edition legend is an added (visible) letter."""
    assert one("שלו<ם> בית", "edition_apparatus").text == "שלום בית"


def test_square_bracket_is_margin_ink_in_ktiv_sigla() -> None:
    """[x] after a note mark & is margin text (KTIV 'j'), kept in reading order."""
    s = one("אמר& [טוב] שלום", "ktiv_sigla")
    assert s.text == "אמר טוב שלום"
    assert s.gaps == 0


def test_inline_correction_kept() -> None:
    """[|x|] in-line correction keeps the corrected letter."""
    assert one("ש[|ל|]ום בית", "ktiv_sigla").text == "שלום בית"


def test_not_body_text_kept_and_flagged() -> None:
    """*x* (later hand, gloss) is visible ink: kept, flagged for review."""
    s = one("ספר *שלום* בית", "ktiv_sigla")
    assert s.text == "ספר שלום בית"
    assert s.flags["not_body_text"] == 1


def test_catchword_kept() -> None:
    """%x% catchword at the page foot is ink."""
    assert one("ספר בית %טוב%", "ktiv_sigla").text == "ספר בית טוב"


def test_middot_becomes_point_and_colon_kept() -> None:
    """The scribe's point · becomes '.'; ':' stays; a lone editorial '.' goes."""
    assert one("בעדים· שלום: בית. ספר").text == "בעדים. שלום: בית ספר"


def test_geresh_and_gershayim_kept() -> None:
    """Abbreviation strokes are ink."""
    assert one("ר' אמר ב\"ה ׳ספר״").text == "ר' אמר ב\"ה ׳ספר״"


def test_detached_geresh_joins_the_preceding_word() -> None:
    """A geresh or gershayim standing alone after a word (export spacing) is attached to it."""
    s = one("אמ ' רב איצ ' א ע \" ב")
    assert s.text == "אמ' רב איצ' א ע\" ב"
    assert s.flags["detached_geresh_joined"] == 3


def test_leading_apostrophe_moved_only_in_ktiv_and_editions() -> None:
    """A geresh displaced to the word start ('אמ) moves after the letters in ktiv_sigla."""
    assert one("'אמ רב 'ר בית", "ktiv_sigla").text == "אמ' רב ר' בית"
    assert one("'אמ רב", "documentary_plain").text == "'אמ רב"


# --------------------------------------------------------------------------- drop spans / gaps


def test_editorial_parenthesis_expansion_dropped() -> None:
    """Outside ktiv_sigla a short Semitic (x) is an editor's expansion: dropped and recorded."""
    s = one("ר' (רבי) אמר")
    assert s.text == "ר' אמר"
    assert ("expansion", "רבי") in s.dropped_spans


@pytest.mark.parametrize("inner", ["קרוע", "...", ". . ."])
def test_loss_note_or_dots_in_parentheses_become_gap(inner: str) -> None:
    """A loss note or a dot span in parentheses stands for lost ink.

    :param inner: Content of the parentheses.
    """
    assert one(f"שלום ({inner}) בית").text == f"שלום {GAP} בית"


@pytest.mark.parametrize("inner", ["?", "!", "3", "sic"])
def test_parenthesis_without_letters_dropped(inner: str) -> None:
    """(?) (!) numbers and Latin notes are not ink.

    :param inner: Content of the parentheses.
    """
    assert one(f"שלום ({inner}) בית").text == "שלום בית"


def test_weiss_format_parentheses_are_restorations() -> None:
    """With an empty ( ) in the section, parentheses are restoration brackets -> counted gaps."""
    s = one("שלום ( ) בית (טוב) ספר")
    assert s.text == f"שלום {GAP} בית {GAP} ספר"
    assert s.recon_letters == 3
    assert s.flags["weiss_parens"] == 1


def test_restoration_bracket_becomes_gap_inside_word() -> None:
    """[x] outside ktiv_sigla is a restoration: gap (inside the word), letters counted."""
    s = one("ש[לו]ם בית")
    assert s.text == f"ש{GAP}ם בית"
    assert s.recon_letters == 2
    assert ("restoration", "לו") in s.dropped_spans


def test_long_bracket_span_is_two_edges_around_ink() -> None:
    """A bracketed span over 40 characters is two paired edge marks: gap + ink + gap."""
    inner = " ".join(["שלום בית ספר"] * 5)
    s = one(f"טוב [{inner}] מים")
    assert s.text == f"טוב {GAP} {inner} {GAP} מים"
    assert s.recon_letters == 0
    assert s.flags["long_bracket_span"] == 1


@pytest.mark.parametrize("raw", ["שלום ] ... [ בית", "שלום [ ] בית", "שלום [...] בית", "שלום ].[ בית"])
def test_blank_bracket_pairs_are_gaps_in_either_orientation(raw: str) -> None:
    """Brackets around dots or blanks are a lacuna whichever way they face.

    :param raw: Synthetic raw section.
    """
    assert one(raw).text == f"שלום {GAP} בית"


@pytest.mark.parametrize("raw,expected", [
    ("יש..יה בית", f"יש{GAP}יה בית"),
    ("שלום ..... .בית", f"שלום {GAP} בית"),
    (".......פרחים בית", f"{GAP}פרחים בית"),
    ("שלום … בית", f"שלום {GAP} בית"),
])
def test_dot_runs_become_gaps(raw: str, expected: str) -> None:
    """Dot runs (one dot per lost letter, with the displaced ' .') become one gap.

    :param raw: Synthetic raw section.
    :param expected: Expected cleaned text.
    """
    assert one(raw).text == expected


def test_underscore_run_gap_and_edition_apparatus_cut() -> None:
    """__ is a gap; a >= 10 underscore rule in an edition cuts the apparatus after it."""
    assert one("שלום __ בית").text == f"שלום {GAP} בית"
    s = one("שלום בית ____________ הערה ספר", "edition_apparatus")
    assert s.text == "שלום בית"
    assert s.flags["apparatus_cut"] == 1
    assert s.dropped_spans[0][0] == "apparatus"


def test_latin_and_digits_dropped_and_counted() -> None:
    """Latin text and digits are not ink; both are counted."""
    s = one("שלום 123 abc בית")
    assert s.text == "שלום בית"
    assert s.flags["digits"] == 3
    assert s.flags["latin_chars"] == 3


# --------------------------------------------------------------------------- dropped marks


def test_paragraph_marks_dropped() -> None:
    """¶ and \\ (paragraph end) are not letters."""
    assert one("שלום¶ בית\\ ספר", "ktiv_sigla").text == "שלום בית ספר"


def test_note_mark_dropped() -> None:
    """& outside a mojibake mapping is a note mark."""
    assert one("ספר& בית", "ktiv_sigla").text == "ספר בית"


def test_separator_marks_become_spaces() -> None:
    """# § = , ; and dashes are layout or editor punctuation."""
    text = "שלום##בית §פזמ§ ספר=טוב, מים; עץ - ארץ"
    assert one(text, "liturgical_markup").text == "שלום בית פזמ ספר טוב מים עץ ארץ"


def test_joining_marks_dropped() -> None:
    """+ | / ! ~ are removed without splitting words."""
    assert one("מלא+ ספר| ב/ית! ~עץ").text == "מלא ספר בית עץ"


def test_dollar_marks_an_uncertain_letter() -> None:
    """Edition 'אב$ג': the letter before $ is doubtful; it is kept and counted."""
    s = one("אב$ג בית", "edition_apparatus")
    assert s.text == "אבג בית"
    assert s.uncertain_letters == 1


def test_reversed_doublet_keeps_one_copy() -> None:
    """'A///B' with B = A reversed is a render artefact: one copy stays, the other is recorded."""
    s = one("שלום///םולש בית", "edition_apparatus")
    assert s.text == "שלום בית"
    assert [k for k, _ in s.dropped_spans] == ["doublet"]


def test_points_stripped_and_presentation_forms_decomposed() -> None:
    """Vocalisation is removed and presentation forms become base letters."""
    s = one("ש\u05b8\u05c1לו\u05b9ם \ufb2aבת בית")
    assert s.text == "שלום שבת בית"
    assert s.flags["vocalised"] >= 3


def test_strip_controls() -> None:
    """Control characters, tabs and carriage returns become spaces."""
    assert L.strip_controls("א\x1bב\tג\rד") == "א ב ג ד"


# --------------------------------------------------------------------------- structure


def test_explicit_line_numbers_become_line_breaks() -> None:
    """'1. 2.' numbering (also the visual '28 .X') breaks lines; numbers are dropped."""
    s = one("1. שלום בית 2. ספר טוב 28 .מים עץ")
    assert s.text == "שלום בית\nספר טוב\nמים עץ"
    assert s.flags["explicit_lines"] == 3


def test_sequential_bare_line_numbers_become_line_breaks() -> None:
    """Bare numbers 1, 2, 3 in order number the lines."""
    assert one("1 שלום בית 2 ספר 3 טוב").text == "שלום בית\nספר\nטוב"


def test_split_sides_at_side_headings() -> None:
    """דף X צד Y headings open sides labelled leaf + r/v."""
    got = [(side, text.strip()) for side, text in L.split_sides("דף א צד א שלום בית דף א צד ב ספר טוב")]
    assert got == [("1r", "שלום בית"), ("1v", "ספר טוב")]


def test_grouped_side_heading_is_one_boundary() -> None:
    """צד א (recto) (P2-F) צד שמאל is one heading; the page code alone opens nothing."""
    got = [(side, text.strip()) for side, text in L.split_sides("צד א( recto) (P2-F ) צד שמאל שלום בית")]
    assert got == [("r", "שלום בית")]
    assert [s for s, _ in L.split_sides("שלום (P2-F ) בית")] == [None]


@pytest.mark.parametrize("raw,labels", [
    ("recto שלום verso בית", ["r", "v"]),
    ("1r שלום 1v בית", ["1r", "1v"]),
    ("1א: שלום 1ב: בית", ["1r", "1v"]),
    ("ע\"א שלום בית", ["r"]),
    ("213 א שלום 213 ב בית", ["r", "v"]),
    ("ד1 א\" ע שלום ד1 ב\" ע בית", ["1r", "1v"]),
    ("1 א\" ע שלום 1 ב\" ע בית", ["1r", "1v"]),
    ("Cat A 12 ע\"ב ) שלום בית", [None, "v"]),
])
def test_side_heading_forms(raw: str, labels: List[Optional[str]]) -> None:
    """Every heading form of the specification yields its side label.

    :param raw: Synthetic raw section.
    :param labels: Expected side labels.
    """
    assert [s for s, _ in L.split_sides(raw)] == labels


@pytest.mark.parametrize("raw", ["שלום לצד אחד בית", "שלום כיצד בית", "כתוב לפני צד א בית", "שלום דף קכ\"ד ע\"ב בית", "ספר צד142 בית"])
def test_running_text_is_not_a_side_heading(raw: str) -> None:
    """Letter-bounded words, citations and page references never split a section.

    :param raw: Synthetic raw section.
    """
    assert [s for s, _ in L.split_sides(raw)] == [None]


def test_text_before_a_verso_heading_is_inferred_recto() -> None:
    """Ink before the first heading that opens a verso belongs to that leaf's recto."""
    sides = L.clean_section("שלום בית ספר טוב צד ב מים עץ", family="documentary_plain")
    assert [(s.side, s.side_source, s.text) for s in sides] == [("r", "inferred", "שלום בית ספר טוב"),
                                                               ("v", "marker", "מים עץ")]


def test_catalogue_header_cut() -> None:
    """A Latin catalogue line before the first heading is a header: dropped and recorded."""
    sides = L.clean_section("Catalogue T-S 1 ספר 1r שלום בית", family="documentary_plain")
    assert [(s.side, s.text) for s in sides] == [("1r", "שלום בית")]
    assert sides[0].flags["header_cut"] == 1
    assert sides[0].dropped_spans[0][0] == "header"
    inv, lost = L.letter_audit("Catalogue T-S 1 ספר 1r שלום בית", sides)
    assert not inv and not lost


def test_columns_split_within_one_side() -> None:
    """טור_X headings are columns of one side, never another side."""
    s = one("טור_א שלום בית טור_ב ספר טוב", "ktiv_sigla")
    assert s.side is None
    assert s.column == "א,ב"
    assert s.text == "שלום בית\n\nספר טוב"
    assert [k for k, _ in s.dropped_spans] == ["column_marker", "column_marker"]


# --------------------------------------------------------------------------- unmatched brackets


@pytest.mark.parametrize("raw", ["שלום (בית ספר", "שלום בית) ספר", "{שלום בית ספר", "שלום <בית ספר"])
def test_unmatched_bracket_flags_and_keeps_text(raw: str) -> None:
    """A bracket without a partner is removed and flagged; no letter is erased.

    :param raw: Synthetic raw section.
    """
    s = one(raw)
    assert s.text == "שלום בית ספר"
    assert s.flags["unmatched_bracket"] == 1


def test_unmatched_square_bracket_is_an_edge_gap_keeping_text() -> None:
    """An unpaired [ or ] marks a damaged edge: a gap, the text stays."""
    s = one("שלום ] בית ספר")
    assert s.text == f"שלום {GAP} בית ספר"
    assert s.flags["edge_bracket"] == 1


def test_brackets_do_not_pair_across_line_breaks() -> None:
    """Brackets on different numbered lines are edge marks, not one restoration."""
    s = one("1. שלום [בית 2. ספר] טוב")
    assert s.text == f"שלום {GAP}בית\nספר{GAP} טוב"
    assert s.recon_letters == 0


def test_flipped_brackets_are_mirrored_back() -> None:
    """A section whose brackets are stored mirrored (]x[) is re-oriented before pairing."""
    s = one("שלום ]טו[ בית ]ספ[ מים")
    assert s.text == f"שלום {GAP} בית {GAP} מים"
    assert s.recon_letters == 4


def test_edition_side_heading_survives_mirrored_parentheses() -> None:
    """An edition header's 'ע"ב )' opens a side even when the section's () pairs are stored mirrored.

    Mirroring turns the ')' after the heading into '(', so headings are found before it.
    """
    raw = "Cat A 12 ע\"ב ) שלום )טוב( בית )ספר( מים עץ"
    assert L.normalise_brackets(raw)[1]["()"] == "flipped"
    s = one(raw, "edition_apparatus")
    assert (s.side, s.side_source) == ("v", "marker")
    assert s.text == "שלום בית מים עץ"
    assert ("expansion", "טוב") in s.dropped_spans


# --------------------------------------------------------------------------- finals


def test_repair_finals_only_word_final() -> None:
    """Only a symbol right after a letter and before a non-letter is a final."""
    text, n = L.repair_finals("שלו\x1c מקו\x1c \x1cאב", {"\x1c": "ם"})
    assert (text, n) == ("שלום מקום \x1cאב", 2)
    with pytest.raises(ValueError):
        L.repair_finals("שלו\x1c", {"\x1c": "מ"})


def test_clean_section_repairs_finals_and_counts() -> None:
    """The section mapping restores finals; the audit counts them as source letters."""
    s = one("שלו\x1c בית מקו\x1c", mapping={"\x1c": "ם"})
    assert s.text == "שלום בית מקום"
    assert s.finals_repaired == 2


def test_unresolved_finals_flagged_in_mojibake_section() -> None:
    """A word-final control without a mapping in a finals-mojibake section is flagged."""
    words = " ".join(["בית", "ספר", "טוב", "דבר"] * 4)
    s = one(f"{words} שלו\x1d")
    assert s.flags["finals_mojibake"] == 1
    assert s.flags["unresolved_finals"] == 1
    assert s.text.endswith("שלו")


def test_infer_finals_mapping_with_lexicon() -> None:
    """The specification rule picks the final letter whose words the lexicon knows."""
    lexicon = {"שלום": 1, "מקום": 1, "יום": 1, "ספר": 1}
    assert L.infer_finals_mapping("שלו\x1c ספר מקו\x1c יו\x1c", lexicon) == {"\x1c": "ם"}
    assert L.infer_finals_mapping("שלו\x1c \x1cשלו \x1cמקו", lexicon) == {}


def test_is_finals_mojibake() -> None:
    """Long sections without real finals are mojibake; short ones are not judged."""
    assert L.is_finals_mojibake(" ".join(["בית", "ספר", "טוב"] * 6))
    assert not L.is_finals_mojibake(" ".join(["שלום", "ספר", "טוב"] * 6))
    assert not L.is_finals_mojibake("בית ספר")


def test_finals_table_keys_are_content_digests() -> None:
    """The module table is keyed by content hash and maps symbols to final letters only."""
    assert L.FINALS_MAPPING
    for key, mapping in L.FINALS_MAPPING.items():
        assert re.fullmatch(r"[0-9a-f]{16}", key)
        assert mapping and set(mapping.values()) <= set(L.FINALS)
    assert L.finals_mapping_for("שלום בית") == {}
    assert L.section_digest("שלום") == L.section_digest("שלום")


# --------------------------------------------------------------------------- reversal


def _logical_section() -> str:
    """A synthetic logical-order section with finals and probe words.

    :return: Text of 24 words.
    """
    return " ".join(["שלום", "אשר", "כל", "מקום", "הוא", "ספר", "בית", "לחם"] * 3)


def test_whole_section_reversal_detected_and_undone() -> None:
    """A section stored right-to-left (finals first, probe words reversed) is restored."""
    logical = _logical_section()
    assert L.unreverse_if_needed(logical) == (logical, False)
    assert L.unreverse_if_needed(logical[::-1]) == (logical, True)
    sides = L.clean_section(logical[::-1], family="documentary_plain")
    assert sides[0].reversed and sides[0].text == logical


def test_local_fragment_reversal_repaired() -> None:
    """Letter runs that start with a final are reversed; a lone final is left alone."""
    assert L.repair_reversed_fragments("ךל שלום ?םא? ן") == ("לך שלום ?אם? ן", 2)


def test_fragment_order_join_needs_lexicon_support() -> None:
    """'ן שאי' is joined to 'שאין' only when the lexicon knows the reversed join."""
    assert L.repair_fragment_order("ספר ן שאי בית", {"שאין"}) == ("ספר שאין בית", 1)
    assert L.repair_fragment_order("ספר ן שאי בית", None) == ("ספר ן שאי בית", 0)
    assert L.repair_fragment_order("חכ ם שאי", {"שאים", "חכם"}) == ("חכ ם שאי", 0)


@pytest.mark.parametrize("raw,lexicon,expected", [
    ("ש?ול?ם בית", {"שלום", "בית"}, ("ש?לו?ם בית", 1)),             # only the reversed word is known
    ("ש?לו?ם בית", {"שלום", "בית"}, ("ש?לו?ם בית", 0)),             # only the forward word is known
    ("ש?ול?ם בית", {"שלום", "שולם", "בית"}, ("ש?ול?ם בית", 0)),     # both known: undecided, unchanged
    ("ש?ול?ם בית", None, ("ש?ול?ם בית", 0)),                         # no lexicon
    ("?בטו? ספר", {"טובב", "ספר"}, ("?בטו? ספר", 0)),                # neither join known
    ("בית ?לש?ום", {"שלום"}, ("בית ?של?ום", 1)),                     # wrap at the word start
    ("בא?ןי?", {"באין"}, ("בא?ןי?", 0)),                             # final-first wraps are left to run repair
])
def test_uncertain_wrap_order_decided_per_wrap_by_lexicon(raw: str, lexicon: Optional[Set[str]],
                                                          expected: Tuple[str, int]) -> None:
    """A ?xy? wrap is reversed only when the whole word reads as a known word reversed and not forward.

    :param raw: Synthetic section text.
    :param lexicon: Synthetic lexicon (or None).
    :param expected: (repaired text, wraps reversed).
    """
    assert L.repair_uncertain_wraps(raw, lexicon) == expected


def test_clean_section_reorders_visual_wraps_in_ktiv_sigla_only() -> None:
    """The wrap repair runs on ktiv_sigla sections, counts as a fragment repair and keeps every letter."""
    raw = "ש?ול?ם בית ספר"
    lexicon = {"שלום", "בית", "ספר"}
    sides = L.clean_section(raw, family="ktiv_sigla", lexicon=lexicon)
    assert [s.text for s in sides] == ["שלום בית ספר"]
    assert sides[0].flags["wrap_reversed"] == 1
    assert sides[0].fragments_repaired == 1
    assert sides[0].uncertain_letters == 2
    inv, lost = L.letter_audit(raw, sides)
    assert not inv and not lost
    other = L.clean_section(raw, family="documentary_plain", lexicon=lexicon)
    assert [s.text for s in other] == ["שולם בית ספר"]
    assert "wrap_reversed" not in other[0].flags


# --------------------------------------------------------------------------- families


@pytest.mark.parametrize("raw,family", [
    ("?ש?לום ?ב?ית ¶ ספר· טוב מים עץ", "ktiv_sigla"),
    ("שלום", "empty"),
    ("שלום ## בית ## ספר טוב מים עץ", "liturgical_markup"),
    ("שלום בית ספר טוב מים ____________ הערה", "edition_apparatus"),
    ("שלום בית ספר טוב מים עץ ארץ", "documentary_plain"),
])
def test_classify_family(raw: str, family: str) -> None:
    """The family precedence of the specification.

    :param raw: Synthetic raw section.
    :param family: Section family under test.
    """
    assert L.classify_family(raw) == family


# --------------------------------------------------------------------------- round trip

_WORDS = ["שלום", "בית", "ספר", "טוב", "מים", "עץ", "ארץ", "דבר", "לחם", "ךל", "ן"]
_MARKS = ["?", "[", "]", "(", ")", "{", "}", "<", ">", "[[", "]]", "((", "))", "^", "@", "*", "%", "&", "$",
          "#", "¶", "\\", "·", ":", "'", "\"", "...", "..... .", "__", "|", "/", "!", "~", "-", ",", ";", "=",
          "§", "+", "1.", "12", "abc", "\x1c", "\t", "טור_א", "טור_ב", "צד א", "צד ב", "דף ב צד א", "recto",
          "1v", "[|", "|]", "( )", "___________", "\u05b9", "ש\u05c1"]
_FORBIDDEN_RE = re.compile(r"[\[\](){}<>?@^*%&$#¶\\|/!~=§+,;_\-\d A-Za-z\x00-\x1f]")


@pytest.mark.parametrize("seed", range(40))
def test_round_trip_never_invents_or_silently_loses_letters(seed: int) -> None:
    """Random mixes of words and marks: letters balance and only ink characters survive.

    :param seed: Random seed of the synthetic mix.
    """
    rng = random.Random(seed)
    tokens = [rng.choice(_WORDS) if rng.random() < 0.55 else rng.choice(_MARKS) for _ in range(rng.randint(5, 60))]
    raw = "".join(tok + (" " if rng.random() < 0.6 else "") for tok in tokens)
    mapping = {"\x1c": "ם"} if seed % 2 else None
    for family in L.FAMILIES:
        sides = L.clean_section(raw, mapping, family=family)
        inv, lost = L.letter_audit(raw, sides, mapping)
        assert not inv, (family, raw, inv)
        if sides:
            assert not lost, (family, raw, lost)
        for s in sides:
            residue = s.text.replace(GAP, "").replace("\n", "").replace(" ", "")
            assert not _FORBIDDEN_RE.search(residue), (family, raw, s.text)
            assert s.letters == len(L.letters_of(s.text))
            assert s.gaps == s.text.count(GAP)
