# File name: test_text27b_lit_segment.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``lit_segment``: stamps, reversed-Hebrew checks, legacy shelfmarks and joins, edition-line
cleaning, the doc-headed / anthology / catalogue segmenters and edition runs.

Synthetic page texts only: no real works, holdings or machine paths.
"""
from src.finetuning.text27b import lit_segment as S

STAMP = ["This content downloaded from", "10.0.0.1 on Tue, 27 Oct 2026 17:48:25 UTC",
         "All use subject to https://about.jstor.org/terms"]


def test_stamps_and_ip() -> None:
    """Stamp lines are recognised, scrubbed, and an IPv4 address is detected."""
    assert all(S.is_stamp_line(x) for x in STAMP)
    assert S.is_stamp_line("Reproduced with permission of the copyright owner. Further reproduction prohibited.")
    text, n = S.scrub_stamps("\n".join(["kept line"] + STAMP))
    assert text == "kept line" and n == 3
    assert S.has_ip_address("from 10.0.0.1 on")
    assert not S.has_ip_address("T-S 12.34 and 1.2.3")


def test_reversed_hebrew_share() -> None:
    """Final-form letters at word starts mark reversed text."""
    good = " ".join(["שלום", "אדם", "ארץ", "כנען", "מלך"] * 3)
    bad = " ".join(w[::-1] for w in good.split())
    assert S.reversed_hebrew_share(good) == 0.0
    assert S.reversed_hebrew_share(bad) == 1.0
    assert S.reversed_hebrew_share("שלום") is None


def test_word_order_reversal() -> None:
    """Fixed-order word pairs found backwards mark a visual-order text layer."""
    fwd = "נכתב על ידי הסופר ראו גם את זה בבית דין של ארץ ישראל " * 2
    rev = " ".join(fwd.split()[::-1])
    assert S.word_order_reversal(fwd) == 0.0
    assert S.word_order_reversal(rev) == 1.0
    assert S.word_order_reversal("מילה אחת") is None


def test_split_joins() -> None:
    """Joins split into parts; bare numbers and folios inherit the first mark's prefix."""
    assert S.split_joins("TS 16.22 and TS 10 J 5, f. 17") == ["TS 16.22", "TS 10 J 5, f. 17"]
    assert S.split_joins("TS NS J 110 and 152") == ["TS NS J 110", "TS NS J 152"]
    assert S.split_joins("TS Misc. Box 28, f. 141 and f. 10") == ["TS Misc. Box 28, f. 141", "TS Misc. Box 28, f. 10"]
    assert S.split_joins("ENA 1234.5") == ["ENA 1234.5"]


def test_legacy_variants() -> None:
    """Old spellings map to the modern forms; raw form first; last-resort variants last."""
    assert S.legacy_variants("TS 8 J 34, f. 8")[1] == "T-S 8J34.8"
    assert S.legacy_variants("TS NS Box 226, f. 11")[1] == "T-S NS 226.11"
    assert S.legacy_variants("TS Arabic 30.123")[1] == "T-S Ar.30.123"
    assert S.legacy_variants("TS 8.223 Verso")[1] == "T-S 8.223"
    assert S.legacy_variants("Bodleian MS Heb. d 66.5")[1] == "Bodl. MS Heb. d 66/5"
    assert S.legacy_variants("ENA 2805, fol. 16B")[1] == "ENA 2805.16b"
    assert S.legacy_variants("TS Misc. Box 28, f. 79, nos. 2, 1,")[1] == "T-S Misc.28.79.2"
    v = S.legacy_variants("TS 8 J 5, f. 1 a, c, b, d (*)")
    assert v[1:] == ["T-S 8J5.1a", "T-S 8J5.1"]
    assert S.legacy_variants("Oxford (Neubauer) 2878/4", {"2878": "MS Heb. d 66"})[1] == "Bodl. MS Heb. d 66/4"
    assert "T-S 8J37.11" in S.legacy_variants("TS 8J 37.1141")


def test_clean_edition_lines() -> None:
    """Line numbers, numeric / debris / stamp lines go; short lines are counted."""
    lines = ["1 בשם רחמן", "2", "w", "אלשיך אבו עלי 3", "דלך", "This content downloaded from", "Latin words only here"]
    out, st = S.clean_edition_lines(lines)
    assert out == ["בשם רחמן", "אלשיך אבו עלי", "דלך"]
    assert st["numeric"] == 1 and st["debris"] == 1 and st["stamp"] == 1 and st["latin"] == 1 and st["short"] == 2


def test_doc_header_and_shelfmark_header() -> None:
    """Header lines are found among the leading lines."""
    lines = ["112", "TS 8.138", "Doc. 16", "Deed of gift."]
    assert S.doc_header(lines) == ("16", "doc", 2)
    assert S.shelfmark_header(lines) == ("TS 8.138", 1)
    assert S.doc_header(["(Docs. 97, 98 and 99)"])[0:2] == ("97,98,99", "docs")
    assert S.doc_header(["Rec. Form. 258"])[0:2] == ("RF258", "rec_form")


def test_segment_doc_headed_continuation_and_whole_page() -> None:
    """A page without a header continues the current document; whole-page mode keeps lines above the header."""
    p1 = S.PageInput("w/p/page_001.json", 1, "112\nTS 8.138\nDoc. 16\nDeed of gift.\nFragment.")
    p2 = S.PageInput("w/p/page_002.json", 2, "113\n(Cont.)\nMore facts here.")
    p3 = S.PageInput("w/p/page_003.json", 3, "line before header\nTS 12.5\nDoc. 17\nCourt record.")
    secs = S.segment_doc_headed([p1, p2, p3])
    assert [s.label for s in secs] == ["16", "17"]
    assert secs[0].shelfmark_raw == "TS 8.138"
    assert "More facts here." in secs[0].text and "Deed of gift." in secs[0].text
    assert "line before header" not in secs[1].text
    whole = S.segment_doc_headed([p3], whole_page=True)
    assert "line before header" in whole[0].text and "Doc. 17" not in whole[0].text


def test_parse_description() -> None:
    """Type, date phrase, stated lines and state come out of a catalogue description."""
    d = S.parse_description("112\nDeed of gift.\nFragment.\nThe date preserved is Shebat, 1125.\nThe script is legible. 14 lines.")
    assert d["document_type"] == "Deed of gift"
    assert d["date_text"] == "Shebat, 1125"
    assert d["lines_stated"] == 14
    assert d["state"] == "fragment"
    assert S.parse_description("Release. Date not preserved.")["date_text"] == "not preserved"


def test_anthology_sections() -> None:
    """Entries split into intro / shelfmark line / translation; running heads, footnotes and list items handled."""
    p1 = S.PageInput("a/b/page_001.json", 1, "\n".join([
        "CHAPTER ONE 12", '1. "I was hungry"', "An intro sentence about the petitioner.",
        "TS 10 J 6.17 (Judaeo-Arabic)", "Your slave asks for bread.", "1. a numbered line inside the letter",
        "6 A footnote that must not enter."]), foot_line=6)
    p2 = S.PageInput("a/b/page_002.json", 2, "\n".join([
        "13 CHAPTER ONE", "And peace.", '2. "Second"', "Second intro.", "ENA 1234.5 (Hebrew)", "Second text.",
        "BIBLIOGRAPHY", "Not part of it."]))
    import re
    secs = S.anthology_sections([p1, p2], stop=re.compile(r"^BIBLIOGRAPHY$"))
    assert [s.label for s in secs] == ["1", "2"]
    assert secs[0].shelfmark_raw == "TS 10 J 6.17"
    assert secs[0].extra["intro_text"] == "An intro sentence about the petitioner."
    assert "Your slave asks for bread." in secs[0].text and "And peace." in secs[0].text
    assert "footnote" not in secs[0].text and "CHAPTER" not in secs[0].text
    assert secs[1].text == "Second text."


def test_anthology_footnote_lines_from_page_json() -> None:
    """Lines whose letters occur in the page's footnote strings are left out wherever they sit."""
    lines = ["Translation line one.", "a note continued from the previous page", "Translation line two."]
    notes = S.footnote_line_set(lines, ["… a note continued from the previous page …"])
    assert notes == {1}


def test_catalogue_entries() -> None:
    """Entry numbers must continue the sequence; page numbers in running heads are skipped."""
    p1 = S.PageInput("c/d/page_001.json", 1, "\n".join(["14", "BIBLE", "1", "First entry text.", "2",
                                                         "Second entry.", "450", "inside text"]))
    p2 = S.PageInput("c/d/page_002.json", 2, "\n".join(["15", "GENIZAH FRAGMENTS", "more of two", "4", "Fourth."]))
    secs = S.catalogue_entries([p1, p2])
    assert [s.label for s in secs] == ["1", "2", "4"]
    assert "more of two" in secs[1].text and "450" in secs[1].text


def test_rtl_runs_and_numbered_share() -> None:
    """Runs of Hebrew lines are found (bridging one number line) and their numbering measured."""
    lines = ["English prose line here.", "1 בשם רחמן רחים", "2 אלשיך אבו עלי", "3", "4 אלחמד ללה וחדה",
             "5 וכתבת אליך", "More English."]
    runs = S.rtl_runs(lines, min_lines=4, min_letters=20)
    assert runs == [(1, 5)]
    assert S.numbered_share(lines[1:6]) == 1.0


def test_anthology_wrapped_title() -> None:
    """A heading title that wraps onto the next line is folded into the title, not the introduction."""
    p = S.PageInput("a/b/page_001.json", 1, "\n".join([
        '1. "I ask you, by God, not to disappoint the', 'request of your slave"', "The writer is from Jerba.",
        "TS 10 J 6.17 (Judaeo-Arabic)", "Your slave writes."]))
    secs = S.anthology_sections([p])
    assert secs[0].extra["title"].endswith('request of your slave"')
    assert secs[0].extra["intro_text"] == "The writer is from Jerba."


def test_anthology_stray_running_head() -> None:
    """A running head the OCR put after the shelfmark line stays out of the translation."""
    p1 = S.PageInput("a/b/page_001.json", 1, "\n".join(['1. "Title"', "Intro.", "TS 10 J 6.17 (Judaeo-Arabic)", "First."]))
    p2 = S.PageInput("a/b/page_002.json", 2, "\n".join(["ENA 1234.5 (Hebrew)", "50 CHAPTER THREE", "Second text."]))
    secs = S.anthology_sections([p1, p2])
    assert secs[0].extra["docs"][1]["text"] == "Second text."


def test_split_piece_lists() -> None:
    """Numbered and lettered piece lists split into one shelfmark per piece; side letters stay together."""
    assert S.split_joins("TS NS J 185, nos. 5, 6, 2") == ["TS NS J 185, no. 5", "TS NS J 185, no. 6", "TS NS J 185, no. 2"]
    assert S.split_joins("TS NS J 185, nos. 12, 8 (*)") == ["TS NS J 185, no. 12", "TS NS J 185, no. 8"]
    assert S.split_joins("TS Misc. Box 28, f. 79, nos. 2, 1,") == ["TS Misc. Box 28, f. 79, no. 2", "TS Misc. Box 28, f. 79, no. 1"]
    assert S.split_joins("TS NS J 67 a, b (*)") == ["TS NS J 67a", "TS NS J 67b"]
    assert S.split_piece_lists("TS 1.2 r, v") == ["TS 1.2 r, v"]
    assert S.split_joins("TS Misc Box 28, f. 46 (and Verso) (*)") == ["TS Misc Box 28, f. 46 (and Verso) (*)"]


def test_piece_of() -> None:
    """A trailing numbered piece is split from its base mark."""
    assert S.piece_of("TS NS J 185, no. 7") == ("TS NS J 185", "no. 7")
    assert S.piece_of("TS NS J 401 no. 18") == ("TS NS J 401", "no. 18")
    assert S.piece_of("TS NS J 401, no. 20 (*)") == ("TS NS J 401", "no. 20")
    assert S.piece_of("TS 12.50") is None


def test_legacy_variants_new_forms() -> None:
    """Piece letters, volumes, Or.1080 boxes, Westminster College, no-without-dot, Penn E and Krengel box numbers."""
    assert S.legacy_variants("TS Misc Box 28, f. 79, no 1")[1] == "T-S Misc.28.79.1"
    assert S.legacy_variants("TS NS J 318(a)")[1] == "T-S NS J318a"
    assert S.legacy_variants("TS NS J 67 a, b (*)")[1] == "T-S NS J67a"
    assert S.legacy_variants("TS Arabic Box 18(1).33")[1] == "T-S Ar.18(1).33"
    assert S.legacy_variants("TS Arabic Box 46.253")[1] == "T-S Ar.46.253"
    assert S.legacy_variants("ULC Or Box 5, f. 13")[1] == "CUL Or.1080 5.13"
    assert S.legacy_variants("TS Or 1080 Box 5, f. 13")[1] == "CUL Or.1080 5.13"
    assert S.legacy_variants("Westminster College, Misc. 34")[1] == "L-G Misc. 34"
    assert S.legacy_variants("West. Coll. (Westminster College), Frag. Cairens., f. 34")[1] == "L-G Misc. 34"
    assert S.legacy_variants("West. Coll. (Westminster College), Ar. (Arabica) I, f. 100")[1] == "L-G Ar. I.100"
    assert S.legacy_variants("E 16.510")[-1] == "UPenn E 16510"
    assert S.legacy_variants("JTS Krengel 5.123")[-1] == "JTS Krengel 123"
    assert S.legacy_variants("TS 8 J 5, f. 3c")[1:] == ["T-S 8J5.3c", "T-S 8J5.3"]


def test_clean_sm_and_alt_classmark() -> None:
    """A volume in parentheses survives cleaning; a second classmark in parentheses is returned completed."""
    assert S._clean_sm("TS Arabic Box 18 (1).336") == "TS Arabic Box 18(1).336"
    assert S._clean_sm("Mosseri L 129.1 (II 127.1) 36") == "Mosseri L 129.1"
    assert S.alt_classmark("Mosseri L 129.1 (II 127.1) 36") == "Mosseri II 127.1"
    assert S.alt_classmark("Mosseri L 291 (Ia, 21) 23") == "Mosseri Ia, 21"
    assert S.alt_classmark("TS 10 J 6.17 (formerly TS 99.1)") == "TS 99.1"
    assert S.alt_classmark("TS Arabic Box 18 (1).33") is None
    assert S._is_lang_line("Mosseri L 129.1 (II 127.1) 36 (Judaeo-Arabic)") == ("Mosseri L 129.1", "Judaeo-Arabic",
                                                                                  "Mosseri II 127.1")
