# File name: test_text27b_textnorm.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``textnorm``: CER normalisation, read / edition cleaning, transliteration,
the B3 / B4 target strips, the project-name strip and the grounding test. Synthetic strings only
(shelfmarks are made-up numbers).
"""
from pathlib import Path

import codecs
import pytest

from src.finetuning.text27b import textnorm as T

#: Scrubbed project names, decoded at runtime from rot13 so this file never spells them (fix pass F12).
SCRUB_SURNAME = codecs.decode("Sevrqoret", "rot_13")
SCRUB_SOCIETY = codecs.decode("SWZF", "rot_13")


# ----------------------------------------------------------------------------------- CER
def test_normalize_for_cer_markers_and_v11() -> None:
    """Editorial signs, label lines, digit lines and leading markers go; U+2212 only under v11."""
    text = "§ recto\n(1–2) [שלום] עליך\n12\n3. ‏אדוני"
    assert T.normalize_for_cer(text) == "שלום עליך ‏אדוני"
    assert T.normalize_for_cer("(3−4) שלום") == "שלום"
    assert T.normalize_for_cer("(3−4) שלום", v11=False) == "3−4 שלום"
    assert T.normalize_for_cer(None) == ""


def test_cer_and_coverage() -> None:
    """CER is edit distance over reference length; coverage is a normalised length ratio."""
    assert T.cer("abc", "abd") == pytest.approx(1 / 3)
    assert T.cer("x", "") == 1.0
    assert T.coverage("ab", "abcd") == 0.5
    assert T.coverage("ab", "§ only label") is None


def test_script_counts_and_dominant() -> None:
    """Letters are counted per script; the dominant script needs 60 %."""
    assert T.script_counts("אב ab سل") == {"hebrew": 2, "arabic": 2, "latin": 2}
    assert T.dominant_script("שלום עליך a") == "hebrew"
    assert T.dominant_script("ab אב") == "mixed"
    assert T.dominant_script("123") == "none"


# ----------------------------------------------------------------------------------- inputs
def test_clean_read_removes_loops() -> None:
    """Placeholder lines, consecutive repeats and frequent short lines are removed."""
    text = "a\na\na\n[...]\nb\nx\ny\nx\nz\nx"
    out, removed = T.clean_read(text)
    assert out == "a\nb\nx\ny\nz" and removed == 5


def test_strip_latin_lines() -> None:
    """Latin header lines go; short side labels become § labels; Hebrew lines stay."""
    text = "Synthetic Shelf 1, f. 2 Letter from Joseph to his son\nשלום עליך\nright margin\nאדוני"
    out, dropped = T.strip_latin_lines(text)
    assert out == "שלום עליך\n§ right margin\nאדוני" and dropped == 1


def test_edition_as_plain() -> None:
    """Label lines and editorial signs are removed, line breaks kept."""
    assert T.edition_as_plain("§ verso\n[שלום] (!) עליך\n⟦מחוק⟧ כתוב") == "שלום עליך\nמחוק כתוב"


def test_transliteration_plain_and_geresh() -> None:
    """The plain JA spelling has no geresh and drops standalone hamza; final forms apply."""
    assert T.transliterate_ar_to_ja_plain("شيء خبز الغرب") == "שי כבז אלגרב"
    assert T.transliterate_ar_to_ja("خبز") == "כ׳בז"
    assert T.transliterate_ar_to_ja_plain("سلام ١٢") == "סלאם 12"
    assert T.transliterate_ar_to_ja_plain("abc") == "abc"


# ----------------------------------------------------------------------------------- line markers
def test_strip_line_markers_gil_ranges() -> None:
    """Gil range markers with any dash (U+2212 included), spaced or reversed, and ``ואילך`` go."""
    text = "(4-1) אני כותב לך\n\n(8–5) ואני בריא\n(14−13) ירוק\n(1 – 2) בשמך\n(12 ואילך) אליו"
    out, ops = T.strip_line_markers(text)
    assert out == "אני כותב לך\n\nואני בריא\nירוק\nבשמך\nאליו"
    assert ops == ["gil_range", "gil_onward"]


def test_strip_line_markers_leading_numbers_need_sequence() -> None:
    """Leading ``N.`` numbers go only when two or more lines carry them in ascending order."""
    seq = "1. which was made\n2. against them\n3. and through all"
    assert T.strip_line_markers(seq) == ("which was made\nagainst them\nand through all", ["leading_line_number"])
    assert T.strip_line_markers("1. Paid for nails") == ("1. Paid for nails", [])
    amounts = "12 1/4 dir. A tip\n11 1/2, for the balance"
    assert T.strip_line_markers(amounts) == (amounts, [])


def test_strip_line_markers_keeps_content_brackets() -> None:
    """Biblical references, fractions, side labels and year ranges are not markers."""
    text = "May it be (Ps. 126: 2) and (10/24).\n§ Left side (1)\nIn the years (1100–1150) prices rose"
    assert T.strip_line_markers(text) == (text, [])


def test_strip_line_markers_refs_and_tidy() -> None:
    """Line references, India Book references and line-start ``(N)`` go, with spacing repaired."""
    text = "Opening of letter (lines 1–5).\n(ln. 9-11) Many resigned\n(1) If they made\nthe word (5–6) next\n[(III,5:4) one absent."
    out, ops = T.strip_line_markers(text)
    assert out == "Opening of letter.\nMany resigned\nIf they made\nthe word next\n[one absent."
    assert ops == ["gil_range", "line_ref", "ib_line_ref", "paren_line_number"]


# ----------------------------------------------------------------------------------- sentences
def test_transliteration_keeps_arabic_indic_digits() -> None:
    """Arabic-Indic digits survive the harakat strip and map to ASCII digits."""
    assert T.transliterate_ar_to_ja("سنة ٤٤٠") == "סנה 440"
    assert T.transliterate_ar_to_ja_plain("بِسْمِ") == "בסם"


def test_sentence_spans_abbreviations_and_protect() -> None:
    """Abbreviations, initials and protected spans do not end sentences; blank lines do."""
    text = "Letter from Abū Naṣr b. Yosef to S. D. Smith. Sent ca. 1100 CE via T-S 99J1.2. Next one.\n\nNew para"
    spans = T.sentence_spans(text, protect=T.find_shelfmarks(text))
    got = [text[a:b] for a, b in spans]
    assert got == ["Letter from Abū Naṣr b. Yosef to S. D. Smith.", "Sent ca. 1100 CE via T-S 99J1.2.", "Next one.",
                   "New para"]


def test_find_shelfmarks() -> None:
    """Common collection prefixes are found with their numbers; plain prose is not."""
    text = "See T-S NS 999.12, ENA 9999.1 and Bodl. MS heb. d 99/7; also Mosseri IX.99.1 and T-S Ar. 99.87. Psalm 23."
    found = [text[a:b] for a, b in T.find_shelfmarks(text)]
    assert found == ["T-S NS 999.12", "ENA 9999.1", "Bodl. MS heb. d 99/7", "Mosseri IX.99.1", "T-S Ar. 99.87"]


# ----------------------------------------------------------------------------------- citations
@pytest.mark.parametrize("text,expected,ops", [
    ("Letter from Ali to Musa. This document is discussed (and cited as T-S 99J9.9) in Q. Author, “A Title,” "
     "Journal 30:4 (2012), p. 1026.", "Letter from Ali to Musa.", ["bibliography"]),
    ("Written by the clerk. Measurements: 13 × 8.5 cm. The verso is blank.", "Written by the clerk. The verso is blank.",
     ["dimensions"]),
    ("He came to a qaysariyya (on which see Goitein, Med Soc I, p. 194; but here an inn (or brothel?)) and left.",
     "He came to a qaysariyya and left.", ["citation_paren"]),
    ("The scribe is known. This scribe wrote T-S 99.1, T-S NS 999.2, and T-S 99J9.3.", "The scribe is known.",
     ["shelfmark_sentence"]),
    ("Is he ill? This document is possibly related to Bodl. MS heb. a 99/1, a letter from the Nagid.", "Is he ill?",
     ["shelfmark_sentence"]),
    ("Legal query about a levirate marriage; verso: blank; published, with translation, by Someone, Title, pp. 144–45.",
     "Legal query about a levirate marriage; verso: blank.", ["see_cf"]),
    ("Letter from Nahray (ENA 9999.15) to his cousin.", "Letter from Nahray to his cousin.", ["shelfmark"]),
    ("Sent 5 dinars to Fustat. A Book Title (Publisher, 2008). Also see https://example.org/x.",
     "Sent 5 dinars to Fustat.", ["bibliography", "see_cf"]),
    ("Paper (20 x 14 cm) letter to his brother.", "Paper letter to his brother.", ["dimensions"]),
])
def test_strip_citations_rules(text: str, expected: str, ops: list) -> None:
    """Each rule removes its sentence, clause or bracket and reports itself once.

    :param text: Input prose.
    :param expected: Expected output.
    :param ops: Expected ops.
    """
    assert T.strip_citations(text) == (expected, ops)


@pytest.mark.parametrize("text", [
    "Marriage contract dated Tishri 1449 Sel. (1137 CE) from Fustat; the bride brings 100 dinars.",
    "The wedding is set for Hanukkah (25 Kislev 5589 AM = December 1 1828 CE).",
    "Our eyes are toward You (Ps. 25:15). The goods described in the letter arrived.",
    "The two partners discussed the matter in detail.",
    "",
])
def test_strip_citations_keeps_content(text: str) -> None:
    """Document dates, biblical references and ordinary prose are untouched.

    :param text: Input prose.
    """
    assert T.strip_citations(text) == (text, [])


def test_strip_citations_local_repairs_only() -> None:
    """Text away from a cut is untouched (ellipses, double dots); a clause after a dropped first clause is capitalised."""
    text = 'Only "mā.. lā yadillu" survives (wa-kataba...). The hand resembles T-S 99J9.9 (same scribe as T-S NS 999.63).'
    out, ops = T.strip_citations(text)
    assert out == 'Only "mā.. lā yadillu" survives (wa-kataba...). The hand resembles.' and ops == ["citation_paren", "shelfmark"]
    text2 = "Ketubba dated 1049 CE. Bodl. MS heb. a 99/32, a ketubba he wrote later; several formulae are shared."
    assert T.strip_citations(text2) == ("Ketubba dated 1049 CE. Several formulae are shared.", ["shelfmark_sentence"])


@pytest.mark.parametrize("text,expected", [
    ("IB IV, 99 (ח79). Letter from Ali to Musa.", "Letter from Ali to Musa."),
    ("A deed. NB: In an older book, Someone mistakenly cites this shelfmark for another one.", "A deed."),
    ("A deed of sale. A. Someone's edition for further information.", "A deed of sale."),
    ('Letter to Fustat. Partial translation from Q. Author, Big Work, 2:271: "I send you 15 dirhems. Buy pepper."',
     'Letter to Fustat. "I send you 15 dirhems. Buy pepper."'),
])
def test_strip_citations_reference_sentences(text: str, expected: str) -> None:
    """India Book prefixes, NB citation notes and pointers to an edition are dropped.

    :param text: Input prose.
    :param expected: Expected output.
    """
    assert T.strip_citations(text)[0] == expected


def test_strip_citations_multiline_analysis() -> None:
    """Paragraph structure survives; a removed paragraph leaves no empty line run."""
    text = "Receipt (423 A.H.) T-S Ar. 99.140.\n\nPaper. 46.5 cm. x 27.5 cm. The verso is blank.\n\nSee the introduction.\n\nNotes follow."
    out, ops = T.strip_citations(text)
    assert out == "Receipt (423 A.H.).\n\nPaper. The verso is blank.\n\nNotes follow."
    assert ops == ["shelfmark", "dimensions", "see_cf"]


# ----------------------------------------------------------------------------------- project names
def test_strip_source_names_sentence_clause_paren() -> None:
    """Hits remove their sentence, ``;`` clause or parenthetical; nothing matching remains."""
    s1 = f"This shelfmark is catalogued in the {SCRUB_SURNAME} metadata as a get. The witnesses sign."
    assert T.strip_source_names(s1) == ("The witnesses sign.", 1)
    host = ".".join([codecs.decode("stc", "rot_13"), codecs.decode("travmnu", "rot_13"), "org"])  # assembled (F12)
    s2 = (f"Images at https://{host}/x?y=1 here. Edition ({SCRUB_SURNAME} Genizah Project, 2010). Wheat; "
          f"{SCRUB_SOCIETY} differs; 1100.")
    out, n = T.strip_source_names(s2)
    assert n == 3 and out == "Edition. Wheat; 1100."
    assert not T.has_source_names(out)


def test_strip_source_names_multiline_and_noop() -> None:
    """A sentence spanning lines goes with all its lines; text without hits is returned unchanged."""
    text = (f"Line one of a page\ncontinues, as noted by the {SCRUB_SURNAME.lower()} team in their\ncatalogue. "
            "Next sentence stays.\nOther line.")
    assert T.strip_source_names(text) == ("Next sentence stays.\nOther line.", 1)
    assert T.strip_source_names("nothing here") == ("nothing here", 0)
    other_host = ".".join([codecs.decode("wrjvfuznahfpevcgf", "rot_13"), "org"])
    surname_he = "".join(chr(c) for c in (0x5E4, 0x5E8, 0x5D9, 0x5D3, 0x5D1, 0x5E8, 0x5D2))
    assert T.has_source_names("see " + other_host) and T.has_source_names(surname_he) and not T.has_source_names("Friday")


def test_scrub_terms_are_encoded_in_the_source() -> None:
    """F12: the scrubber's names are decoded at import and matched, but never spelled in the module source."""
    src = Path(T.__file__).read_text(encoding="utf-8").lower()
    t = T.scrub_terms()
    for k in ("surname", "abbr_society", "abbr_project", "abbr_variant", "surname_he"):
        assert T.has_source_names(f"see {t[k]} here"), k
        assert t[k].lower() not in src, k
    host = f"{t['host']}.{t['tld']}"
    assert T.has_source_names(f"https://{host}/x") and host not in src


# ----------------------------------------------------------------------------------- grounding
def test_grounding_latin_and_skeleton_match() -> None:
    """Names found in the source (directly or via a Hebrew-script skeleton) count as grounded."""
    target = "A letter written in Fustat by Abū Naṣr to Maḥrūz, dated 1140."
    source = "כתאב מן אבו נצר אלי מחרוז סנה 1140 אלפסטאט"
    share, n, miss = T.grounding(target, source)
    assert n == 4 and share == 1.0 and miss == []
    assert T.skeleton_rtl("אלפסטאט") == T.skeleton_latin("fustat") == "fst"
    share2, _, miss2 = T.grounding(target + " He mentions Qayrawān.", source)
    assert share2 < 1.0 and "qayrawan" in miss2


def test_grounding_scholar_names_and_empty() -> None:
    """Scholar surnames are not candidates; a target without candidates is fully grounded."""
    assert T.grounding("as noted by Zzgoitein recently.", "", scholar_names={"zzgoitein"}) == (1.0, 0, [])
    assert T.entity_candidates("the end.") == set()


def test_overlap60() -> None:
    """A 60-character window of the target inside the source is detected."""
    src = "x" * 10 + "A sufficiently long run of words that appears in both texts verbatim here." + "y" * 10
    assert T.overlap60("A sufficiently long run of words that appears in both texts verbatim here. tail", src)
    assert not T.overlap60("short", src)


def test_scholar_surnames(tmp_path: Path) -> None:
    """Surnames come from the ``authors`` column, folded, plus extras.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "sources.csv"
    f.write_text("slug,authors\na,\"Smith, John; Ḥaddād, Ali\"\nb,\n", encoding="utf-8")
    assert T.scholar_surnames(f, extra=["Doe"]) == {"smith", "haddad", "doe"}


def test_strip_line_markers_spaced_line_numbers() -> None:
    """F4: ``N`` + a column gap at the start of three or more ascending lines is line numbering (a translation laid
    out in columns); two lines, a single space after a number, or descending numbers are kept.

    :returns: None.
    """
    t = "1          The first line\n2          and the second\n3\tOn the third line\n10        the tenth came"
    out, ops = T.strip_line_markers(t)
    assert out == "The first line\nand the second\nOn the third line\nthe tenth came" and ops == ["leading_line_number_spaced"]
    for keep in ("12  dinars\n13  dirhams", "1 Paper\n3 note\n4 more", "30   dinars\n12   dirhams\n5   qirats"):
        assert T.strip_line_markers(keep) == (keep, []), keep
    assert T.SCRUB_OP == "source_scrub" and not T.SCRUB_RE.search(T.SCRUB_OP)
