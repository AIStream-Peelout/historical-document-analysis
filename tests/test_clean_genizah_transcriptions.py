# File name: test_clean_genizah_transcriptions.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Unit tests for the diplomatic-transcription cleaner (``clean_genizah_transcriptions``).

Covers the letter-bounded editor-reference regex (``כרך`` inside an ordinary word is not a
volume reference), the section classification it feeds, and the strict cleaning of PGP
editorial marks — sic marks, parenthesised expansions or Weiss-format restorations, and
``|`` join lines — next to the reconstruction, supplied-letter and lacuna behaviour those
fixes must not disturb. Examples follow the shapes found in the merged PGP corpus.
"""
import pytest

from src.datasets.cleaning import clean_genizah_transcriptions as C

GAP = C.GAP


# --------------------------------------------------------------------------- editor references


@pytest.mark.parametrize("text", [
    "כרך ב, עמ' 12",
    "גויטיין, חברה, כרך ב', עמ' 35",
    "תרביץ כרך ט\"ו, חוברת א'",
    "גנזי קדם, כרך שני, תרפ\"ג",
    "כרך 2",
    "מהד' גיל",
    "הדרשה חסרה במהד ' אפשטיין",
    "עמ' 35",
    "בעמ' 35",
    "מדובר בכ-80 ק\"ג גבינה",
    "תרגום ד\"ר פרידמן",
    "השלמתי לפי המקבילה",
    "והשלמתי",
])
def test_edition_reference_tokens_match(text):
    """Standalone editor references (volume, page, edition, translator, restoration) match."""
    assert C.EDITION_REF_RE.search(text)


@pytest.mark.parametrize("text", [
    "ושכרך ירוברב ויותקף",     # ושכרך: ordinary word containing כרך
    "אני זוכרך תמיד",           # זוכרך
    "והשם יכפיל שכרך",
    "כרכים גדולים",
    "בכרך",
    "כרך גדול",                # כרך = "city": no volume designator
    "כרך של רומי",
    "אלשיך אלמהד'ב אבו",       # the name al-Muhadhdhab contains מהד'
    "שאהיה עמ'",               # scribal abbreviation, no page number
    "שאהיה עמ'\n12 ואמר",      # abbreviation at a line end, line number on the next line
    "השלמתיו",
])
def test_edition_reference_needs_letter_boundaries(text):
    """Reference markers glued to Hebrew letters, or without their designator, do not match."""
    assert not C.EDITION_REF_RE.search(text)


def test_classify_section_is_not_tipped_by_a_word_containing_volume_marker():
    """One weak scrape signal plus ושכרך used to reach the ``edition_scrape`` score (regression)."""
    assert C.classify_section("S. D. Goitein", "ושכרך ירוברב ויותקף ]4[ ואמר") == "diplomatic"
    assert C.classify_section("S. D. Goitein", "כרך ב, עמ' 12 ]4[ ואמר") == "edition_scrape"


# --------------------------------------------------------------------------- sic marks


@pytest.mark.parametrize("raw,expected", [
    ("מולי (!) וריסי", "מולי וריסי"),
    ("ותוגיהו(!) אליך", "ותוגיהו אליך"),
    ("ל(!)ך אבג", "לך אבג"),
    ("אבג ( !) דהו", "אבג דהו"),
    ("אבג (!!) דהו", "אבג דהו"),
    ("אבג (?!) דהו", "אבג דהו"),
    ("אבג [!] דהו", "אבג דהו"),           # a bracketed sic mark is not a lacuna
    ("אלאסבוע )!( לאלא", "אלאסבוע לאלא"),   # bidi-mangled
    ("אלבדיה! בעד אן", "אלבדיה בעד אן"),   # bare word-final sic mark
])
def test_sic_marks_dropped(raw, expected):
    """Sic marks vanish without leaving a gap or counting as reconstruction."""
    cleaned, recon, _ = C.clean_diplomatic(raw)
    assert cleaned == expected and recon == 0


# --------------------------------------------------------------------------- parentheses


@pytest.mark.parametrize("raw,expected", [
    ("ר' (רבי) יהודה", "ר' יהודה"),
    ("ויהב(נא) לה", "ויהב לה"),
    ("כ(בוד) ג(דולת) ק(דושת) מר(נו)", "כ ג ק מר"),
    ("אברה(ם) בן", "אברה בן"),                  # supplied letters
    ("אבג (צ\"ל: יחסן) דהו", "אבג דהו"),         # gloss
    ("אבג (כך במקור) דהו", "אבג דהו"),
    ("אבג (רבי?) דהו", "אבג דהו"),
])
def test_expansions_dropped_and_abbreviation_kept(raw, expected):
    """The abbreviated form is the ink; the editor's expansion is dropped, uncounted."""
    cleaned, recon, visible = C.clean_diplomatic(raw)
    assert cleaned == expected and recon == 0
    assert visible == len(C.SEMITIC_RE.findall(expected))


@pytest.mark.parametrize("raw,expected", [
    ("אבג (חתימת יד) דהו", f"אבג {GAP} דהו"),
    ("אבג (השאר אבוד)", f"אבג {GAP}"),
    ("אבג (שש מלים בכתב ערבי, בלתי קריאות) דהו", f"אבג {GAP} דהו"),
    ("אבג (...) דהו", f"אבג {GAP} דהו"),
    ("אבג ( . . . ) דהו", f"אבג {GAP} דהו"),
])
def test_loss_notes_and_dots_are_lacunae(raw, expected):
    """Notes for untranscribed ink and parenthesised dots become one gap token."""
    assert C.clean_diplomatic(raw)[0] == expected


@pytest.mark.parametrize("raw", [
    "זבאד (civet cats) ומעל",     # Latin gloss: left for the Latin-debris gate
    "אבג (12) דהו",               # inline number
])
def test_other_parentheses_untouched(raw):
    """Parenthesised text that is not a Semitic expansion is left for downstream gates."""
    assert C.clean_diplomatic(raw)[0] == raw


def test_long_parenthesised_span_untouched():
    """A span over :data:`MAX_EDITORIAL_SPAN` chars is not read as one expansion."""
    aside = "(" + "א" * (C.MAX_EDITORIAL_SPAN + 1) + ")"
    assert C.clean_diplomatic(f"אבג {aside} דהו")[0] == f"אבג {aside} דהו"
    short = "(" + "א" * C.MAX_EDITORIAL_SPAN + ")"
    assert C.clean_diplomatic(f"אבג {short} דהו")[0] == "אבג דהו"


def test_weiss_format_parentheses_are_restorations():
    """With an empty ``( )`` in the text, ``(xxx)`` is a counted restoration, ``( )`` a gap."""
    raw = "1 ( ) בגמיע אלאלפאט אלמחכמה ואלמעאני אל( )\nמעאני אלאק(ראראת) 3 ( ) ולא גיר דלך"
    cleaned, recon, visible = C.clean_diplomatic(raw)
    assert cleaned == (f"1 {GAP} בגמיע אלאלפאט אלמחכמה ואלמעאני אל {GAP}\n"
                       f"מעאני אלאק {GAP} 3 {GAP} ולא גיר דלך")
    assert recon == 5 and visible == len(C.SEMITIC_RE.findall(cleaned))


def test_weiss_format_torn_edge_and_uncertainty():
    """A trailing space inside the parentheses is a torn edge; ``(?)`` keeps its own rule."""
    assert C.clean_diplomatic("והבתה מעכש(ו ) ( ) דה(?) ו")[0] == f"והבתה מעכש {GAP} דה ו"


def test_same_restoration_in_both_notations():
    """Weiss's ``(הי אלא)ן`` and a bracketed ``[הי אלא]ן`` clean identically."""
    weiss = C.clean_diplomatic("1 (הי אלא)ן פי עצמתה ( )")
    bracket = C.clean_diplomatic("1 [הי אלא]ן פי עצמתה [ ]")
    assert weiss == bracket and weiss[1] == 5


def test_expansion_rule_without_empty_pair():
    """Without an empty ``( )`` the same span is an expansion and is dropped."""
    assert C.clean_diplomatic("1 (הי אלא)ן פי עצמתה")[0] == "1 ן פי עצמתה"


# --------------------------------------------------------------------------- pipes


@pytest.mark.parametrize("raw,expected", [
    ("וקד עלם אללה | ואנת תעלם", "וקד עלם אללה ואנת תעלם"),
    ("לי ענד רב א|הרן בר יוסף", "לי ענד רב אהרן בר יוסף"),   # join line through a word
    ("ואל|גוע || ואלאן", "ואלגוע ואלאן"),
    ("יענך ייי ביום צרה | ישלח עזרך מקודש |", "יענך ייי ביום צרה ישלח עזרך מקודש"),
])
def test_pipes_deleted(raw, expected):
    """``|`` join lines are deleted: glued halves rejoin, spaced ones keep their space."""
    assert C.clean_diplomatic(raw)[0] == expected


# --------------------------------------------------------------------------- unchanged behaviour


def test_reconstructions_still_become_gaps():
    """Bracketed restorations are gaps and their letters count as reconstruction."""
    cleaned, recon, visible = C.clean_diplomatic("וצלני כתא[בך] אלכרים")
    assert cleaned == f"וצלני כתא {GAP} אלכרים" and recon == 2 and visible == 14


def test_supplied_uncertain_and_dots_unchanged():
    """Supplied letters, ``(?)`` and dotted lacunae keep their existing treatment."""
    assert C.clean_diplomatic("אב{ג} דה(?) ו ... ז")[0] == f"אב דה ו {GAP} ז"


def test_marks_combined_across_lines():
    """All three marks on one line, a restoration on the next; line structure is kept."""
    raw = "ר' (רבי) שמואל (!) בן | יעקב\nוצלני כתא[בך] אלכרים"
    cleaned, recon, visible = C.clean_diplomatic(raw)
    assert cleaned == f"ר' שמואל בן יעקב\nוצלני כתא {GAP} אלכרים"
    assert recon == 2 and visible == len(C.SEMITIC_RE.findall(cleaned))


def test_process_document_output_carries_no_editorial_marks():
    """A diplomatic document with sic marks, expansions and pipes cleans to plain ink."""
    lines = [
        "ר' (רבי) שמואל (!) בן | יעקב ושכרך ירוברב ויותקף",
        "כתאבי אליך יא מולאי אטאל אללה בקאך ואדאם עזך",
        "וצלני כתאבך אלכרים ופהמת מא דכרת(ה) מן אמר אלבצאעה",
        "ואנא מנתטר לוצול אלמרכב אלדי פיה אלכתאן | ואלשמע",
    ]
    doc = {"doc_id": "d1", "image_urls": ["u1"],
           "transcriptions": [{"editor": "S. D. Goitein", "text": "\n".join(lines)}]}
    result = C.process_document(doc)
    assert result["triage"] == "A" and result["sections"][0]["kind"] == "diplomatic"
    assert not any(ch in result["cleaned_text"] for ch in "(!)|")
    assert result["cleaned_text"].count("\n") == len(lines) - 1


def test_weiss_document_restoration_share_is_counted():
    """Restorations written in parentheses now raise ``recon_frac`` like bracketed ones."""
    weiss = "\n".join(["1 ( ) בגמיע אלאלפאט אלמחכמה ואלמעאני אל( )",
                       "2 ( ) לליום ובעדה חגה וותאק אנני מקרה ענדכם",
                       "3 (דיואן אלגואמע ואלמסאגד והבתהמא לולדי)",
                       "4 (סיד אלכל דנן במתנה מעכשו מן גמיע) ( )",
                       "5 (מפסדאת אלשהאדה אנני קד והבתה מעכשו)"])
    doc = {"doc_id": "w1", "image_urls": ["u1"],
           "transcriptions": [{"editor": "Weiss, Gershon", "text": weiss}]}
    result = C.process_document(doc)
    assert result["recon_frac"] > 0.5 and "(" not in result["cleaned_text"]
