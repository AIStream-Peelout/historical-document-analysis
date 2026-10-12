# File name: test_semantic_text.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""semantic_v3 document text (src/embeddings/semantic_text.py), the text the fine-tuned embedder was trained on.

Expected outputs below were produced by the reference implementation (genizah_search
evals/embedding_audit/semantic_text.py, branch eval-retrieval) on the same inputs. The v3 functions are frozen: the
2026-10-09 rules (Subjects lines, brackets in alternate titles, the admission list) belong to semantic_v4
(tests/test_semantic_v4.py, which also checks both golden fixtures).
"""

import hashlib
import json

import pytest

from src.embeddings import semantic_text as st

# sha256 of f"{flags}:{pattern}" for each regex of the reference module; any edit to a regex breaks parity with
# training and the frozen eval, so it must fail here (make a new text version instead).
REFERENCE_REGEX_FINGERPRINTS = {
    "EDITOR_RE": "becd3b2dc5e5b8e8c9e4543252d78014ade83aa22c0eb69b8762fc917d520862",
    "PGP_EDITOR_RE": "fad5fc8f50700d150f5ef1706de32d5e0a4a20c458274b2f312f13385ac160e3",
    "EMPTY_FIELD_RE": "18eb71a717c96ebe82baead5b8cb0b0cb9c816398fd3182655e0cd8be90bee28",
    "PROD_SHELFMARK_RE": "3e6644954737692308a03b5206876586b1b115c6f1274909800fc3c6d74881f3",
    "EXTRA_ID_RE": "357b0ce63477fcbab8469355d5f6704ca37d8f12af0f18ad8035ae4477c13b86",
    "PRE_ID_RE": "29e437d35107e97d05edd7c7897af8bf99024b9b2ddc10e347f686cf2d8ec2e4",
}
# sha256 of the frozen boilerplate file as handed off (docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json).
BOILERPLATE_FILE_SHA256 = "d57204f7278cc38bc713d54c05533fe87a40c79e3110cbbd51e42cb10e372677"


@pytest.mark.parametrize("name", sorted(REFERENCE_REGEX_FINGERPRINTS))
def test_regexes_identical_to_reference(name):
    rx = getattr(st, name)
    assert hashlib.sha256(f"{rx.flags}:{rx.pattern}".encode("utf-8")).hexdigest() == REFERENCE_REGEX_FINGERPRINTS[name]


def test_constants_match_reference():
    assert st.ID_PREFIXES == ("Document ID:", "Shelf Mark:")
    assert st.META_PREFIXES == ("Language:", "Document Type:", "Date:")
    assert st.ALT_PREFIX == "Alternate titles:"
    assert st.MIN_CHARS == 40
    assert st.MASK == "[shelfmark]"


def test_boilerplate_file_is_the_frozen_list():
    assert hashlib.sha256(st.BOILERPLATE_PATH.read_bytes()).hexdigest() == BOILERPLATE_FILE_SHA256
    data = json.loads(st.BOILERPLATE_PATH.read_text(encoding="utf-8"))
    labels = st.load_boilerplate_labels()
    assert len(labels) == len(data["boilerplate"]) == 64
    assert "Newly treated and encapsulated, must be examined" in labels
    assert st.load_boilerplate_labels() is labels  # loaded once


def test_boilerplate_file_must_match_min_chars(tmp_path):
    bad = tmp_path / "labels.json"
    bad.write_text(json.dumps({"min_chars": 30, "boilerplate": ["Piyyut"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="min_chars"):
        st.load_boilerplate_labels(bad)


def test_drops_id_lines_and_masks_own_cited_marks_and_pgpids():
    text = ("Document ID: Cambridge_CUL_T_S_13J8_14\nShelf Mark: T-S 13J8.14\n"
            "Description: Letter from Nahray b. Nissim. Join with T-S 13J8.14 and ENA 2558.5; PGPID 12345.\n"
            "Language: Judaeo-Arabic\n")
    assert st.semantic_text(text) == ("Description: Letter from Nahray b. Nissim. Join with [shelfmark] and "
                                      "[shelfmark]; [shelfmark].\nLanguage: Judaeo-Arabic")


def test_own_mark_is_masked_whole_token_only():
    text = ("Shelf Mark: Budapest Kaufmann 123\n"
            "Description: Budapest Kaufmann 123 and Budapest Kaufmann 1234 are different fragments.")
    assert st.semantic_text(text) == "Description: [shelfmark] and Budapest Kaufmann 1234 are different fragments."


def test_own_mark_from_argument_or_line():
    text = "Description: Budapest Kaufmann 123 recto, a list of names."
    assert st.semantic_text(text) == text  # no Shelf Mark line, no regex match: left alone
    assert st.semantic_text(text, own_shelfmark=" Budapest Kaufmann 123 ") == (
        "Description: [shelfmark] recto, a list of names.")
    with_line = "Shelf Mark: Budapest Kaufmann 123\n" + text
    assert st.semantic_text(with_line) == st.semantic_text(with_line, own_shelfmark="Budapest Kaufmann 123")


def test_masks_formats_beyond_the_production_pattern():
    text = "Description: Joins Mosseri VII 155, CUL Add.3339a, T-S NS J401.21, Bodl. Heb. f. 26/1-6 and PGPID: 4567."
    assert st.semantic_text(text) == (
        "Description: Joins [shelfmark], [shelfmark], [shelfmark], [shelfmark] and [shelfmark].")


@pytest.mark.parametrize("transcription", [
    "[Editor: KTIV Academic transcription (Frag. 001r)] בשם רחמנא",
    "בשם רחמנא [Editor: KTIV Acad...",
    "Editor: Gil, Moshe Editor: Goitein, S. D. בשם רחמנא",
    "Editors: Goitein, S. D.; Gil, Moshe בשם רחמנא",
    "Editors: [unknown source] בשם רחמנא",
])
def test_strips_editor_credits(transcription):
    text = f"Document ID: X1\nDescription: Marriage contract.\nTranscription: {transcription}"
    assert st.semantic_text(text) == "Description: Marriage contract.\nTranscription: בשם רחמנא"


def test_drops_field_lines_left_empty():
    text = ("Description: Letter about trade in flax and indigo.\n"
            "Transcription: [Editor: KTIV Academic transcription (Frag. 001r)]\nTranslation:   \n\n"
            "Related People: Nahray b. Nissim")
    assert st.semantic_text(text) == ("Description: Letter about trade in flax and indigo.\n"
                                      "Related People: Nahray b. Nissim")


def test_only_id_lines_leaves_nothing():
    assert st.semantic_text("Document ID: Cambridge_CUL_T_S_NS_1_1\nShelf Mark: T-S NS 1.1") == ""


def test_eligibility_empty_and_length_boundary():
    assert not st.is_semantic_eligible("")
    assert st.semantic_ineligible_reason("") == "empty"
    thirty_nine = "Description: " + "x" * 26
    forty = thirty_nine + "x"
    assert len(thirty_nine) == 39 and len(forty) == 40
    assert not st.is_semantic_eligible(thirty_nine)
    assert st.semantic_ineligible_reason(thirty_nine) == "short"
    assert st.is_semantic_eligible(forty)


def test_boilerplate_label_plus_metadata_is_ineligible():
    stripped = st.semantic_text(
        "Document ID: X\nShelf Mark: T-S NS 1.1\nDescription: Newly treated and encapsulated, must be examined\n"
        "Language: Hebrew\nDocument Type: letter\nDate: standard_date: 1100")
    assert len(stripped) >= st.MIN_CHARS
    assert not st.is_semantic_eligible(stripped)
    assert st.semantic_ineligible_reason(stripped) == "boilerplate_only"
    assert st.is_semantic_eligible(stripped, labels=[])  # it is the label that makes it ineligible


def test_alternate_titles_count_only_when_not_boilerplate():
    boiler_alts = "Description: Piyyut\nAlternate titles: Unidentified; פיוט\nLanguage: Hebrew\nDocument Type: liturgy"
    real_alts = ("Description: Piyyut\nAlternate titles: Unidentified; Hoshana for the seventh day of Sukkot\n"
                 "Language: Hebrew")
    assert not st.is_semantic_eligible(boiler_alts)
    assert st.is_semantic_eligible(real_alts)


def test_real_description_is_eligible_and_labels_can_be_passed():
    assert st.is_semantic_eligible(
        "Description: Letter from Nahray b. Nissim to his partner about the flax trade in Alexandria.")
    letter = "Description: Letter\nLanguage: Judaeo-Arabic\nDate: standard_date: 1050"
    assert not st.is_semantic_eligible(letter)
    assert not st.is_semantic_eligible(letter, labels=["Letter"])
    assert st.is_semantic_eligible(letter, labels=["Piyyut"])


def test_bibliography_variant_drops_only_the_source_pdf_line():
    page = ("Source PDF: cairo_to_manchester_1, Page: 58\nTitle: From Cairo to Manchester\n"
            "Shelf marks mentioned: B 2349-1, T-S 13J8.14\nSummary: Discusses T-S 13J8.14 and PGPID 12345.\n"
            "Main text: Source PDF: quoted in the text stays.")
    out = st.bibliography_semantic_text(page)
    assert out == page.split("\n", 1)[1]
    assert "Shelf marks mentioned: B 2349-1, T-S 13J8.14" in out
    assert "Discusses T-S 13J8.14 and PGPID 12345." in out and st.MASK not in out
    assert out.endswith("Main text: Source PDF: quoted in the text stays.")


def test_bibliography_variant_edge_cases():
    assert st.bibliography_semantic_text("Source PDF: x, Page: 1") == ""
    no_line = "Title: A book\nSummary: A page."
    assert st.bibliography_semantic_text(no_line) == no_line


def test_v3_counts_a_subjects_line_as_content():
    """Frozen reference behaviour: v3 knows no Subjects line, so one with non-boilerplate words is content."""
    keywords_only = ("Description: קטע גניזה\nSubjects: Rabbinic Literature; Talmud Bavli; Talmud Bavli: Texts and "
                     "Anthologies\nLanguage: Aramaic\nDocument Type: talmud")
    assert st.is_semantic_eligible(keywords_only)
    assert st.semantic_ineligible_reason(keywords_only) == "eligible"
    assert st.content_lines(keywords_only, st.load_boilerplate_labels()) == [
        "Subjects: Rabbinic Literature; Talmud Bavli; Talmud Bavli: Texts and Anthologies"]
    # without the line the same record is boilerplate: the label "קטע גניזה" plus metadata
    label_only = "Description: קטע גניזה\nLanguage: Aramaic\nDocument Type: talmud"
    assert st.semantic_ineligible_reason(label_only) == "boilerplate_only"


def test_v3_compares_bracketed_alternate_titles_as_written():
    """Frozen reference behaviour: brackets are not stripped, so a bracketed restatement of a label is content."""
    label = "טוביה בן אליעזר, לקח טוב: אסתר, רות, שיר השירים, קהלת ואיכה"
    assert label in st.load_boilerplate_labels()
    plain = f"Description: לקח טוב\nAlternate titles: {label}"
    assert st.semantic_ineligible_reason(plain) == "boilerplate_only"
    framed = plain + "; [טוביה בן אליעזר, לקח טוב]: אסתר, רות, שיר השירים, קהלת ואיכה"
    assert st.is_semantic_eligible(framed)
    assert st.content_lines(framed, st.load_boilerplate_labels()) == [framed.split("\n")[1]]


def test_v3_measures_the_whole_text_against_the_length_floor():
    """Frozen reference behaviour: every line, a Subjects line included, counts towards MIN_CHARS."""
    short = "Description: אלמקדמאת\nSubjects: Karaites-Sermons; Karaites"
    assert len(short.split("\n")[0]) < st.MIN_CHARS <= len(short)
    assert st.semantic_ineligible_reason(short) == "eligible"
    assert st.is_semantic_eligible(short)
    assert st.semantic_ineligible_reason(short.split("\n")[0]) == "short"
