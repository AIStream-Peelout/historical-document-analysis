# File name: test_semantic_v4.py
# Date: 10/9/26 (genizah_search v4 block mirrored 10/10/26)
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""semantic_v4 text and eligibility (src/embeddings/semantic_text.py), and the golden fixtures of v3 and v4.

The v4 block (the ``semantic_v4 begin`` line through the ``semantic_v4 end`` line) is a byte-for-byte copy of the
genizah_search reference, evals/embedding_audit/semantic_text.py (V10_EMBEDDER_PLAN.md B7: that copy leads and this
one mirrors it; mirrored 2026-10-10 from the eval-retrieval working copy, mtime 2026-10-09 16:36:44 EDT). The pins
below fail on any change to the block, its constants or its data files, and the golden fixtures
(tests/data/semantic_v{3,4}_golden.jsonl, ~200 stratified real records written by
src/embeddings/build_golden_fixtures.py) fail on any change to what the functions produce.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

import pytest

from src.embeddings import semantic_text as st
from src.embeddings.build_golden_fixtures import dedupe_spans_parts

DATA_DIR = Path(__file__).resolve().parent / "data"
V4_GOLDEN = DATA_DIR / "semantic_v4_golden.jsonl"
V3_GOLDEN = DATA_DIR / "semantic_v3_golden.jsonl"
REASONS = {"empty", "short", "boilerplate_only", "eligible"}
# fields every fixture row must carry; further fields (``stratum``, the genizah_search fixture's own) are allowed
V4_REQUIRED_FIELDS = {"doc_id", "v8_text", "v4_text", "v4_sha256", "eligible", "reason"}
V3_REQUIRED_FIELDS = {"doc_id", "v8_text", "v3_sha256", "reason"}
# sha256 of the fixture files as build_golden_fixtures.py wrote them (seed 13, 2026-10-10): a regenerated fixture is
# re-pinned here deliberately, a hand-edited one fails
V4_GOLDEN_SHA256 = "85a5aee7a3ce2dec54b25e835eb8a31bd7611c76d5cb7867722f095226894096"
V3_GOLDEN_SHA256 = "f89c636a01fce573b822ab65da7d6b42e225b1aed5493b4d4cb66a0ca95213bd"

# sha256 of the v4 block ("\n"-joined lines from the begin marker through the end marker, as the reference's
# v4_block_source() cuts it): the genizah_search block mirrored on 2026-10-10
V4_BLOCK_SHA256 = "721e3db0c5a0612b6288179fa9e6bbbacfc925d5d94f9b018028bd8f1a4770a2"
V4_BLOCK_BEGIN = "# semantic_v4 begin"
V4_BLOCK_END = "# semantic_v4 end"
# sha256 of f"{flags}:{pattern}" (the style of tests/test_semantic_text.py's REFERENCE_REGEX_FINGERPRINTS)
TITLE_KEY_RE_FINGERPRINT = "efe4822e065bee69d292201922713fa6927036c8fdff7b27d2bb814792da8fe9"
# sha256 of the two data files (byte-identical copies of the genizah_search files, which pin the same digests)
BOILERPLATE_FILE_SHA256 = "d57204f7278cc38bc713d54c05533fe87a40c79e3110cbbd51e42cb10e372677"
ADMIT_FILE_SHA256 = "2922b7747033c4d1d9b9845fee01b7c8fc0228b532d45c0e359db5786637ec01"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# the frozen label "טוביה בן אליעזר, לקח טוב: ..." and its KTIV scholarly frame
LEQAKH_TOV_LABEL = "טוביה בן אליעזר, לקח טוב: אסתר, רות, שיר השירים, קהלת ואיכה"
LEQAKH_TOV_FRAME = "[טוביה בן אליעזר, לקח טוב]: אסתר, רות, שיר השירים, קהלת ואיכה"


def load_rows(path: Path) -> List[Dict[str, str]]:
    """Read a golden fixture.

    :param path: JSONL file.
    :return: Its rows.
    :rtype: List[Dict[str, str]]
    """
    # one row per "\n"-terminated line (str.splitlines() would also split on U+2028 inside a text)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def v4_block(path: Path) -> str:
    """The ``semantic_v4`` block of a module: its begin-marker line through its end-marker line, joined by ``\\n``.

    :param path: Module file.
    :return: The block.
    :rtype: str
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    begin = [i for i, ln in enumerate(lines) if ln.startswith(V4_BLOCK_BEGIN)]
    end = [i for i, ln in enumerate(lines) if ln.startswith(V4_BLOCK_END)]
    assert len(begin) == 1 and len(end) == 1 and begin[0] < end[0]
    return "\n".join(lines[begin[0]:end[0] + 1])


def write_admit(path: Path, festival_terms: Dict[str, str], topic_rules: Dict[str, str], flags: Dict[str, str]) -> Path:
    """Write an admission file in the reference format.

    :param path: Destination.
    :param festival_terms: ``{topic: regex}``.
    :param topic_rules: ``{topic: regex}``.
    :param flags: ``{"festival_terms": "i" or "", "topic_rules": "i" or ""}``.
    :return: The path.
    :rtype: Path
    """
    data = {"festival_terms": festival_terms, "topic_rules": topic_rules, "flags": flags, "source": "Baz", "version": 1}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- pins (plan B7) -------------------------------------------------------------------------------------------------

def test_the_v4_block_is_the_mirrored_reference():
    assert hashlib.sha256(v4_block(Path(st.__file__)).encode("utf-8")).hexdigest() == V4_BLOCK_SHA256


def test_title_key_regex_fingerprint():
    rx = st.TITLE_KEY_RE
    assert hashlib.sha256(f"{rx.flags}:{rx.pattern}".encode("utf-8")).hexdigest() == TITLE_KEY_RE_FINGERPRINT


def test_v4_constants_are_pinned():
    assert st.FIELD_PREFIXES == ("Document ID:", "Shelf Mark:", "Description:", "Alternate titles:", "Subjects:",
                                 "Title:", "Summary:", "Language:", "Document Type:", "Date:", "Transcription:",
                                 "Translation:", "Related People:", "Related Places:", "Physical:", "Additional Info:")
    assert st.V4_DROP_PREFIXES == ("Subjects:", "Document Type:")
    assert st.TITLE_SEP == "; "
    assert st.MAX_ALT_CHARS == 1000
    assert st.MIN_CHARS == 40
    assert st.DESC_PREFIX == "Description:"
    assert st.SUBJECTS_PREFIX == "Subjects:"
    assert st.ALT_PREFIX == "Alternate titles:"
    assert st.META_PREFIXES == ("Language:", "Document Type:", "Date:")
    assert (st.TEXT_VERSION_SEMANTIC_V4, st.TEXT_VERSION_SEMANTIC_V4_NF) == ("semantic_v4", "semantic_v4_nf")
    assert (st.v4_text_version(True), st.v4_text_version(False)) == ("semantic_v4", "semantic_v4_nf")


def test_data_files_still_hash_to_the_pinned_digests():
    assert st.BOILERPLATE_SHA256 == BOILERPLATE_FILE_SHA256
    assert st.ADMIT_SHA256 == ADMIT_FILE_SHA256
    assert hashlib.sha256(st.BOILERPLATE_PATH.read_bytes()).hexdigest() == st.BOILERPLATE_SHA256
    assert hashlib.sha256(st.ADMIT_PATH.read_bytes()).hexdigest() == st.ADMIT_SHA256
    assert st.file_sha256(st.ADMIT_PATH) == st.ADMIT_SHA256
    assert st.ADMIT_PATH.parent == st.BOILERPLATE_PATH.parent and st.ADMIT_PATH.name == "eligibility_admit_v4.json"


def test_frozen_inputs_are_checked_against_their_pins(tmp_path):
    assert st.frozen_v4_labels() == st.load_boilerplate_labels() and len(st.frozen_v4_labels()) == 64
    assert st.frozen_v4_labels() is st.frozen_v4_labels()                 # checked and loaded once
    assert st.frozen_admit_pattern().pattern == st.admit_pattern(st.ADMIT_PATH).pattern
    assert st.verify_frozen_file(st.ADMIT_PATH, st.ADMIT_SHA256) == st.ADMIT_PATH
    edited = tmp_path / "eligibility_admit_v4.json"
    edited.write_bytes(st.ADMIT_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned"):
        st.verify_frozen_file(edited, st.ADMIT_SHA256)


def test_admission_file_patterns_and_composition(tmp_path):
    data = json.loads(st.ADMIT_PATH.read_text(encoding="utf-8"))
    assert len(data["festival_terms"]) == 10 and len(data["topic_rules"]) == 19
    assert data["flags"] == {"festival_terms": "i", "topic_rules": ""}
    assert st.admit_pattern() is st.admit_pattern()  # compiled once
    # one group per pattern, festival terms first, each block in key order with its own case flag; nothing else is read
    rx = st.admit_pattern(write_admit(tmp_path / "a.json", {"b": "foo", "a": "qux"}, {"x": "Bar"},
                                      {"festival_terms": "i", "topic_rules": ""}))
    assert rx.pattern == "(?i:qux)|(?i:foo)|(?:Bar)"
    assert rx.search("a FOO b") and rx.search("QUX") and rx.search("a Bar b")
    assert not rx.search("bar") and not rx.search("Baz")
    flipped = st.admit_pattern(write_admit(tmp_path / "b.json", {"b": "foo"}, {"x": "Bar"},
                                           {"festival_terms": "", "topic_rules": "i"}))
    assert not flipped.search("FOO") and flipped.search("bar")


# --- golden fixtures (plan B7) --------------------------------------------------------------------------------------

def test_golden_fixture_files_are_the_pinned_ones():
    assert hashlib.sha256(V4_GOLDEN.read_bytes()).hexdigest() == V4_GOLDEN_SHA256
    assert hashlib.sha256(V3_GOLDEN.read_bytes()).hexdigest() == V3_GOLDEN_SHA256


def test_v4_golden_fixture_matches_the_live_functions():
    rows = load_rows(V4_GOLDEN)
    assert 150 <= len(rows) <= 300
    assert all(V4_REQUIRED_FIELDS <= set(row) for row in rows)
    assert {row["reason"] for row in rows} == REASONS
    # every row agrees with itself: the stored text hashes to the stored sha, the flag follows the reason
    assert all(st.text_sha256(row["v4_text"]) == row["v4_sha256"] for row in rows)
    assert all(row["eligible"] is (row["reason"] == "eligible") for row in rows)
    drifted = []
    for row in rows:
        v3 = st.semantic_text(row["v8_text"])
        got = (st.text_sha256(st.semantic_v4_text(v3)), st.semantic_v4_ineligible_reason(v3))
        if got != (row["v4_sha256"], row["reason"]) or st.semantic_v4_record(row["v8_text"])[:2] != (
                st.semantic_v4_text(v3), got[1]):
            drifted.append((row["doc_id"], row["reason"], got[1]))
    assert not drifted, f"{len(drifted)} of {len(rows)} semantic_v4 rows changed: {drifted[:10]}"


def test_v3_golden_fixture_matches_the_live_functions():
    rows = load_rows(V3_GOLDEN)
    assert 150 <= len(rows) <= 300
    assert all(V3_REQUIRED_FIELDS <= set(row) for row in rows)
    assert {row["reason"] for row in rows} == REASONS
    drifted = []
    for row in rows:
        v3 = st.semantic_text(row["v8_text"])
        got = (st.text_sha256(v3), st.semantic_ineligible_reason(v3))
        if got != (row["v3_sha256"], row["reason"]) or st.is_semantic_eligible(v3) != (row["reason"] == "eligible"):
            drifted.append((row["doc_id"], row["reason"], got[1]))
    assert not drifted, f"{len(drifted)} of {len(rows)} semantic_v3 rows changed: {drifted[:10]}"


def test_golden_fixtures_hold_the_same_records_and_cover_the_v4_branches():
    v4_rows, v3_rows = load_rows(V4_GOLDEN), load_rows(V3_GOLDEN)
    assert [(r["doc_id"], r["v8_text"]) for r in v4_rows] == [(r["doc_id"], r["v8_text"]) for r in v3_rows]
    assert [r.get("stratum") for r in v4_rows] == [r.get("stratum") for r in v3_rows]
    assert len({r["doc_id"] for r in v4_rows}) == len(v4_rows)
    v3_texts = [st.semantic_text(r["v8_text"]) for r in v4_rows]
    # continuation lines, an Alternate titles line with a bracketed frame, and the dropped lines are all exercised
    assert any(len(st.fold_field_lines(t).split("\n")) < len(t.split("\n")) for t in v3_texts)
    assert any(ln.startswith(st.ALT_PREFIX) and "[" in ln.replace(st.MASK, "")
               for t in v3_texts for ln in t.split("\n"))
    assert any(ln.startswith(st.V4_DROP_PREFIXES) for t in v3_texts for ln in t.split("\n"))
    # real records whose kept parts span two earlier keys: they tell this block from the audit's draft
    assert any(dedupe_spans_parts(t) for t in v3_texts)
    # admitted short records, and records eligible under v3 that v4 does not admit
    transitions = {(v3_row["reason"], v4_row["reason"]) for v3_row, v4_row in zip(v3_rows, v4_rows)}
    assert ("short", "eligible") in transitions
    assert {("eligible", "short"), ("eligible", "boilerplate_only")} & transitions


# --- the layout -----------------------------------------------------------------------------------------------------

def test_text_sha256_of_empty_and_known_strings():
    assert st.text_sha256("") == EMPTY_SHA256
    assert st.text_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    # the UTF-8 bytes, no normalisation
    assert st.text_sha256("הגדה של פסח") == "0c89e30e85a3a3541a25bf75d1993d85efa1a21ef142eb18ee7effe0dd6c5ef0"


def test_title_key():
    assert st.title_key("[Talmud Bavli]: Sukkah 29 a – b") == "talmud bavli sukkah 29 a b"
    assert st.title_key("  SNAKE_case—and   spaces ") == "snake case and spaces"
    assert st.title_key("...") == ""


def test_continuation_lines_fold_into_the_field_line_before_them():
    v3 = ("Description: Letter about the flax trade.\n   Second paragraph of the description.\nverso: a list\n"
          "Language: Judaeo-Arabic")
    folded = "Description: Letter about the flax trade. Second paragraph of the description. verso: a list"
    assert st.fold_field_lines(v3) == f"{folded}\nLanguage: Judaeo-Arabic"
    assert st.semantic_v4_text(v3) == f"{folded}\nLanguage: Judaeo-Arabic"
    # a first line without a label has nothing to join and stays a line of its own
    assert st.fold_field_lines("a note without a label\nDescription: x") == "a note without a label\nDescription: x"
    # blank continuation lines are dropped, the field line's trailing spaces too
    assert st.fold_field_lines("Description: a  \n\n  b\nLanguage: Hebrew") == "Description: a b\nLanguage: Hebrew"


def test_subjects_and_document_type_lines_are_dropped_with_their_continuations():
    v3 = ("Description: Letter about flax.\nSubjects: trade;\nflax\nLanguage: Judaeo-Arabic\nDocument Type: letter\n"
          "Date: standard_date: 1060")
    assert st.semantic_v4_text(v3) == ("Description: Letter about flax.\nLanguage: Judaeo-Arabic\n"
                                       "Date: standard_date: 1060")


def test_a_text_with_nothing_to_lay_out_is_unchanged():
    v3 = "Description: Letter from Nahray b. Nissim about the flax trade.\nLanguage: Judaeo-Arabic"
    assert st.semantic_v4_text(v3) == v3
    assert st.merge_title_lines(v3.split("\n")) == v3.split("\n")


def test_alternate_titles_fold_into_the_description_with_whole_word_dedupe():
    v3 = ("Description: Piyyut for Sukkot.\n"
          "Alternate titles: PIYYUT; for Sukkot; Hoshana for the seventh day; sukkot; Hoshana for the seventh day!; "
          "[Piyyut]; ...; Piyyutim\n"
          "Language: Hebrew")
    # parts already present as whole words (case, brackets and punctuation aside) are left out, "Piyyutim" is not;
    # ".; " is the cosmetic join of a description ending in a full stop
    assert st.semantic_v4_text(v3) == ("Description: Piyyut for Sukkot.; Hoshana for the seventh day; Piyyutim\n"
                                       "Language: Hebrew")
    assert st.title_parts(v3.split("\n")) == ["PIYYUT", "for Sukkot", "Hoshana for the seventh day", "sukkot",
                                              "Hoshana for the seventh day!", "[Piyyut]", "...", "Piyyutim"]
    # (kept parts, parts left out as duplicates, parts left out by the cap)
    assert st.merge_title_parts("Piyyut for Sukkot.", st.title_parts(v3.split("\n"))) == (
        ["Hoshana for the seventh day", "Piyyutim"], 6, 0)


def test_a_bracketed_frame_restating_an_alternate_title_is_dropped():
    framed = f"Description: לקח טוב\nAlternate titles: {LEQAKH_TOV_LABEL}; {LEQAKH_TOV_FRAME}"
    assert st.semantic_v4_text(framed) == f"Description: לקח טוב; {LEQAKH_TOV_LABEL}"


def test_a_part_whose_words_span_two_earlier_keys_is_kept():
    """Each earlier key is compared on its own (the reference block), so a part made of the end of one earlier key and
    the start of the next is new (14 real records on 2026-10-10, e.g. a haftarah list and a Talmud folio range); the
    audit's draft looked it up in the earlier keys joined by spaces and left it out."""
    haftarot = "Description: S.\nAlternate titles: 3:4; 2:5 – 11; 4:2"
    assert st.semantic_v4_text(haftarot) == "Description: S.; 3:4; 2:5 – 11; 4:2"
    folio = ("Description: Talmud Bavli: Bava Metzi'a 37 b\n"
             "Alternate titles: 38 a; [Talmud Bavli]: Bava Metzi'a 37 b – 38 a")
    assert st.semantic_v4_text(folio) == ("Description: Talmud Bavli: Bava Metzi'a 37 b; 38 a; "
                                          "[Talmud Bavli]: Bava Metzi'a 37 b – 38 a")
    assert dedupe_spans_parts(st.semantic_text(haftarot)) and dedupe_spans_parts(folio)


def test_the_alternate_titles_cap_keeps_whole_parts():
    first, second = "a" * 600, "b" * 600
    v3 = f"Description: Short.\nAlternate titles: {first}; {second}; tiny"
    # appending stops at the first part that does not fit: "tiny" is not appended either
    assert st.semantic_v4_text(v3) == f"Description: Short.; {first}"
    assert st.merge_title_parts("Short.", [first, second, "tiny"]) == ([first], 0, 2)
    lone = "c" * (st.MAX_ALT_CHARS + 50)  # the first part is always kept
    assert st.semantic_v4_text(f"Description: Short.\nAlternate titles: {lone}") == f"Description: Short.; {lone}"
    # each kept part counts with its separator: 500 + 2 + 498 = MAX_ALT_CHARS still fits, 499 does not
    d, e = "d" * 500, "e" * (st.MAX_ALT_CHARS - 502)
    assert st.semantic_v4_text(f"Description: S.\nAlternate titles: {d}; {e}; f") == f"Description: S.; {d}; {e}"
    assert st.semantic_v4_text(f"Description: S.\nAlternate titles: {d}; {e}e; f") == f"Description: S.; {d}"


def test_every_alternate_titles_line_is_merged_at_the_description():
    v3 = "Alternate titles: A\nDescription: D\nLanguage: Hebrew\nAlternate titles: B; a"
    assert st.semantic_v4_text(v3) == "Description: D; A; B\nLanguage: Hebrew"


def test_alternate_titles_without_a_description_become_the_description_line():
    v3 = "Alternate titles: Hoshana for Sukkot; Piyyut; hoshana for sukkot\nLanguage: Hebrew"
    assert st.semantic_v4_text(v3) == "Description: Hoshana for Sukkot; Piyyut\nLanguage: Hebrew"
    # nothing left to say: the line goes
    assert st.merge_title_lines(["Alternate titles: ; ...", "Language: Hebrew"]) == ["Language: Hebrew"]


def test_empty_text():
    assert st.semantic_v4_text("") == ""
    assert st.fold_field_lines("") == ""
    assert st.merge_title_lines([]) == []
    assert st.semantic_v4_ineligible_reason("") == "empty"
    assert st.v4_measured_text("") == ""


# --- eligibility ----------------------------------------------------------------------------------------------------

def test_v4_reasons_and_the_length_boundary():
    assert st.semantic_v4_ineligible_reason("") == "empty"
    thirty_nine = "Description: " + "x" * 26
    assert st.semantic_v4_ineligible_reason(thirty_nine) == "short"
    assert st.semantic_v4_ineligible_reason(thirty_nine + "x") == "eligible"
    boiler = "Description: Newly treated and encapsulated, must be examined\nLanguage: Hebrew"
    assert st.semantic_v4_ineligible_reason(boiler) == "boilerplate_only"
    assert st.semantic_v4_ineligible_reason(boiler, labels=frozenset()) == "eligible"  # the label makes it ineligible
    assert st.semantic_v4_ineligible_reason(
        "Description: Letter from Nahray b. Nissim to his partner about the flax trade.") == "eligible"
    # a text that is nothing but Subjects lines is empty once they are left out
    assert st.semantic_v4_ineligible_reason("Subjects: Passover; Haggadah") == "empty"


def test_v4_measured_text_leaves_out_subjects_lines_and_their_continuations():
    v3 = "Description: Note\nSubjects: Passover;\nHaggadah\nLanguage: Hebrew"
    assert st.v4_measured_text(v3) == "Description: Note\nLanguage: Hebrew"
    no_subjects = "Description: Letter about flax.\nverso: a list\nLanguage: Hebrew"
    assert st.v4_measured_text(no_subjects) == no_subjects


def test_a_subjects_line_never_makes_a_record_eligible_on_its_own():
    keywords_only = ("Description: קטע גניזה\nSubjects: Rabbinic Literature; Talmud Bavli; Talmud Bavli: Texts and "
                     "Anthologies\nLanguage: Aramaic\nDocument Type: talmud")
    assert st.semantic_v4_ineligible_reason(keywords_only) == "boilerplate_only"
    assert st.semantic_ineligible_reason(keywords_only) == "eligible"  # v3 counts the line (frozen)
    with_frame = keywords_only + "\nAlternate titles: [Talmud Bavli]: Sukkah 29 a – b"
    assert st.semantic_v4_ineligible_reason(with_frame) == "eligible"
    assert st.content_lines_v4(st.v4_measured_text(with_frame), st.frozen_v4_labels()) == [
        "Alternate titles: [Talmud Bavli]: Sukkah 29 a – b"]


def test_a_subjects_line_does_not_lift_a_short_text_over_the_length_floor():
    short = "Description: אלמקדמאת\nSubjects: Karaites-Sermons; Karaites"
    assert len(short) >= st.MIN_CHARS
    assert st.semantic_v4_ineligible_reason(short) == "short"
    assert st.semantic_ineligible_reason(short) == "eligible"  # v3 measures the whole text (frozen)
    # a festival named only in the Subjects line, or in its continuation line, does not admit the record either
    assert st.semantic_v4_ineligible_reason("Description: אלמקדמאת\nSubjects: Passover; Haggadah") == "short"
    assert st.semantic_v4_ineligible_reason("Description: אלמקדמאת\nSubjects: Karaites;\nPassover") == "short"
    assert st.semantic_v4_ineligible_reason("Description: Letter from Nahray b. Nissim about the flax trade.\n"
                                            "Subjects: trade; flax") == "eligible"


def test_a_bracketed_frame_restating_a_frozen_label_is_still_boilerplate():
    assert LEQAKH_TOV_LABEL in st.load_boilerplate_labels()
    assert st.unbracketed(f" {LEQAKH_TOV_FRAME} ") == LEQAKH_TOV_LABEL
    framed = f"Description: לקח טוב\nAlternate titles: {LEQAKH_TOV_LABEL}; {LEQAKH_TOV_FRAME}"
    assert st.semantic_v4_ineligible_reason(framed) == "boilerplate_only"
    assert st.semantic_ineligible_reason(framed) == "eligible"  # v3 compares the bracketed part as written (frozen)
    assert st.semantic_v4_ineligible_reason(framed + "; [Talmud Bavli]: Sukkah 29 a – b") == "eligible"


def test_a_short_text_naming_a_festival_is_admitted():
    haggadah = "Description: הגדה של פסח (קטע)"  # Cambridge_CUL_Or_1081_2_33 at HDA HEAD (text-pipeline.md §4)
    assert len(haggadah) < st.MIN_CHARS
    assert st.semantic_ineligible_reason(haggadah) == "short"
    assert st.semantic_v4_ineligible_reason(haggadah) == "eligible"
    assert st.semantic_v4_ineligible_reason("Description: A note on a page") == "short"


@pytest.mark.parametrize("v3_text, reason", [
    ("Description: PASSOVER note", "eligible"),          # festival terms ignore case
    ("Description: Shabbat note", "eligible"),           # topic rules are case-sensitive ...
    ("Description: shabbat note", "short"),              # ... so the lower-case form is not admitted
    ("Description: Note\nTranscription: פסח", "eligible"),  # any content line's value counts
    ("Description: Note\nפסח", "eligible"),               # an unlabelled continuation line counts as written
    ("Description: Note\nDate: Passover", "short"),      # metadata lines never admit
])
def test_admission_reads_the_values_of_content_lines(v3_text, reason):
    assert len(v3_text) < st.MIN_CHARS
    assert st.semantic_v4_ineligible_reason(v3_text) == reason


def test_explicit_labels_and_admission_pattern():
    note = "Description: A note on a page"
    assert st.semantic_v4_ineligible_reason(note, admit=re.compile("note")) == "eligible"
    assert st.semantic_v4_ineligible_reason(note, labels=frozenset({"A note on a page"}), admit=re.compile("note")) == (
        "short")                                                  # a label is no content, so nothing is admitted
    # labelled lines give their value, an unlabelled continuation line its whole stripped text
    assert st.admission_value(["Description: Note", "Transcription:  פסח ", "  verso: a list "]) == (
        "Note ; פסח ; verso: a list")


def test_semantic_v4_record_returns_the_v4_text_its_reason_and_the_v3_text():
    v8 = "Document ID: X_1\nShelf Mark: T-S NS 1.1\nDescription: הגדה של פסח (קטע)\nSubjects: Passover"
    v3 = "Description: הגדה של פסח (קטע)\nSubjects: Passover"
    assert st.semantic_v4_record(v8) == ("Description: הגדה של פסח (קטע)", "eligible", v3)
