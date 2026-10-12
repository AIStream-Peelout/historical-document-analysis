"""KTIV label titles: headings become subjects, the catalogued varying title becomes the description (2026-10-11).

Background: NLI titles most Genizah fragments with subject headings ("פיוט;תפילה וברכות;תפילות קבע"); the merge used
that title as ``description`` for 16,894 records, so result cards showed headings and the embedded text opened with
them while the identification ("Common Prayers: Sukkot Amidah") sat unseen in the varying titles.
"""
from src.datasets.document_models.genizah_document import (
    GenizahDocument,
    _merged_subjects,
    is_ktiv_label_title,
    ktiv_heading_parts,
    ktiv_varying_title,
    merged_description,
)
from src.datasets.merging.merge_shelfmarks import build_merged_record, ktiv_description
from src.embeddings.text_representation import embedding_text

SUKKOT_EN = "Title in English: Common Prayers: Sukkot Amidah"
SUKKOT_HE = "תפילות קבע: סוכות עמידה"
LABELS = "פיוט;תפילה וברכות;תפילות קבע"


def _ktiv(title, varying=(), subjects=("Cairo Genizah fragments",), entries=()):
    return {
        "source": "nli_ktiv", "sys_num": "990051189760205171", "pnx_id": "PNX_MANUSCRIPTS990051189760205171",
        "shelf_mark": "Cambridge University Library, Cambridge, England Ms. T-S H 7.28",
        "basic_catalog": {"title": title, "varying_form_of_title": list(varying), "date": "Unknown/לא ידוע",
                          "language": "Hebrew"},
        "full_catalog": {"subjects": list(subjects)},
        "scholarly_entries": list(entries),
    }


def _merged(ktiv, description=None, pgp=None):
    return {"canonical_id": "Cambridge_CUL_T_S_H7_28", "shelfmark_display": "T-S H 7.28",
            "description": ktiv["basic_catalog"]["title"] if description is None else description,
            "sources": {"ktiv": ktiv, "pgp": pgp, "fjp": []}, "images": {}}


def test_label_title_detection():
    for label in (LABELS, "תפילה וברכות;תפילות קבע", "קטע גניזה", "מקרא [טקסט", "משנה תורה", "הגדה של פסח (קטע)",
                  "מקרא [טקסט];תלמוד בבלי [טקסט"):
        assert is_ktiv_label_title(label), label
    for title in ("Common Prayers: Sukkot Amidah", "Talmud Bavli: Shabbat 56 b – 57 b",
                  "דיני שופר בראש השנה. הקטע לקוח מתוך סידור רס\"ג בעיבוד קל.", "תפילות קבע: סוכות", "", None):
        assert not is_ktiv_label_title(title), title


def test_heading_parts_strip_text_suffix_and_ignore_real_titles():
    assert ktiv_heading_parts(LABELS) == ["פיוט", "תפילה וברכות", "תפילות קבע"]
    assert ktiv_heading_parts("מקרא [טקסט];תלמוד בבלי [טקסט") == ["מקרא", "תלמוד בבלי"]
    assert ktiv_heading_parts("משנה [טקסט]") == ["משנה"]
    assert ktiv_heading_parts("Common Prayers: Sukkot Amidah") == []
    assert ktiv_heading_parts(None) == []


def test_varying_title_prefers_english_then_hebrew_and_skips_labels():
    assert ktiv_varying_title(_ktiv(LABELS, [SUKKOT_EN, SUKKOT_HE])) == "Common Prayers: Sukkot Amidah"
    assert ktiv_varying_title(_ktiv(LABELS, [SUKKOT_HE, SUKKOT_EN])) == "Common Prayers: Sukkot Amidah"
    assert ktiv_varying_title(_ktiv(LABELS, [SUKKOT_HE])) == SUKKOT_HE
    assert ktiv_varying_title(_ktiv("מקרא [טקסט", ["Title in English: Bible: Genesis 12:3 – 14:2"])) == "Bible: Genesis 12:3 – 14:2"
    for generic in ("Title in English: Piyyut", "Title in English: Mishneh Torah", "Title in English: Unidentified",
                    "Title in English: Talmud Bavli", "פיוט"):
        assert ktiv_varying_title(_ktiv(LABELS, [generic])) is None, generic
    assert ktiv_varying_title(_ktiv(LABELS, [])) is None
    assert ktiv_varying_title(None) is None


def test_merged_description_rules():
    assert merged_description(_merged(_ktiv(LABELS, [SUKKOT_EN, SUKKOT_HE]))) == "Common Prayers: Sukkot Amidah"
    assert merged_description(_merged(_ktiv(LABELS, [SUKKOT_HE]))) == SUKKOT_HE
    assert merged_description(_merged(_ktiv(LABELS, []))) == LABELS            # no usable varying title
    assert merged_description(_merged(_ktiv(LABELS, ["Title in English: Piyyut"]))) == LABELS
    real = _ktiv("Common Prayers: Sukkot Amidah", [SUKKOT_HE])
    assert merged_description(_merged(real)) == "Common Prayers: Sukkot Amidah"  # not a label: unchanged
    pgp = _merged(_ktiv(LABELS, [SUKKOT_EN]), description="Letter from Fustat (PGP)")
    assert merged_description(pgp) == "Letter from Fustat (PGP)"              # another source's description stands
    assert merged_description(_merged(_ktiv(LABELS, [SUKKOT_EN]), description="")) == ""
    assert merged_description({"description": LABELS, "sources": {}}) == LABELS  # no KTIV block


def test_headings_become_subjects_and_generic_hebrew_headings_are_dropped():
    assert _merged_subjects(_ktiv(LABELS, [SUKKOT_EN]), []) == ["פיוט", "תפילה וברכות", "תפילות קבע"]
    assert _merged_subjects(_ktiv("קטע גניזה;תחום לא מזוהה", []), []) == []
    assert _merged_subjects(_ktiv("מקרא [טקסט];תלמוד בבלי [טקסט", []), []) == ["מקרא", "תלמוד בבלי"]
    # A real title is not a heading.
    assert _merged_subjects(_ktiv("Common Prayers: Sukkot Amidah", []), []) == []


def test_merge_and_document_model_end_to_end():
    ktiv = _ktiv(LABELS, [SUKKOT_EN, SUKKOT_HE])
    rec = build_merged_record("Cambridge_CUL_T_S_H7_28", None, [], ktiv)
    assert rec["description"] == "Common Prayers: Sukkot Amidah"                # merge applies the rule
    assert ktiv_description(_ktiv("הגדה של פסח (קטע)", [])) == "הגדה של פסח (קטע)"  # label, no varying title: kept
    assert ktiv_description({}) is None

    doc = GenizahDocument.from_merged_format(rec)
    assert doc.description == "Common Prayers: Sukkot Amidah"
    assert doc.subjects == ["פיוט", "תפילה וברכות", "תפילות קבע"]
    assert doc.alt_titles == ["Common Prayers: Sukkot Amidah", SUKKOT_HE]
    text = embedding_text(doc, "semantic_v4", include_scholarly_frames=False)
    assert text.eligible
    first_line = text.text.splitlines()[0]
    assert first_line == f"Description: Common Prayers: Sukkot Amidah; {SUKKOT_HE}"  # title once, no heading prefix
    assert "פיוט;" not in text.text

    # An older merged file still carrying the label as description gets the same document.
    legacy = dict(rec, description=LABELS)
    assert GenizahDocument.from_merged_format(legacy).description == "Common Prayers: Sukkot Amidah"


def test_pgp_description_still_wins_in_the_merge():
    pgp = {"fragment": {"shelfmark": "T-S H 7.28"}, "documents": [{"pgpid": "1", "description": "Legal deed (PGP)"}]}
    rec = build_merged_record("Cambridge_CUL_T_S_H7_28", pgp, [], _ktiv(LABELS, [SUKKOT_EN]))
    assert rec["description"] == "Legal deed (PGP)"
    doc = GenizahDocument.from_merged_format(rec)
    assert doc.description == "Legal deed (PGP)"
    assert doc.subjects == ["פיוט", "תפילה וברכות", "תפילות קבע"]
