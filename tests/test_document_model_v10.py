# File name: test_document_model_v10.py
# Date: 10/9/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Document-model changes for the v10 fragment index: the KTIV language fill, FJP string transcriptions, and the
scholarly-frames render flag of the embedding text.

Pure document-model tests: no Elasticsearch, no model, no network.
"""
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets.document_models.genizah_document import (
    KTIV_LANGUAGE_MAP,
    KTIV_LANGUAGE_SPELLING,
    GenizahDocument,
    TranscriptionSection,
    ktiv_language,
)

CID = "Cambridge_CUL_T_S_NS_219_41"
LETTER = "Letter about flax shipments to Alexandria"


def merged_record(ktiv_value: Optional[str] = None, pgp_language: Optional[str] = None,
                  fjp: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A merged record (``merged_shelfmarks.jsonl`` shape) with a KTIV block and optional PGP / FJP sources.

    :param ktiv_value: KTIV ``basic_catalog.language`` (absent when None).
    :param pgp_language: PGP ``languages_primary`` of the record's one PGP document (no PGP block when None).
    :param fjp: FJP records.
    :return: Record.
    """
    basic = {"title": "קטע גניזה"}
    if ktiv_value is not None:
        basic["language"] = ktiv_value
    return {
        "canonical_id": CID,
        "shelfmark_display": "Cambridge University Library, Cambridge, England Ms. T-S NS 219.41",
        "institution": "Cambridge University Library",
        "sources_present": ["ktiv"],
        "description": LETTER,
        "date": None,
        "images": {"ktiv": {"image_urls": []}, "preferred_source": None},
        "sources": {
            "pgp": {"documents": [{"languages_primary": pgp_language}]} if pgp_language is not None else None,
            "fjp": fjp or [],
            "bodleian": None,
            "ktiv": {"sys_num": "990051590520205171", "basic_catalog": basic},
        },
    }


def text_lines(doc: GenizahDocument, **kwargs: Any) -> List[str]:
    """:param doc: Document.
    :param kwargs: ``create_text_representation`` arguments.
    :return: The lines of its embedding text.
    """
    return doc.create_text_representation(**kwargs).split("\n")


# --------------------------------------------------------------------------------------------- KTIV language fill


@pytest.mark.parametrize("value, expected", [
    ("Hebrew", "Hebrew"),
    ("Judeo-Arabic", "Judaeo-Arabic"),
    ("Hebrew Judeo-Arabic", "Hebrew, Judaeo-Arabic"),
    ("Judeo-Persian Arabic", "Judaeo-Persian, Arabic"),
    ("Greek, ancient", "Greek, ancient"),                  # a comma keeps a two-word name together
    ("Greek, Modern", "Greek, Modern"),
    ("Hebrew Greek, ancient", "Hebrew, Greek, ancient"),
    ("Modern Greek", "Modern Greek"),                      # the catalogue's one two-word name without a comma
    ("Hebrew Modern Greek", "Hebrew, Modern Greek"),
    ("pol", "Polish"),
    ("mul", None),                                         # "multiple languages" names none
    ("Hebrew mul", "Hebrew"),
    ("  Hebrew   Judeo-Arabic ", "Hebrew, Judaeo-Arabic"),
    ("Greek,  ancient", "Greek, ancient"),
    (None, None),
    ("", None),
    ("   ", None),
])
def test_ktiv_language_normalises_catalogue_values(value, expected):
    assert ktiv_language(value) == expected


def test_ktiv_language_map_is_frozen():
    # the Language line is embedded: changing these means re-embedding, so a change must be deliberate
    assert dict(KTIV_LANGUAGE_MAP) == {"pol": "Polish", "mul": None}
    assert KTIV_LANGUAGE_SPELLING == ("Judeo-", "Judaeo-")
    with pytest.raises(TypeError):
        KTIV_LANGUAGE_MAP["heb"] = "Hebrew"  # type: ignore[index]


@pytest.mark.parametrize("ktiv_value, pgp_language, fjp_language, expected", [
    ("Hebrew Judeo-Arabic", None, None, "Hebrew, Judaeo-Arabic"),   # KTIV only: filled
    ("Hebrew", "Judaeo-Arabic", None, "Judaeo-Arabic"),             # PGP first
    ("Hebrew", "", "Aramaic", "Aramaic"),                           # then FJP (an empty PGP value is no value)
    ("Hebrew", None, "Unknown", "Hebrew"),                          # FJP's literal "Unknown" does not hide KTIV
    (None, None, "Unknown", "Unknown"),
    ("mul", None, None, "Unknown"),                                 # nothing left of the KTIV value
    (None, None, None, "Unknown"),
])
def test_language_falls_back_to_the_ktiv_catalogue_language(ktiv_value, pgp_language, fjp_language, expected):
    fjp = [{"language": fjp_language}] if fjp_language else []
    doc = GenizahDocument.from_merged_format(merged_record(ktiv_value, pgp_language, fjp))
    assert doc.language == expected


def test_ktiv_language_reaches_the_text_and_the_index_fields():
    doc = GenizahDocument.from_merged_format(merged_record("Hebrew Judeo-Arabic"))
    assert "Language: Hebrew, Judaeo-Arabic" in text_lines(doc)
    es_doc = doc.to_elasticsearch_document()
    assert es_doc["language"] == "Hebrew, Judaeo-Arabic" and es_doc["main_language"] == "hebrew, judaeo-arabic"
    unknown = GenizahDocument.from_merged_format(merged_record())
    assert unknown.language == "Unknown"
    assert not any(line.startswith("Language:") for line in text_lines(unknown))


# --------------------------------------------------------------------------------------------- FJP string transcriptions

EDITED = {"name": "Editor: Gil, Moshe", "lines": {"1": "בשם רחמנא"}}


def test_string_transcription_items_become_numbered_sections():
    doc = GenizahDocument(description="d", transcriptions=[
        "  שורה ראשונה \n\n שורה שנייה\n", "", "   \n ", EDITED, TranscriptionSection(lines={"1": "x"})])
    assert [t.to_dict() for t in doc.transcriptions] == [
        {"name": None, "lines": {"1": "שורה ראשונה", "2": "שורה שנייה"}},   # one entry per non-empty line
        EDITED,                                                                 # empty strings are dropped
        {"name": None, "lines": {"1": "x"}},
    ]


def test_merged_string_transcriptions_end_as_validated_ones_and_reach_text_and_index():
    items = [EDITED, "שורה ראשונה\nשורה שנייה", ""]
    doc = GenizahDocument.from_merged_format(merged_record(fjp=[{"transcriptions": items}]))
    assert doc.transcriptions == GenizahDocument(description="d", transcriptions=items).transcriptions
    assert [t.to_dict() for t in doc.transcriptions] == [
        EDITED, {"name": None, "lines": {"1": "שורה ראשונה", "2": "שורה שנייה"}}]
    assert "Transcription: [Editor: Gil, Moshe] בשם רחמנא שורה ראשונה שורה שנייה" in text_lines(doc)
    assert "שורה שנייה" in doc.create_full_text_content()
    es_transcriptions = doc.to_elasticsearch_document()["transcriptions"]
    assert es_transcriptions[1]["editor"] is None and es_transcriptions[1]["line_count"] == 2


# --------------------------------------------------------------------------------------------- scholarly-frames render flag


def framed_document(frames: Sequence[str] = ("[Talmud Bavli]: Sukkah 29 a – b", "Sefer ha-Arukh")) -> GenizahDocument:
    """:param frames: Its scholarly frames (by default the first restates the alternate title, the second does not).
    :return: A document with one alternate title, the frames, a subject and a language.
    """
    return GenizahDocument(description=LETTER, alt_titles=["Talmud Bavli: Sukkah 29 a – b"],
                           scholarly_frames=list(frames), subjects=["Jewish law"], language="Hebrew")


def test_frames_follow_the_alternate_titles_in_the_text_by_default():
    doc = framed_document()
    assert "Alternate titles: Talmud Bavli: Sukkah 29 a – b; Sefer ha-Arukh" in text_lines(doc)   # one per key
    assert doc.create_text_representation() == doc.create_text_representation(include_scholarly_frames=True)


def test_without_frames_only_the_alternate_titles_line_changes():
    doc = framed_document()
    with_frames = doc.create_text_representation()
    without = doc.create_text_representation(include_scholarly_frames=False)
    assert "Alternate titles: Talmud Bavli: Sukkah 29 a – b" in without.split("\n")
    assert "Sefer ha-Arukh" not in without
    assert without == with_frames.replace("; Sefer ha-Arukh", "")
    assert {"Subjects: Jewish law", "Language: Hebrew", f"Description: {LETTER}"} <= set(without.split("\n"))
    frames_only = GenizahDocument(description="d", scholarly_frames=["Piyyut"])
    assert "Alternate titles: Piyyut" in text_lines(frames_only)
    assert not any(line.startswith("Alternate titles") for line in text_lines(frames_only, include_scholarly_frames=False))


def test_full_text_index_document_and_cache_key_always_carry_the_frames():
    doc = framed_document()
    full = doc.create_full_text_content()
    assert "Sefer ha-Arukh" in full and "[Talmud Bavli]: Sukkah 29 a – b" in full
    es_doc = doc.to_elasticsearch_document()
    assert es_doc["scholarly_frames"] == doc.scholarly_frames and es_doc["alt_titles"] == doc.alt_titles
    assert es_doc["full_text_content"] == full
    assert framed_document(frames=()).get_embedding_cache_key() != doc.get_embedding_cache_key()


# --- transcription section names and line order (2026-10-09, before the v10 corpus rebuild) --------------------

def test_fjp_dict_transcriptions_are_named_without_the_legacy_store_and_lines_render_in_numeric_order():
    from src.datasets.document_models.genizah_document import GenizahDocument, _line_order
    lines = {str(i): f"line{i}" for i in range(1, 13)}
    merged = {
        "canonical_id": "Cambridge_CUL_T_S_NS_219_99",
        "shelfmark_display": "Cambridge University Library, Cambridge, England Ms. T-S NS 219.99",
        "institution": "Cambridge University Library", "sources_present": ["fjp"],
        "description": "Letter about the flax trade in Alexandria", "date": None,
        "images": {"ktiv": {"image_urls": []}, "preferred_source": None},
        "sources": {"pgp": None, "fjp": [{"transcriptions": {"1": "\n".join(lines.values())}}],
                    "bodleian": None, "ktiv": None},
    }
    doc = GenizahDocument.from_merged_format(merged)
    assert [t.name for t in doc.transcriptions] == ["Transcription 1"]
    es_doc = doc.to_elasticsearch_document()
    assert all("FJP" not in (t.get("editor") or "") for t in es_doc["transcriptions"])
    text = doc.create_text_representation()
    excerpt = text.split("Transcription:", 1)[1]
    assert excerpt.index("line2 ") < excerpt.index("line10 ") < excerpt.index("line11 ") < excerpt.index("line12")
    assert sorted({"10": "", "2": "", "a": "", "1": ""}.items(), key=_line_order) == [("1", ""), ("2", ""), ("10", ""), ("a", "")]
