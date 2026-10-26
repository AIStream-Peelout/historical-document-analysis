# File name: test_text_representation.py
# Date: 10/7/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Versioned embedding text (src/embeddings/text_representation.py): v8 unchanged, semantic_v3 as trained, semantic_v4.

The semantic_v4 expectation for merged_record() is the one genizah_search docs/handoffs/v10_embedder/audits/
text-pipeline.md asks for ("Update the HDA test expectation").
"""

import hashlib
from typing import Any, Dict, List

import pytest

from src.datasets.document_models.bibliography_document import BibliographyDocument
from src.datasets.document_models.genizah_document import GenizahDocument
from src.embeddings.semantic_text import semantic_text
from src.embeddings.text_representation import (
    BIBLIOGRAPHY,
    DEFAULT_TEXT_VERSION,
    TEXT_VERSIONS,
    EmbeddingText,
    create_text_representation,
    embed_text_for_version,
    embedding_text,
    is_text_eligible,
    text_for_version,
)


def merged_record(**over: Any) -> Dict[str, Any]:
    """A merged_shelfmarks.jsonl record (PGP + KTIV sources only).

    :param over: Top-level keys to override.
    :return: Record dict.
    """
    record = {
        "canonical_id": "Cambridge_CUL_T_S_13J8_14",
        "shelfmark_display": "T-S 13J8.14",
        "institution": "Cambridge University Library",
        "sources_present": ["pgp", "ktiv"],
        "description": ("Letter from Nahray b. Nissim to his partner about the flax trade. "
                        "Joins T-S 13J8.14 and ENA 2558.5 (PGPID 12345)."),
        "date": "1060",
        "images": {},
        "sources": {
            "pgp": {"documents": [{"pgpid": "12345", "languages_primary": "Judaeo-Arabic",
                                   "mentioned": "Nahray b. Nissim", "location": "Fustat"}]},
            "ktiv": {"sys_num": "990000000000000001",
                     "basic_catalog": {"varying_form_of_title": ["Title in English: Business letter"]}},
            "ktiv_transcription": {"pages": [{"text": "בשם רחמנא\nכתאבי אליך", "image_name": "Frag. 001r"}],
                                   "page_count": 1},
        },
    }
    record.update(over)
    return record


# GenizahDocument.create_text_representation() of merged_record() as of index v8 (must never change).
V8_TEXT = (
    "Document ID: Cambridge_CUL_T_S_13J8_14\n"
    "Shelf Mark: T-S 13J8.14\n"
    "Description: Letter from Nahray b. Nissim to his partner about the flax trade. "
    "Joins T-S 13J8.14 and ENA 2558.5 (PGPID 12345).\n"
    "Alternate titles: Business letter\n"
    "Language: Judaeo-Arabic\n"
    "Document Type: letter\n"
    "Date: standard_date: 1060\n"
    "Transcription: [Editor: KTIV Academic transcription (Frag. 001r)] בשם רחמנא כתאבי אליך\n"
    "Related People: Nahray b. Nissim\n"
    "Related Places: Fustat"
)
# The reference semantic_text() of V8_TEXT.
SEMANTIC_V3_TEXT = (
    "Description: Letter from Nahray b. Nissim to his partner about the flax trade. "
    "Joins [shelfmark] and [shelfmark] ([shelfmark]).\n"
    "Alternate titles: Business letter\n"
    "Language: Judaeo-Arabic\n"
    "Document Type: letter\n"
    "Date: standard_date: 1060\n"
    "Transcription: בשם רחמנא כתאבי אליך\n"
    "Related People: Nahray b. Nissim\n"
    "Related Places: Fustat"
)
# semantic_v4 of V8_TEXT: the alternate title folded into the Description line, no Document Type line
# (text-pipeline.md, "Update the HDA test expectation"; the ".; " join is cosmetic).
SEMANTIC_V4_TEXT = (
    "Description: Letter from Nahray b. Nissim to his partner about the flax trade. "
    "Joins [shelfmark] and [shelfmark] ([shelfmark]).; Business letter\n"
    "Language: Judaeo-Arabic\n"
    "Date: standard_date: 1060\n"
    "Transcription: בשם רחמנא כתאבי אליך\n"
    "Related People: Nahray b. Nissim\n"
    "Related Places: Fustat"
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def sha256(text: str) -> str:
    """Hex sha256 of a text's UTF-8 bytes.

    :param text: Text.
    :return: Digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bibliography_page(**over: Any) -> BibliographyDocument:
    """A bibliography page document.

    :param over: Field overrides.
    :return: BibliographyDocument.
    """
    fields = dict(
        doc_id="cairo_to_manchester_1_p058", shelf_mark="B 2349-1", pdf_name="cairo_to_manchester_1",
        page_number=58, title="From Cairo to Manchester", authors=["A. Scholar"],
        shelf_marks_mentioned=["B 2349-1", "T-S 13J8.14"], summary="Discusses T-S 13J8.14, a letter on flax.",
        full_main_text="The letter T-S 13J8.14 names the partner.",
    )
    fields.update(over)
    return BibliographyDocument(**fields)


def test_v8_is_todays_text_unchanged():
    doc = GenizahDocument.from_merged_format(merged_record())
    assert doc.create_text_representation() == V8_TEXT
    assert create_text_representation(doc, version="v8") == V8_TEXT


def test_semantic_v4_is_the_default_and_v3_still_matches_training():
    doc = GenizahDocument.from_merged_format(merged_record())
    assert DEFAULT_TEXT_VERSION == "semantic_v4"
    assert create_text_representation(doc) == create_text_representation(doc, version="semantic_v4")
    assert create_text_representation(doc, version="semantic_v3") == SEMANTIC_V3_TEXT
    assert create_text_representation(doc, version="semantic_v3") == semantic_text(V8_TEXT)
    assert SEMANTIC_V3_TEXT == semantic_text(V8_TEXT, own_shelfmark=doc.shelf_mark)


def test_unknown_version_rejected():
    doc = GenizahDocument.from_merged_format(merged_record())
    with pytest.raises(ValueError, match="unknown text representation version"):
        create_text_representation(doc, version="semantic_v5")
    with pytest.raises(ValueError, match="unknown text representation version"):
        embed_text_for_version(V8_TEXT, "semantic_v5")


def test_embedding_text_eligibility_for_fragments():
    rich = GenizahDocument.from_merged_format(merged_record())
    assert embedding_text(rich, "semantic_v3") == (SEMANTIC_V3_TEXT, True, sha256(SEMANTIC_V3_TEXT))
    assert embedding_text(rich) == embed_text_for_version(V8_TEXT, "semantic_v4")
    boiler = GenizahDocument.from_merged_format(merged_record(
        canonical_id="Cambridge_CUL_T_S_NS_1_1", shelfmark_display="T-S NS 1.1",
        description="Newly treated and encapsulated, must be examined", sources={}))
    assert embedding_text(boiler).eligible is False
    assert embedding_text(boiler, version="v8").eligible is True  # v8 embedded every record
    bare = GenizahDocument.from_merged_format(merged_record(
        canonical_id="Cambridge_CUL_T_S_NS_1_2", shelfmark_display="T-S NS 1.2", description="", date=None,
        sources={}))
    assert embedding_text(bare) == ("", False, EMPTY_SHA256)


def test_bibliography_semantic_v3_keeps_shelf_marks_and_drops_the_pdf_line():
    page = bibliography_page()
    v8 = page.create_text_representation()
    assert v8.startswith("Source PDF: cairo_to_manchester_1, Page: 58\n")
    assert create_text_representation(page, version="v8") == v8
    semantic = create_text_representation(page)
    assert semantic == v8.split("\n", 1)[1]
    assert "Shelf marks mentioned: B 2349-1, T-S 13J8.14" in semantic
    assert "T-S 13J8.14, a letter on flax" in semantic and "[shelfmark]" not in semantic
    assert embedding_text(page).eligible is True
    # the verifier rebuilds it from the stored full_text_content (the v8 text)
    stored = page.to_elasticsearch_document()["full_text_content"]
    assert text_for_version(stored, "semantic_v3", BIBLIOGRAPHY) == semantic


def test_bibliography_page_with_nothing_but_the_pdf_line_gets_no_vector():
    page = bibliography_page(title=None, authors=[], shelf_marks_mentioned=[], summary=None, full_main_text="")
    assert embedding_text(page) == ("", False, EMPTY_SHA256)
    assert embedding_text(page, "semantic_v4") == ("", False, EMPTY_SHA256)


def test_documents_must_declare_their_text_kind():
    class Legacy:
        """A document type without embedding_text_kind."""

        def create_text_representation(self) -> str:
            """Legacy text.

            :return: Text.
            """
            return "Document ID: x\nDescription: something"

    assert create_text_representation(Legacy(), version="v8") == "Document ID: x\nDescription: something"
    with pytest.raises(TypeError, match="embedding_text_kind"):
        create_text_representation(Legacy())


# --- semantic_v4 (index v10) ----------------------------------------------------------------------------------------

def test_semantic_v4_is_a_version_but_not_the_default():
    assert TEXT_VERSIONS == ("v8", "semantic_v3", "semantic_v4")
    assert DEFAULT_TEXT_VERSION == "semantic_v4"  # flipped with the v10 model pin (v10-lora-s0211)


def test_semantic_v4_of_the_merged_record_is_the_audited_text():
    doc = GenizahDocument.from_merged_format(merged_record())
    assert create_text_representation(doc, version="semantic_v4") == SEMANTIC_V4_TEXT
    assert text_for_version(V8_TEXT, "semantic_v4") == SEMANTIC_V4_TEXT
    expected = EmbeddingText(SEMANTIC_V4_TEXT, True, sha256(SEMANTIC_V4_TEXT))
    assert embedding_text(doc, "semantic_v4") == expected
    assert embed_text_for_version(V8_TEXT, "semantic_v4") == expected
    assert embedding_text(doc, "semantic_v4").text_sha256 == sha256(SEMANTIC_V4_TEXT)


def test_embed_text_for_version_matches_embedding_text_for_every_version():
    doc = GenizahDocument.from_merged_format(merged_record())
    for version in TEXT_VERSIONS:
        assert embed_text_for_version(V8_TEXT, version) == embedding_text(doc, version)
    assert embed_text_for_version(V8_TEXT, "v8") == (V8_TEXT, True, sha256(V8_TEXT))
    assert embed_text_for_version("", "semantic_v4") == ("", False, EMPTY_SHA256)


def test_semantic_v4_eligibility_is_decided_on_the_v3_text():
    # a short catalogue title that names a festival: "short" under v3, admitted under v4
    haggadah = GenizahDocument.from_merged_format(merged_record(
        canonical_id="Cambridge_CUL_Or_1081_2_33", shelfmark_display="Or.1081 2.33", description="הגדה של פסח (קטע)",
        date=None, sources={}))
    assert embedding_text(haggadah, "semantic_v3") == ("Description: הגדה של פסח (קטע)", False,
                                                       sha256("Description: הגדה של פסח (קטע)"))
    assert embedding_text(haggadah, "semantic_v4") == ("Description: הגדה של פסח (קטע)", True,
                                                       sha256("Description: הגדה של פסח (קטע)"))
    # a bracketed frame restating a frozen label: eligible under v3 (brackets as written), boilerplate under v4,
    # although the v4 text itself no longer shows the bracketed part
    label = "טוביה בן אליעזר, לקח טוב: אסתר, רות, שיר השירים, קהלת ואיכה"
    v8 = (f"Document ID: Cambridge_CUL_T_S_NS_1_3\nShelf Mark: T-S NS 1.3\nDescription: לקח טוב\n"
          f"Alternate titles: {label}; [טוביה בן אליעזר, לקח טוב]: אסתר, רות, שיר השירים, קהלת ואיכה")
    assert embed_text_for_version(v8, "semantic_v3").eligible is True
    item = embed_text_for_version(v8, "semantic_v4")
    assert item == (f"Description: לקח טוב; {label}", False, sha256(f"Description: לקח טוב; {label}"))


def test_is_text_eligible_refuses_semantic_v4_fragments():
    with pytest.raises(ValueError, match="embed_text_for_version"):
        is_text_eligible(SEMANTIC_V4_TEXT, "semantic_v4")
    assert is_text_eligible(SEMANTIC_V3_TEXT, "semantic_v3") is True  # v3 unchanged
    assert is_text_eligible("Title: From Cairo to Manchester", "semantic_v4", BIBLIOGRAPHY) is True
    assert is_text_eligible("  ", "semantic_v4", BIBLIOGRAPHY) is False


def test_bibliography_pages_keep_the_v3_page_text_under_v4():
    page = bibliography_page()
    semantic = create_text_representation(page)
    assert create_text_representation(page, version="semantic_v4") == semantic
    assert embedding_text(page, "semantic_v4") == (semantic, True, sha256(semantic))
    # no layout: the page's unlabelled lines are not folded into the field line before them
    assert "\nShelf marks mentioned: B 2349-1, T-S 13J8.14" in semantic
    stored = page.to_elasticsearch_document()["full_text_content"]
    assert text_for_version(stored, "semantic_v4", BIBLIOGRAPHY) == semantic
    assert embed_text_for_version(stored, "semantic_v4", BIBLIOGRAPHY) == (semantic, True, sha256(semantic))


class FramedDocument:
    """A fragment document whose text depends on the scholarly-frames flag (records how it was called)."""

    embedding_text_kind = "fragment"

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls: List[bool] = []

    def create_text_representation(self, include_scholarly_frames: bool = True) -> str:
        """Production-shaped text; the frame is rendered after the alternate title.

        :param include_scholarly_frames: Render the scholarly frame.
        :return: Text.
        """
        self.calls.append(include_scholarly_frames)
        alt = "Hoshana; [Piyyut]: Hoshanot" if include_scholarly_frames else "Hoshana"
        return f"Document ID: X_1\nDescription: Piyyut for the seventh day of Sukkot\nAlternate titles: {alt}"


def test_the_scholarly_frames_flag_reaches_the_document():
    doc = FramedDocument()
    with_frames = "Description: Piyyut for the seventh day of Sukkot; Hoshana; [Piyyut]: Hoshanot"
    without = "Description: Piyyut for the seventh day of Sukkot; Hoshana"
    assert create_text_representation(doc, "semantic_v4") == with_frames
    assert create_text_representation(doc, "semantic_v4", include_scholarly_frames=False) == without
    assert embedding_text(doc, "semantic_v4", include_scholarly_frames=False) == (without, True, sha256(without))
    assert embedding_text(doc, "v8", include_scholarly_frames=False).text.endswith("Alternate titles: Hoshana")
    assert doc.calls == [True, False, False, False]  # the default path calls it without the keyword
    # the real document model takes the keyword (merged_record() has no scholarly frames, so both agree)
    real = GenizahDocument.from_merged_format(merged_record())
    assert create_text_representation(real, "semantic_v4", include_scholarly_frames=False) == SEMANTIC_V4_TEXT
