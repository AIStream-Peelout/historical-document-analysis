"""Tests for DOI / OCLC / ISBN enrichment of the metadata files.

Two hazards drive most of these tests. A book review carries the reviewed
book's exact title, so title matching alone attaches the review's DOI to the
book. And writing a DOI into the canonical position silently re-keys
``book_uuid``/``page_uuid``, which would break the Elasticsearch-to-Neo4j join.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from src.datasets.document_models.corpus_ids import book_key, book_uuid
from src.datasets.indexing.bibliography.enrich_metadata_identifiers import (
    Identifiers,
    _marc_isbn,
    _marc_oclc,
    apply_identifiers,
    existing_doi,
    expected_kind,
    metadata_files,
    primary_author,
    read_metadata,
)

MARC = """<record xmlns="http://www.loc.gov/MARC21/slim">
  <datafield tag="020"><subfield code="a">9789004154728 (hardback : alk. paper)</subfield></datafield>
  <datafield tag="035"><subfield code="a">(OCoLC)77116920</subfield></datafield>
</record>"""


def _record(xml=MARC):
    """Parse a MARC record fixture.

    :param xml: MARC XML string.
    :returns: The record element.
    """
    return ET.fromstring(xml)


# --------------------------------------------------------------------------
# MARC identifier extraction
# --------------------------------------------------------------------------


def test_marc_oclc_and_isbn():
    record = _record()
    assert _marc_oclc(record) == "77116920"
    assert _marc_isbn(record) == "9789004154728"


def test_marc_isbn_strips_hyphens_and_rejects_junk():
    record = _record("""<record xmlns="http://www.loc.gov/MARC21/slim">
      <datafield tag="020"><subfield code="a">978-90-04-19058-0 (hardback)</subfield></datafield>
    </record>""")
    assert _marc_isbn(record) == "9789004190580"


def test_marc_identifiers_absent():
    empty = _record('<record xmlns="http://www.loc.gov/MARC21/slim"></record>')
    assert _marc_oclc(empty) == ""
    assert _marc_isbn(empty) == ""


# --------------------------------------------------------------------------
# Metadata reading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metadata,expected", [
    ({"authors": ["Marina Rustow", "Someone Else"]}, "Marina Rustow"),
    ({"authors": "Gil, Moshe"}, "Gil, Moshe"),
    # One file in the corpus uses an ad-hoc author object.
    ({"author": {"name": "Nick Posegay", "affiliation": {}}}, "Nick Posegay"),
    ({"authors": []}, ""),
    ({}, ""),
])
def test_primary_author(metadata, expected):
    assert primary_author(metadata) == expected


@pytest.mark.parametrize("metadata,kind", [
    ({"publication": {"type": "Print Book"}}, "book"),
    ({"publication": {"type": "Journal Article"}}, "article"),
    ({"publication": {"type": "Book Chapter"}}, "chapter"),
    ({"publication": {"type": "Doctoral Dissertation"}}, "dissertation"),
    ({"type": "academic_article"}, "article"),
    # Fall back on the shape of the record when no type is stated.
    ({"publication": {"container_title": "A Festschrift"}}, "chapter"),
    ({"publication": {"journal": "Tarbiz"}}, "article"),
    ({"publication": {"publisher": "Brill"}}, "book"),
    ({}, ""),
])
def test_expected_kind(metadata, kind):
    assert expected_kind(metadata) == kind


@pytest.mark.parametrize("metadata,doi", [
    ({"doi": "10.1163/x"}, "10.1163/x"),
    ({"identifiers": {"doi": "10.1163/y"}}, "10.1163/y"),
    ({"publication": {"doi": "10.1163/z"}}, "10.1163/z"),
    # external_ids is deliberately NOT a canonical position.
    ({"external_ids": {"doi": "10.1163/w"}}, ""),
    ({}, ""),
])
def test_existing_doi_only_reads_canonical_positions(metadata, doi):
    assert existing_doi(metadata) == doi


def test_read_metadata_tolerates_broken_json(tmp_path):
    broken = tmp_path / "broken_metadata.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_metadata(broken) is None


def test_read_metadata_rejects_non_objects(tmp_path):
    listy = tmp_path / "list_metadata.json"
    listy.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_metadata(listy) is None


def test_metadata_files_skips_the_template(tmp_path):
    (tmp_path / "example_book_metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "real_metadata.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in metadata_files(tmp_path)] == ["real_metadata.json"]


# --------------------------------------------------------------------------
# Block rendering
# --------------------------------------------------------------------------


def test_block_builds_resolver_urls():
    block = Identifiers(doi="10.1163/x", oclc="77116920", isbn="9789004154728").to_block()
    assert block["doi_url"] == "https://doi.org/10.1163/x"
    assert block["worldcat_url"] == "https://search.worldcat.org/oclc/77116920"
    assert block["isbn"] == "9789004154728"


def test_block_falls_back_to_an_isbn_worldcat_link():
    """WorldCat links by OCLC when there is one, by ISBN otherwise."""
    block = Identifiers(isbn="9789004154728").to_block()
    assert block["worldcat_url"] == "https://search.worldcat.org/search?q=bn:9789004154728"


def test_unverified_doi_is_quarantined_from_the_field_the_frontend_reads():
    """A review's DOI must not be served as the work's DOI."""
    block = Identifiers(doi="10.2307/601931", needs_review=True,
                        notes=["may be a review"]).to_block()
    assert "doi" not in block and "doi_url" not in block
    assert block["doi_candidate"] == "10.2307/601931"
    assert block["needs_review"] is True


def test_empty_identifiers_render_nothing():
    assert Identifiers().to_block() == {}
    assert Identifiers().found is False


# --------------------------------------------------------------------------
# Writing back — and the re-keying hazard
# --------------------------------------------------------------------------


def test_apply_writes_external_ids_without_touching_the_book_key():
    """The whole point: adding a DOI must not change book_uuid by default."""
    metadata = {"title": "A Book"}
    before = book_uuid(book_key(metadata, stem="a_book"))
    apply_identifiers(metadata, Identifiers(doi="10.1163/x", oclc="123"))
    after = book_uuid(book_key(metadata, stem="a_book"))
    assert metadata["external_ids"]["doi"] == "10.1163/x"
    assert after == before


def test_promote_doi_does_change_the_book_key():
    """Opting in re-keys the book — the flag exists so that is a decision."""
    metadata = {"title": "A Book"}
    before = book_uuid(book_key(metadata, stem="a_book"))
    apply_identifiers(metadata, Identifiers(doi="10.1163/x"), promote_doi=True)
    after = book_uuid(book_key(metadata, stem="a_book"))
    assert metadata["identifiers"]["doi"] == "10.1163/x"
    assert after != before


def test_apply_is_idempotent():
    metadata = {"title": "A Book"}
    identifiers = Identifiers(doi="10.1163/x")
    assert apply_identifiers(metadata, identifiers) is True
    assert apply_identifiers(metadata, identifiers) is False


def test_apply_does_nothing_when_nothing_was_found():
    metadata = {"title": "A Book"}
    assert apply_identifiers(metadata, Identifiers()) is False
    assert "external_ids" not in metadata


def test_promote_doi_leaves_an_existing_canonical_doi_alone():
    metadata = {"title": "A Book", "identifiers": {"doi": "10.5479/original"}}
    apply_identifiers(metadata, Identifiers(doi="10.1163/new"), promote_doi=True)
    assert metadata["identifiers"]["doi"] == "10.5479/original"


def test_written_files_stay_valid_json(tmp_path):
    path = tmp_path / "x_metadata.json"
    metadata = {"title": "A Book", "publication": {"type": "Print Book"}}
    apply_identifiers(metadata, Identifiers(doi="10.1163/x", oclc="1"))
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_metadata(path)["external_ids"]["oclc"] == "1"
