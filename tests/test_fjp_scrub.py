"""Tests for the FJP-host URL scrub on bibliography documents served through Elasticsearch."""

import json

import pytest

from src.datasets.document_models.bibliography_document import BibliographyDocument
from src.datasets.document_models.fjp_scrub import contains_fjp_url, scrub_fjp, strip_fjp_urls


@pytest.mark.parametrize("url", [
    "www.genizah.org",
    "genizah.org",
    "https://fgp.genizah.org/FGP/joins?id=12",
    "http://www.jewishmanuscripts.org/",
    "user:pw@fgp.genizah.org/x",
    "WWW.GENIZAH.ORG",
    "fgp.genizah.org:8080/viewer",
    "www.geni-\nzah.org",
    "fgp.genizah.\norg/FGP/joins",
    "www.genizah . org",
    "www.geni\u00adzah.org",
    "jewish-\nmanuscripts.org",
])
def test_every_fjp_url_form_is_removed(url):
    """Scheme-less, subdomain, port, path, userinfo and upper-case forms all go."""
    text = f"see {url} for images"
    assert contains_fjp_url(text)
    assert not contains_fjp_url(strip_fjp_urls(text))


def test_book_sentence_keeps_its_prose():
    """The OBP 0352 sentence loses the URL and its empty brackets, nothing else."""
    text = "can now be found itemised on the Friedberg Genizah website (www.genizah.org), and are tightly connected"
    assert strip_fjp_urls(text) == "can now be found itemised on the Friedberg Genizah website, and are tightly connected"


@pytest.mark.parametrize("text", [
    "https://cairogenizah.ai/search", "cairogenizah.org", "Genizah organisations", "the Cairo Genizah collection",
    "https://cudl.lib.cam.ac.uk/view/MS-TS-AS-00178-00088", "",
])
def test_other_hosts_and_words_untouched(text):
    """Our own site, holding institutions and ordinary words are not touched."""
    assert strip_fjp_urls(text) == text and not contains_fjp_url(text)


def test_edition_brackets_elsewhere_are_kept():
    """Lacuna marks like "[ ]" and "( )" elsewhere in the text survive; only the URL's own brackets go."""
    text = "line 4 [ ] lost; see (www.genizah.org) and [https://fgp.genizah.org/x]; also ( ) here"
    assert strip_fjp_urls(text) == "line 4 [ ] lost; see and; also ( ) here"


def test_check_is_looser_than_the_scrub():
    """The fail-closed check still sees a host the URL pattern cannot match (glued to the next word)."""
    assert contains_fjp_url("visit www.genizah.orgHis page") and not contains_fjp_url("Genizah\norganisations")


def test_scrub_walks_nested_values():
    """Dicts and lists are scrubbed in place of their strings; other types pass through."""
    doc = {"a": "x www.genizah.org y", "b": [{"url": "https://fgp.genizah.org/1"}, 3], "c": None}
    clean = scrub_fjp(doc)
    assert not contains_fjp_url(clean) and clean["b"][1] == 3 and clean["c"] is None


def test_bibliography_es_document_is_scrubbed(tmp_path):
    """A structured page citing genizah.org is indexed without the URL, in stored and embedded text."""
    page = tmp_path / "page_017_structured.json"
    page.write_text(json.dumps({
        "full_main_text": "itemised on the Friedberg Genizah website (www.genizah.org), and connected",
        "summary": "Mentions https://fgp.genizah.org/joins.",
        "transcriptions": {"T-S 1": "line from www.genizah.org/T-S"},
        "shelf_marks_mentioned": {},
        "extracted_page_number": [4],
    }), encoding="utf-8")
    doc = BibliographyDocument.from_structured_json(page, {"title": "Book", "authors": ["A B"], "doi": "10.11647/obp.0352"})
    es_doc = doc.to_elasticsearch_document()
    assert not contains_fjp_url(es_doc)
    assert not contains_fjp_url(doc.create_text_representation())
    assert "Friedberg Genizah website, and connected" in es_doc["full_text_content"]
