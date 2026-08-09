"""Tests for the BookArticle direct-link backfill.

These pin the shape rules in ``docs/BOOK_LINK_INDEXING_CONTRACT.md``. The
serving side does not guess: a URL stored in ``doi`` yields
``https://doi.org/https://doi.org/...`` and a broken link, and a hyphenated
ISBN yields a WorldCat 404. So the cleaners are the contract.
"""

import pytest

from src.datasets.indexing.neo4j.enrich_kg_identifiers import (
    WorkIdentifiers,
    bare_doi,
    clean_isbn,
    clean_url,
    collapse_pairs,
    doi_from_url,
    group_works,
    match_join_keys,
)


# --------------------------------------------------------------------------
# Contract field shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("10.2307/162589", "10.2307/162589"),
    # "Critical: do not store a URL in doi" — strip to the bare form.
    ("https://doi.org/10.2307/162589", "10.2307/162589"),
    ("http://dx.doi.org/10.2307/162589", "10.2307/162589"),
    ("doi:10.2307/162589", "10.2307/162589"),
    ("10.2307/162589.", "10.2307/162589"),
    ("  10.2307/162589  ", "10.2307/162589"),
    # Case is preserved — the contract says store message.DOI verbatim.
    ("10.1163/EJ.9789004190580.I-420.76", "10.1163/EJ.9789004190580.I-420.76"),
    # "If a value does not start with 10., it is not a DOI."
    ("https://www.jstor.org/stable/162589", ""),
    ("not-a-doi", ""),
    ("", ""),
    (None, ""),
])
def test_bare_doi(raw, expected):
    assert bare_doi(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("978-0-691-04869-8", "9780691048698"),
    ("9780691048698", "9780691048698"),
    ("0521451698", "0521451698"),
    ("0-521-45169-8 (hardback : alk. paper)", "0521451698"),
    # ISBN-10 may end in X.
    ("080213825X", "080213825X"),
    ("080213825x", "080213825X"),
    # Wrong lengths are not ISBNs.
    ("12345", ""),
    ("97806910486981234", ""),
    ("", ""),
    (None, ""),
])
def test_clean_isbn(raw, expected):
    assert clean_isbn(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("https://www.jstor.org/stable/162589", "https://www.jstor.org/stable/162589"),
    # "No trailing period."
    ("http://www.jstor.org/stable/23427752.", "http://www.jstor.org/stable/23427752"),
    ("  https://example.org/x  ", "https://example.org/x"),
    # Must be absolute.
    ("www.jstor.org/stable/1", ""),
    ("/relative/path", ""),
    ("", ""),
    (None, ""),
])
def test_clean_url(raw, expected):
    assert clean_url(raw) == expected


@pytest.mark.parametrize("url,expected", [
    ("https://doi.org/10.1515/islam-2023-0026", "10.1515/islam-2023-0026"),
    ("http://dx.doi.org/10.2307/162589", "10.2307/162589"),
    ("https://doi.org/10.2307/162589/", "10.2307/162589"),
    # A JSTOR stable id is NOT 10.2307/<id> — that mapping was tested against
    # Crossref for this corpus's ids and does not hold.
    ("https://www.jstor.org/stable/162589", ""),
    ("https://www.academia.edu/42934528", ""),
    ("", ""),
])
def test_doi_from_url(url, expected):
    assert doi_from_url(url) == expected


# --------------------------------------------------------------------------
# Link tier (contract: "How the link is chosen")
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fields,tier", [
    ({"doi": "10.1/x", "isbn": "9780691048698", "url": "https://e.test"}, "doi"),
    ({"isbn": "9780691048698", "url": "https://e.test"}, "isbn"),
    ({"url": "https://e.test"}, "url"),
    ({}, "search-fallback"),
])
def test_tier_follows_the_contract_priority(fields, tier):
    assert WorkIdentifiers(title="t", **fields).tier == tier


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def _row(article_id, title, **kwargs):
    """Build a BookArticle row.

    :param article_id: Node id.
    :param title: Node title.
    :param kwargs: Other node fields.
    :returns: The row dict.
    """
    row = {"article_id": article_id, "title": title, "citation": None, "year": None,
           "source_type": None, "journal": None, "url": None, "doi": None,
           "isbn": None, "authors": []}
    row.update(kwargs)
    return row


def test_group_works_merges_duplicate_nodes_of_one_work():
    works = group_works([
        _row("a", "Geniza Studies", year=1981),
        _row("b", "geniza studies", year=1981.0),
    ])
    assert len(works) == 1
    assert sorted(works[0].article_ids) == ["a", "b"]


def test_group_works_drops_unpublished_placeholders():
    rows = [_row("a", "unpublished editions"),
            _row("b", "Avraham David.", source_type="Unpublished"),
            _row("c", "A Real Geniza Study")]
    assert [w.title for w in group_works(rows)] == ["A Real Geniza Study"]


def test_group_works_skips_journal_titles():
    """A journal has an ISSN, not a DOI; looking it up is wasted effort."""
    journal = [_row(f"n{i}", "סיני", authors=[f"author {i}"]) for i in range(12)]
    book = [_row("b", "A Mediterranean society", authors=["Goitein"])]
    titles = [w.title for w in group_works(journal + book)]
    assert titles == ["A Mediterranean society"]
    # ...unless explicitly asked for.
    assert len(group_works(journal + book, skip_serials=False)) == 2


def test_group_works_keeps_a_multi_volume_set_apart_by_year():
    works = group_works([
        _row("a", "A Mediterranean society", year=1967),
        _row("b", "A Mediterranean society", year=1971),
    ])
    assert len(works) == 2


# --------------------------------------------------------------------------
# Title join (contract: "Title join")
# --------------------------------------------------------------------------


def test_collapse_pairs_keeps_every_volume_of_a_work():
    """One node, several ES books — the last must not overwrite the rest."""
    rows = collapse_pairs([
        {"article_id": "x", "es_book_id": "india_trader_426_500", "es_title": "India traders"},
        {"article_id": "x", "es_book_id": "india_trader_1_50", "es_title": "India traders"},
        {"article_id": "x", "es_book_id": "india_trader_350_424", "es_title": "India traders"},
    ])
    assert len(rows) == 1
    entry = rows[0]
    assert entry["es_book_ids"] == [
        "india_trader_1_50", "india_trader_350_424", "india_trader_426_500"
    ]
    assert entry["es_book_id"] == "india_trader_1_50"   # scalar primary
    assert entry["ambiguous"] is True


def test_collapse_pairs_single_book_is_not_ambiguous():
    rows = collapse_pairs([
        {"article_id": "x", "es_book_id": "ottomon_era", "es_title": "Ottoman-Era Documents"},
    ])
    assert rows[0]["es_book_id"] == "ottomon_era"
    assert rows[0]["es_titles"] == ["Ottoman-Era Documents"]
    assert rows[0]["ambiguous"] is False


def test_match_join_keys_handles_a_bilingual_es_title():
    """`Ginzei Kedem / גנזי קדם` is what the leading-word probe misses."""
    books = [{"es_book_id": "genizah_kedem_3vol", "es_title": "Ginzei Kedem / גנזי קדם"}]
    rows = [_row("k", "גנזי קדם"), _row("other", "Something Unrelated Entirely")]
    pairs = match_join_keys(books, rows)
    assert len(pairs) == 1
    assert pairs[0]["article_id"] == "k"
    assert pairs[0]["es_title"] == "Ginzei Kedem / גנזי קדם"


def test_match_join_keys_matches_on_exact_normalised_title():
    books = [{"es_book_id": "ottomon_era", "es_title": "Ottoman-Era Documents from the Cairo Genizah"}]
    rows = [_row("o", "Ottoman-Era documents from the Cairo Genizah")]
    pairs = match_join_keys(books, rows)
    assert pairs[0]["article_id"] == "o"


def test_match_join_keys_returns_nothing_when_no_node_matches():
    books = [{"es_book_id": "x", "es_title": "A Book With No Graph Node At All"}]
    rows = [_row("n", "Completely Different Subject Matter Here")]
    assert match_join_keys(books, rows) == []
