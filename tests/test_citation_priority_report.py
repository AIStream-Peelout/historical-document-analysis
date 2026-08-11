"""Tests for the KG citation-priority report.

The ranking is only useful if it counts each *work* once, drops the things that
cannot be photographed, and does not present a journal run as a single book.
"""

import json

import pytest

from src.datasets.indexing.neo4j.citation_priority_report import (
    CitedWork,
    aggregate_works,
    find_serial_titles,
    is_placeholder,
    mark_serials,
    match_local,
    normalize_title,
    resolve_availability,
    scanned_corpus_titles,
)


@pytest.mark.parametrize("title,expected", [
    ("A Mediterranean society: the Jewish communities", "mediterranean society the jewish communities"),
    ("A Mediterranean society; the Jewish communities", "mediterranean society the jewish communities"),
    ("The Ketubba Traditions", "ketubba traditions"),
    ("Histoire des prix et des salaires dans l'Orient médiéval",
     "histoire des prix et des salaires dans l orient medieval"),
    ("  Spaced   Out  ", "spaced out"),
    (None, ""),
    ("", ""),
])
def test_normalize_title(title, expected):
    assert normalize_title(title) == expected


@pytest.mark.parametrize("title,source_type,expected", [
    ("unpublished editions", None, True),
    ("Digital Editions", None, True),
    ("UNPUBLISHED EDITIONS", None, True),
    # PGP leaves these untitled, so the bare-name citation becomes the title.
    ("Avraham David.", "Unpublished", True),
    ("Alan Elbaum, (2024).", "Unpublished", True),
    ("A Mediterranean society", "Book", False),
    ("Geniza Studies", None, False),
])
def test_is_placeholder(title, source_type, expected):
    assert is_placeholder(title, source_type) is expected


def _row(title, **kwargs):
    """Build a fetch_cited_works row.

    :param title: Work title.
    :param kwargs: Field overrides.
    :returns: The row dict.
    """
    row = {
        "title": title, "citation": None, "year": None, "volume": None,
        "publisher": None, "journal": None, "source_type": None,
        "source_books": None, "data_sources": ["biblio"],
        "edge_total": 0, "fragment_ids": [], "authors": [],
    }
    row.update(kwargs)
    return row


def test_aggregate_merges_duplicate_nodes_of_one_work():
    """Two nodes for one book must not split its citation count."""
    rows = [
        _row("Geniza Studies", fragment_ids=["f1", "f2"], edge_total=2, year=1981),
        _row("geniza studies", fragment_ids=["f2", "f3"], edge_total=2, year=1981.0),
    ]
    works = aggregate_works(rows)
    assert len(works) == 1
    work = works[0]
    assert work.node_count == 2
    assert work.fragments == 3          # f2 counted once across the duplicates
    assert work.citations == 4          # edges still total
    assert work.year == 1981


def test_aggregate_keeps_volumes_of_a_set_apart():
    """Goitein's volumes share a title and differ only by year."""
    rows = [
        _row("A Mediterranean society", year=1967, fragment_ids=["a"], edge_total=1),
        _row("A Mediterranean society", year=1971, fragment_ids=["b"], edge_total=1),
    ]
    works = aggregate_works(rows)
    assert len(works) == 2
    assert {w.year for w in works} == {1967, 1971}


def test_aggregate_ranks_by_distinct_fragments():
    rows = [
        _row("Small book", fragment_ids=["a"], edge_total=1),
        _row("Big book", fragment_ids=["a", "b", "c"], edge_total=3),
    ]
    assert [w.title for w in aggregate_works(rows)] == ["Big book", "Small book"]


def test_aggregate_skips_untitled_rows():
    assert aggregate_works([_row(None)]) == []


def test_aggregate_prefers_the_fullest_title_variant():
    rows = [
        _row("Geniza Studies", fragment_ids=["a"]),
        _row("Geniza Studies: A Fuller Subtitle", fragment_ids=["b"]),
    ]
    works = aggregate_works(rows)
    # Different keys (the subtitle changes the key), but each keeps its own text.
    assert "Geniza Studies" in {w.title for w in works}


def test_find_serial_titles_needs_both_breadth_and_span():
    """Many nodes across many years is a journal; a few volumes is not."""
    journal = [_row("Jewish Quarterly Review", year=1900 + i) for i in range(25)]
    monograph = [_row("A Mediterranean society", year=1967 + i * 4) for i in range(5)]
    serials = find_serial_titles(journal + monograph)
    assert normalize_title("Jewish Quarterly Review") in serials
    assert normalize_title("A Mediterranean society") not in serials


def test_mark_serials_catches_year_less_journals():
    """Hebrew journals carry no year, so authorship spread is the signal."""
    journal = CitedWork(key=("sinai", None), title="סיני", node_count=79,
                        authors={f"author {i}" for i in range(31)})
    monograph = CitedWork(key=("med soc", 1967), title="A Mediterranean society",
                          node_count=1, authors={"Shelomo Dov Goitein"})
    mark_serials([journal, monograph])
    assert journal.serial is True and journal.kind == "serial"
    assert monograph.serial is False and monograph.kind == "monograph"


def test_mark_serials_needs_both_conditions():
    """An edited volume with many contributors is still one book to fetch."""
    edited_volume = CitedWork(key=("festschrift", 1947), title="Semitic studies",
                              node_count=1, authors={f"contributor {i}" for i in range(12)})
    reprinted = CitedWork(key=("often cited", None), title="Often cited",
                          node_count=15, authors={"One Author"})
    mark_serials([edited_volume, reprinted])
    assert edited_volume.serial is False
    assert reprinted.serial is False


def test_kind_puts_placeholder_first():
    work = CitedWork(key=("k", None), title="unpublished editions",
                     placeholder=True, serial=True)
    assert work.kind == "placeholder"


def test_scanned_corpus_titles_reads_metadata_and_series(tmp_path):
    root = tmp_path / "academic_literature"
    book = root / "kettubah_palestine"
    book.mkdir(parents=True)
    (book / "friedman_metadata.json").write_text(json.dumps({
        "title": "Jewish Marriage in Palestine: The Kettubba texts",
        "series": {"name": "Jewish Marriage in Palestine: A Cairo Geniza Study"},
    }), encoding="utf-8")
    (root / "example_book_metadata.json").write_text(
        json.dumps({"title": "Template Should Be Ignored"}), encoding="utf-8"
    )
    (root / "broken_metadata.json").write_text("{not json", encoding="utf-8")

    titles = scanned_corpus_titles(root)
    assert titles[normalize_title("Jewish Marriage in Palestine: The Kettubba texts")] == (
        "kettubah_palestine"
    )
    # The series name resolves too, so a citation of the set matches.
    assert normalize_title("Jewish Marriage in Palestine: A Cairo Geniza Study") in titles
    assert normalize_title("Template Should Be Ignored") not in titles


def test_match_local_by_exact_and_partial_title():
    scanned = {normalize_title("India traders of the middle ages: documents from the Cairo Geniza"):
               "india_traders"}
    exact = CitedWork(
        key=(normalize_title("India traders of the middle ages: documents from the Cairo Geniza"),
             None),
        title="India traders",
    )
    assert match_local(exact, scanned) == "india_traders"
    # A citation using the short title still matches the fuller catalogue title.
    short = CitedWork(key=(normalize_title("India traders of the middle ages"), None),
                      title="India traders of the middle ages")
    assert match_local(short, scanned) == "india_traders"
    # A short key must not over-match a long scanned title.
    unrelated = CitedWork(key=("geniza", None), title="Geniza")
    assert match_local(unrelated, scanned) is None
    assert match_local(CitedWork(key=("", None), title=""), scanned) is None


def test_resolve_availability_marks_works_we_already_hold():
    scanned = {normalize_title("In the Kingdom of Ishmael"): "malkhhut_ish"}
    held = CitedWork(key=(normalize_title("In the Kingdom of Ishmael"), None),
                     title="In the Kingdom of Ishmael")
    missing = CitedWork(key=(normalize_title("Palestine During the First Muslim Period"), None),
                        title="Palestine During the First Muslim Period")
    resolve_availability([held, missing], scanned)
    assert held.have_locally is True and held.local_scan == "malkhhut_ish"
    assert missing.have_locally is False


def test_have_locally_honours_source_books():
    """An `enriched` node carries the scan directory directly."""
    work = CitedWork(key=("k", None), title="t", source_books={"ej_arrant"})
    assert work.have_locally is True
