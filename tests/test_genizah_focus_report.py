"""Tests for the Genizah-focus ranking.

Raw citation count rewards big reference works whose Genizah content is a thin
slice. This report exists to correct that, so the tests pin the three judgements
it makes: is the work *about* the Genizah, how big is the unit, and can it be
indexed without splicing pages out of an irrelevant volume.
"""

import pytest

from src.datasets.indexing.neo4j.genizah_focus_report import (
    FocusedWork,
    _already_scanned,
    build_works,
    classify_topicality,
    group_works,
    parse_extent,
)


# --------------------------------------------------------------------------
# Topicality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("title,expected", [
    ("Wills and Deathbed Declarations from the Cairo Geniza", "core"),
    ("קטלוג קטעי גניזת קאהיר", "core"),
    ("Magische Texte aus der Kairoer Geniza", "core"),
    ("Fustat on the Nile: The Jewish Elite in Medieval Egypt", "core"),
    ("Taylor-Schechter New Series", "core"),
    # Cited as a collation base, not for Genizah content.
    ("Biblia Hebraica (Prolegomena)", "reference"),
    ("מילון הערבית-היהודית מימי הביניים", "reference"),
    ("A Grammar of Early Judaeo-Persian", "reference"),
    ("דקדוקי סופרים השלם", "reference"),
    # Genizah-adjacent subject matter, but the title does not say so.
    ("A Fatimid Tax Archive from the Fayyūm", "adjacent"),
    ("מצמון נגיד ארץ תימן וסחר-הודו", "adjacent"),
    ("Some Unrelated Medieval Study", "unclear"),
])
def test_classify_topicality(title, expected):
    assert classify_topicality(title)[0] == expected


def test_core_marker_beats_a_reference_marker():
    """A Genizah *catalogue* is scholarship, not a lookup tool."""
    topicality, markers = classify_topicality(
        "Catalogue of the Cairo Genizah fragments at Westminster College"
    )
    assert topicality == "core"
    assert any("geniza" in m for m in markers)


def test_journal_contributes_to_topicality():
    topicality, _ = classify_topicality("A Letter to Maimonides", journal="Geniza Studies")
    assert topicality == "core"


def test_topicality_reports_which_markers_fired():
    _, markers = classify_topicality("Biblia Hebraica (Prolegomena)")
    assert markers == ["biblia hebraica"]


# --------------------------------------------------------------------------
# Extent parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pages,expected", [
    ("67-104", 38),
    ("281–345", 65),          # en dash
    ("173—238", 66),          # em dash
    ("105-162", 58),
    # Scholarly citation abbreviates the end page: 247–92 means 247–292.
    ("247–92", 46),
    ("1198–204", 7),
    ("pp. 141-142", 2),
    ("141", 1),
    ("", None),
    (None, None),
    ("n.p.", None),
])
def test_parse_extent(pages, expected):
    assert parse_extent(pages) == expected


def test_parse_extent_rejects_an_implausible_range():
    assert parse_extent("1-9999") is None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _work(**kwargs) -> FocusedWork:
    """Build a FocusedWork with defaults.

    :param kwargs: Field overrides.
    :returns: The work.
    """
    defaults = dict(title="t", source_type="Article", fragments=10, extent=20,
                    topicality="core")
    defaults.update(kwargs)
    return FocusedWork(**defaults)


def test_density_and_impact():
    work = _work(fragments=19, extent=38)
    assert work.density == 0.5
    assert work.impact == pytest.approx(9.5)


def test_impact_beats_density_for_ranking():
    """A 2-page note cited 3 times must not outrank a 38-page article cited 19."""
    note = _work(fragments=3, extent=2)
    article = _work(fragments=19, extent=38)
    assert note.density > article.density      # density alone gets this wrong
    assert article.impact > note.impact        # impact gets it right


def test_impact_is_zero_without_an_extent():
    assert _work(extent=None).impact == 0.0
    assert _work(extent=None).density is None


@pytest.mark.parametrize("source_type,unit", [
    ("Article", "article"),
    ("Book Section", "chapter"),
    ("Book", "book"),
    ("Dissertation", "dissertation"),
    ("", "unknown"),
])
def test_unit(source_type, unit):
    assert _work(source_type=source_type).unit == unit


def test_splice_free_for_a_modest_article():
    assert _work(source_type="Article", extent=38).splice_free is True


def test_splice_free_rejects_an_oversized_article():
    """A 400-page 'article' is a volume; it cannot be scanned casually."""
    assert _work(source_type="Article", extent=400).splice_free is False


def test_splice_free_rejects_an_article_of_unknown_extent():
    assert _work(source_type="Article", extent=None).splice_free is False


def test_splice_free_for_books_needs_core_topicality():
    assert _work(source_type="Book", extent=None, topicality="core").splice_free is True
    assert _work(source_type="Book", extent=None, topicality="adjacent").splice_free is False


def test_reference_works_are_never_splice_free():
    assert _work(topicality="reference", extent=10).splice_free is False


@pytest.mark.parametrize("url,hint", [
    ("https://www.jstor.org/stable/123", "institutional (JSTOR)"),
    ("https://www.lib.cam.ac.uk/x", "open (Cambridge Genizah Unit)"),
    ("https://www.academia.edu/42934528", "likely free (Academia.edu)"),
    ("https://www.persee.fr/doc/x", "open (Persée)"),
    ("https://example.invalid/thing", "example.invalid"),
    ("", ""),
])
def test_access_hint(url, hint):
    assert _work(url=url).access_hint == hint


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_group_works_routes_by_acquisition_shape():
    works = [
        _work(title="article", source_type="Article", extent=30, topicality="core"),
        _work(title="book", source_type="Book", extent=None, topicality="core"),
        _work(title="fatimid book", source_type="Book", extent=None, topicality="adjacent"),
        _work(title="dictionary", source_type="Book", extent=None, topicality="reference"),
        _work(title="mystery", source_type="Book", extent=None, topicality="unclear"),
    ]
    groups = group_works(works)
    assert [w.title for w in groups["self_contained_articles"]] == ["article"]
    assert [w.title for w in groups["focused_monographs"]] == ["book"]
    # Everything here is cited by Genizah fragments, so an adjacent marker is a
    # positive signal needing a glance, not a reason to demote to splicing.
    assert [w.title for w in groups["probable_monographs"]] == ["fatimid book"]
    assert [w.title for w in groups["splice_required"]] == ["dictionary"]
    assert [w.title for w in groups["off_topic"]] == ["mystery"]


def test_group_works_drops_uncited_works():
    assert all(not items for items in group_works([_work(fragments=0)]).values())


def test_articles_are_ranked_by_impact_and_books_by_citations():
    articles = [
        _work(title="dense-small", source_type="Article", fragments=3, extent=2),
        _work(title="weighty", source_type="Article", fragments=19, extent=38),
    ]
    books = [
        _work(title="less-cited", source_type="Book", extent=None, fragments=50),
        _work(title="most-cited", source_type="Book", extent=None, fragments=400),
    ]
    groups = group_works(articles + books)
    assert [w.title for w in groups["self_contained_articles"]] == ["weighty", "dense-small"]
    assert [w.title for w in groups["focused_monographs"]] == ["most-cited", "less-cited"]


# --------------------------------------------------------------------------
# Already-held detection
# --------------------------------------------------------------------------


def test_already_scanned_matches_by_title():
    scanned = {"india traders of the middle ages documents from the cairo geniza": "india_traders"}
    assert _already_scanned("India traders of the middle ages: documents from the Cairo Geniza",
                            scanned) is True
    assert _already_scanned("A completely different book", scanned) is False


def test_already_scanned_ignores_short_titles():
    """A short key must not match a long scanned title by containment."""
    scanned = {"india traders of the middle ages documents from the cairo geniza": "india_traders"}
    assert _already_scanned("Geniza", scanned) is False
    assert _already_scanned("", scanned) is False


def test_build_works_skips_works_already_held():
    rows = [
        {"title": "Held Book With A Long Enough Title", "source_books": ["some_dir"],
         "fragments": 100},
        {"title": "India traders of the middle ages", "source_books": None, "fragments": 50},
        {"title": "A Wanted Geniza Study", "source_books": None, "fragments": 10},
    ]
    scanned = {"india traders of the middle ages documents from the cairo geniza": "india_traders"}
    works = build_works(rows, scanned)
    assert [w.title for w in works] == ["A Wanted Geniza Study"]


def test_build_works_skips_untitled_rows():
    assert build_works([{"title": None, "citation": None, "fragments": 5}]) == []
