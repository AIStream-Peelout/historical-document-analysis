"""Tests for the source-availability survey.

The survey is only useful if it (a) turns the KG's messy citation titles into
queries a library catalogue will answer, (b) recognises a romanised Hebrew
record as the same work, and (c) ranks by how much effort acquisition takes.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from src.datasets.indexing.bibliography.availability_survey import (
    ACCESS_TIERS,
    SourceHit,
    WorkAvailability,
    _distinctive_word,
    _marc_oclc,
    _marc_title,
    brandeis_queries,
    catalogue_author,
    load_targets,
    normalize,
    search_variants,
    title_similarity,
    titles_match,
)

MARC_XML = """<?xml version="1.0"?>
<searchRetrieveResponse xmlns="http://www.loc.gov/zing/srw/">
  <numberOfRecords>1</numberOfRecords>
  <records><record><recordData>
  <record xmlns="http://www.loc.gov/MARC21/slim">
    <datafield tag="035"><subfield code="a">(OCoLC)395151</subfield></datafield>
    <datafield tag="245">
      <subfield code="a">Otsar kitve-ha-yad ha-Talmudiyim /</subfield>
      <subfield code="b">a catalogue</subfield>
    </datafield>
    <datafield tag="880"><subfield code="a">אוצר כתבי היד התלמודיים</subfield></datafield>
    <datafield tag="AVA">
      <subfield code="b">MAIN</subfield>
      <subfield code="c">Stacks</subfield>
      <subfield code="d">BM501 .S87 2012</subfield>
      <subfield code="e">available</subfield>
    </datafield>
  </record>
  </recordData></record></records>
</searchRetrieveResponse>
"""


def _record():
    """Parse the fixture MARC record.

    :returns: The ``marc:record`` element.
    """
    ns = {"marc": "http://www.loc.gov/MARC21/slim"}
    return ET.fromstring(MARC_XML).find(".//marc:record", ns)


# --------------------------------------------------------------------------
# Title handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("title,expected", [
    # The KG appends volume designators that no catalogue carries.
    ("A Mediterranean society: the Jewish communities Vol: 2",
     ["A Mediterranean society: the Jewish communities", "A Mediterranean society"]),
    ("אוצר כתבי היד התלמודיים כרך: 2", ["אוצר כתבי היד התלמודיים"]),
    # A trailing parenthetical date range breaks phrase matching.
    ("Palestine During the First Muslim Period (634–1099)",
     ["Palestine During the First Muslim Period"]),
    # An embedded author prefix must not become the "main title".
    ("Shaked, Shaul: A tentative bibliography of Geniza documents",
     ["A tentative bibliography of Geniza documents"]),
    ("", []),
])
def test_search_variants(title, expected):
    assert search_variants(title) == expected


def test_search_variants_skips_a_too_short_main_title():
    """`Geniza: studies` must not spawn a one-word query."""
    assert search_variants("Geniza: studies in the medieval world") == [
        "Geniza: studies in the medieval world"
    ]


def test_search_variants_deduplicates():
    assert search_variants("A single title") == ["A single title"]


@pytest.mark.parametrize("left,right,expected", [
    ("Trade and Institutions", "Trade and institutions in the medieval Mediterranean", 1.0),
    ("A Mediterranean society", "The Jews of Egypt", 0.0),
])
def test_title_similarity(left, right, expected):
    assert title_similarity(left, right) == pytest.approx(expected, abs=0.01)


def test_title_similarity_uses_containment_not_symmetry():
    """A catalogue subtitle is far longer than a citation; that must not hurt."""
    short = "Trade and Institutions in the Medieval Mediterranean"
    long = ("Trade and institutions in the medieval Mediterranean : "
            "the Geniza merchants and their business world")
    assert title_similarity(short, long) > 0.9


def test_title_similarity_handles_empties():
    assert title_similarity("", "anything") == 0.0
    assert title_similarity(None, None) == 0.0


@pytest.mark.parametrize("query,candidate", [
    # Containment scores 1.0 on every one of these and every one is wrong.
    ("Arabica", "100% Arabica"),                       # a film
    ("Textus", "Acta Andreae : textus"),
    ("Jewish Social Studies", "Bibliography of Jewish social studies."),
    ("תרבות", "1900־2000 : מאה שנות תרבות"),
    # Prefixed words mean the catalogue record is a different work.
    ("India Book 4: Halfon the Traveling Merchant Scholar", "Adapted for India Book 4"),
])
def test_titles_match_rejects_a_title_that_merely_contains_the_query(query, candidate):
    matched, _ = titles_match(query, candidate)
    assert matched is False


@pytest.mark.parametrize("query,candidate", [
    ("Geonica", "Geonica."),
    # A catalogue legitimately extends a citation title with a subtitle.
    ("Masoreten des Ostens", "Masoreten des Ostens, die ältesten punktierten Handschriften"),
    ("Ancient Jewish Magic: A History", "Ancient Jewish magic : a history"),
    ("Arabic Documents from Medieval Nubia", "Arabic Documents from Medieval Nubia."),
    ("Bulletin of the School of Oriental and African Studies",
     "Bulletin of the School of Oriental and African Studies, University of London"),
    # A leading article on either side must not block the match.
    ("The Cairo Geniza collection", "Cairo Geniza collection"),
])
def test_titles_match_accepts_a_genuine_extension(query, candidate):
    matched, score = titles_match(query, candidate)
    assert matched is True
    assert score > 0.5


def test_titles_match_handles_empties():
    assert titles_match("", "anything") == (False, 0.0)
    assert titles_match("anything", None) == (False, 0.0)


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Histoire des prix, l'Orient médiéval") == (
        "histoire des prix l orient medieval"
    )


@pytest.mark.parametrize("title,word", [
    ("Palestine During the First Muslim Period", "Palestine"),
    ("The Ketubba Traditions", "Traditions"),
    ("A b c", ""),          # nothing long enough to be distinctive
    ("", ""),
])
def test_distinctive_word(title, word):
    assert _distinctive_word(title) == word


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("authors,expected", [
    ("Gil, Moshe", "Gil, Moshe"),
    ("Goitein, Shelomo Dov; Friedman, Mordechai", "Goitein, Shelomo Dov"),
    # Hebrew-only author forms do not match a romanised creator index.
    ("מרדכי עקיבא פרידמן", ""),
    ("", ""),
])
def test_catalogue_author(authors, expected):
    assert catalogue_author(authors) == expected


def test_brandeis_queries_are_ordered_and_flagged():
    queries = brandeis_queries(
        "Palestine During the First Muslim Period (634–1099)", "Gil, Moshe"
    )
    assert queries[0] == ('alma.title="Palestine During the First Muslim Period"', True)
    # The last resort pairs author with one distinctive word, so an edition
    # published under a different title is still found.
    assert queries[-1] == ('alma.creator="Gil, Moshe" and alma.title="Palestine"', False)
    assert queries[-1][1] is False


def test_brandeis_queries_without_a_usable_author():
    queries = brandeis_queries("אוצר כתבי היד התלמודיים כרך: 2", "יעקב זוסמן")
    assert all(flag for _, flag in queries)     # title queries only
    assert queries[0] == ('alma.title="אוצר כתבי היד התלמודיים"', True)


# --------------------------------------------------------------------------
# MARC parsing
# --------------------------------------------------------------------------


def test_marc_title_reads_both_scripts():
    record = _record()
    assert _marc_title(record, "245") == "Otsar kitve-ha-yad ha-Talmudiyim / a catalogue"
    assert _marc_title(record, "880") == "אוצר כתבי היד התלמודיים"


def test_marc_oclc():
    assert _marc_oclc(_record()) == "395151"


def test_marc_oclc_absent():
    empty = ET.fromstring(
        '<record xmlns="http://www.loc.gov/MARC21/slim"></record>'
    )
    assert _marc_oclc(empty) == ""


def test_hebrew_query_matches_a_romanised_record_via_880():
    """The 880 field is what links a Hebrew citation to a romanised record."""
    record = _record()
    hebrew = "אוצר כתבי היד התלמודיים"
    assert title_similarity(hebrew, _marc_title(record, "245")) == 0.0
    assert title_similarity(hebrew, _marc_title(record, "880")) == 1.0


# --------------------------------------------------------------------------
# Tier classification
# --------------------------------------------------------------------------


def _work(**accesses) -> WorkAvailability:
    """Build a WorkAvailability with the given per-provider accesses.

    :param accesses: provider -> access string.
    :returns: The work.
    """
    work = WorkAvailability(title="t", year="1980", authors="a", fragments=100)
    for provider, access in accesses.items():
        work.hits.append(SourceHit(provider=provider, access=access,
                                   url=f"https://example.test/{provider}"))
    return work


@pytest.mark.parametrize("accesses,tier", [
    ({"internet_archive": "full_download", "brandeis": "holding"}, "open_online"),
    ({"hathitrust": "full_download"}, "open_online"),
    ({"internet_archive": "borrow", "brandeis": "holding"}, "borrow_online"),
    ({"brandeis": "holding"}, "at_brandeis"),
    ({"brandeis": "search_only", "open_library": "search_only"}, "search_only"),
    ({"brandeis": "none"}, "elsewhere"),
    ({}, "elsewhere"),
])
def test_tier_prefers_the_least_effort_route(accesses, tier):
    assert _work(**accesses).tier == tier


def test_tier_does_not_promote_a_catalogued_but_unheld_record():
    """A Brandeis record with no local item is not something you can go get."""
    work = _work(brandeis="search_only")
    assert work.tier == "search_only"
    assert work.brandeis_call_number == ""


def test_all_tiers_are_reachable():
    """Guards the tier list against a classifier that can never emit one."""
    reachable = {
        _work(internet_archive="full_download").tier,
        _work(internet_archive="borrow").tier,
        _work(brandeis="holding").tier,
        _work(open_library="search_only").tier,
        _work().tier,
    }
    assert reachable == set(ACCESS_TIERS)


def test_best_url_follows_the_tier():
    work = WorkAvailability(title="t", year="", authors="", fragments=1, hits=[
        SourceHit(provider="brandeis", access="holding", url="https://brandeis.test"),
        SourceHit(provider="internet_archive", access="none", url="https://ia.test"),
    ])
    assert work.tier == "at_brandeis"
    assert work.best_url == "https://brandeis.test"


def test_best_url_ignores_unmatched_providers():
    """A provider that found nothing must not contribute its search URL."""
    work = WorkAvailability(title="t", year="", authors="", fragments=1, hits=[
        SourceHit(provider="hathitrust", access="none", url="https://search.test/nothing"),
        SourceHit(provider="open_library", access="search_only", url="https://ol.test/found"),
    ])
    assert work.best_url == "https://ol.test/found"


def test_brandeis_call_number_surfaces_the_first_holding():
    work = WorkAvailability(title="t", year="", authors="", fragments=1, hits=[
        SourceHit(provider="brandeis", access="holding",
                  detail={"call_number": "BM501 .S87 2012"}),
    ])
    assert work.brandeis_call_number == "BM501 .S87 2012"


# --------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------


def test_load_targets_filters_and_ranks(tmp_path):
    path = tmp_path / "citation_priority.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("fragments,title,year,authors,have_locally,kind\n")
        handle.write("500,Already held,1980,A,True,monograph\n")
        handle.write("400,Big target,1980,B,False,monograph\n")
        handle.write("300,A journal run,1980,C,False,serial\n")
        handle.write("5,Too minor,1980,D,False,monograph\n")
        handle.write("100,Second target,1980,E,False,monograph\n")
    targets = load_targets(path, limit=10, min_fragments=10, kinds=["monograph"])
    assert [t["title"] for t in targets] == ["Big target", "Second target"]


def test_load_targets_honours_the_limit(tmp_path):
    path = tmp_path / "p.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("fragments,title,year,authors,have_locally,kind\n")
        for n in range(20, 0, -1):
            handle.write(f"{n * 10},Work {n},1980,A,False,monograph\n")
    targets = load_targets(path, limit=3, min_fragments=0, kinds=["monograph"])
    assert [t["title"] for t in targets] == ["Work 20", "Work 19", "Work 18"]
