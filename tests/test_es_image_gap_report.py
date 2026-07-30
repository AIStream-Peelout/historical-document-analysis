"""Tests for the Elasticsearch image-gap report.

The report groups imageless shelfmarks by holding institution. The grouping key
must come from the canonical id (which is built from the institution token), not
from the raw ``institution`` string, because each source spells the same library
differently (``Rylands`` / ``Manchester`` / ``John Rylands Library``).
"""

import pytest

from src.datasets.indexing.es_image_gap_report import (
    GapRecord,
    _to_record,
    build_gap_query,
    institution_group,
    sort_key,
    summarize,
)


@pytest.mark.parametrize("canonical_id,institution,group", [
    # Registry tokens are recovered exactly from the canonical id prefix.
    ("Cambridge_CUL_T_S_10J5_6", "Cambridge University Library", "Cambridge_CUL"),
    ("Cambridge_Mosseri_II_128_1", "Mosseri", "Cambridge_Mosseri"),
    ("Cambridge_Lewis_Gibson_Talmud_1_55", "CUL & Bodl", "Cambridge_Lewis_Gibson"),
    ("New_York_JTS_ENA_2158_40", "Jewish Theological Seminary", "New_York_JTS"),
    ("Manchester_JRL_B_1924", "Rylands", "Manchester_JRL"),
    ("Oxford_Bodleian_MS_Heb_c_28_47", "Bodleian Library, Oxford", "Oxford_Bodleian"),
    ("Budapest_MTA_DK_120", "Hungarian Academy of Sciences", "Budapest_MTA"),
    ("Geneva_119", "Geneva", "Geneva"),
    # A bare token with no core still resolves to itself.
    ("Cambridge_CUL", "", "Cambridge_CUL"),
])
def test_institution_group_from_canonical_prefix(canonical_id, institution, group):
    assert institution_group(canonical_id, institution) == group


def test_institution_group_prefers_longest_token():
    """A registry token must not be shadowed by a shorter one sharing a prefix."""
    assert institution_group("StPetersburg_NLR_Yevr_II_A_1", "") == "StPetersburg_NLR"
    assert institution_group("StPetersburg_IOM_D_55", "") == "StPetersburg_IOM"


def test_institution_group_falls_back_to_raw_institution():
    """Slug-fallback ids (unregistered collections) resolve from institution text."""
    # Not a registry prefix, but the institution text maps to a registry token.
    assert institution_group("Halper_214", "University of Pennsylvania") == "Philadelphia_CAJS"
    # Neither matches the registry: slug the institution rather than guess.
    assert institution_group("Toronto_Fisher_1", "Thomas Fisher Rare Book Library") == (
        "Thomas_Fisher_Rare_Book_Library"
    )
    # No institution text at all: first id segment, never empty.
    assert institution_group("Mittwoch_5", None) == "Mittwoch"


def test_build_gap_query_requires_both_signals():
    """The gap filter must catch an empty ``image_urls`` array, not just the flag."""
    query = build_gap_query()
    assert {"term": {"has_images": False}} in query["bool"]["filter"]
    assert {"exists": {"field": "image_urls"}} in query["bool"]["must_not"]


def _record(**overrides) -> GapRecord:
    """Build a GapRecord with sensible defaults for sort/summary tests.

    :param overrides: Field values to override.
    :returns: A populated record.
    """
    defaults = dict(
        canonical_id="Cambridge_CUL_T_S_1",
        shelf_mark="T-S 1",
        institution_group="Cambridge_CUL",
        collection="Taylor-Schechter",
        institution_raw="Cambridge University Library",
        sources_present="pgp",
        ktiv_manifest_url="",
        ktiv_known=False,
        description_chars=0,
        has_bib=False,
        has_transcriptions=False,
        completeness_score=0.2,
    )
    defaults.update(overrides)
    return GapRecord(**defaults)


def test_to_record_maps_es_source():
    record = _to_record(
        {
            "canonical_id": "Budapest_MTA_Kaufmann_A_433",
            "shelf_mark": "Kaufmann A 433",
            "institution": " Hungarian Academy of Sciences, Budapest ",
            "collection": " David Kaufmann ",
            "sources_present": ["ktiv", "fjp"],
            "description": "abc",
            "has_bib": True,
            "ktiv_iiif_manifest_url": "https://iiif.nli.org.il/x/manifest",
            "completeness_score": 0.2,
        },
        es_id="ignored-when-canonical-id-present",
    )
    assert record.institution_group == "Budapest_MTA"
    assert record.collection == "David Kaufmann"
    assert record.sources_present == "fjp+ktiv"  # sorted, stable across runs
    assert record.description_chars == 3
    assert record.ktiv_ready is True
    assert record.ktiv_known is True
    assert record.metadata_rich is True


def test_to_record_falls_back_to_es_id():
    record = _to_record({"shelf_mark": "T-S 1"}, es_id="Cambridge_CUL_T_S_1")
    assert record.canonical_id == "Cambridge_CUL_T_S_1"
    assert record.institution_group == "Cambridge_CUL"
    assert record.ktiv_ready is False
    assert record.metadata_rich is False


def test_metadata_rich_accepts_any_substantive_signal():
    assert _record(description_chars=10).metadata_rich is True
    assert _record(has_bib=True).metadata_rich is True
    assert _record(has_transcriptions=True).metadata_rich is True
    assert _record().metadata_rich is False


def test_sort_key_orders_by_collection_then_scrape_readiness():
    ready = _record(canonical_id="A", ktiv_manifest_url="https://m")
    rich = _record(canonical_id="B", description_chars=500)
    bare = _record(canonical_id="C")
    unspecified_collection = _record(canonical_id="D", collection="")
    ordered = sorted([bare, unspecified_collection, rich, ready], key=sort_key)
    assert [r.canonical_id for r in ordered] == ["A", "B", "C", "D"]


def test_sort_key_groups_institutions_together():
    rows = [
        _record(canonical_id="Oxford_Bodleian_1", institution_group="Oxford_Bodleian"),
        _record(canonical_id="Cambridge_CUL_2"),
        _record(canonical_id="Oxford_Bodleian_2", institution_group="Oxford_Bodleian"),
        _record(canonical_id="Cambridge_CUL_1"),
    ]
    groups = [r.institution_group for r in sorted(rows, key=sort_key)]
    assert groups == ["Cambridge_CUL", "Cambridge_CUL", "Oxford_Bodleian", "Oxford_Bodleian"]


def test_summarize_counts_and_ranks_by_gap_size():
    records = [
        _record(canonical_id="Cambridge_CUL_1", description_chars=10),
        _record(canonical_id="Cambridge_CUL_2", collection="Oriental Manuscripts"),
        _record(
            canonical_id="New_York_JTS_1",
            institution_group="New_York_JTS",
            collection="Elkan Nathan Adler",
            sources_present="ktiv+pgp",
            ktiv_manifest_url="https://m",
            ktiv_known=True,
            has_bib=True,
        ),
    ]
    summaries = summarize(records)
    assert [s.institution_group for s in summaries] == ["Cambridge_CUL", "New_York_JTS"]
    cul, jts = summaries
    assert cul.total == 2
    assert cul.metadata_rich == 1
    assert cul.ktiv_ready == 0
    assert cul.collections == {"Taylor-Schechter": 1, "Oriental Manuscripts": 1}
    assert jts.total == jts.metadata_rich == jts.ktiv_ready == jts.ktiv_known == 1
    assert jts.source_mix == {"ktiv+pgp": 1}


def test_summarize_labels_missing_collection():
    summaries = summarize([_record(collection="")])
    assert summaries[0].collections == {"(unspecified)": 1}
