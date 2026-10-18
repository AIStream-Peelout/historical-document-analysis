"""Tests for the geocoder guards and curated corrections in geocode_places.py.

These cover the failure that put Goshen (the biblical Land of Goshen, Egypt)
in Goshen, Indiana: a PGP place Princeton left unlocated was geocoded once
book enrichment added a second data source, and the Google hit was accepted
with no geographic plausibility check.
"""

import pytest

from src.datasets.indexing.neo4j.geocode_corrections import (
    INSTITUTION_CORRECTIONS,
    PLACE_CORRECTIONS,
)
from src.datasets.indexing.neo4j.geocode_places import (
    COUNTRY_TO_REGION,
    _apply_all_corrections,
    _apply_correction,
    _geocode_google,
    _has_curated_query,
    _in_genizah_world,
    _pgp_left_unlocated,
)


@pytest.mark.parametrize("label,lat,lng", [
    ("Fustat", 30.0047, 31.2375),
    ("Córdoba", 37.89, -4.78),
    ("London", 51.5097, -0.1181),
    ("Kolkata", 22.5742, 88.3617),
    ("Java (al-Jāwa)", -6.2114, 106.8447),
    ("China (al-Ṣīn)", 33.0994, 105.7689),
])
def test_in_genizah_world_accepts_genizah_places(label, lat, lng):
    assert _in_genizah_world(lat, lng), label


@pytest.mark.parametrize("label,lat,lng", [
    ("Goshen, Indiana", 41.5892, -85.8346),
    ("St. Petersburg, Florida", 27.7671, -82.6384),
    ("Christchurch, New Zealand", -43.5297, 172.6271),
    ("KwaZulu-Natal", -29.7246, 30.7367),
])
def test_in_genizah_world_rejects_homonyms_far_away(label, lat, lng):
    assert not _in_genizah_world(lat, lng), label


@pytest.mark.parametrize("data_sources", [
    ["pgp"],
    ["pgp", "enriched"],          # Goshen: enrichment must not unlock geocoding
    ["pgp", "merged"],
    ["pgp", "enriched", "merged"],
])
def test_pgp_place_without_coords_stays_unlocated(data_sources):
    assert _pgp_left_unlocated(data_sources, has_curated_query=False)


def test_pgp_place_with_curated_query_is_geocoded():
    assert not _pgp_left_unlocated(["pgp", "enriched"], has_curated_query=True)


@pytest.mark.parametrize("data_sources", [["enriched"], [], None])
def test_non_pgp_place_is_geocoded(data_sources):
    assert not _pgp_left_unlocated(data_sources, has_curated_query=False)


def test_curated_query_detection():
    assert _has_curated_query("al-Andalus")          # HISTORICAL_OVERRIDES
    assert _has_curated_query("Tripoli (Libya)")     # parenthetical key itself
    assert _has_curated_query("Fusṭāṭ")
    assert not _has_curated_query("Goshen")
    assert not _has_curated_query("al-Bannā")


def test_placenormalizer_mapping_alone_is_not_curated():
    # PlaceNormalizer maps Ṣūr → 'Tyre, Lebanon', but the Google step only sends
    # HISTORICAL_OVERRIDES queries, so the bare 'Ṣūr' (→ Sur, Oman) would go out.
    assert not _has_curated_query("Ṣūr")


_ALL_CORRECTIONS = ([("place", n, c) for n, c in PLACE_CORRECTIONS.items()]
                    + [("institution", n, c) for n, c in INSTITUTION_CORRECTIONS.items()])


@pytest.mark.parametrize("kind,node,correction", _ALL_CORRECTIONS,
                         ids=[f"{k}:{n}" for k, n, _ in _ALL_CORRECTIONS])
def test_corrections_are_well_formed(kind, node, correction):
    assert node == node.strip() and node
    assert correction.note.strip()
    if correction.is_excluded:
        assert correction.lng is None
        return
    assert -90 <= correction.lat <= 90 and -180 <= correction.lng <= 180
    assert correction.region or COUNTRY_TO_REGION.get(correction.country), \
        f"{node}: no region for country {correction.country!r}"
    if kind == "place":
        assert _in_genizah_world(correction.lat, correction.lng), node


def test_goshen_is_pinned_in_the_nile_delta():
    goshen = PLACE_CORRECTIONS["Goshen"]
    assert goshen.country == "Egypt"
    assert 30 < goshen.lat < 31.5 and 31 < goshen.lng < 32.5


class _Result:
    """Stand-in for a Neo4j result exposing ``data()``."""

    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _RecordingSession:
    """Minimal stand-in for a Neo4j session that records ``run`` calls.

    :param existing: Node names the fake graph contains.
    """

    def __init__(self, existing=()):
        self.calls = []
        self.existing = set(existing)

    def run(self, query, **params):
        self.calls.append((query, params))
        names = params.get("names") or []
        return _Result([{"name": n, "lat": 0.0, "lng": 0.0} for n in names if n in self.existing])

    def writes(self):
        return [(q, p) for q, p in self.calls if "SET" in q]


class _FakeGmaps:
    """Google client returning one canned result per query string."""

    def __init__(self, by_query):
        self.by_query = by_query
        self.asked = []

    def geocode(self, query):
        self.asked.append(query)
        lat, lng, country = self.by_query[query]
        return [{"geometry": {"location": {"lat": lat, "lng": lng}},
                 "types": ["locality", "political"],
                 "address_components": [{"types": ["country"], "long_name": country}],
                 "formatted_address": f"{query}, {country}"}]


def test_google_skips_out_of_world_hit_and_tries_next_query():
    gmaps = _FakeGmaps({"Goshen": (41.5892, -85.8346, "United States"),
                        "Goshen, Egypt": (30.42, 31.56, "Egypt")})
    result = _geocode_google(["Goshen", "Goshen, Egypt"], gmaps, accept=_in_genizah_world)
    assert gmaps.asked == ["Goshen", "Goshen, Egypt"]
    assert result["country"] == "Egypt"


def test_google_without_accept_keeps_first_hit():
    gmaps = _FakeGmaps({"Harvard": (42.3741, -71.1167, "United States")})
    assert _geocode_google(["Harvard"], gmaps)["country"] == "United States"


def test_apply_all_corrections_writes_found_nodes_only():
    session = _RecordingSession(existing={"Goshen", "London"})
    _apply_all_corrections(session, ["Place"], dry_run=False, verbose=False)
    written = {p["name"] for _, p in session.writes()}
    assert written == {"Goshen", "London"}


def test_apply_all_corrections_dry_run_writes_nothing():
    session = _RecordingSession(existing=set(PLACE_CORRECTIONS) | set(INSTITUTION_CORRECTIONS))
    _apply_all_corrections(session, ["Place", "Institution"], dry_run=True, verbose=True)
    assert session.writes() == []


def test_apply_all_corrections_respects_labels():
    session = _RecordingSession(existing=set(INSTITUTION_CORRECTIONS))
    _apply_all_corrections(session, ["Institution"], dry_run=False, verbose=False)
    assert all("n:Institution" in q for q, _ in session.writes())
    assert len(session.writes()) == len(INSTITUTION_CORRECTIONS)


def test_apply_correction_overwrites_stale_location_fields():
    session = _RecordingSession()
    _apply_correction(session, "Place", "Goshen", PLACE_CORRECTIONS["Goshen"])
    query, params = session.calls[0]
    assert "MATCH (n:Place {name: $name})" in query
    assert "n.osm_display_name = null" in query
    assert params["country"] == "Egypt"
    assert params["city"] is None           # was 'Goshen' from the Indiana match
    assert params["region"] == "North Africa"
    assert params["source"] == "manual" and params["excluded"] is False


def test_apply_correction_removes_pin_for_excluded_node():
    session = _RecordingSession()
    _apply_correction(session, "Institution", "Cairo's Janissary corps",
                      INSTITUTION_CORRECTIONS["Cairo's Janissary corps"])
    query, params = session.calls[0]
    assert "MATCH (n:Institution {name: $name})" in query
    assert params["lat"] is None and params["lng"] is None
    assert params["country"] is None
    assert params["source"] == "excluded" and params["excluded"] is True
