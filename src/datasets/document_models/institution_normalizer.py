#!/usr/bin/env python3
"""
Institution name normalization for the Cairo Genizah knowledge graph.

Maps the many variant spellings / abbreviations the LLM produces for holding
institutions onto a single canonical name that is used as the Neo4j
``Institution.name`` merge key.  Longest-matching key wins, so more specific
entries take precedence over shorter ones.

Usage
-----
from src.datasets.document_models.institution_normalizer import InstitutionNormalizer

canonical = InstitutionNormalizer.normalize("JTS Library, New York")
# → "Jewish Theological Seminary"

meta = InstitutionNormalizer.get_metadata("Taylor-Schechter Collection")
# → {"city": "Cambridge", "country": "United Kingdom", "region": "Europe"}
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.datasets.document_models.entity_normalizer import EntityNormalizer


# ---------------------------------------------------------------------------
# Canonical institution registry
#
# Keys are lowercase substrings.  Matching is done by scanning all keys in
# descending length order and returning the first one that is a substring of
# the normalised input.  More specific keys must therefore be longer than their
# more general siblings (e.g. "hebrew union college" before "hebrew union").
#
# Values are the canonical display name used as Neo4j Institution.name.
# ---------------------------------------------------------------------------
_CANONICAL: dict[str, str] = {

    # ── Cambridge University Library ────────────────────────────────────────
    "cambridge university library":         "Cambridge University Library",
    "taylor-schechter genizah collection":  "Cambridge University Library",
    "taylor-schechter collection":          "Cambridge University Library",
    "taylor-schechter":                     "Cambridge University Library",
    "t-s collection":                       "Cambridge University Library",
    "cul genizah":                          "Cambridge University Library",
    "cambridge university library genizah": "Cambridge University Library",

    # ── Jewish Theological Seminary ─────────────────────────────────────────
    "jewish theological seminary of america":   "Jewish Theological Seminary",
    "jewish theological seminary library":      "Jewish Theological Seminary",
    "jewish theological seminary":              "Jewish Theological Seminary",
    "jtsa":                                     "Jewish Theological Seminary",
    "jts library":                              "Jewish Theological Seminary",
    # "jts" alone is short and risky as a substring; keep it but it only
    # matches when no longer key fires first.
    "jts":                                      "Jewish Theological Seminary",

    # ── Bodleian Library ────────────────────────────────────────────────────
    "bodleian library, oxford":     "Bodleian Library, Oxford",
    "bodleian library":             "Bodleian Library, Oxford",
    "bodleian":                     "Bodleian Library, Oxford",

    # ── British Library ─────────────────────────────────────────────────────
    "british library, london":  "British Library, London",
    "british library":          "British Library, London",

    # ── John Rylands Library ────────────────────────────────────────────────
    # "Rylands Genizah"/"Rylands Genizah Fragments" are how the Genizah
    # collection at this library gets named in-text; without these aliases
    # they geocode as bare strings and land on the Cape Town neighbourhood
    # also named "Rylands".
    "john rylands library, university of manchester":   "John Rylands Library, University of Manchester",
    "john rylands library":                             "John Rylands Library, University of Manchester",
    "john rylands":                                     "John Rylands Library, University of Manchester",
    "rylands library":                                  "John Rylands Library, University of Manchester",
    "rylands genizah":                                  "John Rylands Library, University of Manchester",
    "rylands":                                           "John Rylands Library, University of Manchester",

    # ── Alliance Israélite Universelle ──────────────────────────────────────
    "alliance israélite universelle, paris":    "Alliance Israélite Universelle, Paris",
    "alliance israelite universelle":           "Alliance Israélite Universelle, Paris",
    "alliance israélite universelle":           "Alliance Israélite Universelle, Paris",
    "aiu paris":                                "Alliance Israélite Universelle, Paris",

    # ── Russian National Library ─────────────────────────────────────────────
    "russian national library, st. petersburg": "Russian National Library, St. Petersburg",
    "russian national library":                 "Russian National Library, St. Petersburg",
    "rnl st. petersburg":                       "Russian National Library, St. Petersburg",

    # ── Hungarian Academy of Sciences ────────────────────────────────────────
    "hungarian academy of sciences, budapest":  "Hungarian Academy of Sciences, Budapest",
    "hungarian academy of sciences":            "Hungarian Academy of Sciences, Budapest",
    "david kaufmann collection":                "Hungarian Academy of Sciences, Budapest",

    # ── University of Pennsylvania ───────────────────────────────────────────
    "university of pennsylvania, katz center":          "University of Pennsylvania, Katz Center",
    "university of pennsylvania, center for advanced judaic studies":
                                                        "University of Pennsylvania, Katz Center",
    "dropsie college":                                  "University of Pennsylvania, Katz Center",
    "cajs":                                             "University of Pennsylvania, Katz Center",

    # ── Smithsonian / Freer–Sackler ─────────────────────────────────────────
    "freer gallery of art, arthur m. sackler gallery":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",
    "freer gallery of art, arthur m sackler":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",
    "arthur m. sackler gallery":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",
    "freer gallery of art":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",
    "freer gallery":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",
    "sackler gallery":
        "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution",

    # ── Hebrew Union College ─────────────────────────────────────────────────
    "hebrew union college - jewish institute of religion":
        "Hebrew Union College - Jewish Institute of Religion",
    "hebrew union college-jewish institute of religion":
        "Hebrew Union College - Jewish Institute of Religion",
    "hebrew union college":
        "Hebrew Union College - Jewish Institute of Religion",
    "huc-jir":  "Hebrew Union College - Jewish Institute of Religion",
    "huc":      "Hebrew Union College - Jewish Institute of Religion",
    "klau library":
        "Hebrew Union College - Jewish Institute of Religion",

    # ── National Library of Israel ───────────────────────────────────────────
    "national library of israel":   "National Library of Israel",
    "jewish national library":      "National Library of Israel",
    "jnul":                         "National Library of Israel",

    # ── Genizah Research Unit / Other Cambridge ───────────────────────────────
    "genizah research unit":    "Genizah Research Unit, Cambridge University Library",
    "friedberg genizah project": "Friedberg Genizah Project",

    # ── Coptic Museum ────────────────────────────────────────────────────────
    "coptic museum, cairo":     "Coptic Museum, Cairo",
    "coptic museum":            "Coptic Museum, Cairo",

    # ── Columbia University ──────────────────────────────────────────────────
    "columbia university library":  "Columbia University Library",
    "butler library":               "Columbia University Library",
    "columbia university":          "Columbia University Library",

    # ── Brandeis University ──────────────────────────────────────────────────
    "brandeis university":  "Brandeis University",

    # ── Ben-Gurion University ────────────────────────────────────────────────
    "ben-gurion university of the negev":   "Ben-Gurion University of the Negev",
    "ben-gurion university":                "Ben-Gurion University of the Negev",

    # ── Bar-Ilan University ──────────────────────────────────────────────────
    "bar-ilan university":  "Bar-Ilan University",
    "bar ilan university":  "Bar-Ilan University",

    # ── Yad Ben-Zvi ──────────────────────────────────────────────────────────
    "yad izhak ben-zvi":    "Yad Izhak Ben-Zvi Institute",
    "ben-zvi institute":    "Yad Izhak Ben-Zvi Institute",
    "yad ben zvi":          "Yad Izhak Ben-Zvi Institute",

    # ── Jewish Museum ────────────────────────────────────────────────────────
    "jewish museum, new york":  "Jewish Museum, New York",
    "jewish museum":            "Jewish Museum, New York",
}

# ---------------------------------------------------------------------------
# Metadata: canonical name → {city, country, region}
# Used to seed Institution node properties so geocoding starts with good hints.
# ---------------------------------------------------------------------------
_METADATA: dict[str, dict[str, str]] = {
    "Cambridge University Library":
        {"city": "Cambridge", "country": "United Kingdom", "region": "Europe"},
    "Jewish Theological Seminary":
        {"city": "New York", "country": "United States", "region": "North America"},
    "Bodleian Library, Oxford":
        {"city": "Oxford", "country": "United Kingdom", "region": "Europe"},
    "British Library, London":
        {"city": "London", "country": "United Kingdom", "region": "Europe"},
    "John Rylands Library, University of Manchester":
        {"city": "Manchester", "country": "United Kingdom", "region": "Europe"},
    "Alliance Israélite Universelle, Paris":
        {"city": "Paris", "country": "France", "region": "Europe"},
    "Russian National Library, St. Petersburg":
        {"city": "St. Petersburg", "country": "Russia", "region": "Europe"},
    "Hungarian Academy of Sciences, Budapest":
        {"city": "Budapest", "country": "Hungary", "region": "Europe"},
    "University of Pennsylvania, Katz Center":
        {"city": "Philadelphia", "country": "United States", "region": "North America"},
    "Freer Gallery of Art, Arthur M. Sackler Gallery, Smithsonian Institution":
        {"city": "Washington D.C.", "country": "United States", "region": "North America"},
    "Hebrew Union College - Jewish Institute of Religion":
        {"city": "Cincinnati", "country": "United States", "region": "North America"},
    "National Library of Israel":
        {"city": "Jerusalem", "country": "Israel", "region": "Middle East"},
    "Genizah Research Unit, Cambridge University Library":
        {"city": "Cambridge", "country": "United Kingdom", "region": "Europe"},
    "Coptic Museum, Cairo":
        {"city": "Cairo", "country": "Egypt", "region": "North Africa"},
    "Columbia University Library":
        {"city": "New York", "country": "United States", "region": "North America"},
    "Brandeis University":
        {"city": "Waltham", "country": "United States", "region": "North America"},
    "Ben-Gurion University of the Negev":
        {"city": "Beersheba", "country": "Israel", "region": "Middle East"},
    "Bar-Ilan University":
        {"city": "Ramat Gan", "country": "Israel", "region": "Middle East"},
    "Yad Izhak Ben-Zvi Institute":
        {"city": "Jerusalem", "country": "Israel", "region": "Middle East"},
    "Jewish Museum, New York":
        {"city": "New York", "country": "United States", "region": "North America"},
    "Friedberg Genizah Project":
        {"city": "Toronto", "country": "Canada", "region": "North America"},
}

# Pre-sorted keys by descending length for O(n) longest-match scan.
_SORTED_KEYS: list[str] = sorted(_CANONICAL.keys(), key=len, reverse=True)


def _normalise_input(raw: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", raw)
    ascii_str = nfkd.encode("ascii", "ignore").decode()
    return " ".join(ascii_str.lower().split())


class InstitutionNormalizer(EntityNormalizer):
    """Normalize institution name variants to a single canonical display name.

    Matching is case-insensitive, diacritic-insensitive, and substring-based.
    Longer keys take priority over shorter ones to prevent "jts" from swallowing
    "jts library, new york" before the more specific key fires.
    """

    @classmethod
    def normalize(cls, raw: str) -> str:
        """Return the canonical institution name for *raw*.

        :param raw: Institution name in any variant form.
        :returns: Canonical name, or *raw* unchanged if no mapping is known.
        """
        if not raw or not raw.strip():
            return raw
        normalised = _normalise_input(raw)
        for key in _SORTED_KEYS:
            if key in normalised:
                return _CANONICAL[key]
        return raw.strip()

    @classmethod
    def get_metadata(cls, raw: str) -> dict[str, Any]:
        """Return ``{city, country, region}`` for a known institution.

        :param raw: Raw or canonical institution name.
        :returns: Dict with city / country / region strings, or empty dict.
        """
        canonical = cls.normalize(raw)
        return dict(_METADATA.get(canonical, {}))
