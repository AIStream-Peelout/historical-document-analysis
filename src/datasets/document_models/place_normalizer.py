#!/usr/bin/env python3
"""
Historical place name normalization for the Cairo Genizah knowledge graph.

Responsibilities
----------------
1. ``normalize(raw)``       — map variant / transliterated / historical place
                               names to the canonical form used as Neo4j
                               ``Place.name`` merge key.
2. ``geocode_query(raw)``   — return a Nominatim / Google-friendly modern query
                               string for a canonical place name.
3. ``is_historical_institution(name)`` — return True for place-anchored
                               historical academies / yeshivot that should remain
                               as Place nodes, not be reclassified as Institution.

Usage
-----
from src.datasets.document_models.place_normalizer import PlaceNormalizer

PlaceNormalizer.normalize("Fusṭāṭ")       # → "Fustat"
PlaceNormalizer.geocode_query("Fustat")   # → "Old Cairo, Cairo, Egypt"
PlaceNormalizer.is_historical_institution("Sura Academy")  # → True
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.datasets.document_models.entity_normalizer import EntityNormalizer


# ---------------------------------------------------------------------------
# Variant → canonical KG name
#
# The canonical name is what appears as Place.name in Neo4j.  It should be the
# most widely recognised English / transliterated form of the place.
# ---------------------------------------------------------------------------
_CANONICAL: dict[str, str] = {

    # ── Fustat / Old Cairo ───────────────────────────────────────────────────
    "fusṭāṭ":           "Fustat",
    "fustat":           "Fustat",
    "al-fustat":        "Fustat",
    "al-fusṭāṭ":        "Fustat",

    # ── Cairo ────────────────────────────────────────────────────────────────
    "al-qāhira":        "Cairo",
    "al-qahira":        "Cairo",
    "miṣr":             "Cairo",
    "misr":             "Cairo",
    "masr":             "Cairo",

    # ── Alexandria ───────────────────────────────────────────────────────────
    "al-iskandariyya":  "Alexandria",
    "iskandariyya":     "Alexandria",

    # ── Qayrawan ─────────────────────────────────────────────────────────────
    "qayrawān":         "Qayrawan",
    "qayrawan":         "Qayrawan",
    "al-qayrawān":      "Qayrawan",
    "kairouan":         "Qayrawan",
    "ifrīqiya":         "Qayrawan",
    "ifriqiya":         "Qayrawan",

    # ── Ramla ────────────────────────────────────────────────────────────────
    "al-ramla":         "Ramla",
    "ramle":            "Ramla",
    "ramla":            "Ramla",

    # ── Tyre ─────────────────────────────────────────────────────────────────
    "ṣūr":              "Tyre",

    # ── Acre ─────────────────────────────────────────────────────────────────
    "akko":             "Acre",
    "akka":             "Acre",
    "ʿakkā":            "Acre",

    # ── Ascalon ──────────────────────────────────────────────────────────────
    "ascalon":          "Ashkelon",
    "ʿasqalān":         "Ashkelon",

    # ── Damascus ─────────────────────────────────────────────────────────────
    "bilād al-shām":    "Damascus",
    "bilad al-sham":    "Damascus",
    "al-shām":          "Damascus",
    "al-sham":          "Damascus",
    "dimashq":          "Damascus",

    # ── Baghdad ──────────────────────────────────────────────────────────────
    "madīnat al-salām": "Baghdad",
    "madinat al-salam": "Baghdad",

    # ── Sura (Iraq) ──────────────────────────────────────────────────────────
    "sura":             "Sura",
    "סורא":             "Sura",

    # ── Pumbedita (Iraq) ─────────────────────────────────────────────────────
    "pumbedita":        "Pumbedita",
    "pumbeditha":       "Pumbedita",
    "פומבדיתא":         "Pumbedita",

    # ── Constantinople / Istanbul ────────────────────────────────────────────
    "constantinople":   "Istanbul",
    "byzantium":        "Istanbul",

    # ── Al-Andalus ───────────────────────────────────────────────────────────
    "al-andalus":       "Al-Andalus",
    "andalus":          "Al-Andalus",
    "sefarad":          "Al-Andalus",

    # ── Coptic Cairo neighbourhood ───────────────────────────────────────────
    "qaṣr al-shamʿ":    "Qasr al-Sham' (Coptic Cairo)",
    "qasr al-sham'":    "Qasr al-Sham' (Coptic Cairo)",
    "qasr al-shama":    "Qasr al-Sham' (Coptic Cairo)",

    # ── Yemen ────────────────────────────────────────────────────────────────
    "diyār al-yaman":   "Yemen",
    "diyar al-yaman":   "Yemen",
    "diyār al-yemen":   "Yemen",
    "diyar al-yemen":   "Yemen",

    # ── Dahlak Archipelago ───────────────────────────────────────────────────
    "dahlak":           "Dahlak Archipelago",

    # ── Tinnis ───────────────────────────────────────────────────────────────
    "ṭinnīs":           "Tinnis",

    # ── Damietta ─────────────────────────────────────────────────────────────
    "dumyāṭ":           "Damietta",
    "dumyat":           "Damietta",
}

# ---------------------------------------------------------------------------
# Canonical KG name → Nominatim / Google geocoder query string
#
# Kept separate from _CANONICAL because the geocoding query is a modern,
# ASCII-friendly string that differs from both the raw variant and the KG name.
# ---------------------------------------------------------------------------
_GEOCODE_QUERY: dict[str, str] = {
    "Fustat":                       "Old Cairo, Cairo, Egypt",
    "Cairo":                        "Cairo, Egypt",
    "Alexandria":                   "Alexandria, Egypt",
    "Qayrawan":                     "Kairouan, Tunisia",
    "Ramla":                        "Ramla, Israel",
    "Tyre":                         "Tyre, Lebanon",
    "Acre":                         "Akko, Israel",
    "Ashkelon":                     "Ashkelon, Israel",
    "Damascus":                     "Damascus, Syria",
    "Baghdad":                      "Baghdad, Iraq",
    "Sura":                         "Sura, Iraq",
    "Pumbedita":                    "Fallujah, Iraq",
    "Istanbul":                     "Istanbul, Turkey",
    "Al-Andalus":                   "Andalusia, Spain",
    "Qasr al-Sham' (Coptic Cairo)": "Coptic Cairo, Egypt",
    "Yemen":                        "Sana'a, Yemen",
    "Dahlak Archipelago":           "Dahlak Archipelago, Eritrea",
    "Tinnis":                       "Tinnis, Egypt",
    "Damietta":                     "Damietta, Egypt",
}

# ---------------------------------------------------------------------------
# Historical place-anchored institutions
#
# Names that contain institution keywords (academy, yeshiva, etc.) but refer to
# places — they should remain as Place nodes in the KG rather than being
# reclassified as Institution nodes.  Add the canonical Place.name here;
# matching is case-insensitive substring.
# ---------------------------------------------------------------------------
_HISTORICAL_INSTITUTIONS: frozenset[str] = frozenset({
    # Iraqi academies
    "sura academy",
    "yeshiva of sura",
    "academy of sura",
    "pumbedita academy",
    "yeshiva of pumbedita",
    "academy of pumbedita",
    "yeshiva of baghdad",
    "baghdad academy",
    "nehardea academy",
    "yeshiva of nehardea",
    # Palestinian academies
    "yeshiva of tiberias",
    "tiberias academy",
    "yeshiva of caesarea",
    "yeshiva of jerusalem",
    "jerusalem academy",
    # Egyptian
    "yeshiva of fustat",
    "fustat academy",
    # Generic Hebrew prefixes — any "ישיבת X" or "בית מדרש X" refers to a place
    "ישיבת",
    "בית מדרש",
})

# Pre-sorted longest-match keys for _CANONICAL lookup.
_SORTED_KEYS: list[str] = sorted(_CANONICAL.keys(), key=len, reverse=True)


def _nfkd_lower(s: str) -> str:
    """Lowercase and strip diacritics for fuzzy matching."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


class PlaceNormalizer(EntityNormalizer):
    """Normalize historical / variant place names for the Genizah KG.

    Separates two concerns:
    - ``normalize`` — canonical Neo4j Place.name (stable KG identity key)
    - ``geocode_query`` — modern geocoder-friendly string (Nominatim / Google)
    """

    @classmethod
    def normalize(cls, raw: str) -> str:
        """Map a historical or variant place name to the canonical KG name.

        :param raw: Place name in any variant / transliterated form.
        :returns: Canonical Place.name used as Neo4j merge key, or *raw*
                  unchanged when no mapping is known.
        """
        if not raw or not raw.strip():
            return raw
        stripped = raw.strip()
        lower = stripped.lower()
        # Exact lowercase match first (handles Hebrew script too)
        if lower in _CANONICAL:
            return _CANONICAL[lower]
        # Diacritic-stripped fallback
        ascii_lower = _nfkd_lower(stripped)
        if ascii_lower in _CANONICAL:
            return _CANONICAL[ascii_lower]
        return stripped

    @classmethod
    def geocode_query(cls, raw: str) -> str:
        """Return a Nominatim / Google geocoding query string for *raw*.

        Resolves through :meth:`normalize` first so variant forms are handled.

        :param raw: Place name in any variant or canonical form.
        :returns: Modern geocoder-friendly query, or the canonical name when
                  no dedicated geocode override exists.
        """
        canonical = cls.normalize(raw)
        return _GEOCODE_QUERY.get(canonical, canonical)

    @classmethod
    def is_historical_institution(cls, name: str) -> bool:
        """Return True if *name* is a place-anchored historical institution.

        These entities contain keywords like "academy" or "yeshiva" but
        represent geographic places in the KG (Sura, Pumbedita, etc.) and
        should NOT be reclassified as Institution nodes.

        :param name: Entity name as produced by the LLM.
        :returns: True when the name refers to a historical place-academy.
        """
        if not name:
            return False
        lower = name.strip().lower()
        return any(marker in lower for marker in _HISTORICAL_INSTITUTIONS)

    @classmethod
    def get_metadata(cls, raw: str) -> dict[str, Any]:
        """Return ``{geocode_query}`` for the resolved place.

        :param raw: Raw or canonical place name.
        :returns: Dict with ``geocode_query`` key.
        """
        return {"geocode_query": cls.geocode_query(raw)}
