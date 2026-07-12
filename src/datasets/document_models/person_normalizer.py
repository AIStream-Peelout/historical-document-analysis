#!/usr/bin/env python3
"""
Person / scholar name normalization for the Cairo Genizah knowledge graph.

The primary challenge is that the same historical figure appears under:
  - A Latin transliteration   ("Moses Maimonides", "Maimonides")
  - A Hebrew name             ("משה בן מיימון", "רמב"ם")
  - An Arabic kunya / nisba   ("Abū ʿImrān Mūsā ibn Maymūn al-Qurṭubī")
  - A later scholarly short form ("Rambam", "Rambam")

Strategy
--------
1. ``_CANONICAL`` maps every known variant (lowercase, diacritic-stripped) to
   a single canonical display name used as the Neo4j ``Person.name`` merge key.
2. ``normalize()`` applies diacritic stripping + lowercase before lookup so
   "Māymūn" and "Maymun" both hit the same entry.
3. ``_HEBREW_LATIN`` maps single Hebrew script tokens to their canonical Latin
   form for figures whose primary modern identifier is their Hebrew name.
   This handles cases where the LLM returns the Hebrew form verbatim.

Adding new figures
------------------
Add entries to ``_CANONICAL`` (Latin key → canonical) and optionally to
``_HEBREW_LATIN`` (Hebrew script → canonical) if the figure is commonly
referenced in Hebrew.  The canonical should be the most widely used *English*
scholarly form so it appears consistently on the map / timeline UI.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.datasets.document_models.entity_normalizer import EntityNormalizer


# ---------------------------------------------------------------------------
# Variant → canonical (Latin / transliteration keys, lowercase ASCII)
# ---------------------------------------------------------------------------
_CANONICAL: dict[str, str] = {

    # ── Moses Maimonides ─────────────────────────────────────────────────────
    "maimonides":                       "Moses Maimonides",
    "moses maimonides":                 "Moses Maimonides",
    "musa ibn maymun":                  "Moses Maimonides",
    "musa ibn maymun al-qurtubi":       "Moses Maimonides",
    "abu imran musa ibn maymun":        "Moses Maimonides",
    "moshe ben maimon":                 "Moses Maimonides",
    "rambam":                           "Moses Maimonides",

    # ── Abraham Maimonides ───────────────────────────────────────────────────
    "abraham maimonides":               "Abraham Maimonides",
    "abraham ben moses maimonides":     "Abraham Maimonides",
    "avraham ben moshe maimonides":     "Abraham Maimonides",

    # ── Saadia Gaon ──────────────────────────────────────────────────────────
    "saadia gaon":                      "Saadia Gaon",
    "saadiah gaon":                     "Saadia Gaon",
    "saadya gaon":                      "Saadia Gaon",
    "saadia ben joseph":                "Saadia Gaon",
    "saadiah ben joseph":               "Saadia Gaon",
    "rasag":                            "Saadia Gaon",

    # ── Hai Gaon ─────────────────────────────────────────────────────────────
    "hai gaon":                         "Hai Gaon",
    "hay gaon":                         "Hai Gaon",
    "hai ben sherira gaon":             "Hai Gaon",

    # ── Sherira Gaon ─────────────────────────────────────────────────────────
    "sherira gaon":                     "Sherira Gaon",
    "sherira ben hananya":              "Sherira Gaon",

    # ── Samuel ben Ali ───────────────────────────────────────────────────────
    "samuel ben ali":                   "Samuel ben Ali",
    "shmuel ben ali":                   "Samuel ben Ali",
    "samuel ha-levi ben ali":           "Samuel ben Ali",

    # ── Judah ha-Levi ────────────────────────────────────────────────────────
    "judah ha-levi":                    "Judah ha-Levi",
    "judah halevi":                     "Judah ha-Levi",
    "yehuda halevi":                    "Judah ha-Levi",
    "yehudah ha-levi":                  "Judah ha-Levi",
    "jehuda ha-levi":                   "Judah ha-Levi",

    # ── Abraham ibn Ezra ─────────────────────────────────────────────────────
    "abraham ibn ezra":                 "Abraham ibn Ezra",
    "avraham ibn ezra":                 "Abraham ibn Ezra",
    "ibn ezra":                         "Abraham ibn Ezra",

    # ── Moses ibn Ezra ───────────────────────────────────────────────────────
    "moses ibn ezra":                   "Moses ibn Ezra",
    "moshe ibn ezra":                   "Moses ibn Ezra",

    # ── Nahmanides ───────────────────────────────────────────────────────────
    "nahmanides":                       "Nahmanides",
    "ramban":                           "Nahmanides",
    "moses ben nahman":                 "Nahmanides",
    "moshe ben nachman":                "Nahmanides",
    "nachmanides":                      "Nahmanides",

    # ── Goitein ──────────────────────────────────────────────────────────────
    "goitein":                          "S.D. Goitein",
    "s.d. goitein":                     "S.D. Goitein",
    "shelomo dov goitein":              "S.D. Goitein",
    "s. d. goitein":                    "S.D. Goitein",

    # ── Solomon Schechter ────────────────────────────────────────────────────
    "solomon schechter":                "Solomon Schechter",
    "schechter":                        "Solomon Schechter",
    "shlomo schechter":                 "Solomon Schechter",
}

# ---------------------------------------------------------------------------
# Hebrew script tokens → canonical name
#
# Keys are single Hebrew words or short phrases (as the LLM may emit them).
# Matching is done by scanning the raw name for any of these tokens.
# ---------------------------------------------------------------------------
_HEBREW_TOKENS: dict[str, str] = {
    # Maimonides
    "רמב\"ם":       "Moses Maimonides",
    'רמב"ם':        "Moses Maimonides",
    "רמב״ם":        "Moses Maimonides",
    "מיימון":       "Moses Maimonides",
    "מיימוני":      "Moses Maimonides",

    # Nahmanides
    "רמב\"ן":       "Nahmanides",
    'רמב"ן':        "Nahmanides",
    "רמב״ן":        "Nahmanides",
    "נחמנידס":      "Nahmanides",

    # Saadia Gaon
    "רס\"ג":        "Saadia Gaon",
    'רס"ג':         "Saadia Gaon",
    "סעדיה":        "Saadia Gaon",

    # Judah ha-Levi
    "הלוי":         "Judah ha-Levi",   # context-dependent; only fires alone

    # Hai Gaon
    "האי":          "Hai Gaon",
    "האיגאון":      "Hai Gaon",

    # Goitein (uncommon in Hebrew but included)
    "גויטין":       "S.D. Goitein",
}

# Pre-sorted longest-match keys.
_SORTED_KEYS: list[str]     = sorted(_CANONICAL.keys(),      key=len, reverse=True)
_SORTED_HEB:  list[str]     = sorted(_HEBREW_TOKENS.keys(),  key=len, reverse=True)

# Regex that detects any Hebrew Unicode block character.
_HEB_RE = re.compile(r'[֐-׿יִ-ﭏ]')


def _ascii_lower(s: str) -> str:
    """Lowercase and strip diacritics to ASCII for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return nfkd.encode("ascii", "ignore").decode().lower().strip()


def _contains_hebrew(s: str) -> bool:
    return bool(_HEB_RE.search(s))


class PersonNormalizer(EntityNormalizer):
    """Normalize person / scholar name variants for the Genizah KG.

    Handles:
    - Latin transliteration variants (case / diacritic insensitive)
    - Known Hebrew script forms (acronyms, full Hebrew names)

    The canonical name is the standard English scholarly form used as the
    Neo4j ``Person.name`` / ``Scholar.name`` merge key.
    """

    @classmethod
    def normalize(cls, raw: str) -> str:
        """Map any name variant to the canonical person name.

        :param raw: Person name in any transliteration, script, or abbreviation.
        :returns: Canonical name for Neo4j MERGE, or *raw* unchanged if unknown.
        """
        if not raw or not raw.strip():
            return raw
        stripped = raw.strip()

        # ── Hebrew-script path ────────────────────────────────────────────────
        if _contains_hebrew(stripped):
            for token in _SORTED_HEB:
                if token in stripped:
                    return _HEBREW_TOKENS[token]
            # No known Hebrew token — return as-is (may be a newly discovered
            # figure; do not silently discard it)
            return stripped

        # ── Latin / ASCII path ───────────────────────────────────────────────
        key = _ascii_lower(stripped)
        for candidate in _SORTED_KEYS:
            if key == candidate:
                return _CANONICAL[candidate]

        # Substring match as fallback (catches "rambam" inside longer strings)
        for candidate in _SORTED_KEYS:
            if candidate in key:
                return _CANONICAL[candidate]

        return stripped

    @classmethod
    def get_metadata(cls, raw: str) -> dict[str, Any]:
        """Return an empty dict — person metadata is not pre-seeded.

        :param raw: Person name.
        :returns: Empty dict (metadata stored on Neo4j Person nodes directly).
        """
        return {}
