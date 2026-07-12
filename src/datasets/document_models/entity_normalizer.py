#!/usr/bin/env python3
"""
Abstract base class for entity normalization across the Cairo Genizah KG.

Concrete implementations:
  - ShelfmarkNormalizer  — fragment shelfmarks  (genizah_normalizer.py)
  - InstitutionNormalizer — holding institution names (institution_normalizer.py)
  - PlaceNormalizer       — historical place names   (place_normalizer.py)
  - PersonNormalizer      — person / scholar names   (person_normalizer.py)

All normalizers follow the same contract:
  normalize(raw)          → canonical string used as the Neo4j node key
  are_equivalent(a, b)    → True if both map to the same canonical
  get_variants(canonical) → known surface forms (for search / display)
  get_metadata(raw)       → domain-specific dict (institution city, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EntityNormalizer(ABC):
    """Abstract base for entity name / id normalization.

    All methods are class-methods so implementations can be used without
    instantiation, consistent with the existing ShelfmarkNormalizer API.
    """

    @classmethod
    @abstractmethod
    def normalize(cls, raw: str) -> str:
        """Map any variant form of an entity to its canonical string.

        :param raw: Raw / variant string as it appears in source data.
        :returns: Canonical string used as the Neo4j node merge key.
                  Returns *raw* unchanged when no mapping is known.
        """

    @classmethod
    def are_equivalent(cls, a: str, b: str) -> bool:
        """Return True if *a* and *b* normalize to the same canonical form.

        :param a: First entity string.
        :param b: Second entity string.
        :returns: True when both strings resolve to the same canonical key.
        """
        if not a or not b:
            return False
        return cls.normalize(a) == cls.normalize(b)

    @classmethod
    def get_variants(cls, canonical: str) -> list[str]:
        """Return known surface-form variants for a canonical string.

        Used to populate search aliases and display labels.  Subclasses
        that maintain a variant registry should override this.

        :param canonical: Canonical form returned by :meth:`normalize`.
        :returns: List of known variant strings (always includes *canonical*).
        """
        return [canonical]

    @classmethod
    def get_metadata(cls, raw: str) -> dict[str, Any]:
        """Return domain-specific metadata associated with the entity.

        Subclasses may return dicts like ``{city, country, region}`` for
        institutions or ``{geocode_query}`` for places.

        :param raw: Raw or canonical entity string.
        :returns: Metadata dict; empty dict when nothing is known.
        """
        return {}
