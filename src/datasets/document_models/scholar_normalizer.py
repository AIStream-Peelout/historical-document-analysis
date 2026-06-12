#!/usr/bin/env python3
"""
Modern-scholar name normalization and ground-truth author validation.

Solves two problems observed in the knowledge graph:

1. **Variant explosion** — the same scholar appearing as multiple nodes
   ("Estara J Arrant", "Arrant, Estara J", "Arrant E J", "Estara Arrant").
   Fixed by a *structural* name key (surname + first-name initial) that is
   computed algorithmically, so no hard-coded variant dictionary is needed
   for modern scholars.

2. **Hallucinated authors** — the LLM inventing a first name for a surname
   it saw in the text (e.g. "Benjamin Arrant" for an article actually
   co-authored by Estara J Arrant).  Fixed by validating extracted authors
   against the ground-truth ``*_metadata.json`` files that sit alongside
   each book's source PDF.

Usage
-----
::

    registry = ScholarRegistry.load()          # scans academic_literature/
    registry.resolve("Arrant, Estara J")       # → "Estara J Arrant"
    registry.is_known_scholar("Estara Arrant") # → True
    registry.validate_author("Benjamin Arrant", "PosegayandArrantpdf")
    # → AuthorValidation(status="conflict", canonical=None, ...)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Particles that belong to the surname, not a middle name
_SURNAME_PARTICLES = {"van", "von", "de", "del", "della", "di", "da", "le",
                      "la", "den", "der", "ten", "ter", "ha", "ben", "ibn",
                      "abu", "al", "el"}

# Honorifics / suffixes stripped before keying
_STRIP_TOKENS = {"dr", "prof", "professor", "rabbi", "rev", "sir",
                 "jr", "sr", "ii", "iii", "iv", "phd", "ma", "ba"}


def _ascii_lower(s: str) -> str:
    """Strip diacritics and lowercase.

    :param s: Input string.
    :returns: ASCII-folded lowercase string.
    """
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _name_tokens(name: str) -> List[str]:
    """Split a name into cleaned lowercase tokens, honorifics removed.

    Handles "Lastname, Firstname" by flipping around the comma first.

    :param name: Raw person name.
    :returns: Ordered list of name tokens in First-Last order.
    """
    name = name.strip()
    if "," in name:
        # "Arrant, Estara J" → "Estara J Arrant"
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    folded = _ascii_lower(name)
    raw_tokens = re.split(r"[\s.]+", folded)
    return [t for t in (re.sub(r"[^a-z'-]", "", t) for t in raw_tokens)
            if t and t not in _STRIP_TOKENS]


def display_form(name: str) -> str:
    """Return *name* in First-Last display order (flips "Last, First").

    :param name: Raw person name.
    :returns: Name with any comma inversion undone, whitespace collapsed.
    """
    name = name.strip()
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    return re.sub(r"\s+", " ", name)


def surname_of(name: str) -> str:
    """Extract the (folded, lowercase) surname from a name.

    :param name: Raw person name in either "First Last" or "Last, First" form.
    :returns: Lowercase ASCII surname including particles ("van der berg"),
              or empty string when the name has no tokens.
    """
    tokens = _name_tokens(name)
    if not tokens:
        return ""
    surname = [tokens[-1]]
    i = len(tokens) - 2
    while i > 0 and tokens[i] in _SURNAME_PARTICLES:
        surname.insert(0, tokens[i])
        i -= 1
    return " ".join(surname)


def name_key(name: str) -> str:
    """Compute the structural comparison key: ``surname|first-initial``.

    "Estara J Arrant", "Arrant, Estara J", "Arrant E J" and "E. J. Arrant"
    all map to ``arrant|e``.  Single-token names map to ``token|``.

    :param name: Raw person name.
    :returns: Comparison key, or empty string for blank input.
    """
    tokens = _name_tokens(name)
    if not tokens:
        return ""
    surname = surname_of(name)
    n_surname_tokens = len(surname.split())
    given = tokens[:len(tokens) - n_surname_tokens]
    # "Arrant E J" — surname-first with trailing initials (no comma).  If the
    # first token alone matches a known-surname shape we cannot detect that
    # structurally, BUT initials-only given names are a strong signal the
    # name was written surname-first: "Arrant E J" → given=["arrant","e"]?
    # No: tokens=[arrant,e,j], surname=j — wrong.  Guard: when every token
    # after the first is a single letter, treat the FIRST token as surname.
    if len(tokens) > 1 and all(len(t) == 1 for t in tokens[1:]):
        return f"{tokens[0]}|{tokens[1]}"
    first_initial = given[0][0] if given else ""
    return f"{surname}|{first_initial}"


@dataclass
class AuthorValidation:
    """Result of validating an extracted author against ground truth.

    :param status: ``"match"`` (snapped to a known author), ``"conflict"``
        (surname matches a real author but the given name disagrees —
        almost certainly an LLM hallucination), or ``"unknown"`` (no
        metadata overlap; a legitimately cited third-party scholar).
    :param canonical: Canonical metadata name when status is ``"match"``.
    :param detail: Human-readable explanation for logging.
    """
    status: str
    canonical: Optional[str] = None
    detail: str = ""


@dataclass
class ScholarRegistry:
    """Ground-truth registry of scholars from ``*_metadata.json`` files.

    :param key_to_canonical: ``name_key`` → canonical display name.
    :param surname_to_canonicals: folded surname → set of canonical names.
    :param book_authors: book name (structured-dir stem) → canonical authors.
    """

    key_to_canonical: Dict[str, str] = field(default_factory=dict)
    surname_to_canonicals: Dict[str, Set[str]] = field(default_factory=dict)
    book_authors: Dict[str, List[str]] = field(default_factory=dict)

    _instance: "ScholarRegistry | None" = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Optional[Path] = None) -> "ScholarRegistry":
        """Build the registry by scanning metadata files under *root*.

        Each ``*metadata*.json`` contributes its ``authors`` list; the book
        directories sitting next to the metadata file are mapped to those
        authors so per-book validation is possible.

        :param root: academic_literature root.  Defaults to the repo's
            standard location.
        :returns: Populated registry (also cached as the singleton).
        """
        if root is None:
            root = (Path(__file__).resolve().parents[2] / "datasets" / "raw_data"
                    / "cairo_genizah" / "academic_literature")
        reg = cls()
        for meta_path in sorted(root.rglob("*metadata*.json")):
            if meta_path.parent == root:   # skip example_book_metadata.json
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping unreadable metadata {meta_path}: {e}")
                continue
            authors = meta.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            canonicals = [reg._register(a) for a in authors if a and a.strip()]
            # Map every sibling book dir (structured output stem) to these authors
            for sibling in meta_path.parent.iterdir():
                if sibling.is_dir():
                    reg.book_authors.setdefault(sibling.name, []).extend(canonicals)
        logger.info(
            f"ScholarRegistry: {len(reg.key_to_canonical)} author keys "
            f"across {len(reg.book_authors)} books"
        )
        cls._instance = reg
        return reg

    @classmethod
    def instance(cls) -> "ScholarRegistry":
        """Return the cached singleton, loading it on first use.

        :returns: The shared :class:`ScholarRegistry`.
        """
        if cls._instance is None:
            cls.load()
        return cls._instance

    # ------------------------------------------------------------------
    def _register(self, name: str) -> str:
        """Add a ground-truth author, keeping the longest form as canonical.

        :param name: Author name as written in metadata.
        :returns: The canonical form now stored for this author.
        """
        disp = display_form(name)
        key = name_key(disp)
        existing = self.key_to_canonical.get(key)
        if existing is None or len(disp) > len(existing):
            self.key_to_canonical[key] = disp
            existing = disp
        canonical = self.key_to_canonical[key]
        self.surname_to_canonicals.setdefault(surname_of(disp), set()).add(canonical)
        return canonical

    # ------------------------------------------------------------------
    def resolve(self, name: str) -> Optional[str]:
        """Snap a name variant to its canonical ground-truth form.

        :param name: Any surface form of a scholar's name.
        :returns: Canonical metadata name, or None if the name does not
            key-match any known author.
        """
        if not name or not name.strip():
            return None
        key = name_key(name)
        hit = self.key_to_canonical.get(key)
        if hit:
            return hit
        # Surname-only mention ("Arrant") — unambiguous iff exactly one
        # known author has that surname.
        tokens = _name_tokens(name)
        if len(tokens) == 1:
            candidates = self.surname_to_canonicals.get(tokens[0], set())
            if len(candidates) == 1:
                return next(iter(candidates))
        return None

    def is_known_scholar(self, name: str) -> bool:
        """Return True when *name* resolves to a ground-truth author.

        :param name: Any surface form of a person's name.
        :returns: True if the registry can resolve the name.
        """
        return self.resolve(name) is not None

    # ------------------------------------------------------------------
    def validate_author(self, name: str, book: str) -> AuthorValidation:
        """Validate an extracted ``role: author`` person against ground truth.

        :param name: Extracted person name.
        :param book: Book name (structured-dir stem, e.g.
            ``"PosegayandArrantpdf"``).
        :returns: :class:`AuthorValidation` — "match" snaps to the canonical
            name; "conflict" means surname collides with a real author but
            the given name disagrees (LLM hallucination — drop it);
            "unknown" means no overlap (likely a cited third-party scholar).
        """
        canonical = self.resolve(name)
        if canonical:
            return AuthorValidation("match", canonical,
                                    f"'{name}' → '{canonical}'")
        surname = surname_of(name)
        book_canonicals = set(self.book_authors.get(book, []))
        colliding = {c for c in self.surname_to_canonicals.get(surname, set())
                     if not book_canonicals or c in book_canonicals}
        if colliding:
            return AuthorValidation(
                "conflict", None,
                f"'{name}' shares surname with known author(s) "
                f"{sorted(colliding)} but given name disagrees — "
                f"likely hallucinated",
            )
        return AuthorValidation("unknown", None,
                                f"'{name}' not in metadata for '{book}'")
