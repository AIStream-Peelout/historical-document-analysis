#!/usr/bin/env python3
"""Descriptive institution tokens for canonical shelfmark ids.

Every merged canonical id must carry a normalized institution/collection token
so it is never a bare number (``101``) that collides across collections
(Budapest / Geneva / Frankfurt all number from 1). PGP, FJP and KTIV each spell
the same institution differently:

* PGP gives a clean ``library_abbrev`` (``CUL``, ``JTS``, ``JRL``, ``AIU`` …).
* FJP gives ``collection`` / ``institution`` (``Cambridge CUL``, ``Manchester`` …).
* KTIV embeds the holding library in the shelfmark head.

:func:`resolve_token` maps any of those spellings to one descriptive token
(``Cambridge_CUL``, ``New_York_JTS``, ``Paris_AIU``, ``Budapest_MTA`` …) so the
same fragment lands on the same id regardless of source. :func:`canonical_id`
combines the token with the normalized shelfmark core, stripping any leading
core token already represented in the institution token (so ``Manchester_JRL`` +
``JRL_B_1924`` → ``Manchester_JRL_B_1924``, not ``Manchester_JRL_JRL_B_1924``).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

# (descriptive token, [recognition substrings — lowercased]). Order matters:
# more specific entries first, because some institution strings are substrings
# of others (e.g. "Cambridge Lewis-Gibson" and "Cambridge Mosseri" both contain
# "cambridge", so they must be tested before the generic "Cambridge_CUL").
_REGISTRY: List[Tuple[str, List[str]]] = [
    ("Cambridge_Lewis_Gibson", ["lewis-gibson", "lewis gibson", "l-g", "cul & bodl"]),
    ("Cambridge_Mosseri",      ["mosseri", "moss."]),
    ("Cambridge_CUL",          ["cambridge cul", "cambridge university library",
                                "taylor-schechter", "t-s", "ulc", "christ's", "cul"]),
    ("Manchester_JRL",         ["rylands", "manchester", "gaster", "jrl"]),
    ("New_York_JTS",           ["jewish theological", "new york jts", "elkan", "ena", "jts"]),
    ("New_York_Columbia",      ["columbia"]),
    ("New_York_JewishMuseum",  ["jewish museum"]),
    ("Oxford_Bodleian",        ["bodleian", "bodl", "oxford", "ms heb", "ms. heb"]),
    ("Paris_AIU",              ["alliance", "paris aiu", "aiu"]),
    ("Paris_BNF",              ["bibliothèque nationale de france", "paris bnf", "bnf"]),
    ("London_BL",              ["british library", "british museum", "london bl"]),
    ("StPetersburg_NLR",       ["national library of russia", "russian national",
                                "st. petersburg", "st petersburg", "yevr", "rnl", "nlr"]),
    ("StPetersburg_IOM",       ["oriental manuscripts", "spios", "iom"]),
    ("Budapest_MTA",           ["hungarian academy", "budapest", "kaufmann", "mta"]),
    ("Cincinnati_HUC",         ["hebrew union", "cincinnati", "huc"]),
    ("Philadelphia_CAJS",      ["katz center", "penn cajs", "cajs", "halper",
                                "university of pennsylvania", "upenn", "dropsie", "penn"]),
    ("Jerusalem_NLI",          ["national library of israel", "jerusalem nli", "nli"]),
    ("Berlin_JCB",             ["jewish community of berlin"]),
    ("Berlin_SBB",             ["state library of berlin", "staatsbibliothek", "smb",
                                "staatliche museen"]),
    ("Vienna_ONB",             ["austrian national", "vienna", "wien", "papyri", "onb"]),
    ("Geneva",                 ["geneva", "genève", "genf"]),
    ("Strasbourg",             ["strasbourg", "stras."]),
    ("Frankfurt",              ["frankfurt"]),
    ("Cairo_ENL",              ["egyptian national", "dar al-kutub", "dār al-kutub",
                                "museum of islamic", "miac"]),
    ("Cairo_JCC",              ["jewish community of cairo", "jcc"]),
    ("Cairo_Karaite",          ["karaite"]),
    ("TelAviv_TAU",            ["tel aviv"]),
    ("Princeton_PUL",          ["princeton university library", "pul"]),
    ("Heidelberg",             ["heidelberg"]),
    ("Utah",                   ["utah"]),
    ("Washington_Freer",       ["freer", "smithsonian"]),
    ("Birmingham_Mingana",     ["mingana", "birmingham"]),
    ("Reinach",                ["reinach"]),
]


def resolve_token(text: Optional[str]) -> Optional[str]:
    """Map an institution / collection string to its descriptive token.

    :param text: Any institution spelling — a PGP ``library_abbrev``/``library``,
        an FJP ``collection``/``institution``, or a KTIV holding-library head.
    :returns: The descriptive token (e.g. ``Cambridge_CUL``), or ``None`` if no
        registry entry matches.
    """
    if not text:
        return None
    low = text.lower()
    for token, needles in _REGISTRY:
        if any(n in low for n in needles):
            return token
    return None


def _slugify(text: str) -> str:
    """Collapse arbitrary institution text into an underscore token (fallback).

    :param text: Raw institution / collection text.
    :returns: A non-empty underscore slug, or ``"Unknown"``.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug or "Unknown"


def institution_token(text: Optional[str]) -> str:
    """Return a descriptive token for *text*, never empty and never numeric.

    Falls back to a slug of the input so an unmapped institution still yields a
    stable non-numeric prefix rather than a bare number.

    :param text: Institution / collection spelling from any source.
    :returns: A descriptive token.
    """
    return resolve_token(text) or _slugify(text or "Unknown")


def canonical_id(institution_text: Optional[str], shelfmark: str) -> str:
    """Build a fully-qualified canonical id: ``<institution_token>_<core>``.

    The core is :meth:`ShelfmarkNormalizer.to_canonical_id`. Any leading core
    token already present in the institution token is dropped to avoid
    duplication (``Geneva`` + ``Geneva_119`` → ``Geneva_119``).

    :param institution_text: Institution / collection spelling for the token.
    :param shelfmark: The raw shelfmark (any source format).
    :returns: The canonical id, always institution-qualified.
    """
    token = institution_token(institution_text)
    core = ShelfmarkNormalizer.to_canonical_id(shelfmark)
    return combine(token, core)


def combine(token: str, core: str) -> str:
    """Join an institution *token* and a normalized *core*, de-duplicating overlap.

    :param token: Descriptive institution token (e.g. ``Manchester_JRL``).
    :param core: Normalized shelfmark core (e.g. ``JRL_B_1924``).
    :returns: ``<token>_<core>`` with leading core tokens that already appear in
        the token removed (e.g. ``Manchester_JRL_B_1924``).
    """
    if not core:
        return token
    token_parts = set(token.split("_"))
    core_parts = core.split("_")
    while core_parts and core_parts[0] in token_parts:
        core_parts.pop(0)
    return token + ("_" + "_".join(core_parts) if core_parts else "")
