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

Needles match whole words only (see :func:`_needle_pattern`): the JTS needle
``ena`` must find ``ENA 2808.59`` but not the "ena" inside ``Menachem``,
``Benayahu`` or ``Modena``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

OXFORD_TOKEN = "Oxford_Bodleian"
BL_TOKEN = "London_BL"

# (descriptive token, [recognition needles — lowercased]). A needle matches as
# a whole word, never inside a longer word (:func:`_needle_pattern`). Order
# matters: more specific entries first, because some institution strings
# contain others' needles (e.g. "Cambridge Lewis-Gibson" and "Cambridge
# Mosseri" both contain "cambridge", so they must be tested before the generic
# "Cambridge_CUL").
# Libraries whose own Hebrew shelfmarks read "Ms. Heb. …" (NLI, BnF "hebr.",
# Harvard) are tested before Oxford; the bare "ms heb" needle is not here at
# all but in :data:`_FALLBACK_REGISTRY`, so it only decides when nothing else
# is named. ("jerusalem" / "bibliothèque nationale" are deliberately NOT
# needles: they would capture the Schocken / Ben Zvi / private Jerusalem
# collections and Strasbourg's "Bibliothèque Nationale et Universitaire".)
# City needles ("tel aviv", "st. petersburg") would capture every collection
# in that city, so the other holders there are tested first: the Gross,
# Einhorn and Eretz Israel Museum collections before Tel Aviv University, and
# the Institute of Oriental Manuscripts before the National Library of Russia.
# Budapest's "Jewish Theological Seminary - University of Jewish Studies"
# contains the New York JTS needle "jewish theological", so it is tested first.
_REGISTRY: List[Tuple[str, List[str]]] = [
    ("Harvard",                ["harvard", "houghton"]),
    ("Cambridge_Lewis_Gibson", ["lewis-gibson", "lewis gibson", "l-g", "cul & bodl"]),
    ("Cambridge_Mosseri",      ["mosseri", "moss."]),
    ("Cambridge_CUL",          ["cambridge cul", "cambridge university library",
                                "taylor-schechter", "t-s", "ulc", "christ's", "cul"]),
    ("Manchester_JRL",         ["rylands", "manchester", "gaster", "jrl"]),
    ("Budapest_Rabbinical_Seminary", ["university of jewish studies"]),
    ("New_York_JTS",           ["jewish theological", "new york jts", "elkan", "ena", "jts"]),
    ("New_York_Columbia",      ["columbia"]),
    ("New_York_JewishMuseum",  ["jewish museum"]),
    ("Jerusalem_NLI",          ["national library of israel", "jerusalem nli", "nli"]),
    ("Paris_BNF",              ["bibliothèque nationale de france",
                                "bibliotheque nationale de france",
                                "national library of france", "paris bnf", "bnf"]),
    (OXFORD_TOKEN,             ["bodleian", "bodl", "oxford"]),
    ("Paris_AIU",              ["alliance", "paris aiu", "aiu"]),
    (BL_TOKEN,                 ["british library", "british museum", "london bl"]),
    ("London_Sassoon",         ["sassoon"]),
    ("StPetersburg_IOM",       ["oriental manuscripts", "spios", "iom"]),
    ("StPetersburg_NLR",       ["national library of russia", "russian national",
                                "st. petersburg", "st petersburg", "yevr", "rnl", "nlr"]),
    ("Budapest_MTA",           ["hungarian academy", "budapest", "kaufmann", "mta"]),
    ("Cincinnati_HUC",         ["hebrew union", "cincinnati", "huc"]),
    ("Philadelphia_CAJS",      ["katz center", "penn cajs", "cajs", "halper",
                                "university of pennsylvania", "upenn", "dropsie", "penn"]),
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
    ("TelAviv_Gross",          ["gross, william", "william l. gross", "gross family"]),
    ("TelAviv_Einhorn",        ["einhorn, isaac"]),
    ("TelAviv_Eretz_Israel_Museum", ["eretz israel museum"]),
    ("TelAviv_TAU",            ["tel aviv"]),
    ("Princeton_PUL",          ["princeton university library", "pul"]),
    ("Heidelberg",             ["heidelberg"]),
    ("Utah",                   ["utah"]),
    ("Washington_Freer",       ["freer", "smithsonian"]),
    ("Birmingham_Mingana",     ["mingana", "birmingham"]),
    ("Reinach",                ["reinach"]),
    ("Haifa_University",       ["university of haifa"]),
    ("Naples_BNN",             ["victor emmanuel iii national library"]),
    ("Istanbul_Topkapi",       ["topkapu palace", "topkapi palace", "topkapı palace"]),
    ("Warsaw_JHI",             ["ringelblum", "jewish historical institute"]),
    ("Cologne_City_Archive",   ["historical archive of the city of cologne"]),
    ("Modena_State_Archives",  ["state archives of modena"]),
    ("Modena_Capitular_Archives", ["capitular archives of modena"]),
]

# Consulted only when no :data:`_REGISTRY` entry matches: a bare "MS heb." with
# no institution named is the Bodleian's shelfmark style.
_FALLBACK_REGISTRY: List[Tuple[str, List[str]]] = [
    (OXFORD_TOKEN,             ["ms heb", "ms. heb"]),
]


# A letter in any script: a word character that is not a digit or underscore.
_LETTER = r"[^\W\d_]"


def _needle_pattern(needle: str) -> str:
    """Return a regex matching *needle* as a whole word.

    A needle that starts (ends) with a letter must not be preceded (followed)
    by another letter, so ``ena`` matches ``ENA 2808.59``, ``ENA NS 5.29`` and
    ``Ms. ENA …`` but not ``Menachem``, ``Benayahu`` or ``Modena``. Digits and
    punctuation are boundaries (``T-S10J5`` still matches ``t-s``), and a
    needle's own leading/trailing punctuation (``moss.``) needs no boundary.

    :param needle: Lower-cased needle from a registry entry.
    :returns: A regex source string.
    """
    left = f"(?<!{_LETTER})" if needle[:1].isalpha() else ""
    right = f"(?!{_LETTER})" if needle[-1:].isalpha() else ""
    return left + re.escape(needle) + right


def _compile_registry(
    registry: List[Tuple[str, List[str]]],
) -> List[Tuple[str, Pattern[str]]]:
    """Compile each registry entry's needles into one whole-word regex.

    :param registry: ``(token, needles)`` entries.
    :returns: ``(token, compiled_pattern)`` entries in the same order.
    """
    return [
        (token, re.compile("|".join(_needle_pattern(n) for n in needles)))
        for token, needles in registry
    ]


_COMPILED: List[Tuple[str, Pattern[str]]] = _compile_registry(_REGISTRY)
_COMPILED_FALLBACK: List[Tuple[str, Pattern[str]]] = _compile_registry(
    _FALLBACK_REGISTRY
)


def resolve_token(text: Optional[str]) -> Optional[str]:
    """Map an institution / collection string to its descriptive token.

    :param text: Any institution spelling — a PGP ``library_abbrev``/``library``,
        an FJP ``collection``/``institution``, or a KTIV holding-library head.
    :returns: The descriptive token (e.g. ``Cambridge_CUL``), or ``None`` if no
        registry entry's needle occurs in *text* as a whole word.
    """
    if not text:
        return None
    low = text.lower()
    for compiled in (_COMPILED, _COMPILED_FALLBACK):
        for token, pattern in compiled:
            if pattern.search(low):
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
