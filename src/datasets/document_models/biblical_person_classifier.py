#!/usr/bin/env python3
"""
Classify person entities as biblical/canonical vs. historical vs. scholar.

Purpose
-------
Tanakh, Mishnaic, Talmudic, and classical figures (Haman, Mordechai, Hadrian,
the patriarchs, the great sages) recur throughout academic Genizah literature
but are NOT the people the knowledge graph cares about.  Left untyped they
conflate with real Genizah-period individuals and modern scholars, and they
spawn a sprawl of person-to-person edges (a biblical KG we do not want).

We therefore tag them as ``BiblicalPerson`` so they:
  * never pollute the ``Person`` (Genizah-era) or ``Scholar`` nodes, and
  * keep only ``Fragment → BiblicalPerson`` edges (e.g. "this fragment
    discusses Abraham") for future graph-RAG queries.

Detection strategy (conservative — Person/Scholar always win on doubt)
----------------------------------------------------------------------
``lookup(name)`` is a deterministic gazetteer returning one of:
  * ``"biblical"``  — the name is essentially only ever the canonical figure
                      (Haman, Vespasian, Methuselah …).  Tag immediately.
  * ``"ambiguous"`` — the name overlaps with common Genizah given-names
                      (Abraham, Joseph, David, Moses …).  Do NOT auto-tag;
                      defer to context (the Pass-3 LLM fallback decides, and
                      defaults to historical when unsure).
  * ``None``        — not a canonical name; treat as a normal person.

The LLM fallback itself lives in the coreference resolver (it has the page
context); this module only provides the deterministic gazetteer and the
folding helper so both passes agree on what counts as a candidate.
"""

from __future__ import annotations

import unicodedata
from typing import Optional, Set


def _fold(name: str) -> str:
    """Lowercase, strip diacritics, and collapse whitespace for matching.

    :param name: Raw person name.
    :returns: ASCII-folded lowercase comparison key.
    """
    nfkd = unicodedata.normalize("NFKD", name.strip())
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_only.lower().split())


# ---------------------------------------------------------------------------
# Unambiguous canonical figures — names not borne by ordinary Genizah people.
# Folded forms (lowercase, no diacritics).  Extend freely.
# ---------------------------------------------------------------------------
_UNAMBIGUOUS: Set[str] = {
    # Torah / Tanakh narrative figures with distinctive names
    "adam", "eve", "cain", "abel", "noah", "methuselah", "enoch",
    "nimrod", "lot", "esau", "laban", "pharaoh", "balaam", "balak",
    "korah", "og", "sihon", "rahab", "achan", "gideon", "samson",
    "delilah", "sisera", "goliath", "jonathan", "absalom", "bathsheba",
    "jezebel", "ahab", "naboth", "elisha", "gehazi", "naaman",
    "sennacherib", "nebuchadnezzar", "belshazzar", "ahasuerus", "xerxes",
    "esther", "haman", "mordechai", "mordecai", "vashti", "zeresh",
    "haggai", "zechariah", "malachi", "habakkuk", "zephaniah", "nahum",
    "obadiah", "amos", "hosea", "micah", "joel", "jonah", "ezekiel",
    "jeremiah", "isaiah", "job", "boaz", "ruth", "naomi", "hannah",
    "eli", "samuel the prophet", "nathan the prophet",
    # Classical persecutors / Roman emperors in rabbinic lore
    "hadrian", "vespasian", "titus", "trajan", "nero", "caligula",
    "turnus rufus", "tinneius rufus", "tineius rufus", "antoninus",
    "bar kokhba", "bar kochba", "bar koziba",
    # Major Tannaim / Amoraim (clearly rabbinic sages)
    "hillel", "shammai", "akiva", "akiba", "rabbi akiva", "yohanan ben zakkai",
    "johanan ben zakkai", "gamaliel", "rabban gamaliel", "judah ha-nasi",
    "judah hanasi", "rabbi judah ha-nasi", "rabbi meir", "rabbi tarfon",
    "rabbi ishmael", "eliezer ben hyrcanus", "rabbi yehoshua",
    "rav", "shmuel", "rava", "abaye", "resh lakish", "rabbi yohanan",
}

# ---------------------------------------------------------------------------
# Canonical names that ALSO double as ordinary Genizah given-names.  A bare
# match here is NOT enough to tag as biblical — context must confirm it.
# ---------------------------------------------------------------------------
_AMBIGUOUS: Set[str] = {
    "abraham", "avraham", "isaac", "yitzhak", "yitzchak", "jacob", "yaakov",
    "joseph", "yosef", "moses", "moshe", "aaron", "aharon", "david",
    "solomon", "shlomo", "samuel", "shmuel", "saul", "shaul", "joshua",
    "yehoshua", "benjamin", "binyamin", "judah", "yehuda", "levi", "simeon",
    "shimon", "reuben", "reuven", "dan", "gad", "asher", "naphtali",
    "issachar", "zebulun", "ephraim", "manasseh", "daniel", "elijah",
    "eliyahu", "elisha", "nathan", "natan", "ezra", "nehemiah", "phinehas",
    "pinhas", "pinehas", "pinhas", "miriam", "rachel", "leah", "rebecca",
    "rivka", "sarah", "sara", "hezekiah", "josiah", "jeroboam", "rehoboam",
    "gershom", "eleazar", "ithamar", "caleb",
}


def lookup(name: str) -> Optional[str]:
    """Deterministic gazetteer classification of a person name.

    :param name: Raw person name.
    :returns: ``"biblical"`` (unambiguous canonical figure), ``"ambiguous"``
        (canonical name that overlaps with common period names — needs
        context to decide), or ``None`` (not a canonical name).
    """
    key = _fold(name)
    if not key:
        return None
    if key in _UNAMBIGUOUS:
        return "biblical"
    if key in _AMBIGUOUS:
        return "ambiguous"
    # A leading "rabbi "/"rav "/"rabban " honorific on an otherwise ambiguous
    # name leans rabbinic but is still period-ambiguous (many Genizah figures
    # carry it) — treat as ambiguous, not auto-biblical.
    for honorific in ("rabbi ", "rav ", "rabban ", "rabbenu "):
        if key.startswith(honorific) and key[len(honorific):] in _AMBIGUOUS:
            return "ambiguous"
    return None


def is_definitely_biblical(name: str) -> bool:
    """Return True only for unambiguous canonical figures.

    Safe to call at import time (deterministic, no context needed).

    :param name: Raw person name.
    :returns: True if the name is an unambiguous biblical/canonical figure.
    """
    return lookup(name) == "biblical"


def is_biblical_candidate(name: str) -> bool:
    """Return True if the name is biblical or an ambiguous canonical name.

    Used to decide which names need context-based (LLM) disambiguation.

    :param name: Raw person name.
    :returns: True if the name warrants a biblical-vs-historical decision.
    """
    return lookup(name) in ("biblical", "ambiguous")
