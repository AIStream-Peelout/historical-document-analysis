#!/usr/bin/env python
# File name: invert_names.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1 (romanization inverter experiment);
# moved into the repo by Claude Opus 5.5 for the ``pgp_qa_v1`` builder.
"""Invert PGP romanized person names into Hebrew / Judaeo-Arabic script and locate them in text.

PGP records people in romanization only ("Ḥalfon b. Menashshe ha-Levi", "(Abū Yaḥyā) Nahray
b. Nissim"); the editions and the page images are in Hebrew script. This module turns a
romanized name into a small set of Hebrew-script candidate strings and finds where one of them
is written in a document's lines. It is the validator behind the ``qa_person`` / ``qa_party``
/ ``qa_ketubah_parties`` rows of ``build_pgp_qa`` (the answer is always the span as written on
the page; the romanized name never becomes an answer).

The inverter is rule based: a lookup table for frequent (mostly Hebrew) name tokens and titles,
plus a letter-level transliteration for everything else (Judaeo-Arabic orthography: long vowels
written with matres lectionis, short vowels mostly omitted, tāʾ marbūṭa -> ה, article -> אל).
Names are parsed into units (kunya, given name, father, titles, epithets) and only short
"core" queries of two name units are emitted (e.g. "יוסף בן יעקב", "אבו אלפרג", "חלפון הלוי").

Main entry points: :func:`queries_from_romanized` (candidate strings with a tier: 1 = given
name or kunya + connector + father, 2 = kunya + given name, ... 4 = kunya only, 5 = family name
only), :func:`build_doc_index` (normalised, n-gram indexed lines), :func:`person_info` (the
person's given/father spellings used to reject contradicting contexts) and :func:`locate`
(best non-contradicted hit; ``exact_hit`` marks a token-in-order match). Callers that need
certainty should accept only ``status == "located"`` with ``exact_hit`` and tier 1 or kunya+given
(``build_pgp_qa`` does). :data:`NAME_LEXICON` must be filled with :func:`build_name_lexicon`
before building indexes or queries (it splits connectors glued to names).

Experiment CLI (repo root; measures how many relation rows of the usable edition documents
are located)::

    .venv/bin/python -m src.datasets.qa.invert_names [--debug-tokens 150] [--out-dir DIR]

Behaviour is unchanged from the 2026-09-22 experiment (``located.jsonl``: 1,727 exact of 3,560
rows); only the file paths became repo-relative / CLI arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parents[3]
RELATIONS_CSV = Path("/Users/isaac/Documents/GitHub/ktiv-scraper/princeton/pgp_person_document_relations.csv")
PGP_DATA = REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data"
PEOPLE_CSV = PGP_DATA / "people.csv"
FOOTNOTES_CSV = PGP_DATA / "footnotes.csv"
MERGED_JSONL = REPO / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
# served image list snapshot (copied from the 2026-09-22 scratch export to the NAS build inputs)
SERVED_JSONL = Path("/Volumes/home/studio_offload/datasets/pgp_editions_v1_inputs/v6_imaged_docs.jsonl")
BENCHMARK_JSON = REPO / "src/datasets/raw_data/cairo_genizah/decontam/benchmark_ids.json"
OUT_DIR = Path(".")

LOCATED_MIN = 90.0
WEAK_MIN = 80.0
N_WORKERS = 2  # shared production machine: keep the fuzzy matrix modest

HEB_LETTER_RE = re.compile(r"[א-ת]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
FINAL_TO_REGULAR = str.maketrans("ךםןףץ", "כמנפצ")
REGULAR_TO_FINAL = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}

# --------------------------------------------------------------------------------------
# Hebrew-script normalisation (lines, script variants and generated candidates alike)
# --------------------------------------------------------------------------------------

# Characters removed without breaking a word: editorial brackets, quotes, geresh/gershayim,
# niqqud, combining diacritic dots (JA ג̇ / כ' style), zero-width characters.
_DELETE_CHARS = set("'\"`׳״[]⟦⟧{}()<>|/\\?!*_~^+=#​‌‍‎‏﻿")
_YIDDISH_LIGATURES = {"װ": "וו", "ױ": "וי", "ײ": "יי"}
# "son of" connectors, final-form-normalised; JA also writes Arabic ibn as אבן ("מנצור אבן סאלם").
CONNECTORS_NORM = {"בנ", "בר", "ביר", "בירבי", "ברבי", "אבנ"}
CANON_CONNECTOR = "בנ"
# Titles / epithets / blessing abbreviations that sit between a name and the connector
# ("חלפון הלוי ביר מנשה", "אפרים החבר ברבי שמריה", "חלפון הלוי נט' רח' בן ..."). Dropped in the
# "A" normalisation used for queries that contain a connector + father.
PRE_CONNECTOR_SKIP = {
    "הכהנ", "הלוי", "החבר", "הזקנ", "הצעיר", "החזנ", "הנכבד", "היקר", "הנבונ", "הבחור", "השר",
    "התלמיד", "הרב", "הדיינ", "המלמד", "הסופר", "הפרנס", "הנאמנ", "הממונה", "המומחה", "הרופא",
    "החכמ", "הקטנ", "הגדול", "המעולה", "הנשיא", "הגביר", "הפקיד", "השופט", "המשורר", "החסיד",
    "הספרדי", "המערבי", "הירא", "הנגיד", "כהנ", "אלכהנ", "אללוי", "אלחזנ", "אלדיאנ", "אלדיינ", "אלנאמנ",
    "נט", "רח", "נטרה", "רחמנא", "יצו", "ישצו", "יש", "צו", "שצ", "סט", "זל", "נע", "זצל", "יזי",
    "ננ", "נבת", "זלהה", "תנצבה", "הי", "יחי", "בחיר", "הישיבה", "הישיבות", "גדול", "גדו", "בסנ",
    "בסנהדרינ", "בסנהדרי", "בסנהד", "בסג", "הגדולה", "חבר", "אלופ", "ראש", "הנכ",
}
PRE_SKIP_MAX = 4


def is_skippable(tok: str) -> bool:
    """True for a title/blessing token (optionally with a ו- prefix) that may precede a connector.

    :param tok: Normalised token.
    :return: Whether the token can be skipped.
    """
    return tok in PRE_CONNECTOR_SKIP or (tok[:1] == "ו" and tok[1:] in PRE_CONNECTOR_SKIP)

# Normalised Hebrew given-name spellings (filled in main); used to split a connector glued to a
# name ("בריעקב" -> "בנ יעקב") and by the contradiction checks.
NAME_LEXICON: set[str] = set()
# Honorifics that sit between the connector and the father's name ("בר מר ור' יעקב").
HONORIFICS_AFTER_CONNECTOR = {
    "מר", "מרור", "מרר", "כגק", "כבוד", "כב", "כר", "ר", "רב", "רבי", "רבנא", "רבינו", "רבנו",
    "מרנא", "מרנו", "מרינו", "ורבנא", "ורבנו", "ורבינו", "ורב", "ור", "מ", "הר", "גדולת", "קדושת",
    "הרב", "השר", "הזקנ", "החכמ", "הנכבד", "היקר", "רבנ", "אלשיכ", "אלאגל", "אלגליל", "אלרייס", "אלחבר",
}
ALLOWED_PREFIXES = ("ו", "ל", "ד", "ש", "ב", "מ", "ול", "וד", "וב", "ומ", "וש", "אל", "ואל")


def _is_hebrew_mark(ch: str) -> bool:
    """Return True for niqqud / cantillation / combining marks that should be deleted.

    :param ch: A single character.
    :return: True if the character is a diacritic mark to drop.
    """
    cp = ord(ch)
    return (0x0591 <= cp <= 0x05BD) or cp in (0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7) or (
        0x0300 <= cp <= 0x036F
    )


@dataclass
class NormLine:
    """A normalised Hebrew line with a character map back into the original text."""

    original: str
    text: str
    char_orig: list[int]
    tokens: list[str]


def _raw_tokens(text: str) -> list[tuple[str, list[int]]]:
    """Split text into Hebrew-letter tokens, deleting marks/brackets without breaking words.

    :param text: Original line text.
    :return: List of (token letters with final forms regularised, original index per letter).
    """
    toks: list[tuple[str, list[int]]] = []
    cur: list[str] = []
    cur_idx: list[int] = []
    for i, ch in enumerate(text):
        if "א" <= ch <= "ת":
            cur.append(ch.translate(FINAL_TO_REGULAR))
            cur_idx.append(i)
        elif ch in _YIDDISH_LIGATURES:
            for c2 in _YIDDISH_LIGATURES[ch]:
                cur.append(c2)
                cur_idx.append(i)
        elif ch in _DELETE_CHARS or _is_hebrew_mark(ch):
            continue
        else:
            if cur:
                toks.append(("".join(cur), cur_idx))
                cur, cur_idx = [], []
    if cur:
        toks.append(("".join(cur), cur_idx))
    return toks


def normalize_line(text: str, canonicalize: bool = True, skip_pre: bool = False) -> NormLine:
    """Normalise a Hebrew-script line for matching and keep a map back to original offsets.

    Steps: drop marks/brackets/geresh inside words, final letters -> regular forms, other
    non-Hebrew characters -> single spaces; with ``canonicalize`` also join a standalone
    article "אל" to the next word, map every "son of" connector (בן/בר/ביר/בירבי/ברבי/אבן) to
    בנ and drop honorifics directly after a connector (בר מר ור' יעקב -> בנ יעקב); with
    ``skip_pre`` also drop titles/blessings right before a connector (חלפון הלוי ביר -> חלפון בנ).

    :param text: Original line (or candidate) text.
    :param canonicalize: Apply the token-level canonicalisation steps.
    :param skip_pre: Drop :data:`PRE_CONNECTOR_SKIP` tokens that precede a connector.
    :return: The normalised line.
    """
    toks = _raw_tokens(text)
    if canonicalize:
        joined: list[tuple[str, list[int]]] = []
        i = 0
        while i < len(toks):
            t, idx = toks[i]
            if t == "אל" and i + 1 < len(toks):
                t2, idx2 = toks[i + 1]
                joined.append((t + t2, idx + idx2))
                i += 2
                continue
            joined.append((t, idx))
            i += 1
        split: list[tuple[str, list[int]]] = []
        for t, idx in joined:
            glued = next((p for p in ("ביר", "בר", "בנ") if t.startswith(p) and len(t) - len(p) >= 3), None)
            if glued and t not in NAME_LEXICON and t[len(glued):] in NAME_LEXICON:
                split += [(glued, idx[:len(glued)]), (t[len(glued):], idx[len(glued):])]
            else:
                split.append((t, idx))
        canon: list[tuple[str, list[int]]] = []
        skip_honorific = 0
        for t, idx in split:
            if skip_honorific and t in HONORIFICS_AFTER_CONNECTOR:
                skip_honorific -= 1
                continue
            skip_honorific = 0
            if t in CONNECTORS_NORM:
                canon.append((CANON_CONNECTOR, [idx[0]] * (len(CANON_CONNECTOR) - 1) + [idx[-1]]))
                skip_honorific = 3
                continue
            canon.append((t, idx))
        toks = canon
        if skip_pre:
            drop: set[int] = set()
            for j, (t, _) in enumerate(toks):
                if t != CANON_CONNECTOR:
                    continue
                k = j - 1
                while k > 0 and j - k <= PRE_SKIP_MAX and is_skippable(toks[k][0]):
                    drop.add(k)
                    k -= 1
            toks = [tk for j, tk in enumerate(toks) if j not in drop]
    parts: list[str] = []
    char_orig: list[int] = []
    for k, (t, idx) in enumerate(toks):
        if k:
            parts.append(" ")
            char_orig.append(idx[0])
        parts.append(t)
        char_orig.extend(idx)
    return NormLine(original=text, text="".join(parts), char_orig=char_orig, tokens=[t for t, _ in toks])


def display_form(norm: str) -> str:
    """Turn a normalised (no final letters) string into a display form with final letters.

    :param norm: Normalised Hebrew string.
    :return: The string with word-final letters in final form.
    """
    return " ".join(w[:-1] + REGULAR_TO_FINAL.get(w[-1], w[-1]) if w else w for w in norm.split(" "))


ARABIC_TO_HEBREW = {
    "ا": "א", "أ": "א", "إ": "א", "آ": "א", "ٱ": "א", "ء": "א", "ب": "ב", "ت": "ת", "ث": "ת",
    "ج": "ג", "ح": "ח", "خ": "כ", "د": "ד", "ذ": "ד", "ر": "ר", "ز": "ז", "س": "ס", "ش": "ש",
    "ص": "צ", "ض": "צ", "ط": "ט", "ظ": "ט", "ع": "ע", "غ": "ג", "ف": "פ", "ق": "ק", "ك": "כ",
    "ک": "כ", "ل": "ל", "م": "מ", "ن": "נ", "ه": "ה", "ة": "ה", "و": "ו", "ؤ": "ו", "ي": "י",
    "ى": "י", "ئ": "י", "ی": "י",
}


def arabic_to_hebrew_script(text: str) -> str:
    """Map Arabic-script text letter by letter to Judaeo-Arabic Hebrew letters.

    :param text: Arabic-script string.
    :return: Hebrew-script string (diacritics dropped).
    """
    out = []
    for ch in text:
        if ch in ARABIC_TO_HEBREW:
            out.append(ARABIC_TO_HEBREW[ch])
        elif 0x064B <= ord(ch) <= 0x065F or ch in "ٰـ":
            continue
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------------------
# Romanisation -> Hebrew: lookup table
# --------------------------------------------------------------------------------------

# PGP spelling (as written, with diacritics) -> Hebrew spellings, most likely first.
LOOKUP: dict[str, list[str]] = {
    # --- Hebrew given names ---
    "Yosef": ["יוסף"], "Yehosef": ["יהוסף", "יוסף"], "Yaʿaqov": ["יעקב", "יעקוב"], "Yaaqov": ["יעקב"],
    "Avraham": ["אברהם"], "Moshe": ["משה"], "Shelomo": ["שלמה"], "Yehuda": ["יהודה"],
    "Yehūda": ["יהודה"], "Yahūdā": ["יהודה", "יהודא"], "Eliyyahu": ["אליהו"], "Eliyahu": ["אליהו"],
    "Eliyya": ["אליה"], "Natan": ["נתן"], "Nātān": ["נתן"], "Ḥalfon": ["חלפון"],
    "Nahray": ["נהראי", "נהוראי"], "Nahrāy": ["נהראי", "נהוראי"], "Nissim": ["נסים", "ניסים"],
    "Nissīm": ["נסים", "ניסים"], "Yeshuʿa": ["ישועה"], "Shemuʾel": ["שמואל"], "Efrayim": ["אפרים"],
    "Menashshe": ["מנשה"], "Ḥayyim": ["חיים"], "Mevorakh": ["מבורך"], "Yiṣḥaq": ["יצחק"],
    "Yiṣhaq": ["יצחק"], "Yishaq": ["יצחק"], "Iṣḥaq": ["יצחק", "אסחק"], "Iṣḥāq": ["יצחק", "אסחאק"],
    "Isḥāq": ["אסחק", "אסחאק", "יצחק"], "Isḥaq": ["אסחק", "יצחק"], "Sahl": ["סהל"],
    "Mūsā": ["מוסי", "מוסא", "משה"], "Ismāʿīl": ["אסמעיל", "אסמאעיל", "ישמעאל"],
    "Ismaʿīl": ["אסמעיל", "אסמאעיל", "ישמעאל"], "Ibrāhīm": ["אברהים", "אבראהים", "אברהם"],
    "Ibrahim": ["אברהים", "אבראהים", "אברהם"], "Yūsuf": ["יוסף"], "Yaʿqūb": ["יעקוב", "יעקב"],
    "Hillel": ["הלל"], "ʿEli": ["עלי"], "ʿElī": ["עלי"], "ʿAlī": ["עלי"], "ʿAli": ["עלי"],
    "Shemarya": ["שמריה"], "Shemaryā": ["שמריה"], "Shemaryāhu": ["שמריהו", "שמריה"],
    "Yefet": ["יפת"], "Saʿadya": ["סעדיה", "סעדיא"], "Seʿadya": ["סעדיה", "סעדיא"],
    "Sa'adya": ["סעדיה", "סעדיא"], "David": ["דוד", "דויד"], "Dāwūd": ["דאוד", "דוד"],
    "Dāʾūd": ["דאוד", "דוד"], "Dā'ūd": ["דאוד", "דוד"], "Netanʾel": ["נתנאל"], "Nethanel": ["נתנאל"],
    "Aharon": ["אהרן", "אהרון"], "Hārūn": ["הרון", "הארון"], "Meʾīr": ["מאיר"], "Meʾir": ["מאיר"],
    "Yisraʾel": ["ישראל"], "Isrāʾīl": ["אסראיל", "ישראל"], "Ḥananya": ["חנניה"],
    "Ḥananʾel": ["חננאל"], "Elʿazar": ["אלעזר"], "Elʿāzar": ["אלעזר"], "Yehoshuaʿ": ["יהושע"],
    "Peraḥya": ["פרחיה"], "Peraḥyā": ["פרחיה"], "Zekharya": ["זכריה"], "Zakariyyā": ["זכריא", "זכריה"],
    "Zikrī": ["זכרי"], "Mevasser": ["מבשר"], "Mevassēr": ["מבשר"], "Daniʾel": ["דניאל"],
    "Danyāl": ["דניאל", "דאניאל"], "Dāniyāl": ["דאניאל", "דניאל"], "Ṭoviyya": ["טוביה"],
    "Ṭuviyyahu": ["טוביהו", "טוביה"], "Shela": ["שלה"], "Yaʾir": ["יאיר"], "ʿUlla": ["עולא", "עלא"],
    "Somekh": ["סומך"], "Yeḥiʾel": ["יחיאל"], "ʿImmanuel": ["עמנואל"], "Yakhin": ["יכין"],
    "Shalom": ["שלום"], "Yedutun": ["ידותון"], "Binyamin": ["בנימין"], "Evyatar": ["אביתר"],
    "ʿEzra": ["עזרא"], "Shemaʿya": ["שמעיה"], "Shemaʿyahu": ["שמעיהו"], "ʿAzarya": ["עזריה"],
    "Elḥanan": ["אלחנן"], "Sason": ["ששון"], "Simḥa": ["שמחה"], "Yeshaʿyahu": ["ישעיהו"],
    "Shaʿyā": ["שעיא", "שעיה"], "Yoshiyyahu": ["יאשיהו"], "Yoshiyya": ["יאשיה"], "Ṣemaḥ": ["צמח"],
    "Zakkay": ["זכאי"], "Sherira": ["שרירא"], "ʿOvadya": ["עובדיה"], "Elishaʿ": ["אלישע"],
    "Shekhanya": ["שכניה"], "Yequtiʾel": ["יקותיאל"], "Neḥemya": ["נחמיה"], "Ḥesed": ["חסד"],
    "Yashar": ["ישר"], "Shabbetay": ["שבתי"], "Ḥaggay": ["חגי"], "Nadiv": ["נדיב"], "Barukh": ["ברוך"],
    "Banāya": ["בנאיה", "בניה"], "Benaya": ["בניה"], "Tiqva": ["תקוה", "תקווה"], "Ḥiyya": ["חייא", "חייה"],
    "Meshullam": ["משולם"], "Yeḥezqel": ["יחזקאל"], "Ṣadoq": ["צדוק"], "Levi": ["לוי"],
    "Shaʾul": ["שאול"], "Pinḥas": ["פנחס", "פינחס"], "Yonatan": ["יונתן"], "Yishay": ["ישי"],
    "Naḥman": ["נחמן"], "Naḥum": ["נחום"], "Reʾuven": ["ראובן"], "Shimʿon": ["שמעון"],
    "Mishaʾel": ["מישאל"], "Petaḥya": ["פתחיה"], "Dosa": ["דוסא"], "Ḥanina": ["חנינא"],
    "Ḥizqiyya": ["חזקיה"], "Yishmaʿel": ["ישמעאל"], "Hoshaʿna": ["הושענא"], "Ṭarfon": ["טרפון"],
    "Eitan": ["איתן"], "Raḥamim": ["רחמים"], "Seʿadʾel": ["סעדאל"], "Berakha": ["ברכה"],
    "Berakhot": ["ברכות"], "Ṣedaqa": ["צדקה"], "Ṣadaqa": ["צדקה"], "Ṣedeq": ["צדק"], "Ḥefeṣ": ["חפץ"],
    "Maṣlīaḥ": ["מצליח"], "Maṣliaḥ": ["מצליח"], "Mardūkh": ["מרדוך", "מרדכי"], "Sheʾerit": ["שארית"],
    "Tifʾeret": ["תפארת"], "Yiju": ["יגו"], "Yijū": ["יגו"], "Maymūn": ["מימון"], "Maimon": ["מימון"],
    "Maymūnī": ["מימוני"], "Moses": ["משה"], "Judah": ["יהודה"], "Ezekiel": ["יחזקאל"],
    "Nēzer": ["נזר"], "Ḥemdat": ["חמדת"],
    # --- Arabic names where the letter rules need help or an alternate spelling ---
    "Barhūn": ["ברהון"], "ʿArūs": ["ערוס"], "Bundār": ["בנדאר", "בנדר"], "Khalīla": ["כלילה"],
    "Maḍmūn": ["מצמון", "מדמון"], "Manṣūr": ["מנצור"], "Mansūr": ["מנצור"], "Khalaf": ["כלף"],
    "Khalfa": ["כלפה"], "Wahb": ["והב"], "Muḥammad": ["מחמד"], "Naṣr": ["נצר"], "Faraḥ": ["פרח"],
    "Farāḥ": ["פראח", "פרח"], "Farrāḥ": ["פראח"], "Saʿīd": ["סעיד"], "Saʿĩd": ["סעיד"], "Saʿd": ["סעד"],
    "Saʿāda": ["סעאדה", "סעדה"], "Sahlān": ["סהלאן", "סהלן"], "Sahlūn": ["סהלון"], "Makhlūf": ["מכלוף"],
    "Sughmār": ["סגמאר"], "Ṣāliḥ": ["צאלח"], "Aḥmad": ["אחמד"], "Ghālib": ["גאלב"], "Salāma": ["סלאמה"],
    "Sulaymān": ["סלימאן", "סלימן"], "Hiba": ["הבה"], "Hibatallāh": ["הבה אללה", "הבת אללה"],
    "ʿAbdallāh": ["עבד אללה", "עבדאללה"], "Abdallāh": ["עבד אללה", "עבדאללה"], "ʿAbd": ["עבד"],
    "Allāh": ["אללה"], "Hilāl": ["הלאל"], "Barakāt": ["ברכאת"], "ʿUmar": ["עמר"], "Sālim": ["סאלם"],
    "ʿAllān": ["עלאן"], "ʿAllūn": ["עלון"], "Ḥusayn": ["חסין"], "ʿImrān": ["עמראן", "עמרן"],
    "ʿAwkal": ["עוכל"], "Yaḥyā": ["יחיי", "יחיא"], "Mubārak": ["מבארך"], "Kathīr": ["כתיר"],
    "Ḥasan": ["חסן"], "ʿAyyāsh": ["עיאש"], "ʿAmmār": ["עמאר"], "Maḥrūz": ["מחרוז"],
    "Munajjāʾ": ["מנגא"], "Munajjā": ["מנגא"], "Thābit": ["תאבת", "תאבית"], "ʿAmr": ["עמרו", "עמר"],
    "Surūr": ["סרור"], "Sitt": ["סת"], "Umm": ["אם"], "Qaṭāʾif": ["קטאיף"], "Maʿānī": ["מעאני"],
    "Maʿālī": ["מעאלי"], "Ḥayy": ["חי"], "Munā": ["מנא", "מני"], "Ṣāʿid": ["צאעד"],
    # --- titles and epithets ---
    "ha-Levi": ["הלוי"], "Halevi": ["הלוי"], "al-Levi": ["הלוי"], "ha-Kohen": ["הכהן"],
    "Kohen": ["כהן", "הכהן"], "al-Kohen": ["אלכהן", "הכהן"], "he-Ḥaver": ["החבר"], "ha-Ḥaver": ["החבר"],
    "ha-Sefaradi": ["הספרדי"], "Gaʾon": ["גאון"], "Ga'on": ["גאון"], "Nagid": ["הנגיד", "נגיד"],
    "Nasi": ["הנשיא", "נשיא"], "ha-Nasi": ["הנשיא"], "ha-Ḥazzan": ["החזן"], "al-Ḥazzan": ["אלחזן", "החזן"],
    "al-Ḥazzān": ["אלחזאן", "אלחזן"], "ha-Mumḥe": ["המומחה"], "ha-Parnas": ["הפרנס"],
    "ha-Melammed": ["המלמד"], "al-Melammed": ["אלמלמד", "המלמד"], "ha-Shofeṭ": ["השופט"],
    "ha-Dayyan": ["הדיין"], "al-Dayyān": ["אלדיאן", "אלדיין"], "al-Dayyan": ["אלדיין", "אלדיאן"],
    "ha-Rofeʾ": ["הרופא"], "ha-Sofer": ["הסופר"], "ha-Neʾeman": ["הנאמן"], "al-Neʾeman": ["אלנאמן", "הנאמן"],
    "Neʾeman": ["נאמן"], "ha-Meʿulle": ["המעולה"], "he-Meʿulle": ["המעולה"], "ha-Maʿaravi": ["המערבי"],
    "ha-Rav": ["הרב"], "ha-Talmid": ["התלמיד"], "al-Talmid": ["אלתלמיד"], "ha-Seder": ["הסדר"],
    "ha-Yeshiva": ["הישיבה"], "ha-Sar": ["השר"], "ha-Ḥavērīm": ["החברים"], "ha-Nezer": ["הנזר"],
    "ha-Kohanim": ["הכהנים"], "ha-Geʾonim": ["הגאונים"], "ha-Sheviʿi": ["השביעי"], "ha-Memunne": ["הממונה"],
    "ha-Paliṭ": ["הפליט"], "ha-Soḥarim": ["הסוחרים"], "Peqid": ["פקיד"], "Paqid": ["פקיד"],
    "Raʾs": ["ראס"], "al-kull": ["אלכל"], "al-Kull": ["אלכל"], "Rosh": ["ראש"], "Av": ["אב"],
    "Bet": ["בית"], "Din": ["דין"], "Sar": ["שר"], "ha-Sarim": ["השרים"], "ha-Qahal": ["הקהל"],
    "ha-ʿEda": ["העדה"],
}

# Tokens that are titles (never a given/father name) - PGP form, and whether the title can sit
# between the given name and the connector in Hebrew signatures ("חלפון הלוי ביר מנשה").
TITLE_TOKENS = {
    "ha-Levi", "Halevi", "al-Levi", "ha-Kohen", "Kohen", "al-Kohen", "he-Ḥaver", "ha-Ḥaver",
    "ha-Sefaradi", "Gaʾon", "Ga'on", "Nagid", "Nasi", "ha-Nasi", "ha-Ḥazzan", "al-Ḥazzan", "al-Ḥazzān",
    "ha-Mumḥe", "ha-Parnas", "ha-Melammed", "al-Melammed", "ha-Shofeṭ", "ha-Dayyan", "al-Dayyān",
    "al-Dayyan", "ha-Rofeʾ", "ha-Sofer", "ha-Neʾeman", "al-Neʾeman", "ha-Meʿulle", "he-Meʿulle",
    "ha-Maʿaravi", "ha-Rav", "ha-Talmid", "al-Talmid", "ha-Seder", "ha-Yeshiva", "ha-Sar",
    "ha-Ḥavērīm", "ha-Nezer", "ha-Kohanim", "ha-Geʾonim", "ha-Sheviʿi", "ha-Memunne", "ha-Paliṭ",
    "ha-Soḥarim", "Peqid", "Paqid", "Rosh", "Av", "Bet", "Din", "ha-Sarim", "ha-Qahal", "ha-ʿEda",
}
# Honorific epithets listed among PGP name variants ("Tifʾeret ha-Kohanim", "Nēzer ha-Ḥavērīm",
# "Raʾs al-kull"): not names, so such variants are skipped.
EPITHET_HEADS = {"Tifʾeret", "Nēzer", "Nezer", "Raʾs", "Ḥemdat", "Peʾer", "Sar", "Rosh", "Av"}
COMPOUND_NAMES = {("Sar", "Shalom")}
# English / descriptive words: a variant is cut at the first of these (or dropped if it starts so).
ENGLISH_STOP = {
    "of", "the", "in", "or", "from", "and", "de", "son", "daughter", "wife", "widow", "sister",
    "brother", "mother", "father", "brother-in-law", "unnamed", "person", "enslaved", "manumitted",
    "community", "judge", "younger", "elder", "c.", "c", "maimonides", "pseudo", "caliph", "sultan",
    "fustat", "alexandria", "aleppo", "qayrawān", "known", "as", "i", "ii", "iii", "iv",
}
KUNYA_HEADS = {"Abū": ["אבו", "אבי"], "Abu": ["אבו", "אבי"], "Abī": ["אבי", "אבו"], "Abi": ["אבי", "אבו"],
               "Bū": ["בו", "אבו"], "Umm": ["אם"]}
SON_CONNECTORS = {"b.", "ben", "bar", "bin", "b", "Ben"}
DAUGHTER_CONNECTORS = {"bt.", "bint", "bat"}
IBN_CONNECTORS = {"Ibn", "ibn"}
# Compound given names whose second element carries the article ("ʿAbd al-ʿAzīz", "Sitt al-Ahl").
COMPOUND_HEADS = {"ʿAbd", "Sitt", "Thiqat", "Fakhr", "Zayn", "Sadīd", "ʿIzz", "Jamāl", "Sharaf", "Tāj",
                  "Amīn", "Muwaffaq", "Najm", "Shams", "Sayf", "Nūr", "Bahāʾ", "Ḍiyāʾ", "ʿAlam"}


def fold(token: str) -> str:
    """Fold a romanized token for fuzzy lookup: lowercase, strip diacritics and ʿʾ' marks.

    :param token: PGP romanized token.
    :return: Folded key.
    """
    s = unicodedata.normalize("NFKD", token)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[ʿʾ'’‘`]", "", s).lower()


LOOKUP_EXACT = {unicodedata.normalize("NFC", k): v for k, v in LOOKUP.items()}
_fold_groups: dict[str, set[tuple[str, ...]]] = defaultdict(set)
for _k, _v in LOOKUP.items():
    _fold_groups[fold(_k)].add(tuple(_v))
LOOKUP_FOLDED = {k: list(next(iter(v))) for k, v in _fold_groups.items() if len(v) == 1}
TITLE_FOLDED = {fold(t) for t in TITLE_TOKENS}

# --------------------------------------------------------------------------------------
# Romanisation -> Hebrew: letter rules
# --------------------------------------------------------------------------------------

_MACRON = set("āīūēō")
_ARABIC_MARKERS = ("ḍ", "ẓ", "gh", "dh", "th", "j")
_HEBREW_MARKERS = ("e", "o", "v", "p")
_CONSONANTS = {
    "b": [("ב", 0.0)], "v": [("ב", 0.0), ("ו", 1.0)], "p": [("פ", 0.0)], "f": [("פ", 0.0)],
    "t": [("ת", 0.0)], "ṭ": [("ט", 0.0)], "th": [("ת", 0.0)], "d": [("ד", 0.0)], "dh": [("ד", 0.0)],
    "ḍ": [("צ", 0.0), ("ד", 0.6)], "ẓ": [("ט", 0.0), ("צ", 0.6)], "j": [("ג", 0.0)], "g": [("ג", 0.0)],
    "gh": [("ג", 0.0)], "h": [("ה", 0.0)], "ḥ": [("ח", 0.0)], "kh": [("כ", 0.0)], "k": [("כ", 0.0)],
    "q": [("ק", 0.0)], "l": [("ל", 0.0)], "m": [("מ", 0.0)], "n": [("נ", 0.0)], "r": [("ר", 0.0)],
    "s": [("ס", 0.0)], "ś": [("ש", 0.0)], "sh": [("ש", 0.0)], "ṣ": [("צ", 0.0)], "z": [("ז", 0.0)],
    "w": [("ו", 0.0)], "y": [("י", 0.0)], "ʿ": [("ע", 0.0)], "ʾ": [("א", 0.0)], "'": [("א", 0.0)],
    "c": [("כ", 0.0)], "x": [("כס", 0.0)],
}
_VOWELS = set("aeiouāīūēō")


def _token_style(tok: str) -> str:
    """Guess whether a romanized token follows the Arabic or the Hebrew romanization.

    :param tok: Lowercased NFC token.
    :return: "ar", "he" or "both".
    """
    if any(c in _MACRON for c in tok) or any(m in tok for m in _ARABIC_MARKERS):
        return "ar"
    if any(m in tok for m in _HEBREW_MARKERS):
        return "he"
    return "both"


def _segments(tok: str) -> list[str]:
    """Split a lowercased romanized token into letters/digraphs, collapsing geminates.

    :param tok: Lowercased NFC token without article prefix.
    :return: Segment list; doubled consonants appear once, except yy/ww which stay doubled.
    """
    segs: list[str] = []
    i = 0
    while i < len(tok):
        two = tok[i:i + 2]
        if two in ("sh", "kh", "th", "dh", "gh"):
            seg, i = two, i + 2
        else:
            seg, i = tok[i], i + 1
        if segs and seg == segs[-1] and seg not in _VOWELS:
            if seg in ("y", "w"):
                segs[-1] = seg + seg
            continue
        segs.append(seg)
    return segs


def _slot_options(segs: list[str], k: int, style: str) -> list[tuple[str, float]]:
    """Return Hebrew options (string, cost) for segment ``k`` of a romanized token.

    :param segs: Segment list from :func:`_segments`.
    :param k: Index of the segment.
    :param style: Token style ("ar", "he", "both").
    :return: Options for this position, cheapest first.
    """
    seg = segs[k]
    first, last = k == 0, k == len(segs) - 1
    nxt = segs[k + 1] if k + 1 < len(segs) else ""
    he = style == "he"
    if seg in ("yy", "ww"):
        base = "י" if seg == "yy" else "ו"
        return [(base * 2, 0.0), (base, 0.5)] if not he else [(base, 0.0), (base * 2, 0.5)]
    if seg not in _VOWELS:
        opts = _CONSONANTS.get(seg, [("", 0.0)])
        if seg == "s" and he:
            return [("ס", 0.0), ("ש", 0.6)]
        return opts
    if first:
        return {"ī": [("אי", 0.0)], "ū": [("או", 0.0)], "ō": [("או", 0.0)], "o": [("או", 0.0), ("א", 0.5)]}.get(
            seg, [("א", 0.0)])
    if nxt == "yy" and seg == "i":  # Eliyyahu, ʿAṭiyya: the i is carried by the yod
        return [("", 0.0)]
    if seg == "ā":
        return [("י", 0.0), ("א", 0.0)] if last else [("א", 0.0), ("", 1.0)]
    if seg in ("ī", "ē"):
        return [("י", 0.0)]
    if seg in ("ū", "ō"):
        return [("ו", 0.0)]
    if seg == "a":
        if last:
            return [("ה", 0.0), ("א", 0.8)]
        if he and nxt == "y" and k + 2 == len(segs):  # Hebrew final -ay (Nahray -> נהראי)
            return [("", 0.0), ("א", 0.6)]
        return [("", 0.0)]
    if seg == "e":
        return [("ה", 0.0), ("י", 1.0)] if last else [("", 0.0), ("י", 1.0)]
    if seg == "o":
        return [("ה", 0.0), ("ו", 0.5)] if last else [("ו", 0.0), ("", 0.5)]
    if seg == "i":
        if last:
            return [("י", 0.0)]
        return [("", 0.0), ("י", 1.0)] if style == "ar" else [("", 0.0), ("י", 0.4)]
    if seg == "u":
        if last:
            return [("ו", 0.0)]
        return [("", 0.0), ("ו", 1.0)] if style == "ar" else [("ו", 0.0), ("", 0.4)]
    return [("", 0.0)]


def rule_spellings(token: str, k_best: int = 3) -> list[tuple[str, float]]:
    """Letter-rule transliteration of one romanized token (no article) into Hebrew script.

    :param token: PGP romanized token, e.g. "Sughmār".
    :param k_best: Number of spellings to keep.
    :return: Up to ``k_best`` (normalised Hebrew spelling, cost) pairs, cheapest first.
    """
    tok = unicodedata.normalize("NFC", token).lower().replace("’", "ʾ").replace("‘", "ʿ")
    tok = re.sub(r"[^a-zāīūēōḥṣṭḍẓśʿʾ']", "", tok)
    if not tok:
        return []
    style = _token_style(tok)
    segs = _segments(tok)
    slots = [_slot_options(segs, k, style) for k in range(len(segs))]
    combos: list[tuple[str, float]] = [("", 0.0)]
    for opts in slots:
        combos = sorted(((s + o, c + oc) for s, c in combos for o, oc in opts), key=lambda x: x[1])[:12]
    out: dict[str, float] = {}
    for s, c in combos:
        s = re.sub("א+", "א", s)  # ā + ʾ (Dāʾūd) -> single alef
        if s and s not in out:
            out[s] = c
    return sorted(out.items(), key=lambda x: x[1])[:k_best]


def spell_token(token: str) -> list[tuple[str, float]]:
    """Hebrew spellings (normalised, with a cost) for one romanized token, incl. prefixes.

    Order of resolution: exact lookup, folded lookup, article/prefix split + recursion,
    letter rules.

    :param token: PGP romanized token, e.g. "al-Tāhartī", "ha-Levi", "l-Faraj", "Yosef".
    :return: List of (spelling, cost), cheapest first.
    """
    tok = unicodedata.normalize("NFC", token.strip(".,;:\"“”()[]"))
    if not tok:
        return []
    for table, key in ((LOOKUP_EXACT, tok), (LOOKUP_FOLDED, fold(tok))):
        if key in table:
            return [(normalize_line(s, canonicalize=False).text, 0.5 * i) for i, s in enumerate(table[key])]
    m = re.match(r"^(al|el|l|ʾl|'l|ha|he|wa|ve|li|la|bi)-(.+)$", tok, flags=re.IGNORECASE)
    if m:
        pre, rest = m.group(1).lower(), m.group(2)
        heb_pre = {"ha": "ה", "he": "ה", "wa": "ו", "ve": "ו", "li": "ל", "la": "ל", "bi": "ב"}.get(pre, "אל")
        if rest.lower().startswith(("ha-", "he-")):  # ve-ha-Sofer
            return [(heb_pre + s, c) for s, c in spell_token(rest)]
        return [(heb_pre + s.replace(" ", ""), c) for s, c in spell_token(rest)]
    return [(normalize_line(s, canonicalize=False).text, c) for s, c in rule_spellings(tok)]


# --------------------------------------------------------------------------------------
# Name parsing and query generation
# --------------------------------------------------------------------------------------

@dataclass
class Unit:
    """One name unit: a plain name, a kunya ("Abū l-Faraj") or a family name ("Ibn ʿAwkal")."""

    tokens: list[str]
    kind: str  # "name" | "kunya" | "family"
    titles: list[str] = field(default_factory=list)


@dataclass
class ParsedName:
    """Structure of a PGP romanized name."""

    kunya: Unit | None = None
    given: Unit | None = None
    chain: list[tuple[str, Unit]] = field(default_factory=list)  # (connector type, unit)
    titles: list[str] = field(default_factory=list)  # titles attached to the person

@dataclass
class Query:
    """One candidate string to search for in the edition."""

    display: str  # candidate as generated, with final letters
    text: str  # matching form ("A" normalisation for tier 1, "B" otherwise)
    tier: int  # 1 core (has father), 2 kunya+given / kunya+title, 3 given+title, 4 kunya, 5 family
    kind: str
    source: str  # "romanized" | "latin_variant" | "script_variant"
    cost: float = 0.0
    lacks_given: bool = False  # query does not contain the person's given name
    alts: list[str] = field(default_factory=list)  # other spellings with the same matching form (בן/בר/ביר)

    @property
    def exact_only(self) -> bool:
        """Kunya-only, family-only, single-token and given+honorific queries must match exactly.

        :return: Whether fuzzy matching is disabled for this query.
        """
        return self.tier >= 4 or " " not in self.text or self.kind == "given+honorific"


def make_query(raw: str, tier: int, kind: str, source: str, cost: float = 0.0,
               lacks_given: bool = False) -> Query:
    """Build a query, normalising it the same way as the edition lines it is matched against.

    :param raw: Hebrew-script candidate (may contain final letters / geresh).
    :param tier: Query tier (1 = contains connector + father).
    :param kind: Query kind label.
    :param source: Provenance label.
    :param cost: Spelling cost (lower = more likely).
    :param lacks_given: True if the query does not contain the person's given name.
    :return: The query.
    """
    return Query(display=display_form(normalize_line(raw, canonicalize=False).text),
                 text=normalize_line(raw, canonicalize=True, skip_pre=tier == 1).text,
                 tier=tier, kind=kind, source=source, cost=cost, lacks_given=lacks_given)


def _clean_roman(name: str) -> tuple[str, list[str]]:
    """Strip glosses from a romanized name and pull out parenthesised parts.

    :param name: PGP name or Latin name variant.
    :return: (remaining name text, list of parenthesised strings).
    """
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"\[\?\]|\?", "", name)
    name = re.sub(r"[\"“”][^\"“”]*[\"“”]", " ", name)
    parens = re.findall(r"\(([^)]*)\)", name)
    name = re.sub(r"\([^)]*\)", " ", name)
    return re.sub(r"\s+", " ", name).strip(), [p.strip() for p in parens]


def parse_romanized(name: str) -> ParsedName | None:
    """Parse a PGP romanized person name into kunya / given / chain / titles.

    :param name: e.g. "(Abū Yaḥyā) Nahray b. Nissim", "Ḥalfon b. Menashshe ha-Levi".
    :return: The parsed structure, or None if no usable name is left.
    """
    text, parens = _clean_roman(name)
    parsed = ParsedName()
    for p in parens:
        ptoks = p.split()
        if ptoks and ptoks[0] in KUNYA_HEADS and len(ptoks) >= 2:
            parsed.kunya = Unit(tokens=ptoks[:3] if ptoks[1] in ("ʿAbd",) else ptoks[:2], kind="kunya")
        elif p in TITLE_TOKENS or fold(p) in TITLE_FOLDED:
            parsed.titles.append(p)
    raw: list[str] = []
    for t in text.split():
        if t.lower().strip(",.") in ENGLISH_STOP and t not in ("b.", "bt."):
            if not raw:
                return None if parsed.kunya is None else parsed
            break
        if re.match(r"^[IVX]+$", t):
            continue
        if t.startswith("Ben-"):
            raw.extend(["Ben", t[4:]])
        else:
            raw.append(t.strip(","))
    if raw and tuple(raw[:2]) not in COMPOUND_NAMES and (raw[0] in EPITHET_HEADS or raw[0].startswith(("ha-", "he-"))):
        return parsed if parsed.kunya else None
    units: list[tuple[str | None, Unit]] = []  # (connector before, unit)
    i = 0
    pending_conn: str | None = None
    while i < len(raw):
        t = raw[i]
        if t in SON_CONNECTORS or t in DAUGHTER_CONNECTORS or (t in IBN_CONNECTORS):
            pending_conn = "son" if t in SON_CONNECTORS else ("daughter" if t in DAUGHTER_CONNECTORS else "ibn")
            i += 1
            continue
        if t in KUNYA_HEADS and i + 1 < len(raw):
            n = 3 if raw[i + 1] == "ʿAbd" and i + 2 < len(raw) else 2
            unit = Unit(tokens=raw[i:i + n], kind="kunya")
            i += n
        elif t in TITLE_TOKENS or fold(t) in TITLE_FOLDED:
            if units and pending_conn is None:
                units[-1][1].titles.append(t)
            else:
                parsed.titles.append(t)
            i += 1
            continue
        elif t.startswith(("al-", "l-", "Al-")) and units and pending_conn is None:
            units[-1][1].titles.append(t)  # nisba / laqab epithet
            i += 1
            continue
        elif (t in COMPOUND_HEADS and i + 1 < len(raw) and raw[i + 1].startswith(("al-", "Al-"))) or \
                tuple(raw[i:i + 2]) in COMPOUND_NAMES:
            unit = Unit(tokens=raw[i:i + 2], kind="name")
            i += 2
        else:
            unit = Unit(tokens=[t], kind="family" if pending_conn == "ibn" and not units else "name")
            i += 1
        if units and pending_conn is None:
            # a bare name right after a kunya is the given name; otherwise it is an epithet
            if units[-1][1].kind == "kunya" and len(units) == 1 and unit.kind == "name":
                units.append((None, unit))
            else:
                units[-1][1].titles.append(" ".join(unit.tokens))
            continue
        units.append((pending_conn, unit))
        pending_conn = None
    if not units:
        return parsed if parsed.kunya else None
    first_conn, first = units[0]
    rest = units[1:]
    if first_conn is not None:  # "Ibn al-Baʿbāʿ": the whole name is a family name
        parsed.chain = [(first_conn, first)] + [(c or "son", u) for c, u in rest]
        return parsed
    if first.kind == "kunya" and rest and rest[0][0] is None:
        parsed.kunya = parsed.kunya or first
        parsed.given = rest[0][1]
        rest = rest[1:]
    elif first.kind == "kunya" and not rest:
        parsed.kunya = parsed.kunya or first
    elif first.kind == "kunya":  # "Abū Naṣr b. Avraham": the kunya is the only personal name
        parsed.kunya = first
    else:
        parsed.given = first
    parsed.chain = [(c or "son", u) for c, u in rest]
    parsed.titles = (parsed.given.titles if parsed.given else []) + parsed.titles
    if parsed.kunya is not None and parsed.given is None:
        parsed.titles = parsed.kunya.titles + parsed.titles
    for _, u in parsed.chain:
        parsed.titles += [t for t in u.titles if t in TITLE_TOKENS]
    return parsed


CONNECTOR_SPELLINGS = {"son": [("בן", 0.0), ("בר", 0.3), ("ביר", 0.6)], "daughter": [("בת", 0.0)],
                       "ibn": [("בן", 0.0), ("אבן", 0.3)]}


def unit_spellings(unit: Unit, genitive: bool = False, k_best: int = 3) -> list[tuple[str, float]]:
    """Hebrew spellings of a name unit (product over its tokens), cheapest first.

    :param unit: The unit.
    :param genitive: For kunyas after a connector use אבי first ("b. Abī l-Ḥayy").
    :param k_best: Number of spellings to keep.
    :return: List of (spelling, cost).
    """
    per_tok: list[list[tuple[str, float]]] = []
    for j, t in enumerate(unit.tokens):
        if j == 0 and unit.kind == "kunya" and t in KUNYA_HEADS:
            heads = KUNYA_HEADS[t]
            if genitive and "אבי" in heads:
                heads = ["אבי", "אבו"]
            per_tok.append([(h, 0.4 * n) for n, h in enumerate(heads)])
        else:
            sp = spell_token(t)
            if not sp:
                return []
            per_tok.append(sp)
    combos = [(" ".join(s for s, _ in c), sum(x for _, x in c)) for c in product(*per_tok)]
    combos.sort(key=lambda x: x[1])
    seen: dict[str, float] = {}
    for s, c in combos:
        seen.setdefault(s, c)
    return list(seen.items())[:k_best]


def queries_from_romanized(name: str, source: str, max_per_kind: int = 9) -> list[Query]:
    """Generate the small candidate set for one romanized name.

    Emitted kinds (tier): core = given|kunya + connector + father (1), core+title (1),
    kunya+father (1), kunya+given / kunya+title / kunya+epithet (2), given+title (3),
    kunya alone (4), family name "Ibn X" / "Ben X" alone (5).

    :param name: PGP romanized name (or Latin variant).
    :param source: Label for the provenance of the name form.
    :param max_per_kind: Cap on the number of candidate strings per kind.
    :return: Queries, deduplicated on their matching form.
    """
    parsed = parse_romanized(name)
    if parsed is None:
        return []
    out: list[Query] = []
    title_sp: list[tuple[str, float]] = []
    for t in parsed.titles:
        sp = [(s, c) for s, c in spell_token(t) if s.startswith("ה")]
        if sp:
            title_sp = sp[:1]
            break
    given_sp = unit_spellings(parsed.given) if parsed.given else []
    kunya_sp = unit_spellings(parsed.kunya, k_best=2) if parsed.kunya else []

    def add(tier: int, kind: str, parts: list[list[tuple[str, float]]], extra: float = 0.0,
            lacks_given: bool = False) -> None:
        """Add the product of part spellings as queries of one kind.

        :param tier: Query tier.
        :param kind: Query kind label.
        :param parts: Spelling options per part.
        :param extra: Extra cost added to every combination.
        :param lacks_given: Whether the query omits the given name.
        """
        if not parts or any(not p for p in parts):
            return
        combos = sorted(((" ".join(s for s, _ in c), sum(x for _, x in c) + extra) for c in product(*parts)),
                        key=lambda x: x[1])
        for s, c in combos[:max_per_kind]:
            out.append(make_query(s, tier, kind, source, c, lacks_given))

    if parsed.chain:
        conn, father = parsed.chain[0]
        conn_sp = CONNECTOR_SPELLINGS[conn]
        father_sp = unit_spellings(father, genitive=True)
        if given_sp:
            add(1, "core", [given_sp, conn_sp, father_sp])
            if title_sp:
                add(1, "core+title", [given_sp, title_sp, conn_sp, father_sp], extra=0.2)
        if kunya_sp:
            add(1, "kunya+father", [kunya_sp, conn_sp, father_sp], lacks_given=bool(given_sp))
        if not given_sp and not kunya_sp:
            add(5, "family", [conn_sp, father_sp], lacks_given=True)
        # family names ("Ben Yijū", "Ibn ʿAwkal") later in the chain as a unit of their own
        for c2, u2 in parsed.chain[1:]:
            if c2 == "ibn" or u2.kind == "family":
                add(5, "family", [CONNECTOR_SPELLINGS["ibn"], unit_spellings(u2, k_best=2)], lacks_given=True)
    if kunya_sp and given_sp:
        add(2, "kunya+given", [kunya_sp, given_sp])
    if kunya_sp and not given_sp and title_sp:
        add(2, "kunya+title", [kunya_sp, [(s.removeprefix("ה"), c) for s, c in title_sp] + title_sp],
            lacks_given=True)
    if kunya_sp and not given_sp and not title_sp:
        epi = [t for t in parsed.kunya.titles if t not in TITLE_TOKENS]  # "Abū Naṣr al-Ḥalabī"
        if epi:
            add(2, "kunya+epithet", [kunya_sp, spell_token(epi[0])[:2]], lacks_given=True)
    if given_sp and title_sp:
        add(3, "given+title", [given_sp, title_sp])
    if kunya_sp:
        add(4, "kunya", [kunya_sp], lacks_given=True)
    uniq: dict[str, Query] = {}
    for q in out:
        if len(q.text.split()) < 2:
            continue
        if q.text not in uniq or (q.tier, q.cost) < (uniq[q.text].tier, uniq[q.text].cost):
            if q.text in uniq:
                q.alts = [uniq[q.text].display] + uniq[q.text].alts
            uniq[q.text] = q
        elif q.display != uniq[q.text].display and q.display not in uniq[q.text].alts:
            uniq[q.text].alts.append(q.display)
    return list(uniq.values())


SCRIPT_KUNYA = {"אבו", "אבי", "בו"}


def queries_from_script(variant: str) -> list[Query]:
    """Baseline queries from a Hebrew- or Arabic-script PGP name variant.

    Takes the first two name units as written (titles between them kept), plus the kunya.

    :param variant: Hebrew or Arabic script name variant.
    :return: Queries.
    """
    if ARABIC_RE.search(variant):
        variant = arabic_to_hebrew_script(variant)
    toks = normalize_line(variant).tokens
    if not toks:
        return []
    out: list[Query] = []

    def unit_end(k: int) -> int:
        """Index after the name unit starting at ``k`` (kunyas span two tokens).

        :param k: Start index.
        :return: End index (exclusive).
        """
        return min(len(toks), k + 2) if toks[k] in SCRIPT_KUNYA else k + 1

    e0 = unit_end(0)
    if toks[0] in SCRIPT_KUNYA and e0 == 2:
        out.append(make_query(" ".join(toks[:2]), 4, "kunya", "script_variant", lacks_given=True))
    j = e0
    while j < len(toks) and toks[j].startswith("ה") and toks[j] != CANON_CONNECTOR:
        j += 1  # titles between the first unit and the connector
    if j < len(toks) and toks[j] == CANON_CONNECTOR and j + 1 < len(toks):
        out.append(make_query(" ".join(toks[:unit_end(j + 1)]), 1, "core", "script_variant"))
    elif len(toks) >= 2:
        out.append(make_query(" ".join(toks[:max(2, e0)]), 2, "first-two", "script_variant"))
    else:
        out.append(make_query(toks[0], 2, "single", "script_variant"))
    return out


@dataclass
class PersonInfo:
    """Name evidence for one person, used to reject matches that contradict it."""

    given: set[str] = field(default_factory=set)  # normalised given-name tokens
    fathers: set[str] = field(default_factory=set)  # normalised father units


# Honorifics that follow a bare given name ("אפרים החבר", "שלמה הצעיר"); lineage markers
# (הכהן/הלוי) are excluded because they would contradict a PGP name that lacks them.
GENERIC_TITLES = ["החבר", "הזקן", "הצעיר", "הנכבד", "היקר", "השר", "התלמיד", "החכם", "הרב", "המלמד",
                  "החזן", "הדיין", "הסופר", "הפרנס", "הנבון", "הבחור", "המעולה", "הגביר", "הנגיד",
                  "הנשיא", "הממונה", "המומחה", "הרופא", "השופט"]


def honorific_queries(latin_forms: list[str]) -> list[Query]:
    """Given name + any generic honorific (exact-only, tier 3), for names PGP lists without it.

    :param latin_forms: PGP name and Latin variants of one person.
    :return: Queries of kind "given+honorific".
    """
    main = next((p for p in (parse_romanized(f) for f in latin_forms) if p is not None and p.given), None)
    if main is None:
        return []
    return [make_query(f"{g} {t}", 3, "given+honorific", "romanized", c)
            for g, c in unit_spellings(main.given, k_best=2) for t in GENERIC_TITLES]


def cross_form_queries(latin_forms: list[str], source: str = "latin_variant") -> list[Query]:
    """Kunya + given / kunya + father queries when the kunya and the name come from different forms.

    PGP often lists the kunya as a separate variant ("Abū Saʿīd" for Ḥalfon b. Menashshe).

    :param latin_forms: PGP name and Latin variants of one person.
    :param source: Provenance label.
    :return: Extra queries (tier 2 kunya+given, tier 1 kunya+father).
    """
    parsed = [p for p in (parse_romanized(f) for f in latin_forms) if p is not None]
    kunyas = {tuple(p.kunya.tokens): p.kunya for p in parsed if p.kunya}
    main = next((p for p in parsed if p.given), None)
    if not kunyas or main is None:
        return []
    out: list[Query] = []
    given_sp = unit_spellings(main.given)
    father_sp = unit_spellings(main.chain[0][1], genitive=True) if main.chain and main.chain[0][0] != "ibn" else []
    for ku in kunyas.values():
        k_sp = unit_spellings(ku, k_best=2)
        for (a, ca), (b, cb) in sorted(product(k_sp, given_sp), key=lambda x: x[0][1] + x[1][1])[:6]:
            out.append(make_query(f"{a} {b}", 2, "kunya+given", source, ca + cb))
        for (a, ca), (b, cb) in sorted(product(k_sp, father_sp), key=lambda x: x[0][1] + x[1][1])[:6]:
            out.append(make_query(f"{a} בן {b}", 1, "kunya+father", source, ca + cb, lacks_given=True))
    return out


def person_info(latin_forms: list[str], script_variants: list[str]) -> PersonInfo:
    """Collect given-name and father spellings of a person from all its name forms.

    :param latin_forms: PGP name and Latin variants.
    :param script_variants: Hebrew/Arabic-script variants.
    :return: The person's name evidence.
    """
    info = PersonInfo()
    for form in latin_forms:
        parsed = parse_romanized(form)
        if parsed is None:
            continue
        if parsed.given:
            for s, _ in unit_spellings(parsed.given):
                info.given.add(normalize_line(s).tokens[0])
        if parsed.chain and parsed.chain[0][0] != "ibn":
            for s, _ in unit_spellings(parsed.chain[0][1], genitive=True):
                info.fathers.add(normalize_line(s).text)
    for v in script_variants:
        toks = normalize_line(arabic_to_hebrew_script(v)).tokens
        if not toks:
            continue
        if toks[0] not in SCRIPT_KUNYA and toks[0] != CANON_CONNECTOR:
            info.given.add(toks[0])
        if CANON_CONNECTOR in toks[1:]:
            j = toks.index(CANON_CONNECTOR, 1)
            if j + 1 < len(toks):
                info.fathers.add(" ".join(toks[j + 1:j + 3]) if toks[j + 1] in SCRIPT_KUNYA else toks[j + 1])
    return info


# Common JA/Hebrew words that are also given names; never treated as a contradicting name.
NAME_STOP = {"עלי", "סעד", "כיר", "סלאמה", "נצר", "פרג", "סרור", "שלומ", "ברכה", "חסד", "צדקה", "שמחה",
             "טוב", "אלי", "מני", "מנא", "עז", "נגמ", "בדר", "עמר", "פצל", "תקוה", "ישר", "סת", "אמ", "חי"}


def build_name_lexicon(people_rows: list[dict]) -> set[str]:
    """Normalised Hebrew spellings of all PGP given / father names (for contradiction checks).

    :param people_rows: people.csv rows.
    :return: Set of normalised name tokens.
    """
    lex: set[str] = set()
    for r in people_rows:
        for form in [r["name"]] + [v for v in split_variants(r["name_variants"]) if not HEB_LETTER_RE.search(v)
                                   and not ARABIC_RE.search(v)]:
            parsed = parse_romanized(form)
            if parsed is None:
                continue
            units = ([parsed.given] if parsed.given else []) + [u for c, u in parsed.chain if c != "ibn"]
            for u in units:
                if u.kind != "name":
                    continue
                for s, _ in unit_spellings(u):
                    lex.add(normalize_line(s).tokens[0])
    for k, v in LOOKUP.items():
        if k[0].isupper() and k not in TITLE_TOKENS:
            lex.update(normalize_line(s).tokens[0] for s in v)
    return {t for t in lex if len(t) >= 3 and t not in NAME_STOP}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_usable_editions(served_jsonl: Path = SERVED_JSONL) -> tuple[dict[str, str], dict[str, str]]:
    """Load edition texts for usable documents (served image, not in benchmark).

    :param served_jsonl: Served image list (JSONL with ``_id`` = canonical id).
    :return: (pgpid -> longest edition content, pgpid -> canonical_id).
    """
    pgpid_to_canon: dict[str, str] = {}
    with MERGED_JSONL.open() as f:
        for line in f:
            rec = json.loads(line)
            for p in rec.get("pgpids") or []:
                pgpid_to_canon.setdefault(str(p), rec["canonical_id"])
    with served_jsonl.open() as f:
        served = {json.loads(line)["_id"] for line in f}
    bench = set(json.loads(BENCHMARK_JSON.read_text()))
    eds: dict[str, str] = {}
    with FOOTNOTES_CSV.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if "edition" not in (row["doc_relation"] or "").lower():
                continue
            content = row["content"] or ""
            if len(HEB_LETTER_RE.findall(content)) < 100:
                continue
            d = row["document_id"]
            if d not in eds or len(content) > len(eds[d]):
                eds[d] = content
    usable = {d: t for d, t in eds.items()
              if d in pgpid_to_canon and pgpid_to_canon[d] in served and pgpid_to_canon[d] not in bench}
    return usable, {d: pgpid_to_canon[d] for d in usable}


def load_people() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load PGP people keyed by name and by URL slug.

    :return: (name -> row, slug -> row).
    """
    with PEOPLE_CSV.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_name = {r["name"]: r for r in rows}
    by_slug = {r["url"].rstrip("/").split("/")[-1]: r for r in rows if r["url"]}
    return by_name, by_slug


def split_variants(raw: str) -> list[str]:
    """Split a people.csv name_variants cell.

    :param raw: Cell text.
    :return: Variants (stripped, non-empty).
    """
    return [v.strip() for v in re.split(r"[;|,]", raw or "") if v.strip()]


def base_relation(rel: str) -> str:
    """Collapse "(uncertain)" / "(deceased)" qualifiers into the base relation type.

    :param rel: PGP relation label.
    :return: Reporting category.
    """
    base = re.sub(r"\s*\((uncertain|deceased)\)", "", rel).strip()
    keep = {"Sender", "Recipient", "Scribe", "Witness", "Party", "Mentioned", "Validating judge"}
    return base if base in keep else "Other"


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------

@dataclass
class DocIndex:
    """Two normalisations of one edition's lines plus exact token n-gram indexes."""

    lines_a: list[NormLine]  # pre-connector titles dropped (for tier-1 queries)
    lines_b: list[NormLine]  # titles kept (for the other tiers and the contradiction checks)
    ngrams_a: dict[tuple[str, ...], list[tuple[int, int]]]
    ngrams_b: dict[tuple[str, ...], list[tuple[int, int]]]


def _ngram_index(lines: list[NormLine], max_n: int = 6) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    """Index token n-grams of normalised lines; the first token may carry a one/two-letter prefix.

    :param lines: Normalised lines.
    :param max_n: Longest n-gram indexed.
    :return: n-gram -> list of (line index, token index).
    """
    idx: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for li, nl in enumerate(lines):
        toks = nl.tokens
        for s in range(len(toks)):
            firsts = {toks[s]}
            for p in ALLOWED_PREFIXES:
                if toks[s].startswith(p) and len(toks[s]) - len(p) >= 2:
                    firsts.add(toks[s][len(p):])
            for n in range(1, min(max_n, len(toks) - s) + 1):
                for f0 in firsts:
                    idx[(f0,) + tuple(toks[s + 1:s + n])].append((li, s))
    return idx


# Modern editorial commentary that some PGP editions carry after the text (page references,
# scholars' names, modern-Hebrew gloss words, Gregorian years / line ranges in digits).
EDITORIAL_RE = re.compile(
    r"עמ['׳]\s*[0-9]|[0-9]{3,4}|[0-9]+\s*-\s*[0-9]+|גויטיין|פרידמן|ברוקלמן|סזגין|בן-ששון|אשתור|"
    r"(?:^|\s)(?:השוו|הכוונה|כנראה|כלומר|פירושו|מובנו)(?=[\s,.:]|$)")


def is_editorial(line: str) -> bool:
    """Heuristic: True for a modern editorial note rather than document text.

    :param line: Original edition line.
    :return: Whether to exclude the line from matching.
    """
    return bool(EDITORIAL_RE.search(line))


def build_doc_index(edition: str) -> DocIndex:
    """Split an edition into lines (>= 3 Hebrew letters, no editorial notes) and normalise/index them.

    :param edition: Edition content.
    :return: The document index.
    """
    raw_lines = [l for l in edition.split("\n") if len(HEB_LETTER_RE.findall(l)) >= 3 and not is_editorial(l)]
    lines_a = [normalize_line(l, skip_pre=True) for l in raw_lines]
    lines_b = [normalize_line(l) for l in raw_lines]
    return DocIndex(lines_a, lines_b, _ngram_index(lines_a), _ngram_index(lines_b))


def guarded_scores(queries: list[str], lines: list[str]) -> np.ndarray:
    """partial_ratio of each query against each line; plain ratio when the line is shorter.

    Without the guard a short line ("בן יעקב") scores 100 against a longer query.

    :param queries: Normalised query strings.
    :param lines: Normalised line strings.
    :return: Score matrix (queries x lines).
    """
    pr = process.cdist(queries, lines, scorer=fuzz.partial_ratio, workers=N_WORKERS)
    rr = process.cdist(queries, lines, scorer=fuzz.ratio, workers=N_WORKERS)
    ql = np.array([len(q) for q in queries])[:, None]
    ll = np.array([len(l) for l in lines])[None, :]
    return np.where(ll >= ql, pr, rr)


def _close(tok: str, options: set[str], cutoff: float = 75.0) -> bool:
    """True if ``tok`` is (fuzzily) one of ``options``.

    :param tok: Normalised token or unit.
    :param options: Normalised spellings.
    :param cutoff: Minimum fuzz.ratio.
    :return: Whether it matches any option.
    """
    return any(fuzz.ratio(tok, o) >= cutoff for o in options)


def contradicted(q: Query, toks: list[str], s: int, e: int, info: PersonInfo, relation: str) -> bool:
    """Reject a match whose immediate context names a different person.

    Rules: (1) the match sits in patronymic position ("... בן <match>") for a non-"Mentioned"
    relation; for queries without the father: (2) the match is followed (after titles, and the
    given name for kunya-only queries) by "בן Y" with Y not the person's father; (3) a kunya is
    directly followed by another person's given name; (4) a family name is preceded by another
    person's given name.

    :param q: The query.
    :param toks: Tokens of the matched line ("B" normalisation for tiers 2-5, "A" for tier 1).
    :param s: First matched token index.
    :param e: Token index after the match.
    :param info: The person's name evidence.
    :param relation: PGP relation label.
    :return: True if the context contradicts the person.
    """
    if s > 0 and toks[s - 1] == CANON_CONNECTOR and q.kind != "family" and "Mentioned" not in relation:
        return True
    if q.tier == 1:
        return False
    j, steps = e, 0
    while j < len(toks) and steps < PRE_SKIP_MAX and (is_skippable(toks[j]) or (
            q.lacks_given and info.given and _close(toks[j], info.given))):
        j, steps = j + 1, steps + 1
    if info.fathers and j + 1 < len(toks) and toks[j] == CANON_CONNECTOR:
        y = " ".join(toks[j + 1:j + 3]) if toks[j + 1] in SCRIPT_KUNYA else toks[j + 1]
        if not _close(y, info.fathers):
            return True
    if q.kind == "kunya" and info.given and e < len(toks) and toks[e] in NAME_LEXICON \
            and not _close(toks[e], info.given):
        return True
    if q.kind == "family" and info.given:
        for k in range(s - 1, max(-1, s - 5), -1):
            if toks[k] in NAME_LEXICON and not (k > 0 and toks[k - 1] == CANON_CONNECTOR):
                return not _close(toks[k], info.given)
    return False


def _token_starts(nl: NormLine) -> list[int]:
    """Character offset of every token in the normalised text.

    :param nl: Normalised line.
    :return: Start offsets.
    """
    starts = [0]
    for t in nl.tokens[:-1]:
        starts.append(starts[-1] + len(t) + 1)
    return starts


def _snap_tokens(nl: NormLine, cs: int, ce: int) -> tuple[int, int]:
    """Snap a fuzzy character alignment to whole tokens (a token counts if half of it is covered).

    :param nl: Normalised line.
    :param cs: Aligned start offset.
    :param ce: Aligned end offset (exclusive).
    :return: (first token, token after the last one).
    """
    starts = _token_starts(nl)
    inside = [k for k, (st, t) in enumerate(zip(starts, nl.tokens))
              if min(ce, st + len(t)) - max(cs, st) >= max(1, len(t) / 2)]
    if not inside:
        k = max(0, nl.text[:cs].count(" "))
        return k, min(k + 1, len(nl.tokens))
    return inside[0], inside[-1] + 1


def _token_span(nl: NormLine, s: int, e: int, prefix: int = 0) -> str:
    """Original-text span covering normalised tokens ``s..e-1`` of a line.

    :param nl: Normalised line.
    :param s: First token.
    :param e: Token after the last one.
    :param prefix: Number of leading letters of token ``s`` to leave out (prefix particles).
    :return: Substring of the original line.
    """
    starts = _token_starts(nl)
    cs = starts[s] + prefix
    ce = starts[e - 1] + len(nl.tokens[e - 1]) - 1
    return nl.original[nl.char_orig[cs]:nl.char_orig[ce] + 1].strip()


def eval_query(q: Query, sc_row: np.ndarray | None, doc: DocIndex, info: PersonInfo, relation: str) -> dict:
    """Best non-contradicted hit of one query in a document.

    Exact token containment is tried first; kunya-only and family-only queries (tiers 4-5) must
    match exactly, the others may match fuzzily (guarded partial_ratio).

    :param q: The query.
    :param sc_row: Fuzzy scores of this query against every line (None: exact lookup only).
    :param doc: Document index.
    :param info: Person name evidence.
    :param relation: PGP relation label.
    :return: Hit dict (score, line index, span, exact flag).
    """
    lines = doc.lines_a if q.tier == 1 else doc.lines_b
    ngrams = doc.ngrams_a if q.tier == 1 else doc.ngrams_b
    qt = q.text.split()
    for li, s in ngrams.get(tuple(qt), []):
        if not contradicted(q, lines[li].tokens, s, s + len(qt), info, relation):
            prefix = len(lines[li].tokens[s]) - len(qt[0])  # one/two-letter prefix (ו/ל/ד...) left out
            return {"score": 100.0, "li": li, "exact": True,
                    "span": _token_span(lines[li], s, s + len(qt), prefix)}
    if sc_row is None or not len(sc_row):
        return {"score": 0.0, "li": -1, "exact": False, "span": None}
    best = {"score": 0.0, "li": int(sc_row.argmax()), "exact": False, "span": None}
    if q.exact_only:
        best["score"] = min(float(sc_row.max()), LOCATED_MIN - 0.1)
        return best
    for li in np.argsort(-sc_row):
        score = float(sc_row[li])
        if score < WEAK_MIN:
            break
        nl = lines[li]
        if len(nl.text) >= len(q.text):
            al = fuzz.partial_ratio_alignment(q.text, nl.text)
            cs, ce = al.dest_start, max(al.dest_start + 1, al.dest_end)
        else:
            cs, ce = 0, len(nl.text)
        s, e = _snap_tokens(nl, cs, ce)
        if contradicted(q, nl.tokens, s, e, info, relation):
            continue
        return {"score": round(score, 1), "li": int(li), "exact": False, "span": _token_span(nl, s, e)}
    best["score"] = min(float(sc_row.max()), WEAK_MIN - 0.1)
    return best


def locate(queries: list[Query], doc: DocIndex, info: PersonInfo, relation: str) -> dict:
    """Find the best edition line for a set of candidate queries.

    :param queries: Candidate queries for one relation row.
    :param doc: Document index.
    :param info: Person name evidence.
    :param relation: PGP relation label.
    :return: Result dict with status, score, chosen query, line and span.
    """
    res: dict = {"status": "none", "score": 0.0, "exact": False}
    if not queries or not doc.lines_b:
        return res
    hits = []
    for tier1 in (True, False):
        qs = [q for q in queries if (q.tier == 1) == tier1 and q.kind != "given+honorific"]
        if not qs:
            continue
        lines = doc.lines_a if tier1 else doc.lines_b
        sc = guarded_scores([q.text for q in qs], [l.text for l in lines])
        hits += [(q, eval_query(q, sc[i], doc, info, relation)) for i, q in enumerate(qs)]
    hits += [(q, eval_query(q, None, doc, info, relation)) for q in queries if q.kind == "given+honorific"]
    exact = [(q, h) for q, h in hits if h["exact"]]
    res["exact"] = bool(exact)
    good = [(q, h) for q, h in hits if h["score"] >= LOCATED_MIN]
    if good:
        q, h = min(good, key=lambda x: (x[0].tier, -x[1]["score"], -len(x[0].text), x[0].cost))
        res["status"] = "located"
    else:
        q, h = max(hits, key=lambda x: x[1]["score"])
        res["status"] = "weak" if h["score"] >= WEAK_MIN else "none"
    lines = doc.lines_a if q.tier == 1 else doc.lines_b
    res.update(score=h["score"], query=q.display, tier=q.tier, kind=q.kind, source=q.source, line_no=h["li"],
               line=lines[h["li"]].original.strip() if h["li"] >= 0 else None, span=h["span"],
               exact_hit=h["exact"])
    return res


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def person_forms(row: dict, by_name: dict, by_slug: dict) -> tuple[list[str], list[str]]:
    """Script variants and Latin name forms (name + Latin variants) for a relation row.

    :param row: Relation row.
    :param by_name: people.csv keyed by name.
    :param by_slug: people.csv keyed by URL slug.
    :return: (script variants, Latin forms with the PGP name first).
    """
    person = by_name.get(row["person_name"]) or by_slug.get(row["person_slug"])
    script, latin = [], [row["person_name"]]
    for v in split_variants(person["name_variants"]) if person else []:
        if HEB_LETTER_RE.search(v) or ARABIC_RE.search(v):
            script.append(v)
        elif v not in latin:
            latin.append(v)
    return script, latin


def debug_tokens(rels: list[dict], by_name: dict, by_slug: dict, top: int) -> None:
    """Print the most frequent romanized tokens and their Hebrew spellings.

    :param rels: Usable relation rows.
    :param by_name: people.csv keyed by name.
    :param by_slug: people.csv keyed by URL slug.
    :param top: Number of tokens to print.
    """
    cnt: Counter = Counter()
    for r in rels:
        _, latin = person_forms(r, by_name, by_slug)
        cnt.update({t for form in latin for t in _clean_roman(form)[0].split()})
    for t, n in cnt.most_common(top):
        src = "L" if unicodedata.normalize("NFC", t) in LOOKUP_EXACT or fold(t) in LOOKUP_FOLDED else "R"
        print(f"{n:5d} {src} {t:22s} {[display_form(s) for s, _ in spell_token(t)]}")


def main() -> None:
    """Run the inversion + location experiment and write translit/located.jsonl."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug-tokens", type=int, default=0, help="print top-N token spellings and exit")
    ap.add_argument("--no-variants", action="store_true", help="ignore Latin name variants from people.csv")
    ap.add_argument("--no-honorific", action="store_true", help="skip the given+generic-honorific rule")
    ap.add_argument("--seed", type=int, default=7, help="spot-check sample seed")
    ap.add_argument("--spot", type=int, default=25, help="spot-check sample size")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where located.jsonl is written")
    ap.add_argument("--served", type=Path, default=SERVED_JSONL, help="served image list (JSONL, _id)")
    args = ap.parse_args()

    editions, canon = load_usable_editions(args.served)
    with RELATIONS_CSV.open(encoding="utf-8") as f:
        rels = [r for r in csv.DictReader(f) if r["pgpid"] in editions]
    by_name, by_slug = load_people()
    print(f"usable documents: {len(editions)}  relation rows: {len(rels)}")
    if args.debug_tokens:
        debug_tokens(rels, by_name, by_slug, args.debug_tokens)
        return
    NAME_LEXICON.update(build_name_lexicon(list(by_name.values())))

    docs = {d: build_doc_index(t) for d, t in editions.items()}
    q_cache: dict[tuple[str, str], list[Query]] = {}
    results = []
    n_cands: list[int] = []
    for idx, r in enumerate(rels):
        script, latin = person_forms(r, by_name, by_slug)
        if args.no_variants:
            latin = latin[:1]
        info = person_info(latin, script)
        before_q = [q for v in script for q in queries_from_script(v)]
        inv_q: list[Query] = []
        for k, form in enumerate(latin):
            key = (form, "romanized" if k == 0 else "latin_variant")
            if key not in q_cache:
                q_cache[key] = queries_from_romanized(form, key[1])
            inv_q.extend(q_cache[key])
        xkey = ("\t".join(latin), "cross")
        if xkey not in q_cache:
            q_cache[xkey] = cross_form_queries(latin)
        inv_q.extend(q_cache[xkey])
        n_cands.append(len({q.text for q in inv_q}))
        if not args.no_honorific:
            hkey = ("\t".join(latin), "honorific")
            if hkey not in q_cache:
                q_cache[hkey] = honorific_queries(latin)
            inv_q.extend(q_cache[hkey])
        after_q = list({q.text: q for q in inv_q[::-1] + before_q}.values())
        doc = docs[r["pgpid"]]
        results.append({
            "row": idx, "pgpid": r["pgpid"], "canonical_id": canon[r["pgpid"]],
            "person_name": r["person_name"], "person_slug": r["person_slug"], "relation": r["relation"],
            "relation_base": base_relation(r["relation"]), "has_script_variant": bool(script),
            "candidates": list(dict.fromkeys(d for q in sorted(inv_q, key=lambda q: (q.tier, q.cost))
                                             if q.kind != "given+honorific" for d in [q.display] + q.alts)),
            "before": locate(before_q, doc, info, r["relation"]),
            "after": locate(after_q, doc, info, r["relation"]),
        })

    with (args.out_dir / "located.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # ---------------- report ----------------
    cats = ["Sender", "Recipient", "Scribe", "Witness", "Party", "Mentioned", "Validating judge", "Other"]

    def cnt(pred) -> Counter:
        """Count rows per category satisfying ``pred``.

        :param pred: Row predicate.
        :return: Counter by relation_base (+ "ALL").
        """
        c = Counter(r["relation_base"] for r in results if pred(r))
        c["ALL"] = sum(c.values())
        return c

    tot = cnt(lambda r: True)
    b = cnt(lambda r: r["before"]["status"] == "located")
    a = cnt(lambda r: r["after"]["status"] == "located")
    aw = cnt(lambda r: r["after"]["status"] == "weak")
    bx = cnt(lambda r: r["before"]["exact"])
    ax = cnt(lambda r: r["after"]["exact"])
    n_disp = [len(r["candidates"]) for r in results]
    print(f"\ninverter search strings per row: mean {np.mean(n_cands):.1f} (median {np.median(n_cands):.0f}, "
          f"max {max(n_cands)}); candidate spellings incl. בן/בר/ביר: mean {np.mean(n_disp):.1f} "
          f"(median {np.median(n_disp):.0f})")
    print(f"{'relation':18s} {'rows':>5s} | {'before':>6s} {'after':>6s} {'weak':>5s} | {'exact_b':>7s} {'exact_a':>7s}")
    for c in cats + ["ALL"]:
        print(f"{c:18s} {tot[c]:5d} | {b[c]:6d} {a[c]:6d} {aw[c]:5d} | {bx[c]:7d} {ax[c]:7d}")
    docs_b = {r["pgpid"] for r in results if r["before"]["status"] == "located"}
    docs_a = {r["pgpid"] for r in results if r["after"]["status"] == "located"}
    docs_ax = {r["pgpid"] for r in results if r["after"]["exact"]}
    print(f"documents with >=1 located person: before {len(docs_b)}  after {len(docs_a)}  "
          f"(exact after {len(docs_ax)}) of {len({r['pgpid'] for r in results})}")
    print("after-located by kind:", Counter((r["after"]["tier"], r["after"]["kind"]) for r in results
                                            if r["after"]["status"] == "located").most_common())
    print("after-located by source:", Counter(r["after"]["source"] for r in results
                                              if r["after"]["status"] == "located").most_common())
    only_latin = [r for r in results if not r["has_script_variant"]]
    print(f"rows without script variant: {len(only_latin)}; located after: "
          f"{sum(r['after']['status'] == 'located' for r in only_latin)}")

    rng = random.Random(args.seed)
    located = [r for r in results if r["after"]["status"] == "located"]
    print(f"\n--- spot check: {args.spot} random located rows (seed {args.seed}) ---")
    for r in rng.sample(located, min(args.spot, len(located))):
        a_ = r["after"]
        print(f"[{r['row']}] {r['relation']} | {r['person_name']} | T{a_['tier']} {a_['kind']} | "
              f"{a_['query']} | {a_['score']} | exact={a_['exact_hit']}\n      span: {a_.get('span')}\n"
              f"      line: {a_['line'][:150]}")


if __name__ == "__main__":
    main()
