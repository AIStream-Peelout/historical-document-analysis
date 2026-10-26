"""Per-page choice between a born-digital PDF's text layer and Cloud Vision.

Born-digital academic books keep the exact typeset characters in their text
layer, but right-to-left lines come out of it in a broken order: words split at
span boundaries, lines interleaved, punctuation moved. Cloud Vision reads
those lines cleanly, yet on English pages it drops or garbles short Hebrew and
Arabic words set inline (measured on OBP Semitic Languages and Cultures
volumes: Vision kept 99.8% of the Hebrew letters on pages that are mostly
Hebrew, but only 77-87% on pages where Hebrew is under a quarter of the
letters, and 60% of the inline Arabic on Hathaway's English pages).

So the hybrid OCR mode sends a page to Vision only when most of its letters are
Hebrew/Arabic script, and keeps Vision's text only when it recovers the text
layer's Hebrew/Arabic letters (an order-free letter count, so the scrambled
order of the layer does not matter). Everything else stays on the text layer.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Tuple

_SEMITIC_CLASS = "א-תؠ-يٮ-ۓەۺ-ۿ"
_SEMITIC = re.compile(f"[{_SEMITIC_CLASS}]")
_LATIN = re.compile("[A-Za-zÀ-ɏḀ-ỿ]")
# Glyphs a broken ToUnicode map leaves inside Hebrew/Arabic lines: Greek letters
# (Hathaway's PDF maps lam-alef ligatures to λ/μ), private-use code points and
# the replacement character.
_SUSPECT = re.compile("[Ͱ-Ͽ-�]")
_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                           "ى": "ي", "ی": "ي", "ک": "ك"})


@dataclass(frozen=True)
class HybridThresholds:
    """Cut-offs for the hybrid OCR mode.

    :param min_share: Minimum share of Hebrew/Arabic letters among all letters
        on the text layer for a page to be sent to Vision.
    :param min_recall: Minimum share of the text layer's Hebrew/Arabic letters
        that Vision must reproduce for its text to replace the layer's.
    """

    min_share: float = 0.25
    min_recall: float = 0.97


def normalise_letters(text: str) -> str:
    """Fold text so both sources spell letters the same way.

    NFKC turns Arabic presentation forms into base letters; vowel points,
    cantillation, other combining marks and tatweel are dropped; alef, ya and
    kaf variants that Vision and fonts choose differently are unified.

    :param text: Raw page text.
    :returns: Normalised text.
    """
    folded = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    return "".join(c for c in folded if not unicodedata.combining(c) and c != "ـ").translate(_VARIANTS)


def semitic_share(text: str) -> float:
    """Return the share of Hebrew/Arabic letters among the page's letters.

    :param text: Page text (order does not matter).
    :returns: Share in ``[0, 1]``; 0 for a page without letters.
    """
    folded = normalise_letters(text)
    semitic = len(_SEMITIC.findall(folded))
    latin = len(_LATIN.findall(folded))
    return semitic / (semitic + latin) if semitic + latin else 0.0


def letter_agreement(vision_text: str, layer_text: str) -> Tuple[float, float]:
    """Compare the Hebrew/Arabic letters of Vision's text with the text layer's.

    Counts letters, not sequences, so the layer's broken order is irrelevant.
    Recall drops when Vision misses or garbles letters the page really has.
    Precision is reported but not used to reject Vision: a text layer with a
    broken glyph map (see :func:`suspect_glyphs`) lacks letters Vision reads
    correctly.

    :param vision_text: Vision ``full_text``.
    :param layer_text: Text-layer text of the same page.
    :returns: ``(recall, precision)``; 1.0 when the denominator is empty.
    """
    vision = Counter(_SEMITIC.findall(normalise_letters(vision_text)))
    layer = Counter(_SEMITIC.findall(normalise_letters(layer_text)))
    shared = sum((vision & layer).values())
    recall = shared / sum(layer.values()) if layer else 1.0
    precision = shared / sum(vision.values()) if vision else 1.0
    return recall, precision


def suspect_glyphs(text: str) -> int:
    """Count characters on Hebrew/Arabic lines that point to a broken glyph map.

    :param text: Text-layer text.
    :returns: Number of Greek, private-use or replacement characters on lines
        that also hold at least one Hebrew/Arabic letter (presentation forms
        included, so lines are tested after :func:`normalise_letters`).
    """
    return sum(len(_SUSPECT.findall(line)) for line in text.splitlines() if _SEMITIC.search(normalise_letters(line)))


def wants_vision(layer_text: str, thresholds: HybridThresholds) -> bool:
    """Decide whether a page should be read by Vision.

    :param layer_text: Text-layer text of the page.
    :param thresholds: Hybrid cut-offs.
    :returns: True when the page is mostly Hebrew/Arabic script.
    """
    return semitic_share(layer_text) >= thresholds.min_share
