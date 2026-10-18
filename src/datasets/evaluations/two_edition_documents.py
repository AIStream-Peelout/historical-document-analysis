# File name: two_edition_documents.py
# Date: 10/4/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Documents that two different scholars edited: a second human reading of the same text.

The Princeton Geniza Project export holds, for several hundred documents, digital editions from
two different sources (for example Goitein's unpublished typescript and Gil's printed edition).
Two uses:

* The difference between the two editions is the floor a transcription benchmark built on editions
  can reach: where two experts read a word differently, a model cannot be scored reliably.
* A document with two editions is a better test document than one with a single edition, because an
  error can be counted only where the reader disagrees with both.

A pair qualifies when both editions carry at least :data:`MIN_LETTERS` Hebrew letters, are mainly
Hebrew script, are not letter-for-letter identical (a copy adds no second opinion) and share at
least :data:`SAME_TEXT_SHARE` of the shorter edition's letter 5-grams (the same text, not two
different texts filed under one document).

Both editions are reduced to visible ink the way the frozen benchmark reduces its ground truth
(:func:`src.datasets.evaluations.metrics.genizah_visible_ink_gt`).
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.metrics import genizah_visible_ink_gt, normalize_whitespace

_REPO = Path(__file__).resolve().parents[3]
FOOTNOTES_CSV = _REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data/footnotes.csv"
MIN_LETTERS = 200
SAME_TEXT_SHARE = 0.5
MAX_ARABIC_SHARE = 0.2
NGRAM = 5
_HEBREW = re.compile(r"[א-ת]")
_ARABIC = re.compile(r"[ؠ-ي]")


def edition_visible_text(content: str) -> str:
    """Visible-ink text of one edition, single-spaced.

    :param content: ``content`` cell of a ``footnotes.csv`` row.
    :type content: str
    :return: The edition's lines without editorial apparatus, gaps or restorations.
    :rtype: str
    """
    lines = [line for _, section in ar.split_sections(content) for line in section]
    return unicodedata.normalize("NFC", normalize_whitespace(genizah_visible_ink_gt("\n".join(lines))))


def hebrew_letters(text: str) -> str:
    """Hebrew letters of a text, in order.

    :param text: Any text.
    :type text: str
    :return: Its Hebrew letters.
    :rtype: str
    """
    return "".join(_HEBREW.findall(text))


def shared_ngram_share(a: str, b: str, n: int = NGRAM) -> float:
    """Share of the smaller side's distinct letter n-grams that the other side also has.

    :param a: Letters of the first text.
    :type a: str
    :param b: Letters of the second text.
    :type b: str
    :param n: N-gram length.
    :type n: int
    :return: Share in [0, 1]; 0.0 when either side is shorter than ``n``.
    :rtype: float
    """
    grams_a = {a[i:i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / min(len(grams_a), len(grams_b))


def editions_by_source(footnotes_csv: Path = FOOTNOTES_CSV) -> Dict[str, Dict[str, str]]:
    """Edition texts of every document, the longest one per source.

    :param footnotes_csv: PGP export ``footnotes.csv``.
    :type footnotes_csv: Path
    :return: ``{pgpid: {source: content}}`` for rows whose relation contains "Edition".
    :rtype: Dict[str, Dict[str, str]]
    """
    csv.field_size_limit(sys.maxsize)
    out: Dict[str, Dict[str, str]] = {}
    with open(footnotes_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            content = row.get("content") or ""
            if "Edition" not in (row.get("doc_relation") or "") or not content.strip():
                continue
            per_source = out.setdefault(row["document_id"].strip(), {})
            if len(content) > len(per_source.get(row["source"], "")):
                per_source[row["source"]] = content
    return out


def two_edition_documents(footnotes_csv: Path = FOOTNOTES_CSV, min_letters: int = MIN_LETTERS,
                          same_text: float = SAME_TEXT_SHARE,
                          max_arabic_share: float = MAX_ARABIC_SHARE) -> Dict[str, Dict[str, Union[List[str], List[int], float]]]:
    """Documents with two editions of the same Hebrew-script text from different sources.

    The two longest editions from different sources are compared.

    :param footnotes_csv: PGP export ``footnotes.csv``.
    :type footnotes_csv: Path
    :param min_letters: Minimum Hebrew letters of each edition.
    :type min_letters: int
    :param same_text: Minimum :func:`shared_ngram_share` of the pair.
    :type same_text: float
    :param max_arabic_share: Maximum Arabic letters per Hebrew letter in the longer edition.
    :type max_arabic_share: float
    :return: ``{pgpid: {"sources": [first, second], "letters": [n, n], "shared": share}}``.
    :rtype: Dict[str, Dict[str, Union[List[str], List[int], float]]]
    """
    out: Dict[str, Dict[str, Union[List[str], List[int], float]]] = {}
    for pgpid, per_source in editions_by_source(footnotes_csv).items():
        if len(per_source) < 2:
            continue
        (source_a, content_a), (source_b, content_b) = sorted(per_source.items(), key=lambda kv: -len(kv[1]))[:2]
        text_a, text_b = edition_visible_text(content_a), edition_visible_text(content_b)
        letters_a, letters_b = hebrew_letters(text_a), hebrew_letters(text_b)
        if min(len(letters_a), len(letters_b)) < min_letters or text_a == text_b:
            continue
        if len(_ARABIC.findall(text_a)) > max_arabic_share * len(letters_a):
            continue
        shared = shared_ngram_share(letters_a, letters_b)
        if shared < same_text:
            continue
        out[pgpid] = {"sources": [source_a, source_b], "letters": [len(letters_a), len(letters_b)], "shared": round(shared, 4)}
    return out


def pair_texts(pgpid: str, footnotes_csv: Path = FOOTNOTES_CSV) -> Tuple[str, str]:
    """Visible-ink texts of a document's two longest editions from different sources.

    :param pgpid: PGP document id.
    :type pgpid: str
    :param footnotes_csv: PGP export ``footnotes.csv``.
    :type footnotes_csv: Path
    :return: ``(first, second)`` texts, the longer edition first.
    :rtype: Tuple[str, str]
    :raises KeyError: When the document has fewer than two edition sources.
    """
    per_source = editions_by_source(footnotes_csv).get(pgpid, {})
    if len(per_source) < 2:
        raise KeyError(f"document {pgpid} has {len(per_source)} edition source(s)")
    contents = sorted(per_source.values(), key=len, reverse=True)[:2]
    return edition_visible_text(contents[0]), edition_visible_text(contents[1])


def pgpids(documents: Dict[str, Dict[str, Union[List[str], List[int], float]]]) -> Set[str]:
    """PGP ids of a :func:`two_edition_documents` result.

    :param documents: Result of :func:`two_edition_documents`.
    :type documents: Dict[str, Dict[str, Union[List[str], List[int], float]]]
    :return: The document ids.
    :rtype: Set[str]
    """
    return set(documents)
