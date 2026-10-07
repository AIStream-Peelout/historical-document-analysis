#!/usr/bin/env python3
"""KTIV bibliography → ``biblio.json``-style citations (for Neo4j and the search index).

KTIV catalogue records carry a ``bibliography`` list of free-text citation strings such as::

    Davis, Malcolm C.; Knopf, Henry; Outhwaite, Ben, Hebrew Bible manuscripts in the Cambridge
    Genizah Collections. Cambridge University Pressעמוד 405 (איזכור)
    הורביץ, אלעזר, קטלוג קטעי גניזת קאהיר: בספריית ווסטמינסטר קולג', ב: קטלוג קטעי הגניזה. מכון
    גניזת קאהיר - ישיבה אוניב (תשס"ו), עמוד 29-30 (איזכור)

The FJP bibliography feeding the knowledge graph (``biblio.json``) is structured
(``title``, ``author``, ``year``, ``citedonpages``, ``language``, ``mention_type``);
KTIV's is not.  This module parses the KTIV strings into that schema so
``biblio_import.py`` can load them unchanged, tagging every citation with
``source: "ktiv"`` and keeping the raw string.

Parsing is heuristic and conservative: the author list is the leading run of
``Surname, Given`` segments separated by ``;`` (or ``ו`` between two Hebrew names);
the title is the text up to the first ``.``/``(``/page marker after the authors;
pages come from ``עמוד``/``עמ'``/``pp.``/``p.``; the year is a Gregorian year or a
Hebrew year in parentheses; the trailing ``(איזכור|דיון[, יש תמונה])`` tag maps to
``mention_type`` (``mentioned`` / ``discussion``) plus ``has_image``.  Anything
unparsed stays in ``title`` verbatim so no reference is lost.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.neo4j.ktiv_biblio \\
        --out src/datasets/raw_data/cairo_genizah/retained_json/ktiv_biblio.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from typing import Any, Dict, List, Optional

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

DEFAULT_MERGED = "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"

_TAG_RE = re.compile(r"\((איזכור|דיון|עיין אינדקס)([^()]*)\)\s*$")
_PAGES_RE = re.compile(r"(?:עמוד|עמ'|עמ\.|pp?\.)\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?(?:\s*,\s*[0-9]+(?:\s*[-–]\s*[0-9]+)?)*)")
_HEB_PAGES_RE = re.compile(r"(?:עמוד|עמ')\s*([א-ת]{1,4}(?:\s*[-–]\s*[א-ת]{1,4})?)")
_GREG_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
_HEB_YEAR_RE = re.compile(r"\(?\b(ת[א-ת]{0,2}\"[א-ת]|תש[א-ת]{0,2}\"?[א-ת]|תר[א-ת]{0,2}\"?[א-ת])\b\)?")
_AUTHOR_SEG_RE = re.compile(r"^\s*([A-Za-zÀ-žא-ת'’\-\.\s]+?),\s*([A-Za-zÀ-žא-ת'’\-\.\s]+?)\s*(?=;|,|$)")


def _is_hebrew(s: str) -> bool:
    """Return True when the string's first letter is Hebrew.

    :param s: Citation string.
    :type s: str
    :returns: True for Hebrew-script citations.
    :rtype: bool
    """
    m = re.search(r"[A-Za-zא-ת]", s)
    return bool(m) and "א" <= m.group(0) <= "ת"


def _split_authors(head: str) -> List[str]:
    """Split the leading author run into ``Surname, Given`` names.

    :param head: Text before the title.
    :type head: str
    :returns: Author display names in ``Surname, Given`` form.
    :rtype: List[str]
    """
    parts = [p.strip() for p in re.split(r";|\sו(?=[א-ת])", head) if p.strip()]
    out: List[str] = []
    for p in parts:
        m = re.match(r"^([^,]+),\s*([^,]+)$", p)
        out.append(f"{m.group(1).strip()}, {m.group(2).strip()}" if m else p)
    return out


def parse_ktiv_citation(raw: str) -> Dict[str, Any]:
    """Parse one KTIV citation string into the ``biblio.json`` citation schema.

    :param raw: The citation as scraped from KTIV.
    :type raw: str
    :returns: Dict with ``title``, ``author``, ``authors``, ``year``, ``citedonpages``,
        ``language``, ``mention_type``, ``has_image``, ``source``, ``raw`` and ``parsed``
        (True when authors and a title were separated).
    :rtype: Dict[str, Any]
    """
    s = " ".join(raw.split())
    mention: Dict[str, str] = {}
    has_image = False
    m = _TAG_RE.search(s)
    if m:
        tag, extra = m.group(1), m.group(2)
        if tag == "איזכור":
            mention["mentioned"] = "True"
        elif tag == "דיון":
            mention["discussion"] = "True"
        has_image = "תמונה" in extra
        s = s[: m.start()].rstrip(" ,.")
    pages = ""
    pm = _PAGES_RE.search(s) or _HEB_PAGES_RE.search(s)
    if pm:
        pages = pm.group(1).replace(" ", "")
        s_wo_pages = (s[: pm.start()] + s[pm.end():]).strip(" ,.")
    else:
        s_wo_pages = s
    year = ""
    ym = _GREG_YEAR_RE.search(s_wo_pages)
    if ym:
        year = ym.group(1)
    else:
        hm = _HEB_YEAR_RE.search(s_wo_pages)
        if hm:
            year = hm.group(1)
    # authors: leading run of "Surname, Given" segments joined by ';' (or Hebrew ' ו')
    authors: List[str] = []
    rest = s_wo_pages
    head_parts: List[str] = []
    while True:
        am = _AUTHOR_SEG_RE.match(rest)
        if not am:
            break
        surname, given = am.group(1).strip(), am.group(2).strip()
        # a "given" longer than four words is really the title, not a name
        if len(given.split()) > 4 or len(surname.split()) > 4:
            break
        head_parts.append(f"{surname}, {given}")
        rest = rest[am.end():]
        sep = re.match(r"^\s*(;|,|\sו)", rest)
        if not sep:
            break
        rest = rest[sep.end():]
        if sep.group(1) == ",":
            break
    authors = head_parts
    title = rest.strip(" ,.;:") if authors else s_wo_pages.strip()
    # title = up to the first sentence break or opening parenthesis
    tm = re.search(r"\.\s|\s\(", title)
    if tm and tm.start() > 8:
        title = title[: tm.start()].strip(" ,.;:")
    return {
        "title": title,
        "author": "; ".join(authors),
        "authors": authors,
        "year": year,
        "citedonpages": pages,
        "language": "Hebrew" if _is_hebrew(raw) else "English",
        "mention_type": mention,
        "has_image": has_image,
        "source": "ktiv",
        "raw": raw,
        "parsed": bool(authors and title),
    }


def _degenerate(key: str) -> bool:
    """Whether a canonical key is unusable as a fragment identity (numeric or too short).

    :param key: Canonical shelfmark key.
    :type key: str
    :returns: True when the key should be skipped.
    :rtype: bool
    """
    return (not key) or key.isdigit() or len(key) < 4


def build_ktiv_biblio(merged_path: str = DEFAULT_MERGED) -> Dict[str, Any]:
    """Collect every KTIV citation in the merged corpus, keyed like ``biblio.json``.

    :param merged_path: Path to ``merged_shelfmarks.jsonl``.
    :type merged_path: str
    :returns: ``{"data": {canonical_shelfmark: {"citations": [...], "canonical_id": ...}},
        "stats": {...}}``.
    :rtype: Dict[str, Any]
    """
    data: Dict[str, Dict[str, Any]] = {}
    stats: collections.Counter = collections.Counter()
    with open(merged_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ktiv = (rec.get("sources") or {}).get("ktiv") or {}
            refs = ktiv.get("bibliography") or []
            if not refs:
                continue
            stats["records_with_ktiv_bibliography"] += 1
            display = rec.get("shelfmark_display") or ""
            key = ShelfmarkNormalizer.to_canonical_id(display) if display else ""
            if _degenerate(key):
                stats["records_skipped_degenerate_key"] += 1
                continue
            cits = [parse_ktiv_citation(r) for r in refs if isinstance(r, str) and r.strip()]
            stats["citations"] += len(cits)
            stats["citations_parsed"] += sum(1 for c in cits if c["parsed"])
            stats["citations_with_pages"] += sum(1 for c in cits if c["citedonpages"])
            stats["citations_with_year"] += sum(1 for c in cits if c["year"])
            stats["citations_tagged"] += sum(1 for c in cits if c["mention_type"])
            data[key] = {"canonical_id": rec.get("canonical_id"), "shelfmark_display": display, "citations": cits}
    stats["fragments"] = len(data)
    return {"data": data, "stats": dict(stats)}


def main() -> None:
    """CLI: write ``ktiv_biblio.json`` and print parse statistics."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default=DEFAULT_MERGED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = build_ktiv_biblio(args.merged)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result["data"], fh, ensure_ascii=False, indent=1)
    print(json.dumps(result["stats"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
