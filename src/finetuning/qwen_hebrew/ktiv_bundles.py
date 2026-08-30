"""Selection of KTIV transcription bundles across duplicate scrape files.

Re-scraping a manuscript saves a browser-deduplicated sibling next to the
original (``..._transcription(1).json``, ``(2)`` …), so one ``sys_num`` can
hold several bundle files of different vintages and shapes:

* API shape (``source == "nli_ktiv_viewer"``): pages carry an
  ``annotation_page`` with per-word boxes — the clean GT source.
* DOM shape (``source == "nli_ktiv_viewer_dom"``): flat text only.

Every consumer must pick ONE bundle per manuscript with the same rule
(user-set policy 2026-08-28):

    API shape beats DOM  >  newer file mtime beats older  >  richer content.

Note the plain glob ``*_transcription.json`` does NOT match the ``(1)``
siblings — always glob ``*_transcription*.json`` and route through
:func:`pick_best_bundle` / :func:`select_bundles`.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

_SYS_RE = re.compile(r"(\d{18})")


def bundle_shape(doc: dict) -> str:
    """Classify a parsed bundle as ``"api"`` (word boxes) or ``"dom"`` (flat).

    :param doc: Parsed transcription bundle.
    :type doc: dict
    :return: ``"api"`` or ``"dom"``.
    :rtype: str
    """
    if doc.get("source") == "nli_ktiv_viewer":
        return "api"
    if any((p.get("annotation_page") or {}).get("items")
           for p in doc.get("pages") or []):
        return "api"
    return "dom"


def bundle_richness(doc: dict) -> int:
    """Content volume of a bundle, for tie-breaking equal-vintage duplicates.

    :param doc: Parsed transcription bundle.
    :type doc: dict
    :return: Flat-text characters plus annotation-item count over all pages.
    :rtype: int
    """
    n = 0
    for p in doc.get("pages") or []:
        n += len(p.get("text") or "") + sum(len(l) for l in p.get("lines") or [])
        n += len((p.get("annotation_page") or {}).get("items") or [])
    return n


def bundle_sys_num(doc: dict, path: Path) -> Optional[str]:
    """Extract the 18-digit NLI system number for a bundle.

    :param doc: Parsed transcription bundle.
    :type doc: dict
    :param path: File the bundle was read from (filename fallback).
    :type path: Path
    :return: sys_num string, or None when neither source carries one.
    :rtype: Optional[str]
    """
    m = _SYS_RE.search(doc.get("doc_id") or "") or _SYS_RE.search(path.name)
    return m.group(1) if m else None


def pick_best_bundle(paths: Iterable[Path]) -> Optional[Tuple[Path, dict]]:
    """Choose one bundle among duplicate files of ONE manuscript.

    Preference: API shape, then newer file mtime (a deliberate re-scrape
    supersedes its older sibling even if shorter), then richer content.

    :param paths: Candidate ``*_transcription*.json`` files for one sys_num.
    :type paths: Iterable[Path]
    :return: (path, parsed bundle) of the winner, or None if nothing parses.
    :rtype: Optional[Tuple[Path, dict]]
    """
    best: Optional[Tuple[Path, dict]] = None
    best_key: Tuple[bool, float, int] = (False, float("-inf"), -1)
    for path in sorted(paths):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = (bundle_shape(doc) == "api", path.stat().st_mtime, bundle_richness(doc))
        if key > best_key:
            best, best_key = (path, doc), key
    return best


def select_bundles(ktiv_dir: Path) -> Dict[str, Tuple[Path, dict]]:
    """Pick the best transcription bundle for EVERY manuscript in a scrape dir.

    :param ktiv_dir: Directory holding ``*_transcription*.json`` files.
    :type ktiv_dir: Path
    :return: ``sys_num -> (winning path, parsed bundle)`` (both shapes; the
        caller filters on :func:`bundle_shape` as needed).
    :rtype: Dict[str, Tuple[Path, dict]]
    """
    by_sys: Dict[str, list] = {}
    for path in sorted(ktiv_dir.glob("*_transcription*.json")):
        m = _SYS_RE.search(path.name)
        if not m:
            continue
        by_sys.setdefault(m.group(1), []).append(path)
    out: Dict[str, Tuple[Path, dict]] = {}
    for sys_num, paths in by_sys.items():
        winner = pick_best_bundle(paths)
        if winner is not None:
            out[sys_num] = winner
    return out
