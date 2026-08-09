#!/usr/bin/env python3
"""Survey where the corpus's uncollected secondary sources can actually be got.

:mod:`src.datasets.indexing.neo4j.citation_priority_report` says *which* works
are worth acquiring, ranked by how many Genizah fragments cite them. This module
answers the next question: is there a free scan online, is it borrowable, or does
someone have to walk into a library — and if so, does Brandeis hold it, and under
what call number.

Sources queried, all public and unauthenticated:

* **Brandeis** — the Alma SRU endpoint for ``01BRAND_INST``. MARCXML ``AVA``
  fields carry the local holding (library, location, call number, status), which
  is the authoritative "can I go get it" answer. ``035`` yields an OCLC number.
* **HathiTrust** — the Bib API, keyed by the OCLC number Brandeis supplies.
  Distinguishes full view (downloadable) from in-copyright search-only.
* **Internet Archive** — advancedsearch; ``access-restricted-item`` separates
  public-domain downloads from controlled digital lending.
* **Open Library** — search API; reports ``public`` / ``borrowable`` /
  ``printdisabled`` / ``no_ebook``.

This reports *availability only*. Nothing is downloaded, and no paywalled or
restricted material is fetched — the output is links, call numbers and access
tiers for a human to act on.

Two things make naive lookups fail badly here, and both are handled:

* **KG titles carry cruft.** They arrive as ``…of the Cairo Geniza Vol: 2``, and
  an exact-title search on that finds nothing. :func:`search_variants` strips
  volume designators and trailing parentheticals, then falls back to the main
  title before the subtitle.
* **Half the corpus is Hebrew.** Union catalogues frequently hold the record
  under a romanised title (``אוצר כתבי היד התלמודיים`` is catalogued as
  ``Otsar kitve-ha-yad ha-Talmudiyim``), so token overlap against the Hebrew
  scores zero. Matching therefore also scores MARC ``880`` original-script
  fields, and treats a narrow exact-phrase result as evidence in its own right.

Every hit records the catalogue title that was actually matched and a similarity
score. Read ``matched_title`` before trusting a row.

Usage::

    python -m src.datasets.indexing.bibliography.availability_survey --limit 150
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRIORITY_CSV = _REPO_ROOT / "artifacts/citation_priority/citation_priority.csv"
DEFAULT_OUT_DIR = _REPO_ROOT / "artifacts/availability"

USER_AGENT = "genizah-research/0.1 (academic availability survey)"

BRANDEIS_SRU = "https://na01.alma.exlibrisgroup.com/view/sru/01BRAND_INST"
BRANDEIS_DISCOVERY = (
    "https://search.library.brandeis.edu/discovery/search"
    "?vid=01BRAND_INST:BRAND&query=any,contains,"
)
HATHI_BIB = "https://catalog.hathitrust.org/api/volumes/brief/json/oclc:"

MARC_NS = {"srw": "http://www.loc.gov/zing/srw/", "marc": "http://www.loc.gov/MARC21/slim"}

#: Minimum normalised-title similarity for a catalogue hit to count on its own.
MATCH_THRESHOLD = 0.55

#: An exact-phrase title search returning at most this many records is itself
#: evidence of a match, which is what rescues romanised Hebrew records whose
#: title shares no tokens with the Hebrew query.
NARROW_RESULT_COUNT = 3

#: Ordered best-to-worst; the first tier a work qualifies for wins.
ACCESS_TIERS = (
    "open_online",      # free full text, download now
    "borrow_online",    # controlled digital lending
    "at_brandeis",      # physical copy on campus — scan it there
    "search_only",      # located but no readable copy
    "elsewhere",        # not found; ILL or another library
)

#: Trailing volume designators the KG appends to titles.
_VOLUME_SUFFIX = re.compile(
    r"[\s,;:]*(?:vol(?:ume)?|כרך|kerekh|bd|tome)\s*[.:]?\s*[0-9ivxlIVXL]+\s*$", re.IGNORECASE
)
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")

#: Some KG titles embed the author: ``Shaked, Shaul: A tentative bibliography``.
#: Splitting such a title on its colon yields the *name*, which matches nothing.
_AUTHOR_PREFIX = re.compile(r"^\s*[^\s,:]{2,}\s*,\s*[^\s,:]{2,}\s*:\s+(?=\S)")


# ---------------------------------------------------------------------------
# HTTP plumbing: throttling + an on-disk cache so re-runs are cheap and polite
# ---------------------------------------------------------------------------


class HttpClient:
    """Cached, throttled HTTP GET for public bibliographic APIs.

    Responses are cached on disk by URL, so re-runs cost nothing and no provider
    is asked the same question twice.

    :param cache_dir: Directory for cached responses.
    :param min_interval: Minimum seconds between requests to one host.
    :param timeout: Per-request timeout in seconds.
    :param max_attempts: Tries per URL before giving up, including the first.
    :param backoff_base: Seconds for the first retry pause; doubles each time.
    """

    def __init__(
        self,
        cache_dir: Path,
        min_interval: float = 1.0,
        timeout: int = 30,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._min_interval = min_interval
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._last_call: Dict[str, float] = {}
        self.stats: Dict[str, int] = {"hit": 0, "miss": 0, "error": 0, "throttled": 0}

    def _wait(self, url: str) -> None:
        """Sleep so the URL's host is not called faster than the interval.

        :param url: Request URL.
        """
        host = urllib.parse.urlparse(url).netloc
        elapsed = time.monotonic() - self._last_call.get(host, 0.0)
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call[host] = time.monotonic()

    def get(self, url: str) -> Optional[str]:
        """Fetch a URL, using the cache when possible.

        A provider being down must not abort the survey, so failures are logged
        and reported as "not found" for that provider rather than raised.

        :param url: Request URL.
        :returns: Response body, or ``None`` when the request failed.
        """
        path = self.cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.txt"
        if path.exists():
            self.stats["hit"] += 1
            return path.read_text(encoding="utf-8")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        # Providers throttle bulk callers; a 429/503 means "slow down", not
        # "not found", and treating it as a miss would silently blank out a
        # large fraction of a long run.
        for attempt in range(self.max_attempts):
            self._wait(url)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8", "replace")
                self.stats["miss"] += 1
                path.write_text(body, encoding="utf-8")
                return body
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < self.max_attempts:
                    backoff = self.backoff_base * (2 ** attempt)
                    self.stats["throttled"] += 1
                    logger.info("HTTP %s, backing off %.1fs: %s", exc.code, backoff, url[:100])
                    time.sleep(backoff)
                    continue
                self.stats["error"] += 1
                logger.warning("GET failed (HTTP %s): %s", exc.code, url[:120])
                return None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                self.stats["error"] += 1
                logger.warning("GET failed (%s): %s", type(exc).__name__, url[:120])
                return None
        return None


# ---------------------------------------------------------------------------
# Title handling
# ---------------------------------------------------------------------------


def normalize(text: Optional[str]) -> str:
    """Case-fold, strip accents and punctuation, collapse whitespace.

    :param text: Raw text.
    :returns: A normalised comparison string.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text).casefold()
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"[^\w\s]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


#: Shared tokens a full-text match must have before it is believed. Ratio alone
#: is not enough: a three-word query like ``India Book 2`` shares two tokens with
#: any Indian book in the Archive and scores 0.67 on containment.
MIN_SHARED_TOKENS = 3


def _strip_article(tokens: List[str]) -> List[str]:
    """Drop a leading English/French definite or indefinite article.

    :param tokens: Normalised title tokens.
    :returns: The tokens without a leading article.
    """
    return tokens[1:] if tokens and tokens[0] in {"a", "an", "the", "la", "le", "les"} else tokens


def titles_match(query: str, candidate: Optional[str]) -> Tuple[bool, float]:
    """Decide whether a catalogue title is the work being looked for.

    Token overlap alone is far too permissive for a full-text archive. The
    query ``Arabica`` is wholly contained in *100% Arabica* (a film), ``Textus``
    in *Acta Andreae : textus*, and ``Jewish Social Studies`` in *Bibliography
    of Jewish social studies* — all score 1.0 on containment and none is the
    work sought.

    What separates a real match is that one title *begins* with the other: a
    catalogue record legitimately extends a citation's title with a subtitle
    (``Masoreten des Ostens, die ältesten punktierten Handschriften…``) but does
    not prepend words to it. So a match needs an overlap ratio, an absolute
    number of shared tokens, and a prefix relationship in one direction.

    :param query: The title being searched for.
    :param candidate: The catalogue's title.
    :returns: ``(matched, score)``.
    """
    query_words = _strip_article(normalize(query).split())
    candidate_words = _strip_article(normalize(candidate).split())
    if not query_words or not candidate_words:
        return False, 0.0
    query_tokens, candidate_tokens = set(query_words), set(candidate_words)
    shared = query_tokens & candidate_tokens
    score = len(shared) / min(len(query_tokens), len(candidate_tokens))

    shorter, longer = sorted((query_words, candidate_words), key=len)
    if longer[: len(shorter)] != shorter:
        return False, score
    if len(query_tokens) < MIN_SHARED_TOKENS:
        # Too short for overlap to corroborate anything; the prefix rule above
        # already did the work, so just require the tokens to be equal.
        return query_tokens == candidate_tokens, score
    return (len(shared) >= MIN_SHARED_TOKENS and score >= MATCH_THRESHOLD), score


def title_similarity(left: Optional[str], right: Optional[str]) -> float:
    """Score how well two titles agree, by token containment.

    Containment rather than Jaccard, because catalogue records usually carry a
    far longer subtitle than a citation does; requiring symmetry would reject
    correct matches.

    :param left: One title.
    :param right: The other title.
    :returns: A score in ``[0, 1]``.
    """
    left_tokens = set(normalize(left).split())
    right_tokens = set(normalize(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def search_variants(title: str) -> List[str]:
    """Build catalogue query strings for a KG title, most specific first.

    The KG appends volume designators (``Vol: 2``, ``כרך: 2``) that no catalogue
    carries, trailing parenthetical date ranges that break phrase matching, and
    sometimes an author prefix (``Shaked, Shaul: A tentative bibliography…``)
    whose colon would otherwise make the "main title" a personal name.
    Stripping those, then falling back to the main title before the subtitle,
    turns most misses into hits.

    :param title: Raw title from the citation-priority report.
    :returns: Query strings, deduplicated, in decreasing specificity.
    """
    cleaned = _AUTHOR_PREFIX.sub("", title or "").strip()
    cleaned = _VOLUME_SUFFIX.sub("", cleaned).strip(" .,;:")
    cleaned = _TRAILING_PAREN.sub("", cleaned).strip(" .,;:")
    variants = [cleaned] if cleaned else []
    main = re.split(r"\s*[:;]\s*", cleaned)[0].strip() if cleaned else ""
    # Only worth a second query when the main title is substantial on its own.
    if main and main != cleaned and len(main.split()) >= 3:
        variants.append(main)
    seen: set = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class SourceHit:
    """One provider's answer about one work.

    :param provider: Provider name.
    :param access: ``full_download``, ``borrow``, ``search_only``, ``holding``
        or ``none``.
    :param matched_title: The catalogue title that matched — audit this.
    :param score: Title similarity of the match.
    :param url: A link a human can follow.
    :param detail: Provider-specific extras (call number, identifier, status).
    """

    provider: str
    access: str = "none"
    matched_title: str = ""
    score: float = 0.0
    url: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        """:returns: True when the provider found the work at all."""
        return self.access != "none"


@dataclass
class WorkAvailability:
    """A cited work plus everywhere it was found.

    :param title: Title from the citation-priority report.
    :param year: Publication year, when known.
    :param authors: Author string from the priority report.
    :param fragments: Distinct fragments citing the work.
    :param hits: One :class:`SourceHit` per provider queried.
    """

    title: str
    year: str
    authors: str
    fragments: int
    hits: List[SourceHit] = field(default_factory=list)

    def hit(self, provider: str) -> Optional[SourceHit]:
        """Return this work's hit from one provider.

        :param provider: Provider name.
        :returns: The hit, or ``None``.
        """
        return next((h for h in self.hits if h.provider == provider), None)

    @property
    def brandeis_call_number(self) -> str:
        """:returns: The Brandeis call number, or ``""``."""
        hit = self.hit("brandeis")
        return (hit.detail.get("call_number") or "") if hit else ""

    @property
    def tier(self) -> str:
        """Classify how much effort obtaining this work takes.

        A free download beats a loan, a loan beats a trip to the stacks, and a
        trip beats an interlibrary request.

        :returns: One of :data:`ACCESS_TIERS`.
        """
        accesses = {h.provider: h.access for h in self.hits}
        values = set(accesses.values())
        if "full_download" in values:
            return "open_online"
        if "borrow" in values:
            return "borrow_online"
        if accesses.get("brandeis") == "holding":
            return "at_brandeis"
        if "search_only" in values:
            return "search_only"
        return "elsewhere"

    @property
    def best_url(self) -> str:
        """:returns: The most useful link for this work's tier."""
        preference = {
            "open_online": ("internet_archive", "hathitrust", "open_library"),
            "borrow_online": ("internet_archive", "open_library", "hathitrust"),
            "at_brandeis": ("brandeis",),
            "search_only": ("hathitrust", "brandeis", "open_library"),
        }.get(self.tier, ("brandeis", "open_library", "internet_archive"))
        for provider in preference:
            hit = self.hit(provider)
            if hit and hit.matched and hit.url:
                return hit.url
        return next((h.url for h in self.hits if h.matched and h.url), "")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _marc_title(record: ET.Element, tag: str = "245") -> str:
    """Join a MARC title field's ``a``/``b`` subfields.

    :param record: A ``marc:record`` element.
    :param tag: Datafield tag (``245`` transliterated, ``880`` original script).
    :returns: The title string.
    """
    parts = [
        sub.text or ""
        for f in record.findall(f"marc:datafield[@tag='{tag}']", MARC_NS)
        for sub in f.findall("marc:subfield", MARC_NS)
        if sub.get("code") in ("a", "b")
    ]
    return " ".join(parts).strip(" /:,")


def _marc_oclc(record: ET.Element) -> str:
    """Extract an OCLC number from MARC ``035``.

    :param record: A ``marc:record`` element.
    :returns: The bare OCLC number, or ``""``.
    """
    for f in record.findall("marc:datafield[@tag='035']", MARC_NS):
        for sub in f.findall("marc:subfield", MARC_NS):
            text = sub.text or ""
            if "OCoLC" in text:
                digits = re.sub(r"\D", "", text)
                if digits:
                    return digits
    return ""


def catalogue_author(authors: str) -> str:
    """Pick a single author to search a catalogue by.

    The priority report joins co-authors with ``;`` and mixes spellings
    (``Gil, Moshe`` / ``Shaked S``). The first entry is the one catalogues file
    under, so it is the one used.

    :param authors: The ``authors`` cell from the citation-priority CSV.
    :returns: A single author string, or ``""``.
    """
    first = (authors or "").split(";")[0].strip()
    # Hebrew-only author forms rarely match the romanised creator index.
    return first if first and any(ch.isascii() and ch.isalpha() for ch in first) else ""


def _sru_records(client: HttpClient, query: str) -> Tuple[int, List[ET.Element]]:
    """Run one SRU search against Brandeis.

    :param client: HTTP client.
    :param query: A complete CQL query string.
    :returns: ``(total_records, record_elements)``; ``(0, [])`` on failure.
    """
    url = (
        f"{BRANDEIS_SRU}?version=1.2&operation=searchRetrieve&recordSchema=marcxml"
        f"&maximumRecords=8&query={urllib.parse.quote(query)}"
    )
    body = client.get(url)
    if not body:
        return 0, []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        logger.warning("Unparseable SRU response for %r", query[:80])
        return 0, []
    try:
        total = int(root.findtext(".//srw:numberOfRecords", default="0", namespaces=MARC_NS) or 0)
    except ValueError:
        total = 0
    return total, root.findall(".//marc:record", MARC_NS)


def brandeis_queries(title: str, author: str = "") -> List[Tuple[str, bool]]:
    """Build the CQL queries to try for one work, most specific first.

    The last resort pairs the author with the leading words of the title. That
    is what finds a work catalogued under a different title form than the one
    the citation used — Gil's *Palestine During the First Muslim Period* is
    shelved as *A history of Palestine, 634-1099*.

    :param title: Work title from the KG.
    :param author: Author string from the citation-priority report.
    :returns: ``(query, is_title_phrase)`` pairs; the flag marks queries whose
        narrowness may stand in for a title-similarity match.
    """
    variants = search_variants(title)
    queries = [(f'alma.title="{variant}"', True) for variant in variants]
    creator = catalogue_author(author)
    keyword = _distinctive_word(variants[0]) if variants else ""
    if creator and keyword:
        # One distinctive word, not a phrase: the point of this query is to
        # tolerate a title that differs from the citation's, and the author
        # already does the constraining.
        queries.append((f'alma.creator="{creator}" and alma.title="{keyword}"', False))
    return queries


def _distinctive_word(title: str) -> str:
    """Pick the most distinctive early word of a title for a keyword query.

    The longest of the opening words is a good proxy for the most specific one
    (``Palestine`` over ``During``/``the``/``First``).

    :param title: A cleaned title.
    :returns: The chosen word, or ``""``.
    """
    words = [w for w in title.split()[:5] if len(w) > 3]
    return max(words, key=len) if words else ""


def query_brandeis(client: HttpClient, title: str, author: str = "") -> SourceHit:
    """Ask the Brandeis Alma SRU whether the library holds a work.

    Tries each query from :func:`brandeis_queries` and keeps the best-scoring
    record. A record with no ``AVA`` datafield is catalogued but not held as a
    local item, which is reported as ``search_only``, not as a holding.

    :param client: HTTP client.
    :param title: Work title from the KG.
    :param author: Author string, used for the last-resort creator query.
    :returns: The hit, carrying the call number and an OCLC number when found.
    """
    hit = SourceHit(provider="brandeis")
    best: Optional[Tuple[float, ET.Element, str, str]] = None
    for query, is_title_phrase in brandeis_queries(title, author):
        total, records = _sru_records(client, query)
        if not records:
            continue
        via = "title" if is_title_phrase else "author+title"
        for record in records:
            transliterated = _marc_title(record, "245")
            original_script = _marc_title(record, "880")
            score = max(
                title_similarity(title, transliterated),
                title_similarity(title, original_script),
            )
            # A narrow result is evidence in its own right: it is how a Hebrew
            # query legitimately matches a romanised catalogue record, and how
            # the author query finds an edition published under a different
            # title. Only the top record of such a result earns the benefit,
            # and `matched_via` records that the title itself did not agree.
            if total <= NARROW_RESULT_COUNT and record is records[0]:
                score = max(score, MATCH_THRESHOLD)
            label = original_script or transliterated
            if best is None or score > best[0]:
                best = (score, record, label, via)
        if best and best[0] >= MATCH_THRESHOLD:
            break
    if not best or best[0] < MATCH_THRESHOLD:
        return hit

    score, record, label, via = best
    hit.score = round(score, 3)
    hit.matched_title = (label or "")[:200]
    hit.url = BRANDEIS_DISCOVERY + urllib.parse.quote((label or title)[:120])
    holdings = []
    for field_ava in record.findall("marc:datafield[@tag='AVA']", MARC_NS):
        sub = {s.get("code"): s.text for s in field_ava.findall("marc:subfield", MARC_NS)}
        holdings.append({
            "library": sub.get("b") or "",
            "location": sub.get("c") or "",
            "call_number": sub.get("d") or "",
            "status": sub.get("e") or "",
        })
    oclc = _marc_oclc(record)
    if holdings:
        hit.access = "holding"
        hit.detail = {
            "call_number": holdings[0]["call_number"],
            "library": holdings[0]["library"],
            "location": holdings[0]["location"],
            "status": holdings[0]["status"],
            "holdings": holdings,
            "oclc": oclc,
            "matched_via": via,
        }
    else:
        hit.access = "search_only"
        hit.detail = {"note": "catalogued but no local item attached",
                      "oclc": oclc, "matched_via": via}
    return hit


def query_hathitrust(client: HttpClient, oclc: str) -> SourceHit:
    """Check HathiTrust for a digitised copy, keyed by OCLC number.

    HathiTrust's Bib API is identifier-based, so this only runs for works whose
    Brandeis record supplied an OCLC number. ``usRightsString`` beginning
    "Full view" means downloadable; "Limited" means in-copyright search-only.

    :param client: HTTP client.
    :param oclc: OCLC number from the Brandeis MARC record.
    :returns: The hit.
    """
    hit = SourceHit(provider="hathitrust")
    if not oclc:
        return hit
    body = client.get(HATHI_BIB + urllib.parse.quote(oclc))
    if not body:
        return hit
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return hit
    items: List[Dict[str, Any]] = []
    records: Dict[str, Any] = {}
    for entry in payload.values():
        items.extend(entry.get("items") or [])
        records.update(entry.get("records") or {})
    if not items:
        return hit
    rights = [str(item.get("usRightsString") or "") for item in items]
    full_view = next((r for r in rights if r.lower().startswith("full view")), "")
    first_record = next(iter(records.values()), {})
    hit.matched_title = "; ".join(first_record.get("titles") or [])[:200]
    hit.score = 1.0  # matched by identifier, not by title
    hit.url = (
        first_record.get("recordURL")
        or (items[0].get("itemURL") or "")
    )
    hit.access = "full_download" if full_view else "search_only"
    hit.detail = {"oclc": oclc, "items": len(items), "rights": sorted(set(rights))}
    return hit


def query_internet_archive(client: HttpClient, title: str) -> SourceHit:
    """Search the Internet Archive for a scanned copy.

    ``access-restricted-item: true`` marks controlled digital lending; anything
    else with a text mediatype is directly downloadable.

    :param client: HTTP client.
    :param title: Work title from the KG.
    :returns: The hit.
    """
    hit = SourceHit(provider="internet_archive")
    variants = search_variants(title)
    full_title = variants[0] if variants else title
    for variant in variants:
        query = f'title:("{variant[:120]}") AND mediatype:(texts)'
        url = (
            "https://archive.org/advancedsearch.php?q="
            + urllib.parse.quote(query)
            + "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=creator&fl%5B%5D=year"
            + "&fl%5B%5D=access-restricted-item&rows=8&page=1&output=json"
        )
        body = client.get(url)
        if not body:
            continue
        try:
            docs = json.loads(body).get("response", {}).get("docs", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        best = None
        for doc in docs:
            doc_title = doc.get("title")
            if isinstance(doc_title, list):
                doc_title = doc_title[0] if doc_title else ""
            # Score against the full cleaned title, never the truncated variant:
            # a short variant matches far too much of the Archive.
            ok, score = titles_match(full_title, doc_title)
            if ok and (best is None or score > best[0]):
                best = (score, doc, doc_title or "")
        if best:
            score, doc, doc_title = best
            restricted = str(doc.get("access-restricted-item", "")).lower() == "true"
            hit.access = "borrow" if restricted else "full_download"
            hit.score = round(score, 3)
            hit.matched_title = doc_title[:200]
            hit.url = f"https://archive.org/details/{doc.get('identifier', '')}"
            hit.detail = {"identifier": doc.get("identifier"), "restricted": restricted,
                          "year": doc.get("year")}
            return hit
    return hit


def query_open_library(client: HttpClient, title: str) -> SourceHit:
    """Search Open Library and report its ebook access level.

    :param client: HTTP client.
    :param title: Work title from the KG.
    :returns: The hit.
    """
    hit = SourceHit(provider="open_library")
    variants = search_variants(title)
    full_title = variants[0] if variants else title
    for variant in variants:
        url = (
            "https://openlibrary.org/search.json?title="
            + urllib.parse.quote(variant[:150])
            + "&fields=title,author_name,first_publish_year,ebook_access,ia,key&limit=8"
        )
        body = client.get(url)
        if not body:
            continue
        try:
            docs = json.loads(body).get("docs", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        best = None
        for doc in docs:
            ok, score = titles_match(full_title, doc.get("title"))
            if ok and (best is None or score > best[0]):
                best = (score, doc)
        if best:
            score, doc = best
            access = (doc.get("ebook_access") or "no_ebook").lower()
            hit.access = {
                "public": "full_download",
                "borrowable": "borrow",
                "printdisabled": "search_only",
                "no_ebook": "search_only",
            }.get(access, "search_only")
            hit.score = round(score, 3)
            hit.matched_title = (doc.get("title") or "")[:200]
            hit.url = "https://openlibrary.org" + (doc.get("key") or "")
            hit.detail = {"ebook_access": access, "ia": (doc.get("ia") or [])[:3],
                          "first_publish_year": doc.get("first_publish_year")}
            return hit
    return hit


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


def load_targets(
    csv_path: Path,
    limit: int,
    min_fragments: int,
    kinds: Sequence[str],
) -> List[Dict[str, str]]:
    """Read the uncollected works to survey from the citation-priority CSV.

    :param csv_path: Path to ``citation_priority.csv``.
    :param limit: Maximum works to return, highest citation weight first.
    :param min_fragments: Skip works cited by fewer fragments than this.
    :param kinds: Which ``kind`` values to include.
    :returns: Target rows.
    """
    rows = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("have_locally") == "True":
                continue
            if row.get("kind") not in kinds:
                continue
            if int(row.get("fragments") or 0) < min_fragments:
                continue
            rows.append(row)
    rows.sort(key=lambda r: -int(r.get("fragments") or 0))
    return rows[:limit]


def survey_work(client: HttpClient, row: Dict[str, str]) -> WorkAvailability:
    """Query every provider about one work.

    Brandeis runs first because its MARC record supplies the OCLC number that
    HathiTrust's identifier-based API needs.

    :param client: HTTP client.
    :param row: A target row from :func:`load_targets`.
    :returns: The assembled availability record.
    """
    title = (row.get("title") or "").strip()
    work = WorkAvailability(
        title=title,
        year=(row.get("year") or "").strip(),
        authors=(row.get("authors") or "").strip(),
        fragments=int(row.get("fragments") or 0),
    )
    brandeis = _safe_brandeis(client, title, work.authors)
    work.hits.append(brandeis)
    work.hits.append(_safe(query_hathitrust, "hathitrust", client,
                           brandeis.detail.get("oclc", "")))
    work.hits.append(_safe(query_internet_archive, "internet_archive", client, title))
    work.hits.append(_safe(query_open_library, "open_library", client, title))
    return work


def _safe_brandeis(client: HttpClient, title: str, authors: str) -> SourceHit:
    """Run the Brandeis query, converting any failure into an empty hit.

    :param client: HTTP client.
    :param title: Work title.
    :param authors: Author string from the priority report.
    :returns: The hit, empty on failure.
    """
    try:
        return query_brandeis(client, title, authors)
    except Exception as exc:  # noqa: BLE001 — deliberate provider isolation
        logger.warning("brandeis failed for %r: %s", title[:50], exc)
        return SourceHit(provider="brandeis")


def _safe(query, provider: str, client: HttpClient, argument: str) -> SourceHit:
    """Run a provider query, converting any failure into an empty hit.

    One provider erroring must not sink the survey of 150 works.

    :param query: The provider function.
    :param provider: Provider name, for the empty hit.
    :param client: HTTP client.
    :param argument: Title or identifier to pass through.
    :returns: The hit, empty on failure.
    """
    try:
        return query(client, argument)
    except Exception as exc:  # noqa: BLE001 — deliberate provider isolation
        logger.warning("%s failed for %r: %s", provider, str(argument)[:50], exc)
        return SourceHit(provider=provider)


def write_csv(works: Sequence[WorkAvailability], path: Path) -> None:
    """Write the per-work availability table, easiest-to-obtain first.

    :param works: Surveyed works.
    :param path: Destination CSV path.
    """
    columns = [
        "rank", "fragments", "tier", "title", "year", "authors",
        "brandeis", "brandeis_call_number", "brandeis_library", "brandeis_location",
        "hathitrust", "internet_archive", "open_library",
        "best_url", "matched_title", "match_score", "brandeis_matched_via",
    ]
    ordered = sorted(works, key=lambda w: (ACCESS_TIERS.index(w.tier), -w.fragments))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, work in enumerate(ordered, start=1):
            brandeis = work.hit("brandeis")
            detail = brandeis.detail if brandeis else {}
            matched = max(
                (h for h in work.hits if h.matched), key=lambda h: h.score, default=None
            )
            writer.writerow({
                "rank": rank,
                "fragments": work.fragments,
                "tier": work.tier,
                "title": work.title,
                "year": work.year,
                "authors": work.authors,
                "brandeis": brandeis.access if brandeis else "none",
                "brandeis_call_number": detail.get("call_number", ""),
                "brandeis_library": detail.get("library", ""),
                "brandeis_location": detail.get("location", ""),
                "hathitrust": (work.hit("hathitrust") or SourceHit("hathitrust")).access,
                "internet_archive": (
                    work.hit("internet_archive") or SourceHit("internet_archive")
                ).access,
                "open_library": (work.hit("open_library") or SourceHit("open_library")).access,
                "best_url": work.best_url,
                "matched_title": matched.matched_title if matched else "",
                "match_score": matched.score if matched else 0.0,
                "brandeis_matched_via": detail.get("matched_via", ""),
            })


def write_summary(works: Sequence[WorkAvailability], path: Path) -> Dict[str, Any]:
    """Write the machine-readable rollup.

    :param works: Surveyed works.
    :param path: Destination JSON path.
    :returns: The report dict that was written.
    """
    by_tier: Dict[str, List[WorkAvailability]] = {tier: [] for tier in ACCESS_TIERS}
    for work in works:
        by_tier[work.tier].append(work)
    report = {
        "works_surveyed": len(works),
        "fragments_covered": sum(w.fragments for w in works),
        "tiers": {
            tier: {
                "works": len(items),
                "fragments": sum(w.fragments for w in items),
                "entries": [
                    {
                        "title": w.title,
                        "year": w.year,
                        "fragments": w.fragments,
                        "call_number": w.brandeis_call_number,
                        "library": (w.hit("brandeis").detail.get("library", "")
                                    if w.hit("brandeis") else ""),
                        "location": (w.hit("brandeis").detail.get("location", "")
                                     if w.hit("brandeis") else ""),
                        "url": w.best_url,
                        "matched_title": max(
                            (h for h in w.hits if h.matched),
                            key=lambda h: h.score, default=SourceHit("none"),
                        ).matched_title,
                    }
                    for w in sorted(items, key=lambda w: -w.fragments)
                ],
            }
            for tier, items in by_tier.items()
        },
        "detail": [
            {**{k: v for k, v in asdict(w).items() if k != "hits"},
             "tier": w.tier,
             "hits": [asdict(h) for h in w.hits]}
            for w in sorted(works, key=lambda w: -w.fragments)
        ],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


TIER_LABELS = {
    "open_online": "FREE ONLINE — download now, no travel",
    "borrow_online": "BORROWABLE ONLINE — controlled digital lending",
    "at_brandeis": "AT BRANDEIS — physical copy, scan on site",
    "search_only": "LOCATED, NOT READABLE — in copyright / no local item",
    "elsewhere": "NOT LOCATED — ILL or another library",
}


def print_report(report: Dict[str, Any], top: int) -> None:
    """Print the availability report grouped by access tier.

    :param report: The rollup from :func:`write_summary`.
    :param top: Works to list per tier.
    """
    print(
        f"Surveyed {report['works_surveyed']} uncollected works "
        f"covering {report['fragments_covered']} fragment citations.\n"
    )
    for tier in ACCESS_TIERS:
        block = report["tiers"][tier]
        print(f"  {TIER_LABELS[tier]:52s} {block['works']:4d} works, "
              f"{block['fragments']:6d} fragments")
    for tier in ACCESS_TIERS:
        block = report["tiers"][tier]
        if not block["works"]:
            continue
        print(f"\n=== {TIER_LABELS[tier]} ===")
        for row in block["entries"][:top]:
            locator = row["call_number"] or row["url"][:62] or "-"
            print(f"  {row['fragments']:5d}  {row['title'][:56]:56s}  {locator}")


def main() -> None:
    """CLI entry point: survey where uncollected sources can be obtained."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-csv", type=Path, default=DEFAULT_PRIORITY_CSV,
                        help="citation_priority.csv to read targets from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=150, help="Works to survey")
    parser.add_argument("--min-fragments", type=int, default=10,
                        help="Skip works cited by fewer fragments than this")
    parser.add_argument("--kind", action="append", default=None,
                        help="Work kinds to include (default: monograph)")
    parser.add_argument("--top", type=int, default=15, help="Works to print per tier")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Minimum seconds between requests to one host")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    targets = load_targets(
        args.priority_csv, args.limit, args.min_fragments, args.kind or ["monograph"]
    )
    if not targets:
        print("No targets matched the filters.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = HttpClient(args.out_dir / "cache", min_interval=args.interval)
    works: List[WorkAvailability] = []
    for index, row in enumerate(targets, start=1):
        works.append(survey_work(client, row))
        if index % 10 == 0:
            print(f"  … {index}/{len(targets)} surveyed", flush=True)

    csv_path = args.out_dir / "availability.csv"
    json_path = args.out_dir / "availability_summary.json"
    write_csv(works, csv_path)
    report = write_summary(works, json_path)
    print_report(report, args.top)
    print(f"\nHTTP: {client.stats['hit']} cached, {client.stats['miss']} fetched, "
          f"{client.stats['error']} failed")
    print(f"Wrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
