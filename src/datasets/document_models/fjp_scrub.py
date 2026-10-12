"""Remove legacy-catalogue host URLs from text served through Elasticsearch.

Project rule: served data never links to the legacy catalogue's hosts (``genizah.org`` and all its
subdomains, ``jewishmanuscripts.org``). Academic books cite them in running
text, e.g. "the ... website (www.genizah.org)", and that text is
indexed verbatim, so the documents built for the index are scrubbed and then
checked. Scheme-less forms, subdomains, ports, paths and ``user@`` prefixes
are covered, and so are host names that OCR text wraps across a line
(``www.geni-\\nzah.org``, ``fgp.genizah.\\norg``) or spaces around the dot.
Plain name mentions in quoted scholarship are left alone.
"""
from __future__ import annotations

import re
from typing import Any

# A line wrap OCR text may carry inside a host name: soft hyphen, or an optional
# hyphen followed by a line break.
_WRAP = r"(?:­|-?[ \t]*\n[ \t]*)?"


def _wrappable(word: str) -> str:
    """Regex for *word* allowing a line wrap between any two characters, and spaces around dots.

    :param word: Literal host fragment, e.g. ``genizah.org``.
    :returns: Regex source.
    """
    return _WRAP.join(r"[ \t]?\.[ \t]?" if c == "." else re.escape(c) for c in word)


_HOST = rf"(?:{_wrappable('genizah.org')}|{_wrappable('jewishmanuscripts.org')})"
_URL = rf"""
    (?<![a-z0-9.-])                       # not the tail of a longer host name
    (?:https?://)?                        # optional scheme
    (?:[^\s/@<>"'()\[\]]+@)?              # optional userinfo
    (?:[a-z0-9-]+\.{_WRAP})*              # subdomains
    {_HOST}
    (?![a-z0-9-])                         # whole host label
    (?::\d+)?                             # optional port
    (?:/[^\s<>"')\]]*)?                   # optional path
"""
FJP_URL = re.compile(_URL, re.IGNORECASE | re.VERBOSE)
# The URL with the brackets that wrap it, so "website (www.genizah.org), and" loses "(…)" too.
_BRACKETED = re.compile(rf"[ \t]*(?:\(\s*{_URL}\s*\)|\[\s*{_URL}\s*\]|<\s*{_URL}\s*>)", re.IGNORECASE | re.VERBOSE)
# Independent, looser check: any FJP host name, however it is split or glued, fails the document.
_ANY_FJP_HOST = re.compile(r"(?<![a-z])geni[\s­-]*zah[\s­]*\.[\s­]*org"
                           r"|(?<![a-z])jewish[\s­-]*manuscripts[\s­]*\.[\s­]*org", re.IGNORECASE)


def strip_fjp_urls(text: str) -> str:
    """Delete FJP-host URLs from *text*, together with brackets that enclose nothing but the URL.

    :param text: Any text.
    :returns: Text without FJP-host URLs.
    """
    if not text or not FJP_URL.search(text):
        return text
    return FJP_URL.sub("", _BRACKETED.sub("", text))


def scrub_fjp(value: Any) -> Any:
    """Apply :func:`strip_fjp_urls` to every string inside *value*.

    :param value: A string, or a dict/list nesting of JSON-like values.
    :returns: The same structure with FJP-host URLs removed.
    """
    if isinstance(value, str):
        return strip_fjp_urls(value)
    if isinstance(value, dict):
        return {key: scrub_fjp(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_fjp(item) for item in value]
    return value


def contains_fjp_url(value: Any) -> bool:
    """Return True when any string inside *value* still names an FJP host.

    Deliberately looser than :data:`FJP_URL` (it ignores URL structure and
    accepts any split of the host name) so it can catch what the scrub missed.

    :param value: A string, or a dict/list nesting of JSON-like values.
    :returns: Whether an FJP host name is present.
    """
    if isinstance(value, str):
        return bool(_ANY_FJP_HOST.search(value))
    if isinstance(value, dict):
        return any(contains_fjp_url(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_fjp_url(item) for item in value)
    return False
