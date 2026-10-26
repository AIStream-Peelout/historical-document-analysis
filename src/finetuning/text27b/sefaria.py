# File name: sefaria.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Local cache of Sefaria canonical texts (Hebrew / Aramaic) for the canon-comparison families.

Sefaria publishes its whole library as a public bucket indexed by ``books.json`` in the Sefaria-Export GitHub
repository (the repository itself no longer carries the ``json/`` tree). This module pins one ``books.json``
revision (the repository commit), picks ONE Hebrew version per title for each corpus we compare Genizah fragments
against (Talmud Bavli, Mishneh Torah, Mishnah, Ben Sira, ...; preference lists in :data:`CORPORA` avoid
non-commercial versions where an open one exists), downloads the version JSON plus the title's schema, and records
the licence each version file declares. Reading turns a version file into :class:`Segment` records (one per
Sefaria segment: a Talmud line, a halakhah, a mishnah, a verse) with English and Hebrew references and text cleaned
of HTML, footnotes and the editors' biblical-citation parentheses.

Cache layout (``paths.json`` entry ``sefaria_dir``; default :data:`DEFAULT_DIR`, git-ignored)::

    <root>/books.json                                   pinned index
    <root>/json/<categories>/<title>/Hebrew/<version>.json   mirror of the bucket paths
    <root>/schemas/<Title_with_underscores>.json
    <root>/sefaria_manifest.json                        revision, export date, one entry per text (licence, sha256)

Public API (``canon_diff`` imports these names):

* :data:`CORPORA` / :class:`CorpusSpec` -- the corpora and their version preferences.
* :func:`sefaria_dir`, :func:`load_manifest`, :func:`load_corpus`, :func:`load_version`, :func:`iter_segments`.
* :class:`Segment`, :class:`CanonText`.
* :func:`clean_segment`, :func:`strip_citation_parens`, :func:`daf_label`, :func:`daf_index`, :func:`he_numeral`,
  :func:`he_daf`, :func:`licence_class`, :func:`redistributable`.
* :func:`select_versions`, :func:`download` (network; the CLI's ``download``).

CLI::

    python -m src.finetuning.text27b.sefaria download [--corpus bavli --corpus mishneh_torah ...] [--commit SHA]
    python -m src.finetuning.text27b.sefaria show
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import logging
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from src.finetuning.text27b import paths as _paths

LOG = logging.getLogger(__name__)

#: Repository-relative default cache directory (``logs/`` is git-ignored).
DEFAULT_DIR = "logs/text_finetune_27b/sefaria"
#: The export repository (``books.json`` lives there).
GITHUB_REPO = "Sefaria/Sefaria-Export"
#: Public bucket base URL of the export.
BUCKET_BASE = "https://storage.googleapis.com/sefaria-export"
MANIFEST_FILE = "sefaria_manifest.json"
USER_AGENT = "text27b-canon-cache/1.0 (research; contact via repository owner)"


# ------------------------------------------------------------------------------------ corpora
@dataclass(frozen=True)
class CorpusSpec:
    """One canonical corpus: which ``books.json`` entries belong to it and which version to prefer.

    :ivar key: Corpus key (``bavli``, ``mishneh_torah`` ...).
    :ivar work: English work name used in row targets.
    :ivar he_work: Hebrew work name used in row targets.
    :ivar versions: Version titles in order of preference (first present wins; ``merged`` is never used because
        it mixes versions of different licences).
    :ivar match: Predicate over a ``books.json`` entry (entries of :attr:`language` only are offered).
    :ivar language: ``books.json`` language of the versions (``Hebrew``; ``English`` for translation corpora).
    """

    key: str
    work: str
    he_work: str
    versions: Tuple[str, ...]
    match: Callable[[Mapping[str, Any]], bool]
    language: str = "Hebrew"


def _cats(book: Mapping[str, Any]) -> List[str]:
    """Category list of a ``books.json`` entry.

    :param book: Entry.
    :returns: Categories (empty list when absent).
    """
    return list(book.get("categories") or [])


CORPORA: Dict[str, CorpusSpec] = {
    "bavli": CorpusSpec(
        "bavli", "Talmud Bavli", "תלמוד בבלי", ("Wikisource Talmud Bavli",),
        lambda b: _cats(b)[:2] == ["Talmud", "Bavli"] and len(_cats(b)) > 2 and _cats(b)[2].startswith("Seder ")),
    "mishneh_torah": CorpusSpec(
        "mishneh_torah", "Mishneh Torah", "משנה תורה",
        ("Wikisource Mishneh Torah", "Torat Emet 363", "Torat Emet 370", "Mechon Mamre"),
        lambda b: str(b.get("title", "")).startswith("Mishneh Torah, ") and _cats(b)[:2] == ["Halakhah", "Mishneh Torah"]),
    "mishnah": CorpusSpec(
        "mishnah", "Mishnah", "משנה", ("Torat Emet 357", "Mishnah, ed. Romm, Vilna 1913"),
        lambda b: _cats(b)[:1] == ["Mishnah"] and len(_cats(b)) > 1 and _cats(b)[1].startswith("Seder ")),
    "ben_sira": CorpusSpec(
        "ben_sira", "Ben Sira", "בן סירא",
        ("The Wisdom of Ben Sira, Cambridge University Press, 1899", "Ben Sira, David Kahana ed. -- Wikisource "),
        lambda b: b.get("title") == "Ben Sira"),
    "ben_sira_kahana": CorpusSpec(
        "ben_sira_kahana", "Ben Sira", "בן סירא", ("Ben Sira, David Kahana ed. -- Wikisource ",),
        lambda b: b.get("title") == "Ben Sira"),
    "ben_sira_en": CorpusSpec(
        "ben_sira_en", "Ben Sira", "בן סירא", ("The Wisdom of Ben Sira, Cambridge University Press, 1899",),
        lambda b: b.get("title") == "Ben Sira", language="English"),
    "halakhot_gedolot": CorpusSpec(
        "halakhot_gedolot", "Halakhot Gedolot", "הלכות גדולות", ("Halakhot Gedolot, Warsaw 1874",),
        lambda b: b.get("title") == "Halakhot Gedolot"),
}
#: Corpora downloaded by default (what the KTIV transcriptions and the public-domain editions need).
DEFAULT_CORPORA: Tuple[str, ...] = ("bavli", "mishneh_torah", "mishnah", "ben_sira", "ben_sira_kahana", "ben_sira_en",
                                     "halakhot_gedolot")


def sefaria_dir(p: Optional[_paths.Paths] = None) -> Path:
    """Cache root: ``paths.json`` entry ``sefaria_dir`` or the repository-relative :data:`DEFAULT_DIR`.

    :param p: Build paths (default :func:`paths.load_paths`).
    :returns: Absolute cache directory.
    """
    p = p or _paths.load_paths()
    raw = p.get("sefaria_dir")
    if raw:
        q = Path(os.path.expanduser(str(raw)))
        return q if q.is_absolute() else p.repo / q
    return p.repo / DEFAULT_DIR


# ------------------------------------------------------------------------------------ licences
def licence_class(licence: Optional[str]) -> str:
    """Map a Sefaria licence string to the build's licence classes (``rows.LICENCE_CLASSES``).

    Share-alike licences map to ``cc-by`` (redistribution allowed with attribution and the same licence; the exact
    string is kept in ``meta.licence``).

    :param licence: The version file's ``license`` value.
    :returns: ``pd`` / ``cc-by`` / ``cc-by-nc`` / ``copyright`` / ``unknown``.
    """
    s = (licence or "").strip().lower().replace("_", "-")
    if not s:
        return "unknown"
    if "public domain" in s or s in ("pd", "cc0") or "cc0" in s:
        return "pd"
    if "nc" in s.split("-") or "noncommercial" in s or "non-commercial" in s:
        return "cc-by-nc"
    if s.startswith("cc-by") or s.startswith("cc by"):
        return "cc-by"
    if "copyright" in s or "all rights" in s:
        return "copyright"
    return "unknown"


def redistributable(licence: Optional[str]) -> bool:
    """Whether text under this licence may go into a dataset that could be redistributed.

    :param licence: Licence string.
    :returns: True for public-domain and CC-BY(-SA) texts.
    """
    return licence_class(licence) in ("pd", "cc-by")


# ------------------------------------------------------------------------------------ numbers and refs
_HE_ONES = "אבגדהוזחט"
_HE_TENS = "יכלמנסעפצ"
_HE_HUNDREDS = "קרשת"


def he_numeral(n: int) -> str:
    """Hebrew numeral letters of a positive integer (15 → טו, 16 → טז), no geresh marks.

    :param n: Integer ≥ 1.
    :returns: Letters (``""`` for n < 1).
    """
    if n < 1:
        return ""
    out = []
    h = n // 100
    while h >= 4:
        out.append("ת")
        h -= 4
    if h:
        out.append(_HE_HUNDREDS[h - 1])
    rest = n % 100
    if rest == 15:
        out.append("טו")
    elif rest == 16:
        out.append("טז")
    else:
        if rest // 10:
            out.append(_HE_TENS[rest // 10 - 1])
        if rest % 10:
            out.append(_HE_ONES[rest % 10 - 1])
    return "".join(out)


def daf_label(index: int) -> str:
    """Talmud address of a 0-based section index of a Sefaria daf array (0 → ``1a``, 3 → ``2b``).

    :param index: Section index.
    :returns: ``<daf><a|b>``.
    """
    return f"{index // 2 + 1}{'ab'[index % 2]}"


def daf_index(label: str) -> int:
    """Inverse of :func:`daf_label` (``2a`` → 2).

    :param label: ``<daf><a|b>`` (case-insensitive; a missing side means ``a``).
    :returns: 0-based section index.
    :raises ValueError: On a malformed label.
    """
    m = re.fullmatch(r"\s*(\d+)\s*([abAB]?)\s*", label)
    if not m:
        raise ValueError(f"bad daf label {label!r}")
    return (int(m.group(1)) - 1) * 2 + (1 if m.group(2).lower() == "b" else 0)


def he_daf(label: str) -> str:
    """Hebrew form of a daf label (``9b`` → ``ט ע״ב``).

    :param label: ``<daf><a|b>``.
    :returns: Hebrew daf and side.
    """
    i = daf_index(label)
    return f"{he_numeral(i // 2 + 1)} ע״{'אב'[i % 2]}"


# ------------------------------------------------------------------------------------ cleaning
#: Hebrew names of the biblical books as the Wikisource Talmud / Mishneh Torah cite them in parentheses.
BIBLE_BOOKS_HE: Tuple[str, ...] = (
    "בראשית", "שמות", "ויקרא", "במדבר", "דברים", "יהושע", "שופטים", "שמואל", "מלכים", "ישעיהו", "ישעיה", "ירמיהו",
    "ירמיה", "יחזקאל", "הושע", "יואל", "עמוס", "עובדיה", "יונה", "מיכה", "נחום", "חבקוק", "צפניה", "חגי", "זכריה",
    "מלאכי", "תהלים", "תהילים", "משלי", "איוב", "שיר השירים", "רות", "איכה", "קהלת", "אסתר", "דניאל", "עזרא",
    "נחמיה", "דברי הימים", "שם")
_CITE_PAREN = re.compile(r"\s*\((?:%s)(?:[\s,]+[א-ת\"'׳״]{1,6}){0,4}\s*\)" %
                         "|".join(re.escape(b) for b in sorted(BIBLE_BOOKS_HE, key=len, reverse=True)))
_FOOTNOTE = re.compile(r"<sup[^>]*>[^<]*</sup>\s*<i[^>]*class=\"footnote\"[^>]*>.*?</i>", re.S)
_FOOTNOTE_I = re.compile(r"<i[^>]*class=\"footnote\"[^>]*>.*?</i>", re.S)
_COMMENTATOR = re.compile(r"<i[^>]*data-commentator[^>]*>\s*</i>")
_TAG = re.compile(r"<[^>]+>")


def strip_citation_parens(text: str) -> str:
    """Remove parenthesised biblical references the digital editions add (``(ישעיהו מח, יח)``, ``(שם)``).

    Other parentheses (the Vilna deletion marks, ``(והמתין לי)``) keep their words; only the brackets go.

    :param text: Segment text.
    :returns: Text without citation parentheses.
    """
    text = _CITE_PAREN.sub("", text)
    return text.replace("(", "").replace(")", "")


def clean_segment(raw: Any) -> str:
    """Clean one Sefaria segment: footnotes, tags and entities removed, citation parentheses stripped, NFC.

    :param raw: Segment value (non-strings give ``""``).
    :returns: Plain text on one line.
    """
    if not isinstance(raw, str):
        return ""
    t = _FOOTNOTE.sub("", raw)
    t = _FOOTNOTE_I.sub("", t)
    t = _COMMENTATOR.sub("", t)
    t = re.sub(r"<br\s*/?>", " ", t)
    t = _TAG.sub("", t)
    t = html.unescape(t).replace(" ", " ")
    t = strip_citation_parens(t)
    t = re.sub(r"[\[\]{}]", "", t)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", t).strip())


# ------------------------------------------------------------------------------------ reading
@dataclass(frozen=True)
class Segment:
    """One Sefaria segment.

    :ivar title: Sefaria title (``Berakhot``, ``Mishneh Torah, Reading the Shema``).
    :ivar he_title: Hebrew title.
    :ivar section: Section address (``2a`` for Talmud, ``3`` for a chapter; nested addresses joined with ``:``).
    :ivar seg: 1-based segment number within the section.
    :ivar text: Cleaned text.
    :ivar he_section: Hebrew section address (``ב ע״א``, ``ג``).
    """

    title: str
    he_title: str
    section: str
    seg: int
    text: str
    he_section: str = ""

    @property
    def ref(self) -> str:
        """English Sefaria-style reference (``Berakhot 2a:3``).

        :returns: Reference.
        """
        return f"{self.title} {self.section}:{self.seg}"


@dataclass
class CanonText:
    """One downloaded version of one title.

    :ivar title: Sefaria title.
    :ivar he_title: Hebrew title.
    :ivar corpus: Corpus key.
    :ivar version: Version title.
    :ivar licence: Licence string the version file declares.
    :ivar segments: Segments in reading order.
    :ivar path: Local file.
    """

    title: str
    he_title: str
    corpus: str
    version: str
    licence: str
    segments: List[Segment] = field(default_factory=list)
    path: Optional[Path] = None


def _is_talmud(section_names: Sequence[str]) -> bool:
    """Whether the first address level is a daf.

    :param section_names: ``sectionNames`` of the version file.
    :returns: True for Talmud addressing.
    """
    return bool(section_names) and str(section_names[0]).lower() == "daf"


def iter_segments(text: Any, title: str, he_title: str = "", section_names: Sequence[str] = (),
                  prefix: Tuple[str, ...] = (), he_prefix: Tuple[str, ...] = ()) -> Iterator[Segment]:
    """Flatten a Sefaria ``text`` value (jagged array or complex-schema dict) into segments.

    Depth-1 arrays are segments of one unnamed section ``1``; deeper arrays nest sections (``3:2`` addresses).
    Dict nodes (complex schemas) prefix their key to the title (the default node ``""`` adds nothing). Empty
    segments are skipped but keep their numbering.

    :param text: The ``text`` field (or a sub-node).
    :param title: Title used in references.
    :param he_title: Hebrew title.
    :param section_names: ``sectionNames`` (a first level named ``Daf`` gets Talmud addresses).
    :param prefix: Section address so far (recursion).
    :param he_prefix: Hebrew section address so far (recursion).
    :returns: Iterator of segments.
    """
    if isinstance(text, dict):
        for key, node in text.items():
            sub = f"{title}, {key}" if key else title
            yield from iter_segments(node, sub, he_title, section_names, prefix, he_prefix)
        return
    if not isinstance(text, list):
        return
    if all(not isinstance(x, list) for x in text):
        if not prefix:
            prefix, he_prefix = ("1",), ("א",)
        for j, raw in enumerate(text):
            clean = clean_segment(raw)
            if clean:
                yield Segment(title, he_title, ":".join(prefix), j + 1, clean, ":".join(he_prefix))
        return
    talmud = _is_talmud(section_names) and not prefix
    for i, sub in enumerate(text):
        if talmud:
            lab = daf_label(i)
            hlab = he_daf(lab)
        else:
            lab, hlab = str(i + 1), he_numeral(i + 1)
        yield from iter_segments(sub, title, he_title, section_names, prefix + (lab,), he_prefix + (hlab,))


def load_version(path: Union[str, Path], corpus: str = "") -> CanonText:
    """Read one downloaded version file.

    :param path: Version JSON.
    :param corpus: Corpus key recorded on the result.
    :returns: The text with its segments.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    title = str(data.get("title") or Path(path).parent.parent.name)
    he_title = str(data.get("heTitle") or "")
    names = data.get("sectionNames") or []
    if isinstance(names, str):  # some exports stringify list fields
        names = re.findall(r"'([^']+)'", names)
    segs = list(iter_segments(data.get("text"), title, he_title, names))
    return CanonText(title=title, he_title=he_title, corpus=corpus, version=str(data.get("versionTitle") or ""),
                     licence=str(data.get("license") or ""), segments=segs, path=Path(path))


def load_manifest(root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Read ``sefaria_manifest.json``.

    :param root: Cache root (default :func:`sefaria_dir`).
    :returns: The manifest.
    :raises FileNotFoundError: When the cache has not been downloaded.
    """
    root = Path(root) if root is not None else sefaria_dir()
    with open(root / MANIFEST_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(key: str, root: Optional[Union[str, Path]] = None,
                manifest: Optional[Mapping[str, Any]] = None) -> List[CanonText]:
    """Load every downloaded text of one corpus.

    :param key: Corpus key.
    :param root: Cache root.
    :param manifest: Pre-loaded manifest.
    :returns: Texts in manifest order.
    """
    root = Path(root) if root is not None else sefaria_dir()
    man = manifest if manifest is not None else load_manifest(root)
    out = []
    for ent in man.get("texts") or []:
        if ent.get("corpus") != key or not ent.get("path"):
            continue
        q = root / ent["path"]
        if q.exists():
            out.append(load_version(q, key))
    return out


# ------------------------------------------------------------------------------------ download
def _http_get(url: str, timeout: float = 90.0, retries: int = 3) -> bytes:
    """GET a URL (retries with back-off).

    :param url: URL.
    :param timeout: Seconds per attempt.
    :param retries: Attempts.
    :returns: Body bytes.
    :raises urllib.error.URLError: After the last failed attempt.
    """
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:  # a missing object does not come back on retry
                raise
            last = exc
            time.sleep(2.0 * (attempt + 1))
        except OSError as exc:  # URLError / timeouts are OSError subclasses
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write bytes atomically.

    :param dest: Destination.
    :param data: Bytes.
    :returns: None.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name[:40]}.", suffix=".tmp", dir=str(dest.parent))
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)


def bucket_rel_path(url: str) -> str:
    """Local relative path of a bucket URL (the bucket path, percent-decoded).

    :param url: ``https://storage.googleapis.com/sefaria-export/json/...``.
    :returns: ``json/...``.
    """
    path = urllib.parse.urlparse(url).path
    rel = urllib.parse.unquote(path).split("/sefaria-export/", 1)[-1]
    return rel.lstrip("/")


def schema_name(title: str) -> str:
    """File stem of a title's schema in the bucket (spaces become underscores: ``Bava_Kamma``).

    :param title: Sefaria title.
    :returns: Schema file stem.
    """
    return title.replace(" ", "_")


def select_versions(books: Iterable[Mapping[str, Any]], spec: CorpusSpec) -> List[Mapping[str, Any]]:
    """Pick one version per title of a corpus (in the spec's language) by the spec's preference order.

    :param books: ``books.json`` entries.
    :param spec: Corpus spec.
    :returns: Chosen entries (one per title, sorted by title); titles without a preferred version are skipped.
    """
    by_title: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for b in books:
        if b.get("language") != spec.language or not spec.match(b):
            continue
        by_title.setdefault(str(b.get("title")), {})[str(b.get("versionTitle"))] = b
    out = []
    for title in sorted(by_title):
        for v in spec.versions:
            if v in by_title[title]:
                out.append(by_title[title][v])
                break
        else:
            LOG.info("no preferred version for %s (%s)", title, sorted(by_title[title]))
    return out


def pinned_commit() -> str:
    """Current head commit of the export repository's default branch.

    :returns: 40-hex commit sha.
    """
    data = json.loads(_http_get(f"https://api.github.com/repos/{GITHUB_REPO}/commits?per_page=1"))
    return str(data[0]["sha"])


def _sha256(data: bytes) -> str:
    """Hex sha256.

    :param data: Bytes.
    :returns: Digest.
    """
    return hashlib.sha256(data).hexdigest()


def download(root: Path, corpora: Sequence[str] = DEFAULT_CORPORA, commit: Optional[str] = None,
             force: bool = False, limit: int = 0) -> Dict[str, Any]:
    """Download the chosen versions (and schemas) of the given corpora and write the manifest.

    :param root: Cache root.
    :param corpora: Corpus keys.
    :param commit: Export-repository commit to pin (default: the current head).
    :param force: Re-download files that exist.
    :param limit: Download at most this many texts per corpus (0 = all; for smoke runs).
    :returns: The manifest written.
    """
    root.mkdir(parents=True, exist_ok=True)
    commit = commit or pinned_commit()
    books_raw = _http_get(f"https://raw.githubusercontent.com/{GITHUB_REPO}/{commit}/books.json")
    _atomic_write(root / "books.json", books_raw)
    books = json.loads(books_raw)
    try:
        last_export = _http_get(f"{BUCKET_BASE}/last_export.txt").decode("utf-8").strip()
    except OSError:
        last_export = None
    old: Dict[str, Any] = {}
    if (root / MANIFEST_FILE).exists():
        old = {e["path"]: e for e in load_manifest(root).get("texts") or [] if e.get("path")}
    texts: List[Dict[str, Any]] = []
    for key in corpora:
        spec = CORPORA[key]
        chosen = select_versions(books.get("books") or [], spec)
        if limit:
            chosen = chosen[:limit]
        LOG.info("%s: %d titles", key, len(chosen))
        for b in chosen:
            rel = bucket_rel_path(str(b["json_url"]))
            dest = root / rel
            if force or not dest.exists():
                _atomic_write(dest, _http_get(str(b["json_url"])))
            data = dest.read_bytes()
            ver = json.loads(data)
            schema_rel = f"schemas/{schema_name(str(b['title']))}.json"
            if force or not (root / schema_rel).exists():
                try:
                    _atomic_write(root / schema_rel,
                                  _http_get(f"{BUCKET_BASE}/{urllib.parse.quote(schema_rel)}"))
                except OSError as exc:
                    LOG.warning("schema missing for %s: %s", b["title"], exc)
            lic = str(ver.get("license") or "")
            texts.append({"corpus": key, "title": b["title"], "he_title": ver.get("heTitle"),
                          "version": b["versionTitle"], "categories": b.get("categories"), "url": b["json_url"],
                          "path": rel, "bytes": len(data), "sha256": _sha256(data), "licence": lic,
                          "licence_class": licence_class(lic), "redistributable": redistributable(lic),
                          "version_source": ver.get("versionSource"),
                          "downloaded_at": old.get(rel, {}).get("downloaded_at") if not force and rel in old
                          else _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    manifest = {"source": f"github.com/{GITHUB_REPO} books.json + {BUCKET_BASE}", "repo_commit": commit,
                "books_json_generated_at": books.get("generated_at"), "books_json_sha256": _sha256(books_raw),
                "bucket_last_export": last_export, "corpora": list(corpora),
                "version_preferences": {k: list(CORPORA[k].versions) for k in corpora},
                "licence_counts": _licence_counts(texts), "texts": texts}
    _atomic_write(root / MANIFEST_FILE, json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    return manifest


def _licence_counts(texts: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Licence strings per corpus.

    :param texts: Manifest text entries.
    :returns: corpus → licence → titles.
    """
    out: Dict[str, Dict[str, int]] = {}
    for t in texts:
        d = out.setdefault(str(t.get("corpus")), {})
        d[str(t.get("licence"))] = d.get(str(t.get("licence")), 0) + 1
    return out


# ------------------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the chosen versions into the cache")
    d.add_argument("--corpus", action="append", choices=sorted(CORPORA), help="corpus key (repeatable)")
    d.add_argument("--commit", help="export-repository commit to pin (default: head)")
    d.add_argument("--root", help="cache root (default paths.json sefaria_dir)")
    d.add_argument("--force", action="store_true")
    d.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("show", help="print the manifest summary")
    s.add_argument("--root")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.root) if args.root else sefaria_dir()
    if args.cmd == "download":
        man = download(root, tuple(args.corpus or DEFAULT_CORPORA), args.commit, args.force, args.limit)
    else:
        man = load_manifest(root)
    summary = {k: man.get(k) for k in ("repo_commit", "books_json_generated_at", "bucket_last_export",
                                         "licence_counts")}
    summary["texts"] = len(man.get("texts") or [])
    summary["bytes"] = sum(int(t.get("bytes") or 0) for t in man.get("texts") or [])
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
