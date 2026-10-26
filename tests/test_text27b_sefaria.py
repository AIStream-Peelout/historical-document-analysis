# File name: test_text27b_sefaria.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``sefaria``: numerals and Talmud addresses, segment cleaning, flattening of Sefaria text
arrays, licence classes, version selection, the cache manifest and the download loop (network stubbed).

Synthetic version files in ``tmp_path`` only; no network.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.finetuning.text27b import sefaria as S


# ----------------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("n,expected", [(1, "א"), (9, "ט"), (10, "י"), (15, "טו"), (16, "טז"), (17, "יז"),
                                        (100, "ק"), (115, "קטו"), (400, "ת"), (613, "תריג"), (0, "")])
def test_he_numeral(n: int, expected: str) -> None:
    """Hebrew numerals use the ט״ו / ט״ז forms and stack ת for hundreds above 400."""
    assert S.he_numeral(n) == expected


def test_daf_labels_round_trip() -> None:
    """Section index ↔ daf label ↔ Hebrew daf."""
    assert [S.daf_label(i) for i in range(4)] == ["1a", "1b", "2a", "2b"]
    assert S.daf_index("2a") == 2 and S.daf_index("9b") == 17 and S.daf_index("12") == 22
    assert S.he_daf("9b") == "ט ע״ב"
    with pytest.raises(ValueError):
        S.daf_index("x9")


# ----------------------------------------------------------------------------------- cleaning
def test_clean_segment_removes_markup_footnotes_and_citations() -> None:
    """Tags, footnotes, entities and biblical-citation parentheses go; Vilna parentheses keep their words."""
    raw = ("מתני׳ <big><strong>מאימתי</strong></big> קורין<sup>1</sup><i class=\"footnote\">הערה</i> שנאמר "
           "(ישעיהו מח, יח) לוא הקשבת (והמתין לי) &amp; סוף<br>שורה")
    assert S.clean_segment(raw) == "מתני׳ מאימתי קורין שנאמר לוא הקשבת והמתין לי & סוף שורה"
    assert S.clean_segment(None) == ""
    assert S.strip_citation_parens("אמר (שם) כך (דברי הימים א כז) וזה") == "אמר כך וזה"


# ----------------------------------------------------------------------------------- flattening
def test_iter_segments_talmud_and_nested() -> None:
    """A Daf-first array gets ``2a`` addresses; deeper arrays nest; empty segments keep numbering."""
    segs = list(S.iter_segments([[], [], ["א ב", "", "ג"], ["ד"]], "Berakhot", "ברכות", ["Daf", "Line"]))
    assert [s.ref for s in segs] == ["Berakhot 2a:1", "Berakhot 2a:3", "Berakhot 2b:1"]
    assert segs[0].he_section == "ב ע״א"
    deep = list(S.iter_segments([[["x"], ["y", "z"]]], "T", "", ["Chapter", "Halakhah", "Seg"]))
    assert [s.ref for s in deep] == ["T 1:1:1", "T 1:2:1", "T 1:2:2"]
    flat = list(S.iter_segments(["a", "b"], "Flat"))
    assert [s.ref for s in flat] == ["Flat 1:1", "Flat 1:2"]


def test_iter_segments_complex_schema_dict() -> None:
    """Dict nodes prefix their key to the title; the default node adds nothing."""
    segs = list(S.iter_segments({"Introduction": ["פתיחה"], "": [["גוף"]]}, "Book", "ספר", ["Chapter", "Verse"]))
    assert [s.ref for s in segs] == ["Book, Introduction 1:1", "Book 1:1"]


def write_version(path: Path, text: Any, title: str = "Berakhot", licence: str = "CC-BY-SA",
                  names: Any = ("Daf", "Line"), version: str = "Test Edition") -> Path:
    """Write a synthetic Sefaria version file.

    :param path: Destination.
    :param text: ``text`` field.
    :param title: Title.
    :param licence: Licence string.
    :param names: Section names (a list, or a stringified list as some exports carry).
    :param version: Version title.
    :returns: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"title": title, "heTitle": "ברכות", "versionTitle": version, "license": licence,
                                "sectionNames": list(names) if not isinstance(names, str) else names,
                                "text": text}, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_version_and_stringified_section_names(tmp_path: Path) -> None:
    """``sectionNames`` may be a stringified list; Talmud addressing still applies."""
    q = write_version(tmp_path / "v.json", [[], [], ["שלום עולם"]], names="['Daf', 'Line']")
    t = S.load_version(q, "bavli")
    assert t.title == "Berakhot" and t.licence == "CC-BY-SA" and t.corpus == "bavli"
    assert [s.ref for s in t.segments] == ["Berakhot 2a:1"]


# ----------------------------------------------------------------------------------- licences
@pytest.mark.parametrize("lic,cls,redis", [("Public Domain", "pd", True), ("CC0", "pd", True),
                                           ("CC-BY-SA", "cc-by", True), ("CC-BY", "cc-by", True),
                                           ("CC-BY-NC", "cc-by-nc", False), ("CC-BY-NC-SA", "cc-by-nc", False),
                                           ("", "unknown", False), ("Copyright: X", "copyright", False)])
def test_licence_class(lic: str, cls: str, redis: bool) -> None:
    """Sefaria licence strings map to the build's licence classes; NC is not redistributable."""
    assert S.licence_class(lic) == cls
    assert S.redistributable(lic) is redis


# ----------------------------------------------------------------------------------- selection
BOOKS: List[Dict[str, Any]] = [
    {"title": "Berakhot", "language": "Hebrew", "versionTitle": "merged", "categories": ["Talmud", "Bavli", "Seder Zeraim"],
     "json_url": "https://storage.googleapis.com/sefaria-export/json/Talmud/Bavli/Seder%20Zeraim/Berakhot/Hebrew/merged.json"},
    {"title": "Berakhot", "language": "Hebrew", "versionTitle": "Wikisource Talmud Bavli",
     "categories": ["Talmud", "Bavli", "Seder Zeraim"],
     "json_url": "https://storage.googleapis.com/sefaria-export/json/Talmud/Bavli/Seder%20Zeraim/Berakhot/Hebrew/W.json"},
    {"title": "Berakhot", "language": "English", "versionTitle": "Wikisource Talmud Bavli",
     "categories": ["Talmud", "Bavli", "Seder Zeraim"], "json_url": "https://x/e.json"},
    {"title": "Rashi on Berakhot", "language": "Hebrew", "versionTitle": "Vilna Edition",
     "categories": ["Talmud", "Bavli", "Rishonim on Talmud"], "json_url": "https://x/r.json"},
    {"title": "Shabbat", "language": "Hebrew", "versionTitle": "merged", "categories": ["Talmud", "Bavli", "Seder Moed"],
     "json_url": "https://x/s.json"},
]


def test_select_versions_preference_language_and_category() -> None:
    """One preferred Hebrew version per tractate; commentaries and titles without a preferred version are skipped."""
    chosen = S.select_versions(BOOKS, S.CORPORA["bavli"])
    assert [(b["title"], b["versionTitle"]) for b in chosen] == [("Berakhot", "Wikisource Talmud Bavli")]
    en = S.CorpusSpec("x", "X", "", ("Wikisource Talmud Bavli",), lambda b: b["title"] == "Berakhot", "English")
    assert [b["json_url"] for b in S.select_versions(BOOKS, en)] == ["https://x/e.json"]


def test_bucket_paths_and_schema_names() -> None:
    """Bucket URLs map to percent-decoded relative paths; schema stems use underscores."""
    assert S.bucket_rel_path(BOOKS[1]["json_url"]) == "json/Talmud/Bavli/Seder Zeraim/Berakhot/Hebrew/W.json"
    assert S.schema_name("Mishneh Torah, Reading the Shema") == "Mishneh_Torah,_Reading_the_Shema"


def test_sefaria_dir_default_and_override(tmp_path: Path) -> None:
    """``sefaria_dir`` comes from paths.json when set, else the repository-relative default."""
    class P:
        repo = tmp_path

        def __init__(self, extra: Dict[str, Any]) -> None:
            self.extra = extra

        def get(self, key: str, default: Any = None) -> Any:
            return self.extra.get(key, default)

    assert S.sefaria_dir(P({})) == tmp_path / S.DEFAULT_DIR
    assert S.sefaria_dir(P({"sefaria_dir": "cache/s"})) == tmp_path / "cache/s"
    assert S.sefaria_dir(P({"sefaria_dir": str(tmp_path / "abs")})) == tmp_path / "abs"


# ----------------------------------------------------------------------------------- download + manifest
def test_download_writes_mirror_manifest_and_skips_missing_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The download loop mirrors bucket paths, records licence + sha256 per text and survives a missing schema."""
    version = json.dumps({"title": "Berakhot", "heTitle": "ברכות", "versionTitle": "Wikisource Talmud Bavli",
                          "license": "CC-BY-SA", "versionSource": "src", "sectionNames": ["Daf", "Line"],
                          "text": [[], [], ["מתני׳ מאימתי קורין"]]}, ensure_ascii=False).encode("utf-8")
    calls: List[str] = []

    def fake_get(url: str, timeout: float = 90.0, retries: int = 3) -> bytes:
        calls.append(url)
        if url.endswith("books.json"):
            return json.dumps({"generated_at": "2026-10-02", "books": BOOKS}).encode("utf-8")
        if url.endswith("last_export.txt"):
            return b"2026-10-01T00:00:00"
        if "/schemas/" in url:
            raise S.urllib.error.HTTPError(url, 404, "nf", None, None)  # type: ignore[arg-type]
        return version

    monkeypatch.setattr(S, "_http_get", fake_get)
    man = S.download(tmp_path, corpora=("bavli",), commit="a" * 40)
    assert man["repo_commit"] == "a" * 40 and man["bucket_last_export"] == "2026-10-01T00:00:00"
    (ent,) = man["texts"]
    assert ent["licence"] == "CC-BY-SA" and ent["licence_class"] == "cc-by" and ent["redistributable"] is True
    assert (tmp_path / ent["path"]).exists() and len(ent["sha256"]) == 64
    assert any("raw.githubusercontent.com" in c and ("a" * 40) in c for c in calls)
    texts = S.load_corpus("bavli", tmp_path)
    assert [s.ref for s in texts[0].segments] == ["Berakhot 2a:1"]
    assert S.load_corpus("mishnah", tmp_path) == []
    # a second run reuses the file on disk (no second fetch of the version)
    n_before = sum(1 for c in calls if c.endswith("W.json"))
    S.download(tmp_path, corpora=("bavli",), commit="a" * 40)
    assert sum(1 for c in calls if c.endswith("W.json")) == n_before


def test_http_get_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 is raised at once (no back-off sleeps)."""
    n = {"calls": 0}

    def boom(req: Any, timeout: float = 0) -> Any:
        n["calls"] += 1
        raise S.urllib.error.HTTPError("u", 404, "nf", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(S.urllib.request, "urlopen", boom)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    with pytest.raises(S.urllib.error.HTTPError):
        S._http_get("https://example.invalid/x", retries=3)
    assert n["calls"] == 1
