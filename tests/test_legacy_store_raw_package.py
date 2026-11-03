# File name: test_legacy_store_raw_package.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Unit tests for the legacy-store raw packager (``raw_package``).

Roots are built in ``tmp_path``; PDFs are generated with PyMuPDF from synthetic text (common
Hebrew words composed for the test, never transcription text); the bucket is a fake opener that
serves an XML listing and object bodies from memory. Covers name classification (unrelated files
never opened), shelf-mark keys across spellings, listing pages, resumable fetch with MD5 checks,
image SHA-256 dedupe across roots, label-race dating, viewer-sequence token detection, JSON
reading with URL removal and copy dedupe, KTIV lookups and image policy, and an end-to-end
package of two roots. The second verification pass: unsafe bucket keys, ``.part`` files and
hidden listing objects counted, credit wording reduced to its kind, image paths as
``{root, rel}``, side re-downloads folded, edition-reference side files, near-duplicate texts.
"""
import hashlib
import io
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import fitz
import pytest

from src.datasets.indexing.legacy_store import raw_package as R


# ───────────────────────────── fake bucket ─────────────────────────────
class FakeResponse(io.BytesIO):
    """A ``urlopen`` response: a byte stream usable as a context manager."""

    def __enter__(self) -> "FakeResponse":
        """Enter the context.

        :return: Self.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close on exit.

        :param exc: Exception info (ignored).
        """
        self.close()


def listing_xml(objects: List[Dict[str, Any]], next_marker: Optional[str]) -> bytes:
    """One XML listing page.

    :param objects: Objects (``key``, ``body``, optional ``etag``).
    :param next_marker: Marker of the next page, None for the last page.
    :return: XML bytes.
    """
    parts = ["<?xml version='1.0' encoding='UTF-8'?><ListBucketResult xmlns='http://doc.s3.amazonaws.com/2006-03-01'>"]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts.append(f"<IsTruncated>{'true' if next_marker else 'false'}</IsTruncated>")
    for o in objects:
        etag = o.get("etag") or hashlib.md5(o["body"]).hexdigest()
        parts.append(f"<Contents><Key>{o['key']}</Key><LastModified>2025-10-23T23:19:52.164Z</LastModified>"
                     f"<ETag>\"{etag}\"</ETag><Size>{len(o['body'])}</Size></Contents>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


class FakeBucket:
    """Serves a two-page listing and object bodies; records every requested URL."""

    def __init__(self, objects: List[Dict[str, Any]], page_size: int = 2) -> None:
        """Build the fake.

        :param objects: Objects (``key``, ``body``, optional ``etag``).
        :param page_size: Objects per listing page.
        """
        self.objects = objects
        self.page_size = page_size
        self.urls: List[str] = []

    def __call__(self, url: str, timeout: float = 0) -> FakeResponse:
        """Answer one GET.

        :param url: Requested URL.
        :param timeout: Ignored.
        :return: The response.
        """
        self.urls.append(url)
        if "?" in url:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(url).query)
            marker = (q.get("marker") or [""])[0]
            keys = [o["key"] for o in self.objects]
            start = keys.index(marker) + 1 if marker else 0
            page = self.objects[start:start + self.page_size]
            nxt = page[-1]["key"] if start + self.page_size < len(self.objects) else None
            return FakeResponse(listing_xml(page, nxt))
        from urllib.parse import unquote
        key = unquote(url.split("/", 4)[4])
        return FakeResponse(next(o["body"] for o in self.objects if o["key"] == key))


PREFIX = "snap/run_1/"


def bucket_objects() -> List[Dict[str, Any]]:
    """Objects of the fake snapshot: two images, a catalogue JSON, a personal file, a junk JSON.

    :return: Objects.
    """
    return [
        {"key": PREFIX + "a_series/A_10_1r.jpg", "body": b"recto"},
        {"key": PREFIX + "a_series/A_10_1v.jpg", "body": b"verso"},
        {"key": PREFIX + "a_series/cataloging_Manchester_A_10.json", "body": b"{}"},
        {"key": PREFIX + "a_series/Lease_Agreement.docx.pdf", "body": b"personal"},
        {"key": PREFIX + "a_series/joins_Please_select_shelfmarks(3).json", "body": b"{}"},
    ]


def test_list_bucket_pages_and_parses() -> None:
    """The listing follows ``marker`` paging until the page is not truncated."""
    fake = FakeBucket(bucket_objects())
    sleeps: List[float] = []
    objs = R.list_bucket("b", PREFIX, opener=fake, sleep=sleeps.append)
    assert [o["key"] for o in objs] == [o["key"] for o in bucket_objects()]
    assert objs[0]["size"] == 5 and objs[0]["etag"] == hashlib.md5(b"recto").hexdigest()
    assert len(fake.urls) == 3 and len(sleeps) == 2
    assert all("prefix=snap%2Frun_1%2F" in u for u in fake.urls)


def test_fetch_snapshot_skips_unrelated_and_resumes(tmp_path: pathlib.Path) -> None:
    """Only legacy-store shapes are fetched; a second run fetches nothing; a bad MD5 is not kept.

    :param tmp_path: Temporary folder.
    """
    objs = bucket_objects()
    fake = FakeBucket(objs)
    listing = R.list_bucket("b", PREFIX, opener=fake, sleep=lambda s: None)
    dest, state = str(tmp_path / "snap"), str(tmp_path / "snap.state.jsonl")
    stats = R.fetch_snapshot(listing, "b", PREFIX, dest, state, opener=fake, sleep=lambda s: None)
    assert stats["fetched"] == 4 and stats["not_fetched_kind"] == 1 and stats["md5_mismatch"] == []
    assert not os.path.exists(os.path.join(dest, "a_series", "Lease_Agreement.docx.pdf"))
    assert not any("Lease" in u for u in fake.urls if "?" not in u)
    assert open(os.path.join(dest, "a_series", "A_10_1v.jpg"), "rb").read() == b"verso"
    again = R.fetch_snapshot(listing, "b", PREFIX, dest, state, opener=fake, sleep=lambda s: None)
    assert again["fetched"] == 0 and again["skipped_present"] == 4
    bad = [dict(objs[0], key=PREFIX + "a_series/A_11_1r.jpg", etag="0" * 32)]
    fake_bad = FakeBucket(bad)
    listing_bad = R.list_bucket("b", PREFIX, opener=fake_bad, sleep=lambda s: None)
    out = R.fetch_snapshot(listing_bad, "b", PREFIX, dest, state, opener=fake_bad, sleep=lambda s: None)
    assert out["md5_mismatch"] == [PREFIX + "a_series/A_11_1r.jpg"]
    assert not os.path.exists(os.path.join(dest, "a_series", "A_11_1r.jpg"))


def test_verify_snapshot_and_forced_refetch(tmp_path: pathlib.Path) -> None:
    """A file damaged after its fetch fails verification and a forced fetch repairs it.

    :param tmp_path: Temporary folder.
    """
    fake = FakeBucket(bucket_objects())
    listing = R.list_bucket("b", PREFIX, opener=fake, sleep=lambda s: None)
    dest, state = str(tmp_path / "snap"), str(tmp_path / "snap.state.jsonl")
    R.fetch_snapshot(listing, "b", PREFIX, dest, state, opener=fake, sleep=lambda s: None)
    target = os.path.join(dest, "a_series", "A_10_1r.jpg")
    with open(target, "wb") as fh:
        fh.write(b"RECTO")                         # same size, other bytes
    check = R.verify_snapshot(listing, PREFIX, dest, state)
    assert check["bad"] == [PREFIX + "a_series/A_10_1r.jpg"] and len(check["ok"]) == 3 and not check["missing"]
    redo = R.fetch_snapshot(listing, "b", PREFIX, dest, state, opener=fake, sleep=lambda s: None, force=check["bad"])
    assert redo["fetched"] == 1 and open(target, "rb").read() == b"recto"
    assert not R.verify_snapshot(listing, PREFIX, dest, state)["bad"]


# ───────────────────────────── names and keys ─────────────────────────────
@pytest.mark.parametrize("name,kind", [
    ("transcription_manual_Cambridge_CUL_T_S_AS_97_1_2.pdf", "transcription_pdf"),
    ("transcription_manual_Cambridge_CUL_T_S_AS_98_1-1.pdf", "transcription_pdf"),
    ("transcription_response_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf", "transcription_pdf"),
    ("transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf", "transcription_pdf"),
    ("test_transcription.pdf", "unrelated"),
    ("a_12.pdf", "other_shelfmark_pdf"),
    ("T_S_AS_95_124_1v.jpg", "image"),
    ("B_7961_1r(1).jpg", "image"),
    ("Yevr_Arab_II_1202_32r.jpg", "image"),
    ("MS-TS-AS-00189-00002-000-00002.jpg", "unrelated"),
    ("Image(1).jpg", "unrelated"),
    ("cataloging_Cambridge_CUL_T_S_AS_26_2.json", "cataloging_json"),
    ("joins_Please_select_shelfmarks(753).json", "junk_json"),
    ("joins_There_are_no_images_in_the_search_results(41).json", "junk_json"),
    ("bibliography_title_X_shelfmark_T_S_AS_20_151_1(3).json", "bibliography_json"),
    ("metadata_Cambridge__CUL__T_S_AS_22_12.json", "metadata_json"),
    ("ktiv_PNX_MANUSCRIPTS990001.json", "unrelated"),
    ("Backup-codes.txt", "unrelated"),
    ("payslip_full_balances.pdf", "unrelated"),
])
def test_classify_name(name: str, kind: str) -> None:
    """Name shapes map to kinds; everything else is unrelated.

    :param name: File name.
    :param kind: Expected kind.
    """
    assert R.classify_name(name) == kind


def test_parse_image_name() -> None:
    """Stem, token, side and copy number."""
    assert R.parse_image_name("T_S_F6_6_3v(1).jpg") == {"stem": "T_S_F6_6", "token": "3v", "leaf": 3, "side": "v",
                                                        "copy": 1}
    assert R.parse_image_name("notes.txt") is None


@pytest.mark.parametrize("a,b", [
    ("Cambridge, CUL: T-S AS 97.1", "Cambridge_CUL_T_S_AS_97_1"),
    ("T-S AS 97.1", "T_S_AS_97_1"),
    ("Cambridge, CUL: T-S F1(1).33", "T_S_F1_1_33"),
    ("Manchester: B 7961", "B_7961"),
    ("JRL B 7961", "Manchester_B_7961"),
    ("St. Petersburg, St. Peterburg: Yevr.-Arab. II 1201", "Yevr_Arab_II_1201"),
    ("Cambridge, CUL: T-S AS 98.1", "Cambridge__CUL__T_S_AS_98_1"),
])
def test_package_key_agrees_across_spellings(a: str, b: str) -> None:
    """Catalogue strings, file-name stems and image stems share a key.

    :param a: One spelling.
    :param b: Another spelling.
    """
    assert R.package_key(a) == R.package_key(b) != ""


def test_package_key_keeps_digit_boundaries() -> None:
    """``AS 14.46`` and ``AS 144.6`` stay apart."""
    assert R.package_key("T-S AS 14.46") != R.package_key("T-S AS 144.6")


# ───────────────────────────── roots and images ─────────────────────────────
FIX = R.LABEL_RACE_FIX_UTC.timestamp()


def test_label_race_status() -> None:
    """JSON stamps decide first; bucket upload times only prove a pre-fix save."""
    day = 86400.0
    assert R.label_race_status([FIX - day], [FIX + day], False) == "pre_fix"
    assert R.label_race_status([FIX + day, FIX + 2 * day], [], False) == "post_fix"
    assert R.label_race_status([FIX - day, FIX + day], [], False) == "spans_fix"
    assert R.label_race_status([], [FIX - day], True) == "pre_fix"
    assert R.label_race_status([], [FIX + day], True) == "unknown"
    assert R.label_race_status([], [FIX + day], False) == "post_fix"
    assert R.label_race_status([], [], False) == "unknown"


def test_image_token_kind() -> None:
    """Twenty or more tokens and none ending in ``v``: viewer sequence numbers."""
    assert R.image_token_kind([f"{i}r" for i in range(4, 30)]) == "viewer_sequence_suspected"
    assert R.image_token_kind([f"{i}r" for i in range(4, 10)]) == "side_token"
    assert R.image_token_kind(["1r", "1v"] * 15) == "side_token"


def make_root(path: pathlib.Path, files: Dict[str, bytes]) -> pathlib.Path:
    """Write files into a fresh root folder.

    :param path: Folder.
    :param files: Relative name -> bytes.
    :return: The folder.
    """
    for rel, data in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return path


def test_walk_root_skips_hidden_and_app_and_never_opens_unrelated(tmp_path: pathlib.Path) -> None:
    """Unreadable unrelated files are only stat-ed; hidden and ``.app`` folders are skipped.

    :param tmp_path: Temporary folder.
    """
    root = make_root(tmp_path / "r", {"B_7777_1r.jpg": b"x", "secret_notes.txt": b"s", ".hidden/B_1_1r.jpg": b"h",
                                      "Tool.app/B_2_1r.jpg": b"a", "sub/B_3_1r.jpg": b"y"})
    os.chmod(root / "secret_notes.txt", 0)
    flat = R.walk_root(R.RootSpec("r", str(root)))
    assert sorted((e.name, e.kind) for e in flat) == [("B_7777_1r.jpg", "image"), ("secret_notes.txt", "unrelated")]
    deep = R.walk_root(R.RootSpec("r", str(root), recursive=True))
    assert sorted(e.rel for e in deep) == ["B_7777_1r.jpg", "secret_notes.txt", "sub/B_3_1r.jpg"]
    assert next(e for e in deep if e.name == "B_3_1r.jpg").run_folder == "r/sub"
    os.chmod(root / "secret_notes.txt", 0o600)


def test_walk_snapshot_root_uses_listing_times_and_counts_unfetched(tmp_path: pathlib.Path) -> None:
    """Fetched files take the upload time; objects left in the bucket appear unfetched.

    :param tmp_path: Temporary folder.
    """
    root = make_root(tmp_path / "snap", {"a_series/A_10_1r.jpg": b"recto"})
    listing = [{"key": PREFIX + "a_series/A_10_1r.jpg", "size": 5, "etag": "e", "modified": "2025-10-23T23:19:52.164Z"},
               {"key": PREFIX + "a_series/Lease.docx.pdf", "size": 9, "etag": "f", "modified": "2025-10-23T23:19:52Z"}]
    entries = R.walk_root(R.RootSpec("snap", str(root), True, listing, PREFIX))
    img = next(e for e in entries if e.name == "A_10_1r.jpg")
    assert img.time_source == "bucket_upload" and img.mtime < FIX and img.run_folder == "snap/a_series"
    lease = next(e for e in entries if e.name == "Lease.docx.pdf")
    assert lease.fetched is False and lease.kind == "unrelated" and lease.path == ""


def test_dedupe_images_across_roots(tmp_path: pathlib.Path) -> None:
    """One photo under two names in two roots: one group, canonical in the first root.

    :param tmp_path: Temporary folder.
    """
    a = make_root(tmp_path / "a", {"B_7777_1r.jpg": b"photo", "B_7777_1r(1).jpg": b"photo"})
    b = make_root(tmp_path / "b", {"B_7778_1r.jpg": b"photo", "B_7778_1v.jpg": b"other"})
    entries = R.walk_root(R.RootSpec("a", str(a))) + R.walk_root(R.RootSpec("b", str(b)))
    groups = R.dedupe_images([e for e in entries if e.kind == "image"], ["a", "b"])
    assert len(groups) == 2
    g = next(x for x in groups if len(x["paths"]) == 3)
    assert g["canonical"] == {"root": "a", "rel": "B_7777_1r.jpg"} and g["stems"] == ["B_7777", "B_7778"]
    assert g["sha256"] == hashlib.sha256(b"photo").hexdigest()
    assert all("path" not in p and str(tmp_path) not in json.dumps(p) for p in g["paths"])


# ───────────────────────────── JSON ─────────────────────────────
def write_json(path: pathlib.Path, data: Any) -> str:
    """Write JSON.

    :param path: Target.
    :param data: Value.
    :return: The path as a string.
    """
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_read_metadata_json_kinds(tmp_path: pathlib.Path) -> None:
    """Shelf marks and payloads of the four kinds; URLs dropped; placeholders junk; bad JSON unreadable.

    :param tmp_path: Temporary folder.
    """
    cat = R.read_metadata_json(write_json(tmp_path / "c.json", {
        "manuscript": {"shelfmark": "Cambridge, CUL: T-S AS 26.2", "catalogingEntries": [
            {"section": "Content Description", "field": "Title", "value": "Letter see https://example.org/x"}],
            "sources": [{"name": "Some Catalog"}]},
        "metadata": {"pageUrl": "https://example.org/p", "extractedAt": "2025-11-24T02:52:29.999Z"}}), "cataloging_json")
    assert cat["marks"] == ["Cambridge, CUL: T-S AS 26.2"] and not cat["junk"] and not cat["multi"]
    assert cat["payload"]["entries"][0]["value"] == "Letter see" and "example.org" not in json.dumps(cat["payload"])
    assert cat["extracted_at"] == R.iso_to_epoch("2025-11-24T02:52:29.999Z")
    joins = R.read_metadata_json(write_json(tmp_path / "j.json", {
        "joinedManuscripts": [], "mainShelfmark": "Please select shelfmarks", "source": "",
        "metadata": {"extractedAt": "2025-11-23T05:59:00.611Z"}}), "joins_json")
    assert joins["junk"]
    bib = R.read_metadata_json(write_json(tmp_path / "b.json", {"entries": [
        {"key": "shelfmark", "value": "T-S AS 20.151", "links": ["https://example.org"]},
        {"key": "year", "value": "1999"}]}), "bibliography_json")
    assert bib["marks"] == ["T-S AS 20.151"] and bib["payload"] == {"year": "1999"}
    meta = R.read_metadata_json(write_json(tmp_path / "m.json", {
        "shelfmark": "Manchester: B 7777", "source": "Example Transcriptions Team", "inventoryId": "1",
        "pageUrl": "https://example.org"}), "metadata_json")
    assert meta["payload"] == {"credit_kind": "project_team", "inventory_id": "1"}
    (tmp_path / "bad.json").write_text("{\"manuscript\": ", encoding="utf-8")
    assert R.read_metadata_json(str(tmp_path / "bad.json"), "cataloging_json")["unreadable"]


def test_dedupe_json_collapses_redownloads(tmp_path: pathlib.Path) -> None:
    """Copies that differ only in their extraction time are one record.

    :param tmp_path: Temporary folder.
    """
    recs = []
    for i, stamp in enumerate(["2025-11-24T02:52:29Z", "2025-11-25T02:52:29Z"]):
        path = write_json(tmp_path / f"c{i}.json", {"manuscript": {"shelfmark": "Manchester: B 7777"},
                                                    "metadata": {"extractedAt": stamp}})
        recs.append(dict(R.read_metadata_json(path, "cataloging_json"), path=path, root="r"))
    out = R.dedupe_json(recs)
    assert len(out) == 1 and len(out[0]["copies"]) == 2


# ───────────────────────────── corpus join and policy ─────────────────────────────
def merged_line(canonical: str, display: str, legacy_mark: Optional[str], ktiv: bool, held: bool,
                legacy_images: Optional[List[str]] = None) -> Dict[str, Any]:
    """A synthetic ``merged_shelfmarks.jsonl`` record.

    :param canonical: Canonical id.
    :param display: Display shelf mark.
    :param legacy_mark: Legacy-store shelf mark string, or None.
    :param ktiv: Whether the record has a KTIV source (with a IIIF manifest).
    :param held: Whether KTIV images are held.
    :param legacy_images: Legacy-store image file names.
    :return: The record.
    """
    return {"canonical_id": canonical, "shelfmark_display": display, "institution": "x", "sources_present": [],
            "sources": {"pgp": None, R.LEGACY_SOURCE_KEY: [{"shelf_mark": legacy_mark}] if legacy_mark else [],
                        "ktiv": {"shelf_mark": display} if ktiv else None, "ktiv_transcription": None, "bodleian": None},
            "images": {R.LEGACY_SOURCE_KEY: legacy_images or [],
                       "ktiv": {"iiif_manifest_url": "m", "populated": held, "image_urls": ["u"] if held else [],
                                "image_count": 3 if held else 0} if ktiv else None,
                       "bodleian": None, "preferred_source": None}}


def write_merged(path: pathlib.Path) -> str:
    """A two-record merged corpus: B 7777 on KTIV (manifest only), A 10 legacy-only.

    :param path: Target file.
    :return: The path.
    """
    rows = [merged_line("Manchester_JRL_B_7777", "JRL B 7777", "Manchester: B 7777", True, False),
            merged_line("Manchester_JRL_A_10", "JRL A 10", "Manchester: A 10", False, False, ["A_10_1r.jpg"])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_build_merged_index_and_ktiv_status(tmp_path: pathlib.Path) -> None:
    """Keys from every spelling; KTIV flags; legacy image names; policy per rule.

    :param tmp_path: Temporary folder.
    """
    index, names = R.build_merged_index(write_merged(tmp_path / "merged.jsonl"))
    k = R.package_key("B_7777")
    st = R.ktiv_status(k, ["Manchester"], index)
    assert st["matched"] and st["on_ktiv"] and st["ktiv_iiif_manifest"] and not st["ktiv_images_held"]
    assert R.image_policy(st, 0) == R.POLICY_PREFER_KTIV
    a10 = R.ktiv_status(R.package_key("Manchester: A 10"), [], index)
    assert a10["matched"] and not a10["on_ktiv"] and a10["corpus_legacy_images"] == 1
    assert R.image_policy(a10, 2) == R.POLICY_LEGACY_ONLY and R.image_policy(a10, 0) == R.POLICY_NO_IMAGE
    assert names == {"A_10_1r.jpg": ["Manchester_JRL_A_10"]}
    assert not R.ktiv_status("nothing", [], index)["matched"]
    assert not index[R.package_key("A 10")][0]["box_level"]


def test_box_level_corpus_record_is_not_item_evidence(tmp_path: pathlib.Path) -> None:
    """A box record holding its items' images is box-level, so ``T_S_AS_97_4.pdf`` stays item 97.4.

    :param tmp_path: Temporary folder.
    """
    rows = [merged_line("Cambridge_CUL_T_S_AS_97", "Cambridge CUL: T-S AS 97", "Cambridge CUL: T-S AS 97", False,
                        False, ["T_S_AS_97_4_1r.jpg", "T_S_AS_97_5_1r.jpg"])]
    path = tmp_path / "merged.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    index, _ = R.build_merged_index(str(path))
    assert index[R.package_key("T-S AS 97")][0]["box_level"]
    root = make_root(tmp_path / "r", {"transcription_manual_Cambridge_CUL_T_S_AS_97_4.pdf": b"%PDF-1.4",
                                      "T_S_AS_97_4_1r.jpg": b"img"})
    entries = R.walk_root(R.RootSpec("r", str(root)))
    known = {k for k, hs in index.items() if any(not h["box_level"] for h in hs)}
    known |= {R.package_key(R.parse_image_name(e.name)["stem"]) for e in entries if e.kind == "image"}
    parsed = R.P.parse_side_file_name("transcription_manual_Cambridge_CUL_T_S_AS_97_4.pdf", [],
                                      R.side_known_checker(known))
    assert (parsed["shelfmark"], parsed["side_index"], parsed["side_resolution"]) == ("T S AS 97 4", 1, "corpus")
    # a catalogued box is no item either: a known key extended by a number makes it a container
    checker = R.side_known_checker({R.package_key("T-S AS 97"), R.package_key("T-S AS 97.4")})
    assert checker("T_S_AS_97_4") and not checker("T_S_AS_97")


def test_assign_side_prefers_plain_names() -> None:
    """Plain names beat copy names; the lowest side wins, higher sides of one text are re-download names."""
    def parsed(file: str, side: int, copy: Optional[str] = None) -> Dict[str, Any]:
        """A parse stub.

        :param file: File name.
        :param side: Side index.
        :param copy: Copy suffix.
        :return: The stub.
        """
        return {"file": file, "stem": "Manchester_B_7777", "side_index": side, "copy_suffix": copy,
                "side_resolution": "sibling_file", "side_ambiguous": False, "shelfmark": "B 7777"}
    out = R.assign_side([parsed("x_4.pdf", 4), parsed("x.pdf", 1), parsed("x-1.pdf", 2, "-1")])
    assert out["side_index"] == 1 and out["flags"] == ["side_redownload_copies"]
    assert out["redownload_name_sides"] == [4]
    other = dict(parsed("y.pdf", 1), stem="Manchester_B_7778", shelfmark="B 7778")
    both = R.assign_side([parsed("x.pdf", 1), other])
    assert both["shelfmark"] == "B 7777" and both["flags"] == ["shelfmark_name_conflict"]
    only_copy = R.assign_side([parsed("x-2.pdf", 2, "-2")])
    assert only_copy["side_index"] == 2 and only_copy["flags"] == ["side_from_copy_name_unverified"]


# ───────────────────────────── end to end ─────────────────────────────
HEBREW_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
needs_font = pytest.mark.skipif(not os.path.exists(HEBREW_FONT), reason="system TrueType font with Hebrew glyphs not found")


def pdf_bytes(lines: List[str], junk: bool = False) -> bytes:
    """A synthetic PDF: Hebrew lines drawn right to left, or the store's error page.

    :param lines: Hebrew lines (common words composed for the test); a line ``"@<label>"`` draws a
        Latin folio heading instead.
    :param junk: Write the error page instead.
    :return: PDF bytes.
    """
    doc = fitz.open()
    page = doc.new_page()
    if junk:
        page.insert_text((72, 72), "A minor technical problem was encountered while displaying the requested image",
                         fontname="helv", fontsize=8)
        return doc.tobytes()
    font = fitz.Font(fontfile=HEBREW_FONT)
    for i, text in enumerate(lines):
        tw = fitz.TextWriter(page.rect)
        y = 72 + 16 * i
        if text.startswith("@"):
            tw.append((520, y), text[1:], font=font, fontsize=12)
        else:
            tw.append((540 - font.text_length(text, fontsize=12), y), text, font=font, fontsize=12, right_to_left=True)
        tw.write_text(page)
    return doc.tobytes()


POST = "2025-11-20T10:00:00.000Z"


@needs_font
def test_package_roots_end_to_end(tmp_path: pathlib.Path) -> None:
    """Two roots: old-style sides, a header-style text, images in both roots, JSON, junk, a secret file.

    :param tmp_path: Temporary folder.
    """
    side1 = pdf_bytes(["טור_א", "טוב היה כל", "אשר גדול ספר", "טור_ב", "איש בית בא", "על כל יצא"])
    old = make_root(tmp_path / "old", {
        "transcription_manual_Manchester_B_7777.pdf": side1,
        "transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf": side1,
        "transcription_manual_Manchester_B_7777_2.pdf": pdf_bytes(["היה טוב אשר", "כל על בא"]),
        "transcription_manual_Manchester_B_7777_3.pdf": pdf_bytes([], junk=True),
        "transcription_response_11111111-2222-3333-4444-555555555555.pdf": pdf_bytes(["ספר איש טוב", "בית גדול"]),
        "transcription_manual_Manchester_B_7800.pdf": pdf_bytes(["איש טוב היה", "גדול ספר בית"]),
        "B_7777_1r.jpg": b"photo-r", "B_7777_1r(1).jpg": b"photo-r", "B_7777_1v.jpg": b"photo-v",
        "cataloging_Manchester_B_7777.json": json.dumps({"manuscript": {"shelfmark": "Manchester: B 7777"},
                                                         "metadata": {"extractedAt": POST}}).encode(),
        "metadata_Manchester__B_7777.json": json.dumps({"shelfmark": "Manchester: B 7777",
                                                        "source": "Example Transcriptions Team",
                                                        "extractedAt": POST}).encode(),
        "joins_Please_select_shelfmarks(3).json": b"{}",
        "bibliography_title_X_shelfmark_B_7777_1.json": json.dumps({"entries": [
            {"key": "shelfmark", "value": "B 7777"}, {"key": "year", "value": "2001"}]}).encode(),
        "secret_notes.txt": b"never read",
    })
    os.chmod(old / "secret_notes.txt", 0)
    snap = make_root(tmp_path / "snap", {
        "a_series/B_7777_1r.jpg": b"photo-r", "a_series/A_10_1r.jpg": b"a10-photo",
        "a_series/transcription_manual_Manchester_A_10.pdf": pdf_bytes(["@1r", "טוב היה כל", "@1v", "אשר גדול ספר"]),
    })
    listing = [{"key": PREFIX + rel, "size": (snap / rel).stat().st_size, "etag": "e", "modified": "2025-10-23T23:19:52Z"}
               for rel in ("a_series/B_7777_1r.jpg", "a_series/A_10_1r.jpg",
                           "a_series/transcription_manual_Manchester_A_10.pdf")]
    listing.append({"key": PREFIX + "a_series/Lease.docx.pdf", "size": 3, "etag": "e", "modified": "2025-10-23T23:19:52Z"})
    index, names = R.build_merged_index(write_merged(tmp_path / "merged.jsonl"))
    specs = [R.RootSpec("old", str(old)), R.RootSpec("snap", str(snap), True, listing, PREFIX)]
    pkg = R.package_roots(specs, index, names, None)
    os.chmod(old / "secret_notes.txt", 0o600)
    man = {m["key"]: m for m in pkg["manifest"]}
    b, a, b2 = man[R.package_key("B 7777")], man[R.package_key("A 10")], man[R.package_key("B 7800")]
    assert set(man) == {b["key"], a["key"], b2["key"]}

    assert b["image_count"] == 2 and b["image_policy"] == R.POLICY_PREFER_KTIV and b["shelfmark"] == "Manchester: B 7777"
    photo_r = next(i for i in b["images"] if i["sha256"] == hashlib.sha256(b"photo-r").hexdigest())
    assert len(photo_r["paths"]) == 3 and photo_r["canonical"] == {"root": "old", "rel": "B_7777_1r.jpg"}
    assert {"root": "snap", "rel": "a_series/B_7777_1r.jpg"} in photo_r["paths"]
    assert str(tmp_path) not in json.dumps(pkg["manifest"], default=sorted)
    assert str(tmp_path) not in json.dumps([R.image_group_record(g) for g in pkg["images"]])
    assert photo_r["label_race"] == "post_fix" and "label_race_possible" not in photo_r["flags"]
    texts = sorted(b["transcriptions"], key=lambda t: t["units"][0]["side_index"])
    assert [t["style"] for t in texts] == ["side_per_file", "side_per_file"]
    assert [t["units"][0]["side_index"] for t in texts] == [1, 2] and texts[0]["units"][0]["columns"] == 2
    assert b["side_indices"] == [1, 2]
    assert len(texts[0]["files"]) == 2 and texts[0]["provenance"]["credit_kind"] == "project_team"
    assert len(b["catalogue"]) == 1 and b["bibliography"][0]["payload"] == {"year": "2001"}
    assert b["transcription_metadata"][0]["payload"] == {"credit_kind": "project_team", "inventory_id": None}
    side2 = next(t for t in pkg["transcriptions"] if t.get("side_index") == 2 and t["key"] == b["key"])
    assert side2["side_resolution"] == "sibling_file" and side2["cleaned_lines"] == ["היה טוב אשר", "כל על בא"]

    assert a["image_policy"] == R.POLICY_LEGACY_ONLY and a["images"][0]["label_race"] == "pre_fix"
    assert {"needs_label_check", "needs_content_matching", "label_race_possible"} <= set(a["flags"])
    assert a["transcriptions"][0]["style"] == "header_folio"
    assert [u["label"] for u in a["transcriptions"][0]["units"]] == ["1r", "1v"]
    assert a["images"][0]["in_corpus_legacy_names"]

    assert b2["image_policy"] == R.POLICY_NO_IMAGE and "transcription_without_image_anywhere" in b2["flags"]
    assert len(pkg["unresolved"]) == 1 and pkg["unresolved"][0]["style"] == "side_per_file"
    junk = [s for s in pkg["scans"].values() if s["junk"]]
    assert len(junk) == 1
    kinds = {(e.root, e.kind) for e in pkg["entries"]}
    assert ("old", "junk_json") in kinds and ("old", "unrelated") in kinds
    lease = next(e for e in pkg["entries"] if e.name == "Lease.docx.pdf")
    assert not lease.fetched


@needs_font
def test_parent_images_only_for_transcribed_records(tmp_path: pathlib.Path) -> None:
    """``Evr. IIA 223.2`` (header) borrows ``Yevr_II_A_223`` images; a catalogue-only sub-mark does not.

    :param tmp_path: Temporary folder.
    """
    doc = fitz.open()
    page = doc.new_page()
    font = fitz.Font(fontfile=HEBREW_FONT)
    for y, text in ((72, "St. Petersburg, Russian National Library, Firkovich"), (86, "Evr. IIA 223.2"), (128, "1r")):
        tw = fitz.TextWriter(page.rect)
        tw.append((150, y), text, font=font, fontsize=12)
        tw.write_text(page)
    for i, text in enumerate(["טוב היה כל", "אשר גדול ספר"]):
        tw = fitz.TextWriter(page.rect)
        tw.append((540 - font.text_length(text, fontsize=12), 144 + 16 * i), text, font=font, fontsize=12,
                  right_to_left=True)
        tw.write_text(page)
    root = make_root(tmp_path / "r", {
        "transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf": doc.tobytes(),
        "Yevr_II_A_223_5r.jpg": b"leaf", "A_11_1r.jpg": b"a11",
        "cataloging_Manchester_A_11_2.json": json.dumps({"manuscript": {"shelfmark": "Manchester: A 11.2"}}).encode(),
    })
    pkg = R.package_roots([R.RootSpec("r", str(root))], {}, {}, None)
    man = {m["key"]: m for m in pkg["manifest"]}
    sub = man[R.package_key("Evr. IIA 223.2")]
    assert sub["transcriptions"][0]["shelfmark_source"] == "header" and not sub["images"]
    assert [i["names"] for i in sub["parent_images"]] == [["Yevr_II_A_223_5r.jpg"]]
    assert "images_from_parent_shelfmark_unverified" in sub["flags"]
    assert sub["image_policy"] == R.POLICY_LEGACY_ONLY
    cat = man[R.package_key("Manchester: A 11.2")]
    assert not cat["parent_images"] and cat["image_policy"] == R.POLICY_NO_IMAGE


# ───────────────────────────── second verification pass ─────────────────────────────
@pytest.mark.parametrize("key,ok", [
    (PREFIX + "a_series/A_10_1r.jpg", True),
    (PREFIX + "a_series/../../escape.jpg", False),
    ("/etc/A_10_1r.jpg", False),
    (PREFIX + "../A_10_1r.jpg", False),
    (PREFIX + "a_series/A..10_1r.jpg", True),
])
def test_snapshot_rel_refuses_unsafe_keys(key: str, ok: bool) -> None:
    """Keys with a ``..`` component or a leading ``/`` are refused.

    :param key: Bucket key.
    :param ok: Whether the key is accepted.
    """
    if ok:
        assert not R.snapshot_rel(key, PREFIX).startswith("/")
    else:
        with pytest.raises(ValueError):
            R.snapshot_rel(key, PREFIX)


def test_partial_downloads_and_hidden_listing_objects_are_counted(tmp_path: pathlib.Path) -> None:
    """``.part`` files are listed as partial downloads; a hidden object left in the bucket is counted.

    :param tmp_path: Temporary folder.
    """
    assert R.classify_name("B_7777_1r.jpg.part") == "partial_download"
    assert "partial_download" not in R.FETCH_KINDS
    root = make_root(tmp_path / "r", {"B_7777_1r.jpg": b"x", "B_7777_1v.jpg.part": b"", ".DS_Store": b"d"})
    kinds = sorted((e.name, e.kind) for e in R.walk_root(R.RootSpec("r", str(root))))
    assert kinds == [("B_7777_1r.jpg", "image"), ("B_7777_1v.jpg.part", "partial_download")]
    snap = make_root(tmp_path / "snap", {"a_series/A_10_1r.jpg": b"recto"})
    listing = [{"key": PREFIX + "a_series/A_10_1r.jpg", "size": 5, "etag": "e", "modified": "2025-10-23T23:19:52Z"},
               {"key": PREFIX + "a_series/.hidden_notes", "size": 4, "etag": "f", "modified": "2025-10-23T23:19:52Z"}]
    entries = R.walk_root(R.RootSpec("snap", str(snap), True, listing, PREFIX))
    hidden = next(e for e in entries if e.name == ".hidden_notes")
    assert not hidden.fetched and hidden.kind == "unrelated"


@pytest.mark.parametrize("text,out", [
    ("Example Bibliography team", "legacy store project team"),
    ("Jane Roe, Head of Example Bibliography B team", "legacy store project team"),
    ("Site User - Jane Roe", "legacy store site user"),
    ("Jane Roe, Head of the unit", "legacy store other"),
    ("Some Catalog, University Press, 1999", "Some Catalog, University Press, 1999"),
    (None, None),
])
def test_scrub_credit(text: Optional[str], out: Optional[str]) -> None:
    """Credits in the store's formula become their kind; catalogue names are kept.

    :param text: Credit value.
    :param out: Expected value.
    """
    assert R.scrub_credit(text) == out


def test_read_metadata_json_keeps_no_credit_wording(tmp_path: pathlib.Path) -> None:
    """Catalogue Source entries and sources, join and bibliography sources lose the credit wording.

    :param tmp_path: Temporary folder.
    """
    cat = R.read_metadata_json(write_json(tmp_path / "c.json", {"manuscript": {
        "shelfmark": "Manchester: B 7777", "catalogingEntries": [
            {"section": "Shelfmark Description", "field": "Source", "value": "Jane Roe, Head of Example team"},
            {"section": "Content Description", "field": "Title", "value": "Letter of a team of scribes"}],
        "sources": [{"name": "Example Talmud team"}, {"name": "Some Catalog"}]}}), "cataloging_json")
    assert [e["value"] for e in cat["payload"]["entries"]] == ["legacy store project team",
                                                               "Letter of a team of scribes"]
    assert cat["payload"]["sources"] == ["legacy store project team", "Some Catalog"]
    joins = R.read_metadata_json(write_json(tmp_path / "j.json", {
        "mainShelfmark": "Manchester: B 7777", "joinedManuscripts": [{"shelfmark": "Manchester: B 7778"}],
        "source": "Example Joins team"}), "joins_json")
    assert joins["payload"] == {"joined": ["Manchester: B 7778"], "source": "legacy store project team"}
    bib = R.read_metadata_json(write_json(tmp_path / "b.json", {"entries": [
        {"key": "shelfmark", "value": "B 7777"}, {"key": "source", "value": "Jane Roe, Head of Example team"},
        {"key": "year", "value": "1999"}]}), "bibliography_json")
    assert bib["payload"] == {"year": "1999", "source_kind": "project_team"}
    assert "Jane" not in json.dumps([cat["payload"], joins["payload"], bib["payload"]])


def test_pdf_name_key() -> None:
    """File names give the shelf mark's key; copy and side suffixes are not part of it; uuids give none."""
    assert R.pdf_name_key("transcription_manual_Manchester_A_152.pdf") == R.package_key("A 152")
    assert R.pdf_name_key("transcription_manual_Manchester_A_71-1.pdf") == R.package_key("A 71")
    assert R.pdf_name_key("a_76.pdf") == R.package_key("A 76")
    assert R.pdf_name_key("transcription_0573c6a5-23c7-4601-8090-b6e5ab3de69b.pdf") == ""


SIDE_A = ["טוב היה כל", "אשר גדול ספר", "איש בית בא", "על כל יצא"]
LONG_LINES = ["שלום עולם גדול מאד", "אדם טוב בבית אביו", "היה איש אחד בעיר", "כל העם הלך אל ההר",
              "ספר תורה קטן ויפה", "בא יום חדש וטוב", "עץ פרי עושה פרי", "דבר אל העם כולו"]


@needs_font
def test_package_folds_redownloads_and_flags_near_duplicates(tmp_path: pathlib.Path) -> None:
    """A re-download under ``_3`` is a copy of side 1; edition headings stay one side; a near-duplicate is flagged.

    :param tmp_path: Temporary folder.
    """
    edition = fitz.open()
    page = edition.new_page()
    font = fitz.Font(fontfile=HEBREW_FONT)
    y = 72.0
    for parts in ((("דף פא עמוד ב", True), (" )גמרא(", True)), (("טוב היה כל", True),), (("אשר גדול ספר", True),)):
        tw = fitz.TextWriter(page.rect)
        x = 540.0
        for text, rtl in parts:
            x -= font.text_length(text, fontsize=12)
            tw.append((x, y), text, font=font, fontsize=12, right_to_left=rtl)
        tw.write_text(page)
        y += 16
    near_lines = LONG_LINES[:-1] + ["דבר אל העם כולם"]
    old = make_root(tmp_path / "old", {
        "transcription_manual_Manchester_B_7777.pdf": pdf_bytes(SIDE_A),
        "transcription_manual_Manchester_B_7777_2.pdf": pdf_bytes(["היה טוב אשר", "כל על בא"]),
        "transcription_manual_Manchester_B_7777_3.pdf": pdf_bytes([SIDE_A[2], SIDE_A[0], SIDE_A[3], SIDE_A[1]]),
        "transcription_manual_Manchester_B_7777_4.pdf": edition.tobytes(),
        "transcription_manual_Manchester_B_7800.pdf": pdf_bytes(LONG_LINES),
        "transcription_manual_Manchester_B_7801.pdf": pdf_bytes(near_lines),
    })
    os.utime(old / "transcription_manual_Manchester_B_7800.pdf", (1000, 1000))
    os.utime(old / "transcription_manual_Manchester_B_7801.pdf", (2000, 2000))
    pkg = R.package_roots([R.RootSpec("old", str(old))], {}, {}, None)
    man = {m["key"]: m for m in pkg["manifest"]}
    b = man[R.package_key("B 7777")]
    assert len(b["transcriptions"]) == 3 and b["side_indices"] == [1, 2, 3]
    assert [(c["key"], c["suffix_side_index"], c["copy_of_suffix_side_index"]) for c in pkg["side_redownloads"]] == \
        [(b["key"], 3, 1)]
    s1 = next(t for t in b["transcriptions"] if t["units"][0]["suffix_side_index"] == 1)
    assert {"root": "old", "rel": "transcription_manual_Manchester_B_7777_3.pdf", "copy": "side_redownload",
            "suffix_side_index": 3} in s1["files"]
    s3 = next(t for t in b["transcriptions"] if t["units"][0]["suffix_side_index"] == 4)
    assert s3["style"] == "side_per_file" and s3["units"][0]["side_index"] == 3
    unit = next(r for r in pkg["transcriptions"] if r["key"] == b["key"] and r["suffix_side_index"] == 4)
    assert unit["flags"]["edition_ref_heading"] == 1 and unit["cleaned_lines"] == SIDE_A[:2]
    assert all("side_index_from_download_order_unverified" in u["flags"] for t in b["transcriptions"]
               for u in t["units"])
    assert not any("side_sequence_gap" in u["page_row_exclusions"] for t in b["transcriptions"] for u in t["units"])
    assert len([r for r in pkg["transcriptions"] if r["key"] == b["key"]]) == 3
    near = man[R.package_key("B 7801")]["transcriptions"][0]
    assert near["near_duplicate_of"]["key"] == R.package_key("B 7800") and near["near_duplicate_of"]["ratio"] >= 0.98
    assert all("near_duplicate_document" in u["page_row_exclusions"] for u in near["units"])
    orig = man[R.package_key("B 7800")]["transcriptions"][0]
    assert orig["near_duplicate_of"] is None and not orig["units"][0]["page_row_exclusions"]
    assert [(d["key"], d["of_key"], d["units"]) for d in pkg["near_duplicates"]] == \
        [(R.package_key("B 7801"), R.package_key("B 7800"), 1)]


@needs_font
def test_same_text_under_a_higher_suffix_is_a_redownload(tmp_path: pathlib.Path) -> None:
    """One text saved as ``x.pdf`` and ``x_2.pdf`` is side 1 with a re-download name, not a conflict.

    :param tmp_path: Temporary folder.
    """
    data = pdf_bytes(SIDE_A)
    old = make_root(tmp_path / "old", {"transcription_manual_Manchester_B_7777.pdf": data,
                                       "transcription_manual_Manchester_B_7777_2.pdf": data,
                                       "transcription_manual_Manchester_B_7777_3.pdf": pdf_bytes(list(reversed(SIDE_A)))})
    pkg = R.package_roots([R.RootSpec("old", str(old))], {}, {}, None)
    units = [r for r in pkg["transcriptions"] if r["key"] == R.package_key("B 7777")]
    assert len(units) == 1 and units[0]["side_index"] == 1 and units[0]["redownload_name_sides"] == [2]
    assert units[0]["flags"]["side_redownload_copies"] == 2 and not units[0]["page_row_exclusions"]
    assert len(units[0]["files"]) == 3
