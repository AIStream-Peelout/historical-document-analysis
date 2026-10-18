# File name: test_build_arabic_benchmark.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script benchmark builder (queue reading, ground truth, IIIF handling)."""
import csv
import io
import json

import pytest
from PIL import Image

from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab

QUEUE_COLUMNS = ["rank", "tier", "pgpid", "shelfmark", "single_fragment", "canonical_id", "library", "doc_type", "side", "doc_date",
                 "iiif_urls", "suggested_iiif"]
BASMALA = "بسم الله الرحمن الرحيم"


def _jpeg(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _queue(tmp_path, rows):
    path = tmp_path / "queue.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "True" if c == "single_fragment" else "") for c in QUEUE_COLUMNS})
    return path


def test_tier1_documents_merges_rows_of_one_document(tmp_path):
    path = _queue(tmp_path, [
        {"tier": "1", "pgpid": "10", "canonical_id": "Cambridge_CUL_T_S_Ar_38_31", "shelfmark": "T-S Ar.38.31",
         "library": "Cambridge University Library", "iiif_urls": "https://a/m1 ; https://a/m2", "side": "recto"},
        {"tier": "1", "pgpid": "11", "canonical_id": "Cambridge_CUL_T_S_Ar_38_31", "iiif_urls": "https://a/m2",
         "shelfmark": "T-S Misc.29.23 + T-S Ar.38.31", "single_fragment": "False"},
        {"tier": "2", "pgpid": "12", "canonical_id": "Other_doc"},
        {"tier": "1", "pgpid": "13", "canonical_id": "", "iiif_urls": "https://a/m3"},
        {"tier": "1", "pgpid": "14", "canonical_id": "Oxford_X", "suggested_iiif": "https://b/m4"},
    ])
    docs = bab.tier1_documents(path)
    assert list(docs) == ["Cambridge_CUL_T_S_Ar_38_31", "Oxford_X"]
    assert docs["Cambridge_CUL_T_S_Ar_38_31"]["pgpids"] == ["10", "11"]
    assert docs["Cambridge_CUL_T_S_Ar_38_31"]["manifests"] == ["https://a/m1", "https://a/m2"]
    assert docs["Cambridge_CUL_T_S_Ar_38_31"]["pgp_side"] == "recto"
    assert not docs["Cambridge_CUL_T_S_Ar_38_31"]["single_fragment"]          # one of its PGP records is a join
    assert docs["Oxford_X"]["manifests"] == ["https://b/m4"] and docs["Oxford_X"]["single_fragment"]


def test_training_documents_reads_manifests_and_id_lists(tmp_path):
    (tmp_path / "pgp_editions_v1").mkdir()
    (tmp_path / "documentary_grounding_v1").mkdir()
    editions, lines, ids = tmp_path / "pgp_editions_v1/manifest.jsonl", tmp_path / "documentary_grounding_v1/manifest.jsonl", tmp_path / "clean_v2_ids.json"
    editions.write_text(json.dumps({"pgpid": "2771", "canonical_id": "Cambridge_CUL_T_S_10J10_10"}) + "\n\n"
                        + json.dumps({"pgpid": 446, "canonical_id": None}) + "\n", encoding="utf-8")
    lines.write_text(json.dumps({"doc_id": "Cambridge_CUL_T_S_13J13_2", "stem": "dg_x"}) + "\n"
                     + json.dumps({"doc_id": "Cambridge_CUL_T_S_10J10_10"}) + "\n", encoding="utf-8")
    ids.write_text(json.dumps(["Cambridge_CUL_Add_3341"]), encoding="utf-8")
    seen = bab.training_documents([editions, lines], [ids])
    assert seen == {"Cambridge_CUL_Add_3341": "clean_v2_ids", "Cambridge_CUL_T_S_10J10_10": "pgp_editions_v1", "pgp:2771": "pgp_editions_v1",
                    "pgp:446": "pgp_editions_v1", "Cambridge_CUL_T_S_13J13_2": "documentary_grounding_v1"}
    assert bab.training_set_of("Cambridge_CUL_T_S_13J13_2", ["9"], seen) == "documentary_grounding_v1"
    assert bab.training_set_of("Some_other_id", ["9", "446"], seen) == "pgp_editions_v1"          # caught by its PGP id
    assert bab.training_set_of("Cambridge_CUL_T_S_13J13_20", ["27710"], seen) == ""               # ids are compared whole
    with pytest.raises(FileNotFoundError):                                                        # no check, no benchmark
        bab.training_documents([tmp_path / "missing/manifest.jsonl"], [])


def test_load_editions_keeps_only_edition_rows(tmp_path):
    path = tmp_path / "footnotes.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document", "document_id", "doc_relation", "content"])
        writer.writeheader()
        writer.writerow({"document_id": "10", "doc_relation": "Edition", "content": BASMALA})
        writer.writerow({"document_id": "10", "doc_relation": "Edition, Translation", "content": "second"})
        writer.writerow({"document_id": "10", "doc_relation": "Translation", "content": "In the name of God"})
        writer.writerow({"document_id": "10", "doc_relation": "Edition", "content": "   "})
        writer.writerow({"document_id": "99", "doc_relation": "Edition", "content": "not wanted"})
    assert bab.load_editions(path, ["10", "11"]) == {"10": [BASMALA, "second"]}


def test_ground_truth_takes_the_fullest_edition_and_only_visible_text():
    short = "Recto\n" + BASMALA
    full = "Recto\n" + BASMALA + " [الحمد لله]\nهذا ما استاجر (؟) افرايم\nVerso\nالى الشيخ <ا>بو سعد\nלכבוד מרנא ורבנא"
    gt = bab.ground_truth([short, full])
    assert [s["side"] for s in gt["sections"]] == ["recto", "verso"] and gt["editions"] == 2
    assert gt["sections"][0]["lines"] == [BASMALA, "هذا ما استاجر افرايم"]
    assert gt["text"].splitlines()[2] == "الى الشيخ بو سعد"
    assert gt["arabic_letters"] == len("بسماللهالرحمنالرحيمهذامااستاجرافرايمالىالشيخبوسعد".replace("ى", "ي"))
    assert gt["hebrew_letters"] == 14
    assert bab.ground_truth([])["arabic_letters"] == 0


V2 = {"sequences": [{"canvases": [
    {"label": "1r", "width": 3013, "height": 3943,
     "images": [{"resource": {"@id": "https://img/x-1.jp2", "service": {"@id": "https://img/x-1.jp2/"}}}]},
    {"label": "1v", "width": 8131, "height": 11823,
     "images": [{"resource": {"@id": "https://img/x-2/full/full/0/default.jpg", "service": [{"@id": "https://img/x-2"}]}}]},
    {"label": "plain", "width": 500, "height": 400, "images": [{"resource": {"@id": "https://files/plain.jpg"}}]},
]}]}
V3 = {"items": [{"label": {"en": ["16 verso"]}, "width": 6000, "height": 3000,
                 "items": [{"items": [{"body": {"id": "https://img3/y/full/max/0/default.jpg", "service": [{"id": "https://img3/y"}]}}]}]}]}


def test_iiif_canvases_reads_both_presentation_versions():
    v2 = bab.iiif_canvases(V2)
    assert [c["label"] for c in v2] == ["1r", "1v", "plain"]
    assert v2[0]["service"] == "https://img/x-1.jp2" and v2[1]["service"] == "https://img/x-2" and v2[2]["service"] == ""
    v3 = bab.iiif_canvases(V3)
    assert v3 == [{"label": "16 verso", "width": 6000, "height": 3000, "service": "https://img3/y",
                   "resource": "https://img3/y/full/max/0/default.jpg"}]
    assert bab.iiif_canvases({}) == []


def test_iiif_image_url_caps_the_long_side():
    v2 = bab.iiif_canvases(V2)
    assert bab.iiif_image_url(v2[0], 4000) == "https://img/x-1.jp2/full/full/0/default.jpg"              # already small enough
    assert bab.iiif_image_url(v2[1], 4000) == "https://img/x-2/full/2751,/0/default.jpg"                 # portrait: height becomes 4000
    assert bab.iiif_image_url(bab.iiif_canvases(V3)[0], 4000) == "https://img3/y/full/4000,/0/default.jpg"   # landscape: width 4000
    assert bab.iiif_image_url(v2[2], 4000) == "https://files/plain.jpg"                                  # no image service


@pytest.mark.parametrize("label, side", [("1r", "recto"), ("1v", "verso"), ("16 recto", "recto"), ("16 Verso", "verso"),
                                         ("f. 2 v", "verso"), ("p. 1", ""), ("1", ""), ("Inside upper board", ""), ("", "")])
def test_side_of_label(label, side):
    assert bab.side_of_label(label) == side


def test_planned_images_prefers_the_image_store_and_skips_whole_volumes(monkeypatch):
    doc = {"manifests": ["https://lib/volume", "https://lib/folio", "https://lib/broken"]}
    assert bab.planned_images("d", doc, ["https://store/a.jpg", "https://store/b.jpg"], True, 4000, 8, 0)[0] == [
        {"url": "https://store/a.jpg", "label": "", "side": "", "source": "image_store"},
        {"url": "https://store/b.jpg", "label": "", "side": "", "source": "image_store"}]
    assert bab.planned_images("d", doc, [], False, 4000, 8, 0) == ([], "no image in the index (IIIF disabled)")
    assert bab.planned_images("d", {"manifests": []}, [], True, 4000, 8, 0) == ([], "no image in the index and no manifest")
    volume = {"sequences": [{"canvases": V2["sequences"][0]["canvases"][:1] * 121}]}

    def fake_get(url, timeout=90):
        if url.endswith("broken"):
            raise OSError("refused")
        return json.dumps(volume if url.endswith("volume") else V2).encode()

    monkeypatch.setattr(bab, "http_get", fake_get)
    images, note = bab.planned_images("d", doc, [], True, 4000, 8, 0)
    assert [(i["label"], i["side"], i["source"]) for i in images] == [("1r", "recto", "https://lib/folio"), ("1v", "verso", "https://lib/folio"),
                                                                      ("plain", "", "https://lib/folio")]
    assert "whole volume (121 canvases)" in note and "https://lib/broken: OSError" in note


def test_save_image_downloads_once_and_reports_the_hash(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bab, "http_get", lambda url, timeout=90: calls.append(url) or b"jpeg-bytes")
    first = bab.save_image("https://store/a.jpg", tmp_path / "d__0.jpg")
    again = bab.save_image("https://store/a.jpg", tmp_path / "d__0.jpg")
    import hashlib
    assert first == again == {"file": "d__0.jpg", "bytes": 10, "sha256": hashlib.sha256(b"jpeg-bytes").hexdigest()}
    assert calls == ["https://store/a.jpg"]                                # the second call read the file instead
    assert not list(tmp_path.glob("*.part"))


def test_build_excludes_joins_training_documents_and_short_editions(tmp_path, monkeypatch):
    from src.datasets.consensus import two_reader_lines

    long_edition = "Recto\n" + "\n".join([BASMALA + " هذا ما استاجر افرايم بن عالي"] * 8) + "\nVerso\nالى الشيخ ابو سعد"
    rows = [{"tier": "1", "pgpid": str(n), "canonical_id": doc_id, "shelfmark": doc_id, "library": "Lib", "doc_type": "Letter",
             "side": "recto and verso", **extra}
            for n, (doc_id, extra) in enumerate([("kept_store", {}), ("kept_iiif", {"iiif_urls": "https://lib/folio"}),
                                                 ("a_join", {"single_fragment": "False"}), ("trained_by_id", {}), ("trained_by_pgpid", {}),
                                                 ("too_short", {}), ("no_image", {}), ("store_404", {}), ("too_dense", {})], start=1)]
    monkeypatch.setattr(bab, "QUEUE_CSV", _queue(tmp_path, rows))
    footnotes = tmp_path / "footnotes.csv"
    with open(footnotes, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document_id", "doc_relation", "content"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"document_id": row["pgpid"], "doc_relation": "Edition",
                             "content": BASMALA if row["canonical_id"] == "too_short" else long_edition})
    monkeypatch.setattr(bab, "FOOTNOTES_CSV", footnotes)
    (tmp_path / "lines_v1").mkdir()
    manifest = tmp_path / "lines_v1/manifest.jsonl"
    manifest.write_text(json.dumps({"doc_id": "trained_by_id"}) + "\n" + json.dumps({"pgpid": "5"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(two_reader_lines, "es_get_docs", lambda ids, index, fields: {
        "kept_store": {"image_urls": ["https://store/a.jpg", "https://store/gone.jpg"]}, "store_404": {"image_urls": ["https://store/gone.jpg"]},
        "a_join": {"image_urls": ["https://store/j.jpg"]}, "trained_by_id": {"image_urls": ["https://store/t.jpg"]},
        "too_dense": {"image_urls": ["https://store/thumbnail.jpg"]}})
    fetched = []

    def fake_get(url, timeout=90):
        fetched.append(url)
        if url.endswith("gone.jpg"):
            raise OSError("404")
        if url == "https://lib/folio":
            return json.dumps(V2).encode()
        return _jpeg(300, 400) if url.endswith("thumbnail.jpg") else _jpeg(900, 1200)      # 0.12 and 1.08 megapixels

    monkeypatch.setattr(bab, "http_get", fake_get)
    out = tmp_path / "bench"
    report = bab.build(out, delay_s=0, training_manifests=[manifest], training_id_lists=[], registry=tmp_path / "registry.json")
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert registry["benchmark"] == "bench" and registry["ids"] == ["kept_iiif", "kept_store"] and registry["pgpids"] == ["1", "2"]
    assert report["queue_documents"] == 9 and report["benchmark_documents"] == 2 and report["images"] == 4 and report["failed_images"] == 2
    assert report["excluded_documents"] == {
        "a_join": "join: the edition covers several fragments", "trained_by_id": "in a training source set (lines_v1)",
        "trained_by_pgpid": "in a training source set (lines_v1)", "too_short": "fewer than 200 visible Arabic letters",
        "no_image": "no image in the index and no manifest", "store_404": "every image failed to download",
        "too_dense": "text too dense for the images (over 1000 letters per megapixel)"}
    assert bab.megapixels(out / "images" / "kept_store__0.jpg") == pytest.approx(1.08)
    assert report["excluded"]["in a training source set (lines_v1)"] == 2 and report["image_sources"] == {"image_store": 1, "library_iiif": 3}
    assert not {"https://store/j.jpg", "https://store/t.jpg"} & set(fetched)       # nothing is downloaded for an excluded document
    records = [json.loads(line) for line in (out / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    import hashlib
    assert [r["id"] for r in records] == sorted(["kept_store", "kept_iiif"], key=lambda d: hashlib.sha1(d.encode()).hexdigest())
    by_id = {r["id"]: r for r in records}
    assert by_id["kept_store"]["images"] == [{"file": "kept_store__0.jpg", "image_index": 0, "label": "", "side": "", "source": "image_store"}]
    assert [(i["image_index"], i["side"]) for i in by_id["kept_iiif"]["images"]] == [(0, "recto"), (1, "verso"), (2, "")]
    assert by_id["kept_iiif"]["pgp_side"] == "recto and verso" and by_id["kept_iiif"]["gt"]["arabic_letters"] >= 200
    assert by_id["kept_store"]["letters_per_megapixel"] == round(by_id["kept_store"]["gt"]["arabic_letters"] / 1.08)
    assert [s["side"] for s in by_id["kept_iiif"]["gt"]["sections"]] == ["recto", "verso"]
    assert sorted(p.name for p in (out / "images").iterdir()) == ["kept_iiif__0.jpg", "kept_iiif__1.jpg", "kept_iiif__2.jpg", "kept_store__0.jpg",
                                                                 "too_dense__0.jpg"]
    logged = json.loads((out / "images_manifest.json").read_text(encoding="utf-8"))["images"]
    assert sum("error" in e for e in logged) == 2 and json.loads((out / "build_report.json").read_text(encoding="utf-8")) == report
