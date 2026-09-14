"""Tests for the Bodleian direct-scrape image pointers and downstream consumers.

The merge writes ``BODLEIAN/<canonical_id>/<stem>.jpg`` object paths only for
files that exist on disk, and the uploader resolves the same files through the
same function, so the two can never disagree about which objects exist.
"""

import json

from src.datasets.document_models.genizah_document import GenizahDocument, _merged_image_urls
from src.datasets.indexing.es_image_gap_report import _to_record, summarize
from src.datasets.merging.bodleian_images import (
    bodleian_image_manifest,
    bodleian_image_sources,
    gcs_object_path,
    load_bodleian_records,
    tei_date,
    tei_description,
)

CID = "Oxford_Bodleian_Bodl_MS_heb_d_66_137"


def _record(images):
    return {"canonical_id": CID, "match": "part", "tei": None,
            "images": [{"stem": s, "local_path": f"images/{CID}/{s}.jpg"} for s in images],
            "image_count": len(images)}


def test_gcs_object_path_layout():
    assert gcs_object_path(CID, "MS_HEB_d_66_137a") == f"BODLEIAN/{CID}/MS_HEB_d_66_137a.jpg"


def test_image_sources_only_point_at_files_on_disk(tmp_path):
    img_dir = tmp_path / "images" / CID
    img_dir.mkdir(parents=True)
    (img_dir / "MS_HEB_d_66_137a.jpg").write_bytes(b"\xff\xd8")
    records = {CID: _record(["MS_HEB_d_66_137a", "MS_HEB_d_66_137b"]),  # b never downloaded
               "Oxford_Bodleian_X": _record([])}
    sources = bodleian_image_sources(records, str(tmp_path))
    assert list(sources) == [CID]
    (object_path, local_file), = sources[CID].entries
    assert object_path == f"BODLEIAN/{CID}/MS_HEB_d_66_137a.jpg"
    assert local_file == str(img_dir / "MS_HEB_d_66_137a.jpg")
    assert bodleian_image_manifest(records, str(tmp_path)) == {CID: [object_path]}


def test_load_records_missing_dir_and_dedupe(tmp_path):
    assert load_bodleian_records(str(tmp_path / "none" / "*.json")) == {}
    (tmp_path / "a.json").write_text(json.dumps(_record([])), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_record(["s1"])), encoding="utf-8")
    by_cid = load_bodleian_records(str(tmp_path / "*.json"))
    assert by_cid[CID]["image_count"] == 1


def test_tei_description_titles_then_note_skipping_placeholder():
    assert tei_description({"titles": ["Contract", "Contract", "Letter"]}) == "Contract; Letter"
    assert tei_description({"titles": [], "items": [{"notes": ["Temporary record."]},
                                                    {"notes": ["Liturgy; vellum."]}]}) == "Liturgy; vellum."
    assert tei_description({"titles": [], "items": [{"notes": ["Temporary record."]}]}) is None
    assert tei_description(None) is None


def test_tei_date_text_then_not_before():
    assert tei_date({"orig_date": {"text": "12xx", "not_before": "1189"}}) == "12xx"
    assert tei_date({"orig_date": {"text": None, "not_before": "1189"}}) == "1189"
    assert tei_date({}) is None


# ── document model / index consumers ─────────────────────────────────────────

def _images(preferred):
    return {
        "fjp": ["f.jpg"],
        "ktiv": {"image_urls": ["https://gcs/KTIV/1/k.jpg"]},
        "bodleian": {"image_urls": ["https://gcs/BODLEIAN/x/b.jpg"],
                     "catalogue_url": "https://hebrew.bodleian.ox.ac.uk/catalog/volume_9#p",
                     "tei_part_id": "p", "match": "part"},
        "preferred_source": preferred,
    }


def test_merged_image_urls_follow_preference_then_fixed_order():
    fjp = "https://storage.googleapis.com/cairo-genizah-es-json/images/f.jpg"
    assert _merged_image_urls(_images("bodleian")) == [
        "https://gcs/BODLEIAN/x/b.jpg", "https://gcs/KTIV/1/k.jpg", fjp]
    assert _merged_image_urls(_images("ktiv")) == [
        "https://gcs/KTIV/1/k.jpg", "https://gcs/BODLEIAN/x/b.jpg", fjp]
    assert _merged_image_urls(_images("fjp")) == [
        fjp, "https://gcs/KTIV/1/k.jpg", "https://gcs/BODLEIAN/x/b.jpg"]


def test_from_merged_format_surfaces_bodleian_fields_and_tei_date():
    merged = {
        "canonical_id": CID, "shelfmark_display": "Bodl. MS heb. d 66/137",
        "institution": "Bodleian Library, Oxford", "sources_present": ["pgp", "bodleian"],
        "description": "Contract", "date": "1154",
        "images": _images("bodleian"),
        "sources": {"pgp": {"documents": [{"pgpid": "1"}]}, "fjp": [], "ktiv": None,
                    "bodleian": {"match": "part"}},
    }
    doc = GenizahDocument.from_merged_format(merged)
    assert doc.image_urls[0] == "https://gcs/BODLEIAN/x/b.jpg"
    assert doc.date == {"standard_date": "1154"}
    meta = doc.full_metadata
    assert meta["image_preferred_source"] == "bodleian"
    assert meta["bodleian_images"] == ["https://gcs/BODLEIAN/x/b.jpg"]
    assert meta["bodleian_catalogue_url"].startswith("https://hebrew.bodleian.ox.ac.uk/")
    assert meta["bodleian_tei_part_id"] == "p" and meta["bodleian_match"] == "part"
    es_doc = doc.to_elasticsearch_document()
    assert es_doc["has_bodleian_images"] is True
    assert es_doc["bodleian_catalogue_url"] == meta["bodleian_catalogue_url"]
    assert es_doc["date_start"].startswith("1154")


def test_gap_report_counts_bodleian_known():
    record = _to_record({"canonical_id": CID, "sources_present": ["pgp", "bodleian"]}, es_id=CID)
    assert record.bodleian_known is True
    assert record.sources_present == "bodleian+pgp"
    (summary,) = summarize([record])
    assert summary.bodleian_known == 1
    assert summary.source_mix == {"bodleian+pgp": 1}
