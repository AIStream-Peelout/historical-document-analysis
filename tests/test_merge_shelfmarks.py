"""Tests for FJP multi-shelfmark splitting in the merge pipeline.

The vast majority of FJP records carry a single shelfmark, but ~2% enumerate
several fragments in one string (joins / ranges). Those must be exploded so each
constituent shelfmark is captured rather than lost to a garbage canonical id.
"""

import json

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer as SN
from src.datasets.merging.institution_tokens import (
    combine,
    institution_token,
    resolve_token,
)
from src.datasets.merging.merge_shelfmarks import (
    _is_blocked_scrape,
    _transcription_page_lines,
    annotation_page_lines,
    _ktiv_ena_alias,
    _ktiv_fallback_shelfmarks,
    _ktiv_manifest_url,
    _write_ktiv_backlog,
    build_merged_record,
    diff_against_snapshot,
    index_ktiv_zips,
    load_ktiv_transcriptions,
    looks_multi,
    record_has_image,
    record_is_rich,
    split_shelfmarks,
)


def test_single_shelfmark_not_split():
    assert split_shelfmarks("Cambridge CUL: T-S 10J5.6") == ["Cambridge CUL: T-S 10J5.6"]
    assert split_shelfmarks("New York JTS: ENA 4020.50") == ["New York JTS: ENA 4020.50"]
    assert not looks_multi("Cambridge CUL: T-S 10J5.6")


def test_multi_same_institution_explodes_and_carries_prefix():
    raw = "Cambridge CUL: T-S AS 62.64 T-S AS 62.645"
    marks = split_shelfmarks(raw)
    assert [SN.to_canonical_id(m) for m in marks] == ["T_S_AS_62_64", "T_S_AS_62_645"]
    # The institution context is carried forward to the second mark.
    assert all(m.startswith("Cambridge CUL:") for m in marks)


def test_range_endpoints_become_separate_marks():
    raw = "Cambridge CUL: T-S NS 266 - T-S NS 270"
    cids = [SN.to_canonical_id(m) for m in split_shelfmarks(raw)]
    assert cids == ["T_S_NS_266", "T_S_NS_270"]


def test_multi_institution_explodes_with_each_context():
    raw = ("Cambridge CUL: T-S A38.2 T-S NS 247.4 "
           "Oxford: MS heb. e.43/3 Paris AIU: IX.A.13")
    cids = [SN.to_canonical_id(m) for m in split_shelfmarks(raw)]
    assert cids == ["T_S_A38_2", "T_S_NS_247_4", "MS_heb_e_43_3", "IX_A_13"]


def test_parenthetical_alt_not_treated_as_second_shelfmark():
    # "(Alt: 1)" must not split on its inner colon into a bogus "1)" mark.
    assert split_shelfmarks("Cincinnati HUC: 1001 (Alt: 1)") == ["Cincinnati HUC: 1001"]
    marks = split_shelfmarks("Paris Mosseri: Moss. VIII399.1 (Alt: 2nd Series: P 764)")
    assert [SN.to_canonical_id(m) for m in marks] == ["Mosseri_VIII399_1"]


# ── image / richness predicates ───────────────────────────────────────────────

def test_record_has_image():
    assert record_has_image({"images": {"fjp": ["x_1r.jpg"], "ktiv": None}})
    assert record_has_image({"images": {"fjp": [], "ktiv": {"pnx_id": "PNX_1"}}})
    assert not record_has_image({"images": {"fjp": [], "ktiv": None}})
    assert not record_has_image({"images": {"fjp": [], "ktiv": {"pnx_id": None}}})


def test_record_is_rich():
    assert record_is_rich({"description": "Letter from X", "sources": {}})
    assert record_is_rich({"description": None, "sources": {"fjp": [{"transcriptions": [1]}]}})
    assert not record_is_rich({"description": None, "sources": {"fjp": [{}], "ktiv": None}})


# ── run-over-run diff ─────────────────────────────────────────────────────────

# ── KTIV JTS dual-shelfmark (ENA) bridge ──────────────────────────────────────

def test_ktiv_lutzki_keys_under_ena_from_additional():
    # A Lutzki/MS item carries its old ENA number in shelfmarks.additional;
    # it must re-key onto the ENA form so it joins PGP/FJP.
    doc = {
        "shelf_mark": "The Jewish Theological Seminary of America, New York Ms. Lutzki 825, fol. 55",
        "shelfmarks": {"additional": "Adler, Elkan Nathan Ms. 1205.55"},
    }
    assert _ktiv_ena_alias(doc) == "New_York_JTS_ENA_1205_55"


def test_ktiv_ena_alias_absent_without_adler():
    assert _ktiv_ena_alias({"shelfmarks": {"additional": "Film MSS-D"}}) is None
    assert _ktiv_ena_alias({"shelfmarks": {}}) is None


# ── KTIV image-zip pointers ───────────────────────────────────────────────────

def test_index_ktiv_zips_groups_by_sysnum(tmp_path):
    for name in [
        "ktiv_PNX_MANUSCRIPTS990051236050205171-1_images.zip",
        "ktiv_PNX_MANUSCRIPTS990051236050205171-1_images(1).zip",
        "ktiv_PNX_MANUSCRIPTS990039474250205171-1_images.zip",
    ]:
        (tmp_path / name).write_bytes(b"")
    index = index_ktiv_zips(str(tmp_path / "*.zip"))
    assert len(index["990051236050205171"]) == 2          # two downloads of one ms
    assert index["990039474250205171"][0].endswith("images.zip")


def test_merged_record_points_at_ktiv_images():
    ktiv = {"pnx_id": "PNX_1", "sys_num": "990051236050205171", "shelf_mark": "x"}
    zips = {"990051236050205171": ["ktiv_PNX_MANUSCRIPTS990051236050205171-1_images.zip"]}
    images = {"990051236050205171": ["KTIV/990051236050205171/0001_FL1.jpg",
                                     "KTIV/990051236050205171/0002_FL2.jpg"]}
    rec = build_merged_record("Cambridge_CUL_T_S_10J5_6", None, [], ktiv, zips, images)
    blk = rec["images"]["ktiv"]
    assert blk["zip_files"] == zips["990051236050205171"]
    assert blk["image_count"] == 2
    assert blk["populated"] is True
    assert blk["image_urls"][0].startswith(
        "https://storage.googleapis.com/cairo-genizah-es-json/KTIV/990051236050205171/")
    # Routes to KTIV once images are populated.
    assert rec["images"]["preferred_source"] == "ktiv"


def test_ktiv_only_record_gets_institution_from_shelfmark_head():
    ktiv = {"pnx_id": "PNX_3", "sys_num": "999999999999999998",
            "shelf_mark": "Berlin, Staatsbibliothek, Germany Ms. Or. 123"}
    rec = build_merged_record("Berlin_Or_123", None, [], ktiv, {}, {})
    assert rec["institution"] == "Berlin"


def test_merged_record_falls_back_to_fjp_when_ktiv_not_populated():
    ktiv = {"pnx_id": "PNX_2", "sys_num": "999999999999999999", "shelf_mark": "x"}
    fjp = [("Cambridge CUL: T-S 1.1", {"images": ["x_1r.jpg"]})]
    rec = build_merged_record("X_1", None, fjp, ktiv, {}, {})
    assert rec["images"]["ktiv"]["populated"] is False
    assert rec["images"]["preferred_source"] == "fjp"


# ── KTIV null-shelfmark handling (joins, blocked scrapes) ─────────────────────

def test_blocked_scrape_detected_by_challenge_title_or_empty_content():
    assert _is_blocked_scrape({"page_title": "Just a moment..."})
    assert _is_blocked_scrape({"page_title": "x", "shelfmarks": {},
                               "scholarly_entries": []})
    assert not _is_blocked_scrape({"page_title": "x",
                                   "shelfmarks": {"system_no": "1"}})


def test_join_page_title_yields_all_constituent_shelfmarks():
    # Join pages (צירוף) render no shelfmark element; the title enumerates the
    # constituents, each of which must canonicalize to its PGP-joinable id.
    doc = {"page_title": ("Ms. T-S 10 J 5.6, Ms. T-S 20.113, "
                          "Cambridge University Library, צירוף מכתב | Ktiv | Item page")}
    marks = _ktiv_fallback_shelfmarks(doc)
    cids = [
        combine(resolve_token(m) or institution_token(m), SN.to_canonical_id(m))
        for m in marks
    ]
    assert cids == ["Cambridge_CUL_T_S_10J5_6", "Cambridge_CUL_T_S_20_113"]


def test_fallback_prefers_shelfmarks_block_over_title():
    doc = {
        "shelfmarks": {"shelfmark": {
            "value": "Cambridge University Library, Cambridge, England Ms. T-S 10 K 6"
        }},
        "page_title": "Ms. Something Else | Ktiv",
    }
    assert _ktiv_fallback_shelfmarks(doc) == [
        "Cambridge University Library, Cambridge, England Ms. T-S 10 K 6"
    ]


def test_fallback_empty_when_title_has_no_marks():
    assert _ktiv_fallback_shelfmarks({"page_title": "כתב יד | Ktiv"}) == []
    assert _ktiv_fallback_shelfmarks({}) == []


# ── KTIV transcription bundles ────────────────────────────────────────────────

def _write_bundle(tmp_path, name, doc_id, pages):
    bundle = {
        "source": "nli_ktiv_viewer",
        "doc_id": doc_id,
        "ie": "IE202603440",
        "pages": [
            {"fl": fl, "annotation_page": {
                "items": [{"body": {"value": w}} for w in words]}}
            for fl, words in pages
        ],
    }
    (tmp_path / name).write_text(json.dumps(bundle), encoding="utf-8")


def test_load_ktiv_transcriptions_flattens_words_per_page(tmp_path):
    _write_bundle(
        tmp_path,
        "ktiv_PNX_MANUSCRIPTS990051173350205171-1_transcription.json",
        "PNX_MANUSCRIPTS990051173350205171-1",
        [("FL111", ["בשעת", "עשייה"]), ("FL222", ["אשר"])],
    )
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    summary = by_sysnum["990051173350205171"]
    assert summary["page_count"] == 2
    assert summary["pages"][0]["text"] == "בשעת עשייה"
    assert summary["pages"][0]["words"] == 2
    assert summary["ie"] == "IE202603440"


def test_load_ktiv_transcriptions_splits_lines_on_break_markers(tmp_path):
    bundle = {
        "doc_id": "PNX_MANUSCRIPTS990000000000000002-1",
        "ie": "IE9",
        "pages": [{"fl": "FL1", "annotation_page": {"items": [
            {"id": "0 oldVer", "body": {"value": "שורה"}},
            {"id": "1 oldVer", "body": {"value": "ראשונה"}},
            {"id": "/supplementing/t0-BreackLine", "body": {"value": ""}},
            {"id": "2 oldVer", "body": {"value": "שנייה"}},
        ]}}],
    }
    (tmp_path / "ktiv_x_transcription.json").write_text(
        json.dumps(bundle), encoding="utf-8")
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    page = by_sysnum["990000000000000002"]["pages"][0]
    assert page["text"] == "שורה ראשונה\nשנייה"
    assert page["lines"] == 2


def test_load_ktiv_transcriptions_accepts_dom_captured_lines(tmp_path):
    # The viewer-panel DOM fallback saves pre-built lines instead of an
    # AnnotationPage; both shapes must land in the same summary format.
    bundle = {
        "doc_id": "PNX_MANUSCRIPTS990000000000000003-1",
        "ie": None,
        "pages": [{"fl": None, "image_name": "Frag. 001r",
                   "lines": ["שורה אחת", "שורה שתיים"],
                   "text": "שורה אחת\nשורה שתיים"}],
    }
    (tmp_path / "ktiv_dom_transcription.json").write_text(
        json.dumps(bundle), encoding="utf-8")
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    page = by_sysnum["990000000000000003"]["pages"][0]
    assert page["text"] == "שורה אחת\nשורה שתיים"
    assert page["image_name"] == "Frag. 001r"
    assert page["words"] == 4


def test_load_ktiv_transcriptions_keeps_richest_duplicate(tmp_path):
    _write_bundle(tmp_path, "ktiv_a_transcription.json",
                  "PNX_MANUSCRIPTS990000000000000001-1", [("FL1", ["א"])])
    _write_bundle(tmp_path, "ktiv_a(1)_transcription.json",
                  "PNX_MANUSCRIPTS990000000000000001-1",
                  [("FL1", ["א", "ב", "ג"])])
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    assert by_sysnum["990000000000000001"]["pages"][0]["words"] == 3


# ── API-shape fragments → words (geometry) ───────────────────────────────────
# Boxes below are the live values served for T-S F 7.54 (FL202603457) and for
# Frag. 004r of sys_num 990055620790205171 (FL203116737): NLI splits a word into
# one item per sigla run, and the fragments' boxes abut exactly.

def _ann(item_id, text, x1, x2, y1, y2, sigla=None):
    return {
        "id": item_id, "type": "Annotation",
        "body": {"type": "TextualBody", "value": text, "sigla": sigla},
        "target": {"selector": {"type": "SvgSelector", "value":
                   f'<svg><path d="M{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2} {x1},{y1}z"/></svg>'}},
    }


_BREAK = {"id": "/supplementing/t0-BreackLine", "type": "Annotation",
          "body": {"type": "TextualBody", "value": "", "sigla": None},
          "target": {"id": None, "selector": None}}


def test_load_ktiv_transcriptions_prefers_api_shape_over_dom_at_equal_pages(tmp_path):
    # The same manuscript scraped both ways: the DOM copy has more "words"
    # (NLI's fragments joined with spaces), the API copy has whole words.
    dom = {
        "source": "nli_ktiv_viewer_dom",
        "doc_id": "PNX_MANUSCRIPTS990000000000000004-1", "ie": "IE4",
        "pages": [{"fl": None, "image_name": "Frag. 001r",
                   "lines": ["בשעת ע שייה"], "text": "בשעת ע שייה"}],
    }
    api = {
        "source": "nli_ktiv_viewer",
        "doc_id": "PNX_MANUSCRIPTS990000000000000004-1", "ie": "IE4",
        "pages": [{"fl": "FL1", "annotation_page": {"items": [
            _ann("0 oldVer", "בשעת", 3056.55, 3421, 471, 711),
            _ann("1 oldVer", "ע", 2921.97, 3005.55, 471, 711),
            _ann("2 oldVer", "שייה", 2653.73, 2921.96, 471, 711, "t h"),
        ]}}],
    }
    (tmp_path / "ktiv_d_transcription.json").write_text(json.dumps(dom), encoding="utf-8")
    (tmp_path / "ktiv_a_transcription.json").write_text(json.dumps(api), encoding="utf-8")
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    summary = by_sysnum["990000000000000004"]
    assert summary["shape"] == "api"
    assert summary["file"] == "ktiv_a_transcription.json"
    assert summary["pages"][0]["text"] == "בשעת עשייה"
    assert summary["pages"][0]["words"] == 2
    assert summary["pages"][0]["fragments"] == 3
    assert "_rank" not in summary

    # But a DOM copy covering more pages still wins over a partial API copy.
    dom["pages"].append({"fl": None, "image_name": "Frag. 001v",
                         "lines": ["שורה"], "text": "שורה"})
    (tmp_path / "ktiv_d_transcription.json").write_text(json.dumps(dom), encoding="utf-8")
    by_sysnum = load_ktiv_transcriptions(str(tmp_path / "*_transcription.json"))
    assert by_sysnum["990000000000000004"]["shape"] == "dom"
    assert by_sysnum["990000000000000004"]["page_count"] == 2


def test_annotation_page_lines_merges_split_words_by_geometry():
    page = {"fl": "FL202603457", "annotation_page": {"items": [
        _ann("0 oldVer", "בשעת", 3056.55, 3421, 471, 711),
        _ann("1 oldVer", "ע", 2921.97, 3005.5499999999997, 471, 711),
        _ann("2 oldVer", "שייה", 2653.73, 2921.96, 471, 711, "t h"),
        _ann("3 oldVer", "אשר", 2352.09, 2605.6200000000003, 471, 711),
        _BREAK,
        _ann("12 oldVer", "מפ", 2204.58, 2353.56, 711, 934),
        _ann("13 oldVer", "ני", 2116.23, 2204.57, 711, 934, "t h"),
        _ann("14 oldVer", "שיש", 1854.63, 2071.19, 711, 934),
        _BREAK,
    ]}}
    lines = annotation_page_lines(page)
    assert [[w["text"] for w in line] for line in lines] == [
        ["בשעת", "עשייה", "אשר"], ["מפני", "שיש"], []]
    merged = lines[0][1]
    assert merged["fragments"] == 2
    assert merged["sigla"] == ["t", "h"]
    assert merged["bbox"] == [2653.73, 471, 3005.5499999999997, 711]
    assert lines[0][0]["sigla"] == [] and lines[0][0]["fragments"] == 1
    assert _transcription_page_lines(page) == ["בשעת עשייה אשר", "מפני שיש"]


def test_annotation_page_lines_dots_join_the_word_they_abut():
    # "..." (illegible letters, sigla t h) directly precedes אמתי on the image.
    page = {"annotation_page": {"items": [
        _ann("19 oldVer", "...", 2938.13, 3038.13, 934, 1210, "t h"),
        _ann("20 oldVer", "אמתי", 2700.0, 2938.13, 934, 1210),
        _ann("21 oldVer", "מברך", 2400.0, 2654.6, 934, 1210),
    ]}}
    assert _transcription_page_lines(page) == ["...אמתי מברך"]


def test_annotation_page_lines_orders_fragments_by_geometry_not_served_order():
    # FL203116737 line 6: the illegible run was served after ינן but sits to
    # its right on the image, so the word reads ......ינן (RTL), and the box
    # union spans both fragments.
    page = {"annotation_page": {"items": [
        _ann("65 oldVer", "דאמ", 2293.13, 2480.36, 2637.96, 2815.61, "t h"),
        _ann("66 oldVer", "ינן", 1995.84, 2101.12, 2612, 2773.64),
        _ann("67 oldVer", "......", 2101.13, 2293.13, 2620.36, 2794.89, "t h"),
    ]}}
    lines = annotation_page_lines(page)
    # The dots also abut דאמ on their right, bridging the whole thing into
    # one word: דאמ[רי]נן.
    assert [w["text"] for w in lines[0]] == ["דאמ......ינן"]
    assert lines[0][0]["fragments"] == 3
    assert lines[0][0]["bbox"] == [1995.84, 2612, 2480.36, 2815.61]


def test_annotation_page_lines_bridges_a_run_served_at_line_end():
    # FL203116737 line 2: "...." (served last) abuts both ת and חת → בכת....חת;
    # the word keeps the served position of its first fragment.
    page = {"annotation_page": {"items": [
        _ann("11 oldVer", "בכ", 3258.31, 3362.87, 2021, 2196),
        _ann("12 oldVer", "ת", 3191.48, 3258.31, 2021, 2196, "t h"),
        _ann("13 oldVer", "חת", 2925.91, 3084.49, 2021, 2196),
        _ann("14 oldVer", "אף", 2767.98, 2887.9, 2021, 2193.69),
        _ann("31 oldVer", "....", 3084.48, 3191.48, 2021, 2196, "t h"),
    ]}}
    assert _transcription_page_lines(page) == ["בכת....חת אף"]


def test_annotation_page_lines_keeps_touching_same_sigla_words_apart():
    # Two clean words whose boxes happen to abut are not one word: NLI only
    # splits a word where its editorial status changes.
    page = {"annotation_page": {"items": [
        _ann("0 oldVer", "שלום", 3000, 3400, 100, 300),
        _ann("1 oldVer", "עליכם", 2600, 3000, 100, 300),
    ]}}
    assert _transcription_page_lines(page) == ["שלום עליכם"]


def test_annotation_page_lines_merges_a_gloss_written_down_the_margin():
    # Frag. 004r: a marginal gloss (sigla j) runs vertically; its fragments
    # share the column exactly and abut top-to-bottom. A 22px gap separates
    # the next word.
    page = {"annotation_page": {"items": [
        _ann("a", "..", 1815, 1910, 2013, 2035, "j t h"),
        _ann("b", "קיד", 1815, 1910, 2035, 2117, "j"),
        _ann("c", "..", 1815, 1910, 2117, 2139, "j t h"),
        _ann("d", "שין", 1815, 1910, 2139, 2222, "j"),
        _ann("e", "ג", 1815, 1910, 2244, 2270, "j"),
    ]}}
    lines = annotation_page_lines(page)
    assert [w["text"] for w in lines[0]] == ["..קיד..שין", "ג"]
    assert lines[0][0]["sigla"] == ["j", "t", "h"]


def test_annotation_page_lines_does_not_glue_an_interlinear_word_to_its_line():
    # A word written above the line overlaps its neighbour horizontally and
    # touches it vertically, but does not share the column — it stays a word.
    page = {"annotation_page": {"items": [
        _ann("a", "מלה", 1000, 1300, 700, 900),
        _ann("b", "תוספת", 1050, 1250, 640, 700, "s"),
    ]}}
    assert _transcription_page_lines(page) == ["מלה תוספת"]


def test_annotation_page_lines_without_geometry_falls_back_to_spaces():
    page = {"annotation_page": {"items": [
        {"id": "0 oldVer", "body": {"value": "א"}},
        {"id": "1 oldVer", "body": {"value": "ב"}},
    ]}}
    assert _transcription_page_lines(page) == ["א ב"]


def test_merged_record_carries_transcription_and_counts_as_rich():
    ktiv = {"pnx_id": "PNX_9", "sys_num": "990051173350205171", "shelf_mark": "x"}
    trans = {"990051173350205171": {"file": "t.json", "ie": "IE1",
                                    "page_count": 1,
                                    "pages": [{"fl": "FL1", "text": "שלום", "words": 1}]}}
    rec = build_merged_record("X_1", None, [], ktiv, {}, {}, trans)
    assert rec["sources"]["ktiv_transcription"]["pages"][0]["text"] == "שלום"
    assert record_is_rich(rec)


# ── KTIV images-missing backlog ───────────────────────────────────────────────

def test_ktiv_manifest_url_scraped_then_constructed():
    assert _ktiv_manifest_url(
        {"iiif_manifest_url": "https://iiif.nli.org.il/x/manifest"}
    ) == "https://iiif.nli.org.il/x/manifest"
    assert _ktiv_manifest_url({"sys_num": "990051236050205171"}) == (
        "https://iiif.nli.org.il/IIIFv21/DOCID/"
        "PNX_MANUSCRIPTS990051236050205171-1/manifest"
    )
    assert _ktiv_manifest_url({}) is None


def test_write_ktiv_backlog_emits_sorted_worklist(tmp_path):
    rows = [
        {"institution": "B-inst", "canonical_id": "B_2", "shelfmark_display": "B 2",
         "sys_num": "2", "pnx_id": "PNX_2", "iiif_manifest_url": None,
         "page_url": None},
        {"institution": "A-inst", "canonical_id": "A_1", "shelfmark_display": "A 1",
         "sys_num": "1", "pnx_id": "PNX_1", "iiif_manifest_url": "https://m",
         "page_url": "https://p"},
    ]
    section = _write_ktiv_backlog(str(tmp_path), rows)
    assert section["scraped_without_images"] == 2
    assert section["by_institution"] == {"A-inst": 1, "B-inst": 1}
    lines = [json.loads(l) for l in
             open(section["worklist_jsonl"], encoding="utf-8")]
    assert [r["canonical_id"] for r in lines] == ["A_1", "B_2"]
    assert open(section["worklist_csv"], encoding="utf-8").readline().startswith(
        "institution,canonical_id")


def test_diff_baseline_when_no_previous():
    state = {"all_ids": ["A"], "pgp_covered_ids": [], "ktiv_only_ids": [], "no_image_ids": ["A"]}
    assert diff_against_snapshot(state, None)["baseline"] is True


def test_diff_detects_new_ktiv_and_coverage_and_imaging():
    prev = {
        "all_ids": ["A", "B"],
        "pgp_covered_ids": [],
        "ktiv_only_ids": [],
        "no_image_ids": ["A", "B"],
    }
    cur = {
        "all_ids": ["A", "B", "C"],   # C is newly scraped
        "pgp_covered_ids": ["A"],     # A just gained FJP/KTIV coverage
        "ktiv_only_ids": ["C"],       # C is KTIV-only
        "no_image_ids": ["B"],        # A just gained an image
    }
    diff = diff_against_snapshot(cur, prev)
    assert diff["new_records"] == ["C"]
    assert diff["new_ktiv_only"] == ["C"]
    assert diff["newly_pgp_covered"] == ["A"]
    assert diff["newly_imaged"] == ["A"]
    assert diff["counts"]["new_records"] == 1
