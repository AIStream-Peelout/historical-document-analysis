# File name: test_build_sealed_benchmark.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the sealed confirmation-set builder (pure functions; no NAS, no network, no model)."""
import collections
import json
import types
import zipfile

import pytest

from src.datasets.evaluations.helper_eval_scripts import build_sealed_benchmark as sb
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
from src.finetuning.qwen_hebrew import build_pgp_editions as bpe

GATE = types.SimpleNamespace(loose_key=DecontamGate.loose_key, bench_keys=set(), clean_keys=set())
RECTO = ["אברהם בן יצחק שלום רב לאחינו", "ידוע יהיה לכם כי הגיע המכתב", "ושמחנו מאד בשלומכם ובשלום ביתכם"]
VERSO = ["ועתה בקשתי ממך לשלוח את הספרים", "אשר השארתי בבית הכנסת הגדול", "והשלום עליך ועל כל בני ביתך"]


def _reader(lines):
    """A reader that read exactly these lines.

    :param lines: VLM lines.
    :return: The reader.
    """
    return bpe.Reader(vlm=list(lines), rows=[], grams=bpe.grams(lines))


def _merged(pgpids_of=None, sys_of=None, records=None):
    """A small merged index.

    :param pgpids_of: ``canonical id -> pgpids``.
    :param sys_of: ``canonical id -> sys_num``.
    :param records: Full records.
    :return: The index.
    """
    pgpids_of = pgpids_of or {}
    canon = collections.defaultdict(list)
    for c, ps in pgpids_of.items():
        for p in ps:
            canon[p].append(c)
    sys_of = sys_of or {}
    by_sys = collections.defaultdict(list)
    for c, s in sys_of.items():
        by_sys[s].append(c)
    return sb.MergedIndex(pgpids_of, dict(canon), sys_of, dict(by_sys), records or {})


def test_parse_stem_reads_every_training_row_family():
    assert sb.parse_stem("pgp_446_0_page") == ("pgpid", "446")
    assert sb.parse_stem("pgpar_8317_1_page") == ("pgpid", "8317")
    assert sb.parse_stem("pgpqa_499_0_qa_date_0") == ("pgpid", "499")
    assert sb.parse_stem("vqa_pgpqa_41322_1_qa_abstain_0_w0") == ("pgpid", "41322")
    assert sb.parse_stem("vqa_446_0_parse") == ("pgpid", "446")
    assert sb.parse_stem("dg_Cambridge_CUL_T_S_13J6_27__0_loc3") == ("canonical_id", "Cambridge_CUL_T_S_13J6_27")
    assert sb.parse_stem("ktiv_990001340990205171_FL202576915_loc0") == ("sys_num", "990001340990205171")
    for other in ("01_16", "synv3_000001", "muharaf_af_295v", "agapet_sa418_0005_sa_418_23577"):
        assert sb.parse_stem(other) == ("", "")
    assert sb.stem_keys(["pgp_1_0_page", "dg_X__0_rb0", "ktiv_990001340990205171_FL1"]) == \
        ({"X"}, {"1"}, {"990001340990205171"})


def test_inventory_keeps_the_first_source_and_counts_each_source():
    inv = sb.Inventory()
    inv.add("training:a", "training", "a.json", ids=["X", "Y"], pgpids=[5, "5"])
    inv.add("benchmark:b", "benchmark", "b.json", ids=["Y", "Z"], sys_nums=["990000000000000001"])
    inv.missing("decision:c", "decision", "c.json", "absent")
    assert inv.ids == {"X": "training:a", "Y": "training:a", "Z": "benchmark:b"}
    assert inv.pgpids == {"5": "training:a"}
    assert [s["status"] for s in inv.sources] == ["read", "read", "missing"]
    assert inv.sources[0]["ids"] == 2 and inv.sources[0]["pgpids"] == 1 and inv.sources[1]["sys_nums"] == 1


def test_expand_inventory_closes_joins_and_crosses_to_ktiv():
    merged = _merged({"A": ["7"], "B": ["7"], "C": ["8"], "K": []}, sys_of={"C": "990000000000000009",
                                                                            "K": "990000000000000010"})
    inv = sb.Inventory()
    inv.add("benchmark:x", "benchmark", "x", ids=["A"])                   # its join partner B goes too
    inv.add("training:k", "training", "k", sys_nums=["990000000000000010"])  # KTIV ms -> merged record K
    inv.add("training:c", "training", "c", ids=["C"])                     # record C -> its KTIV ms
    sb.expand_inventory(inv, merged, {}, {}, {}, GATE)
    assert {"A", "B", "C", "K"} <= set(inv.ids) and inv.ids["B"] == "benchmark:x"
    assert inv.pgpids["7"] == "benchmark:x"
    assert inv.sys_nums["990000000000000009"] == "training:c"
    assert bpe.fragment_key("A", GATE) in inv.keys


def test_doc_exclusion_layers():
    merged = _merged({"P_1": ["9"], "P": ["10"]}, sys_of={"S": "990000000000000001"})
    inv = sb.Inventory()
    inv.add("training:t", "training", "t", ids=["Cambridge_CUL_T_S_12_1", "P_1"], pgpids=["3"],
            urls=["https://storage.googleapis.com/b/x.jpg"], sys_nums=["990000000000000001"])
    sb.expand_inventory(inv, merged, {}, {}, {}, GATE)
    parents = {bpe.strip_suffix(c): s for c, s in inv.ids.items() if bpe.strip_suffix(c) != c}
    check = lambda pid, cid, names=(), urls=(): sb.doc_exclusion(pid, cid, names, urls, inv, merged, GATE, parents)
    assert check("3", "Q").endswith("(pgpid)")
    assert check("4", "Cambridge_CUL_T_S_12_1").endswith("(canonical id)")
    assert check("4", "Cambridge_CUL_T_S_12_1_2").endswith("(sub-part of an excluded id)")
    assert check("4", "P_2").endswith("(sibling sub-part)")
    assert check("4", "S") == "training:t (canonical id)"           # the manuscript's merged record was expanded
    raw = sb.Inventory()                                              # unexpanded: the KTIV layer itself
    raw.add("training:k", "training", "k", sys_nums=["990000000000000001"])
    assert sb.doc_exclusion("4", "S", (), (), raw, merged, GATE, {}) == "training:k (KTIV manuscript of the record)"
    assert check("4", "Q", names=["T-S 12.1"]).endswith("(fragment key)")
    assert check("4", "Q", urls=["https://storage.googleapis.com/b/x.jpg"]).endswith("(image url)")
    assert check("4", "Q", names=["T-S 12.11"]) is None


def test_strata_and_genres():
    assert sb.language_stratum("Hebrew, Judaeo-Arabic") == "Judaeo-Arabic"
    assert sb.language_stratum("Arabic") == "Arabic"
    assert sb.language_stratum("Hebrew") == "Hebrew"
    assert sb.language_stratum("Aramaic, Hebrew") == "other" and sb.language_stratum("") == "other"
    assert [sb.type_stratum(t) for t in ("Letter", "Legal document", "List or table", "State document", "")] == \
        ["letter", "legal", "list", "other", "other"]
    assert sb.genre_of("תלמוד בבלי סדר נשים (יבמות עז ע\"ב)") == "talmud/halakha"
    assert sb.genre_of("משנה תורה (ספר זמנים)") == "talmud/halakha"
    assert sb.genre_of("מדרשי אגדה;תלמוד בבלי [טקסט]") == "midrash/aggadah"
    assert sb.genre_of("פיוט;תפילה וברכות") == "liturgy/piyyut"
    assert sb.genre_of("קטע גניזה", "Bible [Text]") == "bible/masorah"
    assert sb.genre_of("קטע גניזה") == "other"


def test_allocate_is_proportional_capped_and_exact():
    assert sb.allocate({"a": 10, "b": 30}, 100) == {"a": 10, "b": 30}
    out = sb.allocate({"a": 50, "b": 30, "c": 20}, 10)
    assert out == {"a": 5, "b": 3, "c": 2}
    out = sb.allocate({"a": 1, "b": 1, "c": 98}, 5)
    assert sum(out.values()) == 5 and out["c"] >= 4
    assert sb.allocate({}, 5) == {}


def test_seeded_order_ignores_input_order():
    items = [{"id": str(i)} for i in range(20)]
    a = sb.seeded_order(items, "id", 1)
    assert a == sb.seeded_order(list(reversed(items)), "id", 1) and a != sb.seeded_order(items, "id", 2)


def test_merged_image_urls_follow_the_index_order():
    images = {"fjp": ["a_1r.jpg", "https://storage.googleapis.com/x/b.jpg"], "preferred_source": "bodleian",
              "bodleian": {"image_urls": ["https://storage.googleapis.com/c.jpg"]},
              "ktiv": {"image_urls": ["https://storage.googleapis.com/k.jpg"]}}
    assert sb.merged_image_urls(images) == ["https://storage.googleapis.com/c.jpg", "https://storage.googleapis.com/k.jpg",
                                            "https://storage.googleapis.com/cairo-genizah-es-json/images/a_1r.jpg",
                                            "https://storage.googleapis.com/x/b.jpg"]
    assert sb.merged_image_urls(None) == []
    assert sb.is_gcs("https://storage.googleapis.com/b/x.jpg")
    assert not sb.is_gcs("https://images.lib.cam.ac.uk/iiif/x.jp2/full/4000,/0/default.jpg")
    assert not sb.is_gcs("http://storage.googleapis.com/b/x.jpg")


def test_side_evidence_assigns_the_side_the_readers_saw():
    ed = bpe.parse_edition("Recto\n" + "\n".join(RECTO) + "\n\nVerso\n" + "\n".join(VERSO))
    assert ed.labelled and sb.edition_sides(ed) == ["recto", "verso"]
    nulls = [_reader(["מה נשתנה הלילה הזה מכל הלילות"])]
    ev = sb.side_evidence(ed, _reader(RECTO), nulls)
    assert [(e.side, e.present) for e in ev] == [("recto", True), ("verso", False)] and ev[0].coverage == 1.0
    side, reason = sb.assign_side(ev)
    assert side.side == "recto" and reason == ""
    assert sb.side_answer(ed, "recto").lines == RECTO
    assert sb.assign_side(sb.side_evidence(ed, _reader(RECTO + VERSO), nulls))[1] == "several_sides_present"
    assert sb.assign_side(sb.side_evidence(ed, _reader(["שורה אחרת לגמרי שאין בה כלום"]), nulls))[1] == "no_side_present"
    unlabelled = bpe.parse_edition("\n".join(RECTO) + "\n\n" + "\n".join(VERSO))
    ev = sb.side_evidence(unlabelled, _reader(RECTO), nulls)
    assert len(ev) == 1 and ev[0].side is None and ev[0].present and ev[0].coverage == pytest.approx(0.5, abs=0.02)
    assert sb.assign_side(ev)[1] == "side_coverage_low" and sb.assign_side(ev, 0.4)[0] is not None


def test_gt_gate():
    long_text = "\n".join(RECTO + VERSO)
    assert sb.gt_gate(long_text, [frozenset()] * 6, 6) is None
    assert sb.gt_gate(long_text, [frozenset({"parentheses"})], 6) == "notation_parentheses"
    assert sb.gt_gate("\n".join(RECTO), [frozenset()] * 3, 3) == "too_few_letters"
    assert sb.gt_gate(long_text, [frozenset()] * 2, 2) == "too_few_lines"
    arabic = "\n".join(["بسم الله الرحمن الرحيم والصلاة على النبي"] * 8 + RECTO)
    assert sb.gt_gate(arabic, [frozenset()] * 11, 11) == "mostly_arabic_script"


def _doc_page(cid, language, doc_type, n_images=1):
    """A documentary page stand-in for the sampler.

    :param cid: Canonical id.
    :param language: Language stratum.
    :param doc_type: Type stratum.
    :param n_images: Images of the document.
    :return: The page.
    """
    cand = sb.DocCandidate(cid, cid, ["u"] * n_images, "", None, {}, "", language, doc_type)
    return sb.DocPage(cand, 0, None, None, None, "", "")


def test_sample_documentary_is_stratified_and_prefers_single_images():
    pages = [_doc_page(f"ja{i}", "Judaeo-Arabic", "letter", 1 + i % 2) for i in range(60)] + \
            [_doc_page(f"he{i}", "Hebrew", "legal") for i in range(20)]
    sel, table = sb.sample_documentary(pages, 40, seed=3)
    assert len(sel) == 40 and table == {"Hebrew|legal": {"pool": 20, "selected": 10},
                                        "Judaeo-Arabic|letter": {"pool": 60, "selected": 30}}
    assert all(len(p.cand.urls) == 1 for p in sel if p.cand.language == "Judaeo-Arabic")
    assert len(sb.sample_documentary(pages, 500)[0]) == 80


def _lit(sys_num, fl, genre, cols):
    """A literary page stand-in.

    :param sys_num: Manuscript.
    :param fl: Page id.
    :param genre: Genre.
    :param cols: Columns.
    :return: The page.
    """
    return sb.LitPage(sys_num, fl, "", genre, cols, "api_clean", "", ("zip", "z|m"), "", False, False)


def test_sample_literary_meets_the_multi_column_floor_and_the_manuscript_cap():
    pages = [_lit(f"9900000000000000{i % 30:02d}", f"FL{i}", "talmud/halakha" if i % 4 else "liturgy/piyyut",
                  2 if i % 3 == 0 else 1) for i in range(300)]
    sel, table = sb.sample_literary(pages, n=40, min_multi=15, max_per_ms=2, seed=1)
    assert len(sel) == 40 and sum(p.multi for p in sel) >= 15
    assert max(collections.Counter(p.sys_num for p in sel).values()) <= 2
    assert {"multi|liturgy/piyyut", "multi|talmud/halakha", "single|liturgy/piyyut", "single|talmud/halakha"} <= set(table)
    assert sel == sb.sample_literary(list(reversed(pages)), n=40, min_multi=15, max_per_ms=2, seed=1)[0]


def test_sample_literary_tops_up_a_class_the_manuscript_cap_starved():
    multi = [_lit("990000000000000001", f"FL{i}", "talmud/halakha", 2) for i in range(10)]   # one manuscript only
    single = [_lit(f"9900000000000001{i:02d}", f"FL{100 + i}", "talmud/halakha", 1) for i in range(30)]
    sel, table = sb.sample_literary(multi + single, n=12, min_multi=6, max_per_ms=2, seed=1)
    assert len(sel) == 12 and sum(p.multi for p in sel) == 2 and table["top_up"]["selected"] == 4


def test_page_image_source_prefers_the_zip_member(tmp_path):
    members = [("a.zip", "0001_FL123.jpg"), ("a.zip", "0002_FL1234.jpg")]
    assert sb.page_image_source("FL1234", members, []) == ("zip", "a.zip|0002_FL1234.jpg")
    urls = ["https://storage.googleapis.com/cairo-genizah-es-json/KTIV/99/0003_FL999.jpg"]
    assert sb.page_image_source("FL999", members, urls) == ("url", urls[0])
    assert sb.page_image_source("FL5", members, urls) is None
    zpath = tmp_path / "m.zip"
    from PIL import Image
    Image.new("RGB", (4000, 1000), "white").save(tmp_path / "p.jpg")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(tmp_path / "p.jpg", "0001_FL7.jpg")
    meta = sb.materialise_ktiv_image(("zip", f"{zpath}|0001_FL7.jpg"), tmp_path / "out.jpg", None)
    assert meta["ok"] and (meta["width"], meta["height"]) == (sb.KTIV_TARGET_WIDTH, 700)


def test_fetch_refuses_anything_but_the_bucket(tmp_path):
    with pytest.raises(ValueError):
        sb.fetch_bucket_image("https://iiif.nli.org.il/IIIFv21/FL1/full/2800,/0/default.jpg", tmp_path / "x.jpg", None)


def test_registry_entries_cover_both_halves():
    merged = _merged({"Cambridge_CUL_T_S_1": ["11"], "Cambridge_CUL_T_S_2": ["11"], "Oxford_X": []},
                     sys_of={"Cambridge_CUL_T_S_1": "990000000000000077", "Oxford_X": "990000000000000088"})
    doc = sb.DocPage(sb.DocCandidate("11", "Cambridge_CUL_T_S_1", ["u"], "", None, {}, "", "Hebrew", "letter"),
                     0, None, None, None, "", "")
    lit = _lit("990000000000000088", "FL1", "talmud/halakha", 2)
    ids, pgpids, sys_nums = sb.registry_entries([doc], [lit], merged)
    assert ids == {"Cambridge_CUL_T_S_1", "Cambridge_CUL_T_S_2", "KTIV_990000000000000088_FL1", "Oxford_X"}
    assert pgpids == {"11"} and sys_nums == {"990000000000000077", "990000000000000088"}


def test_documentary_doc_has_the_verified_format_keys():
    ed = bpe.parse_edition("\n".join(RECTO + VERSO))
    cand = sb.DocCandidate("11", "Cambridge_CUL_T_S_1", ["https://storage.googleapis.com/b/1.jpg"], "", ed,
                           {"shelfmark": "T-S 1", "languages_primary": "Hebrew", "type": "Letter"}, "Cambridge University Library",
                           "Hebrew", "letter")
    ev = sb.SideEvidence(None, 0.8, 0.1, 40, True, 1.0)
    entry = sb.documentary_doc(sb.DocPage(cand, 0, None, sb.side_answer(ed, None), ev, "m", "abc"))
    spec_keys = json.loads((sb.PGP131_DIR / "genizah_test_v1_verified.json").read_text())["docs"][0].keys() \
        if (sb.PGP131_DIR / "genizah_test_v1_verified.json").exists() else \
        {"doc_id", "shelf_mark", "institution", "image_url", "gt", "visible_chars", "gaps", "recon_frac", "_repaired_duplication"}
    assert set(spec_keys) <= set(entry)
    assert entry["pgpid"] == "11" and entry["language"] == "Hebrew" and entry["type"] == "letter" and entry["source"] == "sealed_v1"
    assert entry["gt"].splitlines() == RECTO + VERSO and entry["visible_chars"] > 100


def test_literary_doc_has_the_religious_format_keys(tmp_path):
    entry = sb.literary_doc(_lit("990000000000000088", "FL1", "talmud/halakha", 2), tmp_path)
    for key in ("doc_id", "sys_num", "fl", "shelf_mark", "genre", "is_talmud", "n_columns", "gt_shape", "gt",
                "gt_letters", "image", "image_name", "gt_mash"):
        assert key in entry
    assert entry["doc_id"] == "KTIV_990000000000000088_FL1" and entry["is_talmud"]
    assert entry["image"] == str(tmp_path / "990000000000000088_FL1.jpg")


def test_arrow_stems_reads_only_the_stem_column(tmp_path):
    import pyarrow as pa
    table = pa.table({"image": [b"\xff" * 10, b"\x00" * 10], "stem": ["ktiv_990000000000000001_FL1", "pgp_5_0_page"]})
    path = tmp_path / "data-00000-of-00001.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=1)
    assert sb.arrow_stems([path]) == ["ktiv_990000000000000001_FL1", "pgp_5_0_page"]


def test_prune_stale_images_keeps_what_the_spec_lists(tmp_path):
    for name in ("a.jpg", "b.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert sb.prune_stale_images(tmp_path, {"a.jpg"}) == ["b.jpg"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.jpg", "notes.txt"]
    assert sb.prune_stale_images(tmp_path / "missing", set()) == []
