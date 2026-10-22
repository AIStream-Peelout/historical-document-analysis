# File name: test_text27b_rag_l3.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``rag_l3``: span config, heading / running-head / label styles, the crosswalk and
page citations, span detection and resolution, the page index (kinds, tiers, evidence, leaked keys),
the E6 test-doc builder, the site source format and the ``rag_context`` row builder (page drop rules,
single-page and union leak rules, distractors, twins, length trimming, determinism).

Everything runs on a synthetic corpus, a synthetic literature tree and synthetic rows in ``tmp_path``;
no real work names or holdings are used.
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import pages as G
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rag_l3 as L
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import shelfmarks as S

SM1, SM2, SM3 = "T-S 99J1.2", "Bodl. MS heb. d 99/7", "T-S NS 999.5"
PROSE = ("This letter was sent by a merchant in the capital to his partner on the coast, and it describes "
         "the arrival of the ships, the prices of flax and pepper in the market, and the troubles of the "
         "community with the new governor of the province, which the editor discusses at length in the "
         "introduction to the volume, comparing it with the other letters of the same family archive and "
         "with the accounts that survive from the following decade in the same collection of documents.")
HEB = ["שלום עליך אחי היקר והנכבד", "אלדי כתבת אליה פי אמר אלכתאן", "ואנא אעלמך אן אלמראכב קד וצלת",
       "ולם יבק פי אלבלד שי מן אלפלפל", "ואללה יחפטך ויבקיך ויסלמך דאימא"]


def wrap(text: str, width: int = 74) -> str:
    """Break prose into lines of about ``width`` characters.

    :param text: Prose.
    :param width: Target line width.
    :returns: Wrapped text.
    """
    out, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    out.append(cur)
    return "\n".join(out)


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    """Write a CSV with a BOM, as the PGP export does.

    :param path: Destination.
    :param header: Column names.
    :param rows: Rows.
    :returns: None.
    """
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_page(path: Path, text: str, page_number: int, printed: Any = None,
               shelf_marks: Optional[Dict[str, str]] = None, cls: str = "general_academic") -> None:
    """Write one structured page JSON.

    :param path: Destination.
    :param text: ``full_main_text``.
    :param page_number: ``metadata.page_number``.
    :param printed: ``extracted_page_number``.
    :param shelf_marks: ``shelf_marks_mentioned``.
    :param cls: ``classification``.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"full_main_text": text, "metadata": {"page_number": page_number, "model_used": "gemini_3.5"},
            "classification": cls, "extracted_page_number": printed, "shelf_marks_mentioned": shelf_marks or {},
            "footnotes": {}, "transcriptions": {}}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


SPAN_CFG = {"hebrew_works": [], "works": {
    "work_docs": {"heading": "document_n_colon", "markers": "generic", "monotone": False, "max_pages": 10,
                  "pgp_source_slugs": ["slug-docs"], "location_label": "doc_n", "key_label": "doc_or_no_n"},
    "work_cites": {"pgp_source_slugs": ["slug-cites"]},
}}


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> P.Paths:
    """Synthetic config, corpus, literature tree, span config and v0_1 test rows.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch (points ``TEXT27B_PATHS`` at the synthetic config).
    :returns: Loaded :class:`Paths`.
    """
    data = tmp_path / "data"
    pgp = data / "pgp"
    (data / "reads").mkdir(parents=True)
    pgp.mkdir(parents=True)
    merged = [
        {"canonical_id": "Cambridge_CUL_T_S_99J1_2", "shelfmark_display": SM1, "pgpids": [9001], "sources_present": ["pgp"]},
        {"canonical_id": "Oxford_Bodleian_Bodl_MS_heb_d_99_7", "shelfmark_display": SM2, "pgpids": [9002],
         "sources_present": ["pgp"]},
        {"canonical_id": "Cambridge_CUL_T_S_NS_999_5", "shelfmark_display": SM3, "pgpids": [9004], "sources_present": ["pgp"]},
        {"canonical_id": "New_York_JTS_ENA_9999_1", "shelfmark_display": "ENA 9999.1", "pgpids": [], "sources_present": ["ktiv"]},
    ]
    with open(data / "merged.jsonl", "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    _write_csv(pgp / "documents.csv", ["pgpid", "shelfmark", "type", "description", "has_transcription", "has_translation"],
               [{"pgpid": "9001", "shelfmark": SM1}, {"pgpid": "9002", "shelfmark": SM2}, {"pgpid": "9004", "shelfmark": SM3}])
    _write_csv(pgp / "footnotes.csv", ["document_id", "source_slug", "location", "doc_relation", "content"],
               [{"document_id": "9001", "source_slug": "slug-docs-2001", "location": "Doc. 1, pp. 1-2", "doc_relation": "Edition"},
                {"document_id": "9002", "source_slug": "slug-cites-1999", "location": "p. 7", "doc_relation": "Discussion"}])
    _write_csv(pgp / "fragments.csv", ["shelfmark", "pgpids", "shelfmarks_historic"],
               [{"shelfmark": SM1, "pgpids": "9001"}, {"shelfmark": SM3, "pgpids": "9004"}])
    _write_csv(pgp / "sources.csv", ["slug", "title", "year", "authors"], [{"slug": "slug-docs-2001", "title": "T"}])
    lit = data / "lit"
    base = lit / "work_docs" / "stem_d" / "stem_d_structured_gemini_3.5"
    write_page(base / "page_001_structured.json",
               f"DOCUMENT 1: {SM1}\nDiscussion\n{wrap(PROSE)}\nText\n" + "\n".join(HEB), 1, [1], {SM1: "letter"})
    write_page(base / "page_002_structured.json",
               "Letters of the merchants\nTranslation\n" + wrap("I am writing to you, my brother, to inform you that the ships arrived. " * 4)
               + "\nNotes\n" + wrap(PROSE), 2, [2])
    write_page(base / "page_003_structured.json", f"DOCUMENT 2: {SM2}\nDiscussion\n{wrap(PROSE)}", 3, [3])
    write_page(base / "page_004_structured.json", f"DOCUMENT 4: {SM3}\nDiscussion\n{wrap(PROSE)}", 4, [4])
    cbase = lit / "work_cites" / "stem_c" / "stem_c_structured_gemini_3.5"
    mid = PROSE[:200] + f" The fragment {SM1} belongs to the same archive. " + PROSE[200:]
    write_page(cbase / "page_007_structured.json", wrap(mid), 7, [7],
               {SM1: "letter", "T-S A43.1": "example", "ENA 9999.1": "not in text"})
    idx_text = "\n".join([f"{SM3} 12, 14, 15"] + [f"{i}, {i + 3}, {i + 9}" for i in range(20, 34)])
    write_page(cbase / "page_008_structured.json", idx_text, 8, [8], {SM3: "index"})
    v01 = data / "v01"
    v01.mkdir()
    test_rows = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                               {"role": "assistant", "content": PROSE}],
                  "meta": {"task": "describe_pgp", "pgpid": "9002", "split": "test", "condition": "edition",
                           "held_out_vision_benchmark": True, "shelfmark": SM2}}]
    R.write_jsonl(v01 / "test.jsonl", test_rows)
    (data / "span.json").write_text(json.dumps(SPAN_CFG))
    cfg: Dict[str, Any] = {"repo": str(tmp_path), "merged_shelfmarks": "data/merged.jsonl", "pgp_export": "data/pgp",
                           "ai_reads": "data/reads", "out_dir": "out", "academic_literature": "data/lit",
                           "text_ft_v0_1": "data/v01", "rag_span_config": "data/span.json"}
    for k in ["probe_dir", "pgp_parallel", "merged_pgpid_map", "document_treatments", "analysis_pages_index",
              "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "book_editions_nas", "tokenizer_json",
              "chat_template", "scratch"]:
        cfg[k] = f"data/{k}"
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg))
    monkeypatch.setenv(P.ENV_VAR, str(f))
    L.clear_caches()
    return P.load_paths(f)


@pytest.fixture()
def corpus(env: P.Paths) -> S.Corpus:
    """The synthetic corpus.

    :param env: Synthetic configuration.
    :returns: Loaded corpus (no cache).
    """
    return S.load_corpus(env, cache=False)


# ===================================================================================== config


def test_load_span_config_and_validation(tmp_path: Path) -> None:
    """Styles resolve, booleans and markers load, unknown names raise, a missing file gives an empty config.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"hebrew_works": ["w_he"], "works": {
        "w": {"heading": "number_caps_title", "markers": "he_side", "extra_markers": [["^X$", "notes"]],
              "monotone": False, "chapter_heads": ["A"], "label_series": "roman"}}}))
    c = L.load_span_config(f)
    w = c.get("w")
    assert w is not None and w.heading == "number_caps_title" and not w.monotone and w.chapter_heads == {"A"}
    assert w.markers == L.HE_SIDE_MARKERS + [("^X$", "notes")]
    assert c.hebrew_works == {"w_he"} and set(c.span_works()) == {"w"}
    for bad in ({"heading": "nope"}, {"markers": "nope"}, {"surprise": 1}, {"label_series": "nope"}):
        f.write_text(json.dumps({"works": {"w": bad}}))
        with pytest.raises(ValueError):
            L.load_span_config(f)
    assert L.load_span_config(tmp_path / "missing.json").works == {}


def test_span_config_path_from_paths(env: P.Paths) -> None:
    """The ``rag_span_config`` entry wins; otherwise the file sits next to ``paths.json``.

    :param env: Synthetic configuration.
    """
    assert L.span_config_path(env).name == "span.json"
    assert L.load_span_config(p=env).get("work_docs") is not None


# ===================================================================================== styles


def test_heading_styles(corpus: S.Corpus) -> None:
    """Every heading layout finds its label and (where it has one) the shelfmark line.

    :param corpus: Synthetic corpus.
    """
    h = L.h_roman_volume_number(["II, 30 Letter from the coast", "Aden, 1135", SM1], corpus)
    assert [(x.label, x.sms[0]["pgpids"]) for x in h] == [("II, 30", ["9001"])]
    h = L.h_number_material(["12", SM1, "Paper, 20 x 10 cm."], corpus)
    assert [x.label for x in h] == ["12"] and h[0].sms[0]["pgpids"] == ["9001"]
    h = L.h_document_n_colon([f"DOCUMENT 3: {SM1} AND", SM2], corpus)
    assert h[0].label == "DOCUMENT 3" and sorted(m["pgpids"][0] for m in h[0].sms) == ["9001", "9002"]
    assert L.h_no_n_caps_title(["No. 5. FUSTAT, EGYPT, 1100"], corpus)[0].label == "5"
    h = L.h_number_shelfmark_dims_he(["123", SM1, 'נייר, 100X200 מ"מ'], corpus)
    assert [x.label for x in h] == ["123"]
    assert L.h_number_caps_title(["7", "A LETTER FROM THE WEST"], corpus)[0].title == "A LETTER FROM THE WEST"
    assert L.h_roman_caps_title(["XXII. A BUSINESS LETTER"], corpus)[0].label == "XXII"
    h = L.h_series_code_title(["C.1.d: A deed of sale", SM2], corpus)
    assert h[0].label == "C.1.d" and h[0].sms[0]["pgpids"] == ["9002"]
    h = L.h_number_alef_title_he(["א", "12", "שטר מכר", SM1], corpus)
    assert [x.label for x in h] == ["A-12"]
    assert L.sm_line("12", corpus) == [] and L.sm_line("this is a long sentence about the letter 12", corpus) == []


def test_runheads_labels_and_numbers() -> None:
    """Running heads, location / key label parsers, label keys and Roman numerals."""
    assert L.rh_no_n(["• No. 14", "text"]) == "14"
    assert L.rh_he_series_numbers(["כתבי גניזה מס' 334-333"]) == "333,334"
    cfg = L.WorkCfg("w", location_label="roman_dash_number", key_label="roman_comma_number", label_series="roman_comma_number")
    assert L.label_from_location(cfg, "II-30a") == "II, 30a"
    assert L.key_label(cfg, "u_II_30") == "II, 30"
    assert L.label_key(cfg, "II, 30a") == ("II", 30, "a", 30)
    assert L.label_key(L.WorkCfg("w"), "II, 30a") is None
    assert L.label_key(L.WorkCfg("w"), "X: DOCUMENTS 4-6") == ("X", 4, "", 6)
    assert L.label_key(L.WorkCfg("w", label_series="roman"), "XIV") == ("", 14, "", 14)
    assert L.key_label(L.WorkCfg("w", key_label="doc_or_no_n"), "Doc_012") == "12"
    assert L.key_label(L.WorkCfg("w", key_label="no_or_doc_roman"), "no. xxii") == "XXII"
    assert L.key_label(None, "Doc 12") is None
    for style, loc, want in (("doc_n", "Doc. 12, p. 3", "12"), ("hash_n", "#012", "12"), ("doc_hash_n", "doc #7", "7"),
                             ("doc_roman", "Doc. XIV", "XIV"), ("doc_a_n", "Doc. A-3", "A-3")):
        assert L.label_from_location(L.WorkCfg("w", location_label=style), loc) == want
    assert L.roman_to_int("XIV") == 14 and L.roman_to_int("XQ") == 0


def test_page_number_drops_unconfirmed_example_label() -> None:
    """A raw prompt-example page label survives only when the neighbour check confirms it; spreads keep both."""
    def rec(raw: Any, num: Optional[int], src: Optional[str]) -> G.PageRec:
        """Build a page record with a raw and a checked printed page.

        :param raw: ``extracted_page_number``.
        :param num: ``printed_page_num``.
        :param src: ``printed_page_source``.
        :returns: Page record.
        """
        return G.PageRec("w", "p", "page_001_structured.json", 1, "x", "s", None, None, None, "t", None, [],
                         printed_page_num=num, printed_page_source=src, data={"extracted_page_number": raw})

    assert L.page_number(rec([119], None, None)) == []
    assert L.page_number(rec([119], 119, "extracted")) == [119]
    assert L.page_number(rec([119], 23, "inferred")) == [23]
    assert L.page_number(rec([12, 13], 12, "extracted")) == [12, 13]


def test_crosswalk_and_page_citations(corpus: S.Corpus) -> None:
    """Footnote locations map labels and printed pages; historic shelfmarks map labels; page_cites can be off.

    :param corpus: Synthetic corpus.
    """
    cfg = L.load_span_config(Path("/nonexistent"))
    cfg.works = {"work_docs": L.work_cfg_from_dict("work_docs", SPAN_CFG["works"]["work_docs"]),
                 "work_cites": L.work_cfg_from_dict("work_cites", SPAN_CFG["works"]["work_cites"])}
    cw = L.build_crosswalk(corpus, cfg)
    assert cw["work_docs"]["1"] == {"9001"}
    cites = L.page_citations(corpus, cfg)
    assert cites["work_docs"][1] == {"9001"} and cites["work_docs"][2] == {"9001"} and cites["work_cites"][7] == {"9002"}
    cfg.works["work_cites"].page_cites = False
    assert "work_cites" not in L.page_citations(corpus, cfg)
    corpus.fragments.append({"shelfmark": SM2, "pgpids": ["9002"], "historic": "Old-Series XIV"})
    cfg.works["work_docs"].historic_label_re = r"Old-Series ([IVXL]+)\b"
    assert L.build_crosswalk(corpus, cfg)["work_docs"]["XIV"] == {"9002"}


# ===================================================================================== spans + index


def test_detect_resolve_and_flag_spans(env: P.Paths, corpus: S.Corpus) -> None:
    """Spans follow DOCUMENT headings across pages, resolve to pgpids and flag the numbering gap.

    :param env: Synthetic configuration.
    :param corpus: Synthetic corpus.
    """
    cfg = L.load_span_config(p=env)
    pages = G.iter_pages(root=env.academic_literature, with_ocr=False)
    spans = L.detect_spans(pages, corpus, cfg)
    # a span keeps the page on which the next heading starts (zero characters there), as in the prototype
    assert [(s.label, s.pages) for s in spans] == [
        ("DOCUMENT 1", ["page_001_structured.json", "page_002_structured.json", "page_003_structured.json"]),
        ("DOCUMENT 2", ["page_003_structured.json", "page_004_structured.json"]),
        ("DOCUMENT 4", ["page_004_structured.json"])]
    s1 = spans[0]
    assert sum(s1.per_page["page_003_structured.json"].values()) == 0
    assert s1.per_page["page_001_structured.json"]["intro"] >= 300 and s1.per_page["page_001_structured.json"]["edition"] > 0
    assert s1.per_page["page_002_structured.json"]["translation"] >= 150
    by_key = {(p.book, p.part, p.page_file): p for p in pages}
    cw = L.build_crosswalk(corpus, cfg)
    rows = [L.resolve_span(s, cw, by_key, corpus, L.page_citations(corpus, cfg), cfg) for s in spans]
    L.flag_boundaries(rows, cfg)
    assert [r["pgpids"] for r in rows] == [["9001"], ["9002"], ["9004"]]
    assert rows[0]["join_basis"] == "heading_shelfmark" and rows[0]["pgp_page_citation_agrees"] is True
    assert [r["boundary_clean"] for r in rows] == [True, False, True]


def test_resolve_span_label_crosswalk(env: P.Paths, corpus: S.Corpus) -> None:
    """A heading without a shelfmark resolves through the label crosswalk, verified by locating the PGP shelfmark.

    :param env: Synthetic configuration.
    :param corpus: Synthetic corpus.
    """
    cfg = L.load_span_config(p=env)
    pages = G.iter_pages(root=env.academic_literature, with_ocr=False)
    by_key = {(p.book, p.part, p.page_file): p for p in pages}
    sp = L.Span("work_docs", "stem_d", "1", "", [], pages=["page_001_structured.json"], page_nums=[1])
    cw = {"work_docs": {"1": {"9001"}}}
    r = L.resolve_span(sp, cw, by_key, corpus, {}, cfg)
    assert r["join_basis"] == "label_crosswalk" and r["label_crosswalk_agrees"] is True and r["pgpids"] == ["9001"]
    assert r["merged_doc_ids"] == ["Cambridge_CUL_T_S_99J1_2"] and r["shelfmarks"] == [SM1]
    sp2 = L.Span("work_docs", "stem_d", "1", "", [], pages=["page_003_structured.json"], page_nums=[3])
    assert L.resolve_span(sp2, cw, by_key, corpus, {}, cfg)["label_crosswalk_agrees"] is False
    cfg.works["work_docs"].trust_label_crosswalk = True
    r = L.resolve_span(sp2, cw, by_key, corpus, {}, cfg)
    assert r["label_crosswalk_agrees"] is None and r["shelfmarks"] == [SM1]


def test_build_index_tiers_evidence_and_leaked(env: P.Paths, corpus: S.Corpus) -> None:
    """Located commentary, clean / unclean span rows, key-only mentions, index pages and dropped prompt examples.

    :param env: Synthetic configuration.
    :param corpus: Synthetic corpus.
    """
    pages = G.iter_pages(root=env.academic_literature, with_ocr=False)
    rows, spans, stats = L.build_index(pages, corpus, L.load_span_config(p=env))
    got = {(r["page_file"], r["shelfmark_raw"]): r for r in rows}
    r = got[("page_007_structured.json", SM1)]
    assert "commentary" in r["kinds"] and (r["tier"], r["evidence"]) == ("discuss", "located")
    assert r["page_key"] == "work_cites/stem_c/page_007_structured.json" and r["printed_page"] == 7
    r = got[("page_007_structured.json", "ENA 9999.1")]
    assert (r["tier"], r["evidence"]) == ("mention", "key_only") and r["merged_doc_ids"] == ["New_York_JTS_ENA_9999_1"]
    assert ("page_007_structured.json", "T-S A43.1") not in got and stats["leaked_dropped"] == 1
    r = got[("page_008_structured.json", SM3)]
    assert r["index_page"] and r["tier"] == "mention"
    r = got[("page_001_structured.json", SM1)]
    assert r["evidence"] == "located" and r["via_span"] == "DOCUMENT 1" and r["tier"] == "discuss"
    span_rows = {r["page_file"]: r for r in rows if r["match_how"] == "span_label"}
    assert span_rows["page_002_structured.json"]["evidence"] == "clean_span"
    assert span_rows["page_003_structured.json"]["evidence"] == "unclean_span"
    assert span_rows["page_002_structured.json"]["pgpids"] == ["9001"]
    rep = L.index_report(rows, spans, stats, len(pages))
    assert rep["pages"] == 6 and rep["spans"]["total"] == 3 and rep["tiers"]["discuss"]["pgpids"] == 3
    assert rep["strong_discuss_doc_page_pairs"] == 4  # 9001 on pages 1, 2, 7; 9004 on page 4
    assert rep["resolved_strings_by_method"]["span_label"] == 3


def test_tier_and_evidence_rules() -> None:
    """Tier and evidence precedence on hand-made rows."""
    assert L.tier_of({"kinds": ["commentary"], "index_page": True}) == "mention"
    assert L.tier_of({"kinds": ["catalogue_entry", "passing_mention"]}) == "catalogue"
    assert L.tier_of({"kinds": ["edition_quote"]}) == "discuss"
    assert L.evidence_of({"located_in_text": True, "via_span": "X", "span_clean": False}) == "located"
    assert L.evidence_of({"located_in_text": False, "via_span": "X", "span_clean": True}) == "clean_span"
    assert L.evidence_of({"located_in_text": False, "via_span": None}) == "key_only"


def test_run_build_index_writes_outputs(env: P.Paths) -> None:
    """The CLI writes the index, spans, report and E6 test docs (leak share computed on the test target).

    :param env: Synthetic configuration.
    """
    rep = L.run_build_index(env)
    for f in ("rag_index.jsonl", "rag_spans.jsonl", "rag_index_report.json", "rag_test_docs.json"):
        assert env.out(f).exists()
    assert rep["rows"] == len(R.load_jsonl(env.out("rag_index.jsonl")))
    td = json.loads(env.out("rag_test_docs.json").read_text())
    assert td["source"] == "v0_1 test.jsonl" and [d["pgpid"] for d in td["docs"]] == ["9002"]
    doc = td["docs"][0]
    assert doc["held_out_vision_benchmark"] and doc["strong_pages"] == []  # page 3 is an unclean span only
    assert doc["discuss_pages"][0]["evidence"] == "unclean_span"
    assert L.run_test_docs(env)["docs_with_discuss"] == 1


def test_build_test_docs_leak_share() -> None:
    """A strong page holding the target verbatim is not leak-safe; a clean page is."""
    idx = [{"tier": "discuss", "pgpids": ["5"], "page_key": "w/p/a", "work": "w", "evidence": "located", "kinds": ["commentary"]},
           {"tier": "discuss", "pgpids": ["5"], "page_key": "w/p/b", "work": "w", "evidence": "clean_span", "kinds": ["translation"]},
           {"tier": "mention", "pgpids": ["5"], "page_key": "w/p/c", "work": "w", "evidence": "located", "kinds": ["passing_mention"]}]
    ev = [{"messages": [{}, {}, {"content": PROSE}], "meta": {"pgpid": "5", "task": "describe_pgp", "condition": "edition",
                                                              "split": "test_indist"}}]
    texts = {"w/p/a": "unrelated words " * 30, "w/p/b": "intro " + PROSE}
    out = L.build_test_docs(idx, ev, lambda k: texts.get(k, ""), source="x")
    d = out["docs"][0]
    assert d["strong_pages"] == ["w/p/a", "w/p/b"] and d["leak_share"]["describe_pgp"]["w/p/b"] == 1.0
    assert d["leak_safe_pages"] == {"describe_pgp": ["w/p/a"]} and out["counts"]["docs_any_leak_safe"] == 1


# ===================================================================================== rendering


def test_render_source_caps_and_citation() -> None:
    """Site layout: citation stub, 1,800 / 600 character caps, at most ten shelf marks, no summary line."""
    assert L.citation("A. Author", "A Title", 1999, 12) == "A. Author, *A Title*, 1999, p. 12"
    assert L.citation(None, "", None, None) == "Unknown, *Untitled*"
    text, n = L.render_source(2, "X, *Y*", "word " * 1000, [f"T-S {i}.1" for i in range(12)])
    lines = text.split("\n")
    assert lines[0] == "[SOURCE 2] X, *Y*" and n == L.QUOTE_CAP
    assert lines[1].startswith("Original page text (quoteable): word")
    assert lines[-1].count(",") == 9 and "Generated catalog summary" not in text
    _, n_he = L.render_source(1, "c", "שלום עליך " * 400, [])
    assert n_he == L.HEBREW_QUOTE_CAP == 600
    block, chars = L.render_block([("c1", "abc", []), ("c2", "def", ["T-S 1.1"])])
    assert block.startswith("Literature:\n[SOURCE 1] c1\nOriginal page text (quoteable): abc\n[SOURCE 2] c2")
    assert chars == [3, 3] and L.hebrew_char_ratio("abc שלום") == pytest.approx(4 / 7)


def test_literature_manifest_shapes(tmp_path: Path) -> None:
    """List, ``{"works": [...]}`` and keyed shapes load; part overrides win; a missing file gives ``{}``.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "m.json"
    for raw in ([{"work_dir": "w", "author": "A", "title": "T", "year": 2000}],
                {"works": [{"work_dir": "w", "author": "A", "title": "T", "year": 2000}]},
                {"w": {"author": "A", "title": "T", "year": 2000}}):
        f.write_text(json.dumps(raw))
        assert L.work_meta(L.load_literature_manifest(path=f), "w") == ("A", "T", 2000)
    f.write_text(json.dumps({"w": {"author": "A", "title": "T", "parts": {"p2": {"title": "T2"}}}}))
    man = L.load_literature_manifest(path=f)
    assert L.work_meta(man, "w", "p2") == ("A", "T2", None)
    assert L.load_literature_manifest(path=tmp_path / "none.json") == {}


def test_nontrain_and_quality_adapters() -> None:
    """Direct sets, a holdouts-like object and the quality-result coercion."""
    class FakeHoldouts:
        """Stand-in for ``holdouts.Holdouts``."""

        def is_benchmark(self, pgpid: Optional[str], shelfmark: Optional[str], canonical: Optional[str]) -> Optional[str]:
            """Benchmark hit for one canonical id.

            :param pgpid: pgpid.
            :param shelfmark: Shelfmark display.
            :param canonical: Canonical id.
            :returns: Reason or None.
            """
            return "pgp131" if canonical == "C_X" else None

        def is_eval_doc(self, pgpid: Optional[str], shelfmark: Optional[str], canonical: Optional[str]) -> Optional[str]:
            """Eval-doc hit for one pgpid.

            :param pgpid: pgpid.
            :param shelfmark: Shelfmark display.
            :param canonical: Canonical id.
            :returns: Reason or None.
            """
            return "dev" if pgpid == "77" else None

    nt = L.NonTrain(pgpids={"1"}, canonical_ids={"C_Y"}, holdouts=FakeHoldouts())
    assert nt.reason(["1"], []) == "nontrain_pgpid" and nt.reason([], ["C_Y"]) == "nontrain_canonical"
    assert nt.reason([], ["C_X"]) == "is_benchmark:pgp131" and nt.reason(["77"], []) == "is_eval_doc:dev"
    assert nt.reason(["2"], ["C_Z"]) is None
    page = object()
    assert L._quality_pass(None, page) and not L._quality_pass(lambda _: (False, ["short"]), page)
    assert L._quality_pass(lambda _: {"ok": True}, page) and not L._quality_pass(lambda _: False, page)


# ===================================================================================== build_rows


def _page(key: str, text: str, printed: Optional[int] = 5, marks: Optional[List[str]] = None) -> G.PageRec:
    """A page record for a page key.

    :param key: ``work/part/page_file``.
    :param text: Page text.
    :param printed: Printed page.
    :param marks: ``shelf_marks_mentioned`` keys.
    :returns: Page record.
    """
    w, part, pf = G.parse_page_key(key)
    return G.PageRec(w, part, pf, 1, "x", "s", None, None, None, text, None, list(marks or []),
                     printed_page_num=printed, printed_page_source="extracted", data={})


def _idx(key: str, pgpids: List[str], evidence: str = "located", tier: str = "discuss") -> Dict[str, Any]:
    """An index row.

    :param key: Page key.
    :param pgpids: Documents.
    :param evidence: Evidence label.
    :param tier: Tier.
    :returns: Row.
    """
    return {"page_key": key, "work": key.split("/")[0], "pgpids": pgpids, "merged_doc_ids": [], "tier": tier,
            "evidence": evidence, "kinds": ["commentary"]}


def _base(pgpid: str, task: str, condition: str, target: str, read_model: Optional[str] = None) -> Dict[str, Any]:
    """A v0_2-style train row with every v0_1 meta field.

    :param pgpid: Document.
    :param task: Family.
    :param condition: Input condition.
    :param target: Assistant target.
    :param read_model: Reader.
    :returns: Row.
    """
    meta = R.new_meta(task, condition, "train", pgpid, read_model, lang_class="JA", pgpid=pgpid, shelfmark="X 1",
                      shelfmark_group=f"g{pgpid}", held_out_vision_benchmark=False, target_chars=len(target),
                      target_tokens=10, tokens=100, input_cer=None, input_source="s", input_truncated=False,
                      licence="l", provenance={}, source="s")
    user = R.user_prompt(task, condition, "hebrew", "אלכתאב")
    return {"messages": R.make_messages(user, target), "meta": meta}


FILLER = " ".join(f"w{i}" for i in range(400))


def _target(tag: str) -> str:
    """A 100-word target with words unique to ``tag``.

    :param tag: Prefix.
    :returns: Text.
    """
    return " ".join(f"{tag}{i}" for i in range(100))


@pytest.fixture()
def rag_inputs() -> L.RagInputs:
    """Doc 1 (train) with good, leaking, unclean, non-train-mentioning and low-quality pages; doc 2 (train) elsewhere.

    :returns: Inputs.
    """
    t1 = _target("a")
    t1_words = t1.split()
    pages = {
        "w1/p/good": _page("w1/p/good", FILLER + " T-S 99J1.2", marks=["T-S 99J1.2", "T-S A43.1"]),
        "w1/p/span": _page("w1/p/span", FILLER + " more"),
        "w1/p/leak": _page("w1/p/leak", "intro " + t1),
        "w1/p/unclean": _page("w1/p/unclean", FILLER),
        "w1/p/mixed": _page("w1/p/mixed", FILLER),
        "w1/p/poor": _page("w1/p/poor", "POOR " + FILLER),
        "w1/p/part1": _page("w1/p/part1", " ".join(t1_words[:11]) + " " + FILLER),
        "w1/p/part2": _page("w1/p/part2", " ".join(t1_words[50:61]) + " " + FILLER),
        "w2/p/other": _page("w2/p/other", "other " + FILLER),
    }
    idx = [_idx("w1/p/good", ["1"]), _idx("w1/p/span", ["1"], "clean_span"), _idx("w1/p/leak", ["1"]),
           _idx("w1/p/unclean", ["1"], "unclean_span"), _idx("w1/p/mixed", ["1"]), _idx("w1/p/mixed", ["99"], tier="mention"),
           _idx("w1/p/poor", ["1"]), _idx("w2/p/other", ["2"])]
    base = [_base("1", "catalogue_json", "edition", t1), _base("1", "describe_pgp", "edition", t1),
            _base("1", "extract_relations", "read", t1, read_model="r1"), _base("1", "extract_relations", "noised_edition", t1),
            _base("2", "describe_pgp", "edition", _target("b")), _base("3", "describe_pgp", "edition", _target("c"))]
    return L.RagInputs(index_rows=idx, pages=pages, base_rows=base, nontrain=L.NonTrain(pgpids={"99"}),
                       manifest={"w1": {"author": "A. Author", "title": "Work One", "year": 1990}},
                       quality_fn=lambda pg: "POOR" not in pg.text)


def _count(row: Dict[str, Any]) -> int:
    """Cheap token count for tests (characters / 3).

    :param row: Row.
    :returns: Tokens.
    """
    return sum(len(m["content"]) for m in row["messages"]) // 3


def test_build_rows_page_rules_and_schema(rag_inputs: L.RagInputs) -> None:
    """Only strong, allowed, leak-safe pages are joined; one row per document group; schema and twins hold.

    :param rag_inputs: Synthetic inputs.
    """
    rows, man = L.build_rows(rag_inputs, distractor_rate=0.0, token_fn=_count)
    by_task = {r["meta"]["task"]: r for r in rows}
    assert set(by_task) <= {"catalogue_json", "describe_pgp", "extract_relations"} and len(rows) == 3
    assert sum(t in by_task for t in ("catalogue_json", "describe_pgp")) == 1  # B5: one row per document for the pair
    assert "describe_pgp" in by_task and by_task["describe_pgp"]["meta"]["pgpid"] == "2"
    for r in rows:
        assert R.validate_row(r) == []
        m = r["meta"]
        assert m["family"] == "rag_context" and m["condition"] == "literature" and m["split"] == "train"
        keys = {s["page_key"] for s in m["lit_sources"]}
        assert not keys & {"w1/p/leak", "w1/p/unclean", "w1/p/mixed", "w1/p/poor"}
        user = r["messages"][1]["content"]
        assert "\n\nLiterature:\n[SOURCE 1] " in user and r["messages"][2]["content"] == r["messages"][2]["content"].strip()
    rel = by_task["extract_relations"]["meta"]
    assert rel["input_condition"] == "read" and rel["read_model"] == "r1"
    twin = next(b for b in rag_inputs.base_rows if b["meta"]["task"] == "extract_relations" and b["meta"]["condition"] == "read")
    assert rel["twin_row_id"] == twin["meta"]["row_id"] and rel["row_id"] != twin["meta"]["row_id"]
    assert len({r["meta"]["row_id"] for r in rows}) == 3
    doc1 = [r for r in rows if r["meta"]["pgpid"] == "1"][0]
    assert "A. Author, *Work One*, 1990, p. 5" in doc1["messages"][1]["content"]
    assert "Shelf marks cited in this source: T-S 99J1.2" in doc1["messages"][1]["content"]
    assert "T-S A43.1" not in doc1["messages"][1]["content"]  # prompt-example key not located in the text
    assert man["page_drops"]["by_reason"] == {"nontrain_pgpid": 1, "quality": 1}
    assert man["stats"]["leak_drop_page"] >= 1 and man["rows"] == 3 and man["by_task"]["extract_relations"]["docs"] == 1


def test_build_rows_union_leak_rule(rag_inputs: L.RagInputs) -> None:
    """Two pages each under 5 % of the target's 8-grams but together over it: only one is kept.

    :param rag_inputs: Synthetic inputs.
    """
    rag_inputs.index_rows += [_idx("w1/p/part1", ["1"]), _idx("w1/p/part2", ["1"])]
    rag_inputs.index_rows = [r for r in rag_inputs.index_rows if r["page_key"] not in ("w1/p/good", "w1/p/span")]
    rows, man = L.build_rows(rag_inputs, distractor_rate=0.0, token_fn=_count)
    for r in rows:
        if r["meta"]["pgpid"] == "1":
            keys = [s["page_key"] for s in r["meta"]["lit_sources"]]
            assert len(keys) == 1 and keys[0] in ("w1/p/part1", "w1/p/part2")
            assert r["meta"]["lit_leak_share"] < 0.05
    assert man["stats"]["leak_drop_union"] >= 1


def test_build_rows_distractors_and_determinism(rag_inputs: L.RagInputs) -> None:
    """Distractors come from another work and another document; the seeded build is reproducible.

    :param rag_inputs: Synthetic inputs.
    """
    rows, man = L.build_rows(rag_inputs, distractor_rate=1.0, token_fn=_count)
    mentions: Dict[str, set] = {}
    for x in rag_inputs.index_rows:
        mentions.setdefault(x["page_key"], set()).update(x["pgpids"])
    for r in rows:
        dis = [s for s in r["meta"]["lit_sources"] if s["distractor"]]
        own = [s for s in r["meta"]["lit_sources"] if not s["distractor"]]
        assert len(dis) == 1 and dis[0]["work"] not in {s["work"] for s in own}
        assert r["meta"]["pgpid"] not in mentions[dis[0]["page_key"]]
        if r["meta"]["pgpid"] == "1":
            assert dis[0]["page_key"] == "w2/p/other"
    assert man["stats"].get("distractor_not_found", 0) == 0
    assert sum(any(s["distractor"] for s in r["meta"]["lit_sources"]) for r in rows) == len(rows)
    again, man2 = L.build_rows(rag_inputs, distractor_rate=1.0, token_fn=_count)
    assert man2["fingerprint"] == man["fingerprint"]


def test_build_rows_length_trim_and_skip(rag_inputs: L.RagInputs) -> None:
    """Rows over the cap lose own sources until they fit; a row that cannot fit is skipped.

    :param rag_inputs: Synthetic inputs.
    """
    def by_sources(row: Dict[str, Any]) -> int:
        """Tokens grow with the number of sources.

        :param row: Row.
        :returns: Tokens.
        """
        return 100 * row["messages"][1]["content"].count("[SOURCE ")

    rows, man = L.build_rows(rag_inputs, distractor_rate=0.0, token_fn=by_sources, max_tokens=100)
    assert rows and all(len(r["meta"]["lit_sources"]) == 1 and r["meta"]["tokens"] == 100 for r in rows)
    assert man["stats"]["sources_trimmed_length"] >= 1
    rows, man = L.build_rows(rag_inputs, distractor_rate=0.0, token_fn=by_sources, max_tokens=50)
    assert rows == [] and man["stats"]["rows_skipped_length"] == 3


def test_build_rows_excluded_pages_and_analyse_span(rag_inputs: L.RagInputs) -> None:
    """``excluded_pages.json`` keys never join; analyse rows skip the document's own span pages.

    :param rag_inputs: Synthetic inputs.
    """
    rag_inputs.excluded = {"w1/p/good": ["eval_mention_window"]}
    rag_inputs.base_rows.append(_base("1", "analyse_document", "edition", _target("a")))
    rag_inputs.span_pages = {"1": {"w1/p/span"}}
    rows, man = L.build_rows(rag_inputs, distractor_rate=0.0, token_fn=_count,
                             families=L.RAG_FAMILIES + ("analyse_document",))
    keys = {s["page_key"] for r in rows for s in r["meta"]["lit_sources"]}
    assert "w1/p/good" not in keys and man["page_drops"]["by_reason"]["excluded_pages"] == 1
    assert not any(r["meta"]["task"] == "analyse_document" for r in rows)
    assert man["stats"]["groups_no_safe_page"] >= 1


def test_run_build_rows_end_to_end(env: P.Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase-2 CLI on the synthetic tree: index → families → ``rag_context.jsonl`` + manifest.

    :param env: Synthetic configuration.
    :param monkeypatch: Pytest monkeypatch (token count, sibling modules).
    """
    L.run_build_index(env, test_docs=False)
    fam = env.out("families", "describe_pgp.jsonl", mkdir=True)
    R.write_jsonl(fam, [_base("9001", "describe_pgp", "edition", _target("q")),
                        {**_base("9004", "describe_pgp", "edition", _target("r")),
                         "meta": {**_base("9004", "describe_pgp", "edition", "x")["meta"], "split": "dev"}}])
    monkeypatch.setattr(L._rows, "tokens_of_row", lambda row, *a, **k: 500)
    monkeypatch.setattr(L, "_load_holdouts", lambda *a, **k: None)
    monkeypatch.setattr(L, "_load_quality_fn", lambda *a, **k: None)
    man = L.run_build_rows(env, distractor_rate=0.0)
    out = R.load_jsonl(env.out("families", "rag_context.jsonl"))
    assert man["rows"] == len(out) == 1 and out[0]["meta"]["pgpid"] == "9001"
    keys = [s["page_key"] for s in out[0]["meta"]["lit_sources"]]
    assert "work_docs/stem_d/page_003_structured.json" not in keys  # unclean span
    assert man["inputs"]["family_files"] == ["describe_pgp.jsonl"]
    assert json.loads(env.out("families", "rag_context_manifest.json").read_text())["rows"] == 1
    # D5: a base row listed in excluded_rows.json is never a base (the document then has no rag row)
    (env.out_dir / "excluded_rows.json").write_text(json.dumps({out[0]["meta"]["twin_row_id"]: "ngram_target:x#w8"}))
    man2 = L.run_build_rows(env, distractor_rate=0.0)
    assert man2["rows"] == 0 and man2["inputs"]["base_rows_excluded"] == 1 and man2["inputs"]["excluded_rows_listed"] == 1
    assert L.load_excluded_rows(env.out_dir / "none.json") == set()


# ===================================================================================== adapters


def test_rag_excluded_pages_ignores_l1_only_reasons() -> None:
    """Pages excluded only by the L1 train-target dedupe stay usable as RAG context; any other reason binds."""
    ex = {"a/b/p1": ["train_target_dup:translate_en:1"], "a/b/p2": ["train_target_dup:x:2", "ngram:tgt:y#w8"],
          "a/b/p3": ["window:mention:rag:test"]}
    assert L.rag_excluded_pages(ex) == {"a/b/p2": ex["a/b/p2"], "a/b/p3": ex["a/b/p3"]}


def test_load_excluded_pages_shapes(tmp_path: Path) -> None:
    """Plain mapping, ``pages`` / ``excluded`` wrappers and a bare key list load; a missing file gives ``{}``.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "x.json"
    for raw, want in (({"a/b/c": ["held"]}, {"a/b/c": ["held"]}), ({"pages": {"a/b/c": "held"}}, {"a/b/c": ["held"]}),
                      ({"excluded": ["a/b/c"]}, {"a/b/c": ["listed"]}), (["a/b/c"], {"a/b/c": ["listed"]})):
        f.write_text(json.dumps(raw))
        assert L.load_excluded_pages(f) == want
    assert L.load_excluded_pages(tmp_path / "none.json") == {}


def test_eval_rows_source_switch(env: P.Paths) -> None:
    """v0_1 test rows until every E6 family file exists; then the v0_2 eval splits only.

    :param env: Synthetic configuration.
    """
    src, rows = L._iter_eval_rows(env)
    assert src == "v0_1 test.jsonl" and len(list(rows)) == 1
    for fam in L.E6_FAMILIES[:-1]:
        R.write_jsonl(env.out("families", f"{fam}.jsonl", mkdir=True), [])
    assert L._iter_eval_rows(env)[0] == "v0_1 test.jsonl"
    rows_v2 = [{**_base("9002", "translate_en", "edition", "t"), "meta": {**_base("9002", "translate_en", "edition", "t")["meta"],
                                                                          "split": split}}
               for split in ("train", "dev", "test_indist", "heldout_vision", "heldout")]
    R.write_jsonl(env.out("families", "translate_en.jsonl"), rows_v2)
    src, rows = L._iter_eval_rows(env)
    assert src.startswith("v0_2") and sorted(r["meta"]["split"] for r in rows) == ["heldout", "heldout_vision", "test_indist"]


def test_optional_sibling_adapters(env: P.Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """The quality closure passes each part's running heads; a missing ``holdouts.json`` gives None.

    :param env: Synthetic configuration.
    :param monkeypatch: Pytest monkeypatch (a fake ``literature_l1``).
    """
    import sys
    import types

    import src.finetuning.text27b as pkg

    seen: List[Any] = []
    fake = types.ModuleType("src.finetuning.text27b.literature_l1")

    def quality_ok(page: Any, heads: Any = None) -> bool:
        """Record the call.

        :param page: Page.
        :param heads: Running heads.
        :returns: True.
        """
        seen.append(heads)
        return True

    def running_heads_by_part(pages: Any) -> Dict[Any, Any]:
        """One head set per part.

        :param pages: Pages.
        :returns: ``{(work, part): heads}``.
        """
        return {(p.work, p.part): frozenset({f"head of {p.part}"}) for p in pages}

    fake.quality_ok = quality_ok  # type: ignore[attr-defined]
    fake.running_heads_by_part = running_heads_by_part  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.finetuning.text27b.literature_l1", fake)
    monkeypatch.setattr(pkg, "literature_l1", fake, raising=False)
    pages = G.iter_pages(root=env.academic_literature, with_ocr=False)
    fn = L._load_quality_fn(pages)
    assert fn is not None and L._quality_pass(fn, pages[0]) and seen == [frozenset({f"head of {pages[0].part}"})]
    assert L._load_holdouts(env) is None


def test_block_check_uses_contamination_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows whose literature block fails ``contamination.block_ok`` are dropped.

    :param monkeypatch: Pytest monkeypatch (a fake ``contamination``).
    """
    import sys
    import types

    import src.finetuning.text27b as pkg

    fake = types.ModuleType("src.finetuning.text27b.contamination")
    fake.block_ok = lambda target, block: target not in block  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.finetuning.text27b.contamination", fake)
    monkeypatch.setattr(pkg, "contamination", fake, raising=False)
    ok = {"messages": R.make_messages("doc\n\nLiterature:\n[SOURCE 1] c\nOriginal page text (quoteable): other", "target")}
    bad = {"messages": R.make_messages("doc\n\nLiterature:\n[SOURCE 1] c\nOriginal page text (quoteable): target", "target")}
    kept, dropped = L._block_check([ok, bad])
    assert kept == [ok] and dropped == 1
