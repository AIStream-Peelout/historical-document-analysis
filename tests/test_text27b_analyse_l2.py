# File name: test_text27b_analyse_l2.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for :mod:`src.finetuning.text27b.analyse_l2` (synthetic documents, spans, entries and hold-outs only)."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.text27b import analyse_l2 as L
from src.finetuning.text27b import holdouts as H
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import shelfmarks as SM

EDITION = "\n".join(["אלשיך זבריק כתב אלי קזואר פי אמר אלפלפל", "ואלסלאם עליך ורחמה אללה"] * 4)
SM_TRAIN = "T-S 10J5.12"
SM_BENCH = "T-S 13J1.5"
SM_EVAL = "T-S 8J20.3"
SM_NEW = "T-S 24.77"


def ws_tokens() -> L.TokenFns:
    """Whitespace token counters (no tokenizer needed).

    :returns: Counters.
    """
    def render_count(msgs: Sequence[Dict[str, Any]]) -> int:
        """Rendered length: words plus four tokens of scaffold per turn.

        :param msgs: Chat turns.
        :returns: Token count.
        """
        return sum(len(m["content"].split()) + 4 for m in msgs)

    return L.TokenFns(count=lambda s: len(s.split()), render_count=render_count)


def small_cfg(**kw: Any) -> L.L2Config:
    """A config with thresholds scaled for short synthetic texts.

    :param kw: Overrides.
    :returns: Config.
    """
    base = dict(min_commentary_chars=40, min_target_chars=60, min_target_after_strip=40, min_book_edition_chars=20,
                min_page_chars=20, max_row_tokens=6144)
    base.update(kw)
    return L.L2Config(**base)


def holdouts() -> H.Holdouts:
    """A hold-out set: one benchmark document (pgpid + key), one eval document.

    :returns: Hold-outs.
    """
    return H.Holdouts.from_dict({
        "benchmark": {"900": ["pgp131"]}, "eval_docs": {"800": ["test"]},
        "canonical": {}, "eval_canonical": {},
        "keys": {SM.key(SM_BENCH): ["pgp131"]}, "eval_keys": {SM.key(SM_EVAL): ["test"]}})


def doc(pgpid: str, split: str, shelfmark: str, edition: Optional[str] = EDITION,
        catalogue: Optional[Dict[str, Any]] = None, variants: Optional[List[L.Variant]] = None) -> L.DocInfo:
    """A builder document.

    :param pgpid: PGP id.
    :param split: Split.
    :param shelfmark: Shelfmark.
    :param edition: Edition text (None: no edition).
    :param catalogue: Catalogue record.
    :param variants: Input variants.
    :returns: The document.
    """
    return L.DocInfo(pgpid=pgpid, split=split, group=f"g{pgpid}", lang_class="JA", shelfmark=shelfmark,
                     held_out_vision_benchmark=False, holdout_reasons=[], script="hebrew", edition=edition,
                     edition_row=edition is not None, variants=variants or [], catalogue=catalogue)


def index(*docs: L.DocInfo) -> L.DocIndex:
    """A document index over the given documents.

    :param docs: Documents.
    :returns: Index.
    """
    ix = L.DocIndex()
    for d in docs:
        ix.by_pgpid[d.pgpid] = d
        for k in SM.keys(d.shelfmark):
            ix.by_key[k].add(d.pgpid)
    return ix


def cand(kind: str, label: str, intro: Any, notes: Any = "", pgpids: Sequence[str] = (), shelfmarks: Sequence[str] = (),
         edition_text: Optional[str] = None, edition_pgpid: Optional[str] = None, work: str = "work_a",
         hints: Optional[Dict[str, str]] = None, page_keys: Sequence[str] = ()) -> L.Candidate:
    """A candidate.

    :param kind: Source kind.
    :param label: Label.
    :param intro: Intro lines (spans) or text (entries).
    :param notes: Notes lines or text.
    :param pgpids: PGP ids.
    :param shelfmarks: Shelfmarks.
    :param edition_text: Book edition text.
    :param edition_pgpid: PGP-edition document.
    :param work: Work / book slug.
    :param hints: Heading hints.
    :param page_keys: Span page keys.
    :returns: Candidate.
    """
    return L.Candidate(kind=kind, source=f"book:{work}", work=work, part="p", label=label, title="", pgpids=list(pgpids),
                       merged_ids=[], shelfmarks=list(shelfmarks), intro=intro, notes=notes, edition_text=edition_text,
                       script="hebrew", lang_hint="JA", page_keys=list(page_keys), hints=hints or {}, licence="test",
                       licence_class="unknown", provenance={"book": work, "doc_label": label}, edition_pgpid=edition_pgpid)


CAT = {"document_type": "Letter", "date": None, "origin": None, "destination": None, "places_mentioned": [],
       "summary": "Letter from Zibriq to Qazwar about pepper."}
GOOD_INTRO = ["This letter was written by the merchant Zibriq to his partner Qazwar at the end of the",
              "season and it reports on a shipment of pepper and the price obtained for it in the market."]


# ============================================================================== splits
def test_hash_split_is_deterministic_and_bucketed() -> None:
    """The D14 hash split is a pure function of the key with 80/10/10 buckets."""
    keys = [f"key{i}" for i in range(2000)]
    splits = collections.Counter(L.hash_split(k) for k in keys)
    assert L.hash_split("abc") == L.hash_split("abc")
    assert set(splits) == {"train", "dev", "test_indist"}
    assert 0.75 < splits["train"] / 2000 < 0.85
    assert 0.07 < splits["dev"] / 2000 < 0.13


def test_hash_split_agrees_with_book_pairs() -> None:
    """L2 and L5 put the same document key into the same split (D14 consistency)."""
    bp = pytest.importorskip("src.finetuning.text27b.book_pairs")
    if not hasattr(bp, "hash_split"):
        pytest.skip("book_pairs has no hash_split")
    for k in ("ts10j5_12", "mshebc72_18", "x1", "x2", "book:a#3"):
        assert L.hash_split(k) == bp.hash_split(k)


def test_assign_split_order() -> None:
    """Benchmark first, then the builder split, eval tier, shelfmark-key siblings, then the hash."""
    ho = holdouts()
    docs = index(doc("101", "train", SM_TRAIN), doc("102", "dev", "T-S 6J1.1"), doc("900", "train", "T-S 1.1"))
    bench = L.assign_split(cand(L.SPAN_PGP, "1", [], pgpids=["900"], edition_pgpid="900"), docs, ho, "k")
    assert (bench.split, bench.reason.split(":")[0]) == ("heldout", "benchmark") and bench.held_out_vision_benchmark
    by_key = L.assign_split(cand(L.SPAN_BOOK, "2", [], shelfmarks=[SM_BENCH]), docs, ho, "k")
    assert by_key.split == "heldout" and "pgp131" in by_key.reason
    assert L.assign_split(cand(L.SPAN_PGP, "3", [], pgpids=["101"], edition_pgpid="101"), docs, ho, "k").split == "train"
    assert L.assign_split(cand(L.SPAN_PGP, "4", [], pgpids=["102"], edition_pgpid="102"), docs, ho, "k").split == "dev"
    ev = L.assign_split(cand(L.SPAN_BOOK, "5", [], shelfmarks=[SM_EVAL]), docs, ho, "k")
    assert ev.split == "heldout" and ev.reason.startswith("eval:")
    sib = L.assign_split(cand(L.NAS, "6", "", shelfmarks=[SM_TRAIN]), docs, ho, "k")
    assert (sib.split, sib.reason) == ("train", "shelfmark_key:train")
    sib_dev = L.assign_split(cand(L.NAS, "7", "", shelfmarks=["T-S 6J1.1"]), docs, ho, "k")
    assert sib_dev.split == "heldout"
    new = L.assign_split(cand(L.NAS, "8", "", shelfmarks=[SM_NEW]), docs, ho, SM.key(SM_NEW))
    assert (new.split, new.reason) == (L.hash_split(SM.key(SM_NEW)), "hash")


def test_doc_key_prefers_shelfmark_key() -> None:
    """The document key is the shelfmark key, else source#label."""
    assert L.doc_key_of(cand(L.NAS, "3", "", shelfmarks=[SM_NEW]), None) == SM.key(SM_NEW)
    assert L.doc_key_of(cand(L.NAS, "3", ""), None) == "book:work_a#3"
    assert L.doc_key_of(cand(L.NAS, "3", ""), doc("1", "train", SM_TRAIN)) == SM.key(SM_TRAIN)


# ============================================================================== document index
def _family_row(family: str, condition: str, pgpid: str, split: str, text: str, target: str, **meta: Any) -> Dict[str, Any]:
    """A builder-style family row.

    :param family: Family.
    :param condition: Condition.
    :param pgpid: PGP id.
    :param split: Split.
    :param text: Document text.
    :param target: Target.
    :param meta: Further meta.
    :returns: Row.
    """
    m = {"family": family, "task": family, "condition": condition, "pgpid": pgpid, "split": split, "lang_class": "JA",
         "shelfmark": SM_TRAIN, "shelfmark_group": "g" + pgpid, "held_out_vision_benchmark": False,
         "holdout_reasons": [], **meta}
    return {"messages": R.make_messages(R.user_prompt(family, condition, "hebrew", text), target), "meta": m}


def test_load_doc_index(tmp_path: Path) -> None:
    """Editions (untruncated preferred), variants, catalogue, script and the want filter."""
    fam = tmp_path / "families"
    R.write_jsonl(fam / "catalogue_json.jsonl", [
        _family_row("catalogue_json", "edition", "101", "train", "short [...]", json.dumps(CAT), input_truncated=True,
                    input_source="pgp_digital_edition:x"),
        _family_row("catalogue_json", "read", "101", "train", "a read", json.dumps(CAT), read_model="a50", input_cer=0.3,
                    reader_exposed=False, exposure_level="none"),
        _family_row("catalogue_json", "edition", "202", "dev", EDITION, json.dumps(CAT))])
    R.write_jsonl(fam / "translate_en.jsonl", [
        _family_row("translate_en", "edition", "101", "train", EDITION, "a translation"),
        _family_row("translate_en", "edition", "303", "train", "side text", "t", side="r")])
    ix = L.load_doc_index(fam, want={"101"})
    d = ix.get("101")
    assert d is not None and d.edition == EDITION and not d.edition_truncated
    assert [v.condition for v in d.variants] == ["edition", "read"]
    assert d.variants[1].meta["read_model"] == "a50" and d.variants[1].text == "a read"
    assert d.catalogue == CAT and d.script == "hebrew" and d.split == "train"
    other = ix.get("202")
    assert other is not None and other.edition is None and other.has_edition
    assert not ix.get("303").has_edition
    assert ix.by_shelfmarks([SM_TRAIN]) >= {"101"}


# ============================================================================== spans and entries
def test_attach_sections_checks_alignment() -> None:
    """Full sections are attached only when labels and section sizes agree."""
    row = {"book": "w", "part": "p", "doc_label": "1", "intro_chars": 5, "notes_chars": 0, "translation_chars": 0,
           "edition_chars": 0}
    sp = SimpleNamespace(book="w", part="p", label="1", samples={"intro": ["abcde"]})
    out = L.attach_sections([row], [sp])
    assert out[0]["sections"] == {"intro": ["abcde"]}
    with pytest.raises(ValueError):
        L.attach_sections([dict(row, intro_chars=6)], [sp])
    with pytest.raises(ValueError):
        L.attach_sections([row], [])


def test_uncapped_detector_runs() -> None:
    """The patched detector compiles and runs (no pages → no spans)."""
    assert L.detect_full_spans([], None, SimpleNamespace(span_works=lambda: {})) == []


def test_clean_book_edition() -> None:
    """Line numbers, colons and Latin lines go; Hebrew lines stay."""
    text = L.clean_book_edition(["1 אלשיך זבריק", ": כתב אלי 2", "Translation follows", "  ", "ואלסלאם"])
    assert text.split("\n") == ["אלשיך זבריק", "כתב אלי", "ואלסלאם"]


def test_render_transcription_labels() -> None:
    """Side / part labels appear as § lines when the entry has several groups."""
    lines = [{"n": 1, "side": "recto", "part": None, "text": "א ב"}, {"n": 2, "side": "recto", "part": None, "text": "ג"},
             {"n": 1, "side": "verso", "part": "address", "text": "ד"}]
    assert L.render_transcription(lines) == "§ Recto\nא ב\nג\n§ Verso, address\nד"
    assert L.render_transcription([{"n": 1, "side": "single", "text": "א"}]) == "א"


def test_pgp_matches_suffix_and_own() -> None:
    """Entries match PGP by their own cross-check and by an inventory-number suffix."""
    corpus = SimpleNamespace(docs={"40150": {"shelfmark": "MIAC 23973.66"}, "1": {"shelfmark": "T-S 1.1"}})
    suffix = L.pgp_suffix_index(corpus)
    assert L.pgp_matches({"shelfmark": "23973.66"}, None, suffix)[0] == ["40150"]
    assert L.pgp_matches({"shelfmark": "Add. 01", "pgp": {"pgpids": [7]}}, None, suffix)[0] == ["7"]


def test_nas_candidates_filters() -> None:
    """Short commentary, unsupported scripts and (opt-in) PGP overlap are dropped."""
    lines = [{"n": i, "side": "recto", "text": "אלשיך זבריק כתב אלי קזואר"} for i in range(4)]
    long_text = "The merchant Zibriq wrote this letter to Qazwar about pepper. " * 3
    books = {"bk": [
        {"entry_no": 1, "script": "hebrew", "shelfmark": "X 1.1", "discussion": long_text, "transcription_lines": lines},
        {"entry_no": 2, "script": "hebrew", "shelfmark": "X 1.2", "discussion": "short", "transcription_lines": lines},
        {"entry_no": 3, "script": "syriac", "shelfmark": "X 1.3", "discussion": long_text,
         "transcription_lines": [{"n": 1, "text": "ܐܒܓ ܕܗܘ ܙܚܛ ܝܟܠ ܡܢܣ ܥܦܨ ܩܪܫ ܬ"}]},
        {"entry_no": 4, "script": "hebrew", "shelfmark": "X 1.4", "discussion": long_text, "transcription_lines": lines,
         "pgp": {"pgpids": ["55"]}}]}
    cands, drops = L.nas_candidates(books, None, small_cfg())
    assert [c.label for c in cands] == ["1", "4"] and drops["nas_commentary_short"] == 1
    assert drops["nas_script_unsupported"] == 1
    cands2, drops2 = L.nas_candidates(books, None, small_cfg(skip_pgp_overlap=("bk",)))
    assert [c.label for c in cands2] == ["1"] and drops2["nas_pgp_overlap"] == 1
    assert cands[0].edition_text.startswith("§ Recto") is False and "אלשיך" in cands[0].edition_text


def test_nas_candidates_entry_page_keys() -> None:
    """F11: with the shared entry → page mapper every book-edition candidate carries its entry's page keys (an
    unmapped book is flagged), the row meta repeats them, and the page rule drops a train entry on an excluded page.

    :returns: None.
    """
    from src.finetuning.text27b import book_pairs as BP
    lines = [{"n": i, "side": "recto", "text": "אלשיך זבריק כתב אלי קזואר"} for i in range(4)]
    long_text = "The merchant Zibriq wrote this letter to Qazwar about pepper. " * 3
    books = {"bk": [{"entry_no": 1, "script": "hebrew", "shelfmark": "X 1.1", "discussion": long_text,
                     "transcription_lines": lines, "pdf_page": 3},
                    {"entry_no": 2, "script": "hebrew", "shelfmark": "X 1.2", "discussion": long_text,
                     "transcription_lines": lines, "pdf_page": 5}],
             "lost": [{"entry_no": 1, "script": "hebrew", "shelfmark": "X 2.1", "discussion": long_text,
                       "transcription_lines": lines, "pdf_page": 3}]}
    pages = [SimpleNamespace(work="bk_work", page_no=n, page_key=f"bk_work/p/page_{n:03d}_structured.json")
             for n in range(1, 9)]
    ep = BP.EntryPages.from_pages({"bk": "bk_work", "lost": None}, pages)
    cands, _ = L.nas_candidates(books, None, small_cfg(), ep)
    by = {(c.source, c.label): c for c in cands}
    assert by[("book:bk", "1")].page_keys == [f"bk_work/p/page_{n:03d}_structured.json" for n in (3, 4, 5)]
    assert by[("book:bk", "1")].provenance["page_keys_source"] == L.NAS_PAGE_SOURCE == BP.NAS_PAGE_SOURCE
    assert by[("book:lost", "1")].page_keys == [] and by[("book:lost", "1")].provenance["page_keys_unmapped"] == "no_work"
    assert L.nas_candidates(books, None, small_cfg())[0][0].page_keys == []          # no mapper: no keys
    inp = L.L2Inputs(candidates=[by[("book:bk", "1")], by[("book:lost", "1")]], docs=index(), holdouts=holdouts(),
                     scholar_names=set())
    rows, rep = L.build_rows(inp, small_cfg(), ws_tokens())
    metas = {r["meta"]["source"]: r["meta"] for r in rows}
    assert set(metas) == {"book:bk", "book:lost"} and metas["book:bk"]["split"] == "train", rep["drops"]
    assert metas["book:bk"]["page_keys"] == by[("book:bk", "1")].page_keys and "page_keys_unmapped" not in metas["book:bk"]
    assert metas["book:lost"]["page_keys_unmapped"] == "no_work" and metas["book:lost"]["page_keys"] == []
    assert all(m["page_keys_source"] == L.NAS_PAGE_SOURCE for m in metas.values())
    inp.excluded = {"bk_work/p/page_005_structured.json": ["extra:int:l5:dev:book:bk:2:entry"]}
    rows2, rep2 = L.build_rows(inp, small_cfg(), ws_tokens())
    assert rep2["drops"]["span_page_excluded"] == 1 and not any(r["meta"]["source"] == "book:bk" for r in rows2)


def test_span_candidates_kinds() -> None:
    """A span with one PGP edition → span_pgp_edition; a span printing its edition → span_book_edition."""
    docs = index(doc("101", "train", SM_TRAIN))
    base = {"part": "p", "boundary_clean": True, "intro_chars": 100, "notes_chars": 0, "title": "t", "merged_doc_ids": [],
            "shelfmarks": [], "page_keys": [], "printed_pages": []}
    rows = [dict(base, book="w", doc_label="1", pgpids=["101"], sections={"intro": GOOD_INTRO}),
            dict(base, book="w", doc_label="2", pgpids=[], sections={"intro": GOOD_INTRO, "edition": EDITION.split("\n")}),
            dict(base, book="w", doc_label="3", pgpids=[], sections={"intro": GOOD_INTRO}),
            dict(base, book="w", doc_label="4", pgpids=[], boundary_clean=False, sections={}),
            dict(base, book="x", doc_label="5", pgpids=["101"], sections={"intro": GOOD_INTRO})]
    cands, drops = L.span_candidates(rows, docs, small_cfg(excluded_books=("x",)))
    assert [(c.label, c.kind) for c in cands] == [("1", L.SPAN_PGP), ("2", L.SPAN_BOOK)]
    assert cands[1].edition_text and drops == {"span_no_input": 1, "span_boundary_unclean": 1, "span_excluded_book": 1}


# ============================================================================== targets
def test_span_target_filters_lines() -> None:
    """Furniture, edition, garbled and input-copying lines are dropped; thin paragraphs too."""
    cfg = small_cfg()
    intro = GOOD_INTRO + ["©2025 Some Author, CC BY-NC 4.0", "12", "אלשיך זבריק כתב", "praise to God .2 alone here",
                          EDITION.split("\n")[0]]
    notes = ["1 The pepper was sent from the coast in two bales and arrived late.", "2 Line 9:"]
    target, acts, units = L.span_target(intro, notes, EDITION, cfg)
    assert "Zibriq" in target and "©" not in target and "אלשיך" not in target and ".2" not in target
    assert "\n\nNotes\n\n" in target and "Line 9" not in target
    assert acts["furniture_line"] == 2 and acts["edition_line_in_commentary"] >= 1 and acts["bidi_garbled_line"] == 1
    assert all("©" not in u for u in units)


def test_nas_target_drops_garbled_and_copied() -> None:
    """Book-entry commentary loses garbled and input-copying sentences."""
    intro = ("The merchant Zibriq wrote to Qazwar about the pepper trade in that year. "
             "Here :ويسلمها the two letters are reversed in the manuscript.")
    target, acts = L.nas_target(intro, "", EDITION, small_cfg())
    assert "Zibriq" in target and "reversed" not in target and acts["bidi_garbled_sentence"] == 1


def test_strip_physical() -> None:
    """Measurement pairs and material lists go; amounts stay."""
    assert L.strip_physical("Paper. 21.5 × 14.6; paper; 1 leaf.")[0] == "Paper."
    assert L.strip_physical("He paid 3 × 4 dirhams. Fine.") == ("He paid 3 × 4 dirhams. Fine.", [])
    assert L.strip_physical("Torn, holes and stains on paper. Next sentence.") == ("Next sentence.",
                                                                                  ["L2:physical_description"])


def test_thin_paragraphs() -> None:
    """Paragraphs with fewer than three content words are thin."""
    assert L.thin("m. Eruvin 5.1; b. Eruvin 52b.", "latin")
    assert L.thin("Recto: (a) Verso: (b)", "latin")
    assert not L.thin("Unfortunately there is no date.", "latin")
    assert L.thin("כתב", "rtl") and not L.thin("כתב אלשיך זבריק", "rtl")


def test_ocr_containment_per_line() -> None:
    """Containment is measured within lines, so dropped lines create no foreign shingles."""
    ocr = "the merchant wrote to his partner about the\npepper trade in that year and the price"
    units = ["the merchant wrote to his partner about the", "pepper trade in that year and the price"]
    assert L.ocr_containment(units, [ocr]) == 1.0
    assert L.ocr_containment(units, [None]) is None
    assert L.ocr_containment(["entirely different words appear in this line"], [ocr]) == 0.0


def test_held_mentions_ignore_own_shelfmark() -> None:
    """A target naming a held document is caught; the document's own shelfmark is not."""
    ho = holdouts()
    text = f"Compare {SM_BENCH}, which mentions the same judge. See also {SM_TRAIN}."
    hits = L.held_mentions(text, ho, [SM_TRAIN])
    assert len(hits) == 1 and "pgp131" in hits[0][1]
    assert L.held_mentions(text, ho, [SM_BENCH, SM_TRAIN]) == []


def test_context_for_withholds_copied_summary() -> None:
    """The catalogue summary is withheld when the target copies it; heading hints lose shelfmarks."""
    summary = "Letter from Zibriq to Qazwar about a shipment of pepper that reached the capital late in the season."
    tgt = "As the catalogue says: " + summary + " Nothing else is known."
    ctx, note = L.context_for(tgt, dict(CAT, summary=summary))
    assert note == "description_withheld_overlap" and "Catalogue description" not in ctx and "Type: Letter" in ctx
    ctx2, note2 = L.context_for("Unrelated commentary text that copies nothing at all from the record.", None,
                                {"title": f"{SM_TRAIN}: A letter to Qazwar", "date": "1480s"})
    assert note2 == "hint" and "Title: A letter to Qazwar" in ctx2 and "10J5" not in ctx2 and "Date: 1480s" in ctx2


def test_context_for_withholds_leaking_summary() -> None:
    """A catalogue summary that leaks an eval reference is withheld; a context that still leaks is dropped."""
    ctx, note = L.context_for("Unrelated commentary.", CAT, leak_fn=lambda t: "pepper" in t)
    assert note == "description_withheld_ngram" and "pepper" not in ctx and "Type: Letter" in ctx
    ctx2, note2 = L.context_for("Unrelated commentary.", CAT, leak_fn=lambda t: "Letter" in t)
    assert (ctx2, note2) == ("", "context_dropped_ngram")


# ============================================================================== literature block and tiers
def _pool(texts: Dict[str, str]) -> L.LitPool:
    """A literature pool over synthetic pages.

    :param texts: page key → text.
    :returns: Pool.
    """
    keys = list(texts)
    loc = {k: ("w/p", i) for i, k in enumerate(keys)}
    return L.LitPool(work="w", keys=keys, texts=[texts[k] for k in keys], cites=[f"A, *T*, 2000, p. {i}" for i in range(len(keys))],
                     loc=loc, index=L.BM25([L.bm25_tokens(texts[k]) for k in keys]))


def test_bm25_ranks_matching_page() -> None:
    """The page sharing rare query terms scores highest; stop words do not count."""
    bm = L.BM25([L.bm25_tokens(t) for t in ("pepper shipment from aden", "the and of", "silk from spain")])
    s = bm.scores(L.bm25_tokens("the pepper of aden"))
    assert s[0] > 0 and s[1] == 0 and s[0] > s[2]


def test_select_block_rules() -> None:
    """Window pages, pages copying the target and leaking pages are skipped; k_max is honoured."""
    target = "Barhun of Tahert sent the pepper to Zibriq when the caravan from the coast had arrived in the capital city."
    pool = _pool({"w/p/a": "pepper caravan Barhun Tahert merchants of the coast",
                  "w/p/b": "pepper " + target,
                  "w/p/c": "pepper Tahert family history of the west",
                  "w/p/d": "pepper Barhun window page",
                  "w/p/e": "pepper leaking page text"})
    cfg = small_cfg(k_max=2)
    src, meta, stats = L.select_block(pool, "pepper Barhun Tahert caravan", {"w/p/d"}, target, cfg,
                                      leak_fn=lambda t: "leaking" in t)
    keys = [m["page_key"] for m in meta]
    assert "w/p/b" not in keys and "w/p/d" not in keys and "w/p/e" not in keys and len(keys) == 2
    assert stats["block_page_target_overlap"] == 1 and stats["block_page_in_window"] == 1
    assert len(src) == 2 and src[0][2] == []


def test_decide_tier_a_c_and_fail() -> None:
    """A′ passes at 0.5 on input + context; C needs the block; otherwise no tier."""
    cfg = small_cfg()
    tgt = "The letter of Zibriq names Qazwar and also Barhun and Tahert as partners."
    ctx = "Known catalogue context:\nCatalogue description: Letter from Zibriq to Qazwar."
    a = L.decide_tier(tgt, EDITION, ctx, set(), cfg)
    assert a.tier == L.TIER_A and a.share_a >= 0.5
    tgt2 = "The letter of Zibriq names Barhun, Tahert, Sijilmasa and Wargla as partners."
    fail = L.decide_tier(tgt2, EDITION, ctx, set(), cfg)
    assert fail.tier is None
    block = ([("A, *T*, 2000, p. 4", "Barhun of Tahert traded with Sijilmasa and Wargla.", [])],
             [{"work": "w", "page_key": "w/p/x", "page": "4", "bm25": 1.0}], collections.Counter())
    c = L.decide_tier(tgt2, EDITION, ctx, set(), cfg, block_fn=lambda: block)
    assert c.tier == L.TIER_C and c.share_c >= 0.6 and "[SOURCE 1]" in c.extra and c.lit_sources[0]["page_key"] == "w/p/x"
    no_c = L.decide_tier(tgt2, EDITION, ctx, set(), small_cfg(tiers=(L.TIER_A,)), block_fn=lambda: block)
    assert no_c.tier is None


def test_fit_row_truncates_target_then_input() -> None:
    """Trailing target paragraphs go first, then input lines; impossible rows return None."""
    tok = ws_tokens()
    target = "one two three four five six\n\nNotes\n\nseven eight nine ten eleven twelve"
    fn = lambda text: "prompt " + text  # noqa: E731
    cap = tok.render_count(R.make_messages(fn("a b c"), "one two three four five six"))
    out = L.fit_row(fn, "a b c", target, tok, cap=cap, min_target_chars=5)
    assert out is not None and out[3] is True and "seven" not in out[1] and out[2] <= cap
    long_input = "\n".join(["w w w w w"] * 20)
    cap2 = tok.render_count(R.make_messages(fn("\n".join(["w w w w w"] * 12) + "\n[...]"), "x y z"))
    out2 = L.fit_row(fn, long_input, "x y z", tok, cap=cap2, min_target_chars=1)
    assert out2 is not None and out2[4] is True and out2[0].endswith("[...]") and out2[2] <= cap2
    assert L.fit_row(fn, "a", "q " * 100, tok, cap=10, min_target_chars=1) is None


# ============================================================================== end to end
def _inputs() -> L.L2Inputs:
    """Synthetic inputs covering every source kind and the main drop reasons.

    :returns: Inputs.
    """
    read = L.Variant("read", "אלשיך זבריק כתב", {"input_source": "read:a50", "read_model": "a50", "input_cer": 0.3,
                                                  "reader_exposed": False, "exposure_level": "none"})
    docs = index(doc("101", "train", SM_TRAIN, catalogue=CAT,
                     variants=[L.Variant("edition", EDITION, {"input_source": "pgp_digital_edition:x"}), read]),
                 doc("102", "train", "T-S 6J2.2", catalogue=CAT),
                 doc("103", "train", "T-S 6J3.3", catalogue=CAT))
    citations_only = " ".join(["See Goitein, Mediterranean Society, vol. 2, p. 23."] * 8)
    cands = [
        cand(L.SPAN_PGP, "1", GOOD_INTRO, ["2 The bales of pepper reached Qazwar after a long delay at sea."],
             pgpids=["101"], edition_pgpid="101", page_keys=["w/p/1"]),
        cand(L.SPAN_PGP, "2", ["The writer Zibriq complains that Barhun of Tahert and the people of Sijilmasa",
                               "and Wargla have not paid for the pepper that was sent to them last year."],
             pgpids=["102"], edition_pgpid="102"),
        cand(L.NAS, "3", f"The merchant Zibriq wrote this letter about pepper; compare {SM_BENCH} for the same judge.",
             pgpids=["103"], shelfmarks=["T-S 6J3.3"], edition_text=EDITION),
        cand(L.NAS, "4", citations_only, pgpids=["103"], shelfmarks=["T-S 6J3.3"], edition_text=EDITION, work="other"),
        cand(L.SPAN_BOOK, "5", GOOD_INTRO, shelfmarks=[SM_BENCH], edition_text=EDITION),
        cand(L.SPAN_BOOK, "6", GOOD_INTRO, shelfmarks=[SM_NEW], edition_text=EDITION, hints={"title": "Letter of Zibriq"}),
    ]
    return L.L2Inputs(candidates=cands, docs=docs, holdouts=holdouts(), scholar_names=set(),
                      ocr_by_key={"w/p/1": "\n".join(GOOD_INTRO + ["2 The bales of pepper reached Qazwar after a long "
                                                                   "delay at sea."])})


def test_build_rows_requires_ocr_for_spans() -> None:
    """Span targets without OCR text are dropped unless explicitly allowed."""
    rows, rep = L.build_rows(_inputs(), small_cfg(), ws_tokens())
    assert rep["drops"]["target_no_ocr"] == 3
    assert {r["meta"]["provenance"]["doc_label"] for r in rows} == {"1"}


def test_build_rows_end_to_end() -> None:
    """Rows validate, carry the read note and exposure tags, and every drop reason is counted."""
    rows, rep = L.build_rows(_inputs(), small_cfg(allow_missing_ocr=True), ws_tokens())
    assert rows and all(R.validate_row(r) == [] for r in rows)
    assert len({r["meta"]["row_id"] for r in rows}) == len(rows)
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["meta"]["provenance"]["doc_label"]].append(r)
    assert sorted(r["meta"]["condition"] for r in by_label["1"]) == ["edition", "read"]
    read_row = next(r for r in by_label["1"] if r["meta"]["condition"] == "read")
    assert R.READ_NOTE_V2 in read_row["messages"][1]["content"] and read_row["meta"]["reader_exposed"] is False
    assert all(r["meta"]["grounding_tier"] == L.TIER_A and r["meta"]["split"] == "train" for r in by_label["1"])
    assert "Known catalogue context" in by_label["1"][0]["messages"][1]["content"]
    assert by_label["5"][0]["meta"]["split"] == "heldout"
    assert by_label["6"][0]["meta"]["split"] == L.hash_split(SM.key(SM_NEW))
    assert by_label["6"][0]["meta"]["input_source"].startswith("book_edition:")
    drops = rep["drops"]
    assert drops["grounding_fail"] == 1 and drops["target_mentions_held_doc"] == 1 and drops["strip_emptied"] == 1
    assert all(not R.READ_NOTE_V2 in r["messages"][1]["content"] for r in rows if r["meta"]["condition"] == "edition")


def test_build_rows_page_level_exclusion() -> None:
    """Page-level exclusion is on by default (fix pass F2): a train target whose span page is excluded for any reason
    but the L1-only ``train_target_dup`` is dropped; switched off, only the text-level gates apply."""
    excluded = {"w/p/1": ["mention:rag_index:test"], "w/p/2": ["train_target_dup:x"]}
    assert L.L2Config().page_level_exclusion is True and L.L2Config().page_exclusion_ignore == ("train_target_dup",)
    rows, rep = L.build_rows(_inputs_with(excluded), small_cfg(allow_missing_ocr=True, page_level_exclusion=False),
                             ws_tokens())
    assert any(r["meta"]["provenance"]["doc_label"] == "1" for r in rows)
    rows2, rep2 = L.build_rows(_inputs_with(excluded), small_cfg(allow_missing_ocr=True), ws_tokens())
    assert rep2["drops"]["span_page_excluded"] == 1
    assert not any(r["meta"]["provenance"]["doc_label"] == "1" for r in rows2)
    for reasons in (["ngram_pair:test:9#w8"], ["extra:int:l2:dev:x:window"], ["window:mention:text:key:registry"]):
        rows3, rep3 = L.build_rows(_inputs_with({"w/p/1": reasons}), small_cfg(allow_missing_ocr=True), ws_tokens())
        assert rep3["drops"]["span_page_excluded"] == 1, reasons
    rows4, rep4 = L.build_rows(_inputs_with({"w/p/1": ["train_target_dup:analyse_document:1"]}),
                               small_cfg(allow_missing_ocr=True), ws_tokens())
    assert "span_page_excluded" not in rep4["drops"]
    assert any(r["meta"]["provenance"]["doc_label"] == "1" for r in rows4)


def _inputs_with(excluded: Dict[str, List[str]]) -> L.L2Inputs:
    """:func:`_inputs` with an exclusion map.

    :param excluded: page key → reasons.
    :returns: Inputs.
    """
    inp = _inputs()
    inp.excluded = excluded
    return inp


def test_build_rows_tier_c_with_pool() -> None:
    """A target failing A′ passes C when its work's pool supplies the missing names."""
    inp = _inputs()
    inp.candidates = [c for c in inp.candidates if c.label == "2"]
    inp.candidates[0].work = "w"
    inp.pools = {"w": _pool({"w/p/9": "pepper trade of Barhun of Tahert with Sijilmasa and Wargla merchants"})}
    rows, rep = L.build_rows(inp, small_cfg(allow_missing_ocr=True), ws_tokens())
    assert rows and rows[0]["meta"]["grounding_tier"] == L.TIER_C
    assert rows[0]["meta"]["lit_sources"][0]["page_key"] == "w/p/9" and "w/p/9" in rows[0]["meta"]["page_keys"]
    assert "[SOURCE 1]" in rows[0]["messages"][1]["content"]


def test_build_rows_ngram_gate() -> None:
    """Train targets that leak an eval reference are dropped."""
    inp = _inputs()
    inp.leak_fn = lambda text: "Qazwar" in text
    rows, rep = L.build_rows(inp, small_cfg(allow_missing_ocr=True), ws_tokens())
    assert rep["drops"].get("target_ngram_leak", 0) >= 1
    assert not any("Qazwar" in r["messages"][2]["content"] and r["meta"]["split"] == "train" for r in rows)


def test_summarise_and_audit() -> None:
    """The manifest counts tiers / drops / tokens and the audit sample takes edition rows only."""
    rows, rep = L.build_rows(_inputs(), small_cfg(allow_missing_ocr=True), ws_tokens())
    carried = [{"meta": {"pgpid": "101", "split": "train"}}, {"meta": {"pgpid": "555", "split": "train"}}]
    man = L.summarise(rows, rep, {"span_no_input": 2}, {"x": 1}, carried)
    assert man["documents"]["kept"] == len({r["meta"]["doc_key"] + r["meta"]["source"] for r in rows})
    assert man["tokens"]["total"]["rows"] == len(rows) and man["drops"]["candidate_extraction"] == {"span_no_input": 2}
    assert man["v0_1_coverage"]["covered"] == 1 and man["v0_1_coverage"]["lost"] == 1
    sample = L.audit_sample(rows, rep["records"], n=3)
    assert 0 < len(sample) <= 3 and all(s["target"] for s in sample)


def test_parse_tiers() -> None:
    """Tier aliases parse; unknown tiers fail."""
    assert L._parse_tiers("A',C") == (L.TIER_A, L.TIER_C)
    assert L._parse_tiers("A,B") == (L.TIER_A, L.TIER_B)
    with pytest.raises(Exception):
        L._parse_tiers("Z")


def test_local_render_contract() -> None:
    """With the local tokenizer and template, an L2 row renders under the contract (skipped elsewhere)."""
    try:
        p = P.load_paths()
    except (FileNotFoundError, KeyError):
        pytest.skip("no paths.json on this machine")
    if not (Path(p.tokenizer_json).exists() and Path(p.chat_template).exists()):
        pytest.skip("tokenizer / template not on this machine")
    rows, _ = L.build_rows(_inputs(), small_cfg(allow_missing_ocr=True),
                           L.default_token_fns(p.tokenizer_json, p.chat_template))
    for r in rows[:3]:
        rendered = R.render_chat(r["messages"], template_path=p.chat_template)
        assert rendered.count(R.EMPTY_THINK) == 1 and R.EFFORT_MARKER not in rendered
        assert r["meta"]["tokens"] <= 6144


#: An absolute directory in a string or comment (``/root/…``), not a URL, ``a/b`` or ``{x}/y`` (rule 3).
ABS_DIR = re.compile(r"(?<![\w.~{}:/\\-])/([A-Za-z][\w.-]*)/")


def test_module_names_no_machine_paths(tmp_path: Path) -> None:
    """The module source names no absolute or home-relative machine path, not this machine's home, and no host.

    :param tmp_path: Pytest temporary directory (positive control for the pattern).
    """
    src = Path(L.__file__).read_text(encoding="utf-8")
    assert ABS_DIR.findall(src) == [] and "~/" not in src and str(Path.home()) not in src
    for bad in ("localhost", ":9200", "studio_offload"):
        assert bad not in src
    # positive control: a synthetic absolute model folder is caught; URLs and relative joins are not
    assert ABS_DIR.findall(f'TOK = "{tmp_path / "fake-model-store" / "tokenizer.json"}"')
    assert ABS_DIR.findall('u = "https://example.test/a/b"; rel = "a/b/c"; f"{root}/x/y"') == []
