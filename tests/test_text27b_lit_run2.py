# File name: test_text27b_lit_run2.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the run-2 literature mining modules (``lit_run2``, ``lit_run2_build``, ``lit_run2_report``):
Neubauer crosswalk, resolution with legacy variants / joins / hints, hold-out reasons and splits, glued footnote
digits, edition quality flags, bylines, passages, row rendering, gates, the anthology spans source and the
image-index cache.

Synthetic fixtures only (a fake corpus matcher and a fake hold-out set): no real corpus, holdings or machine paths,
no network.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.text27b import lit_run2 as L
from src.finetuning.text27b import lit_run2_build as B
from src.finetuning.text27b import lit_segment as S
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import shelfmarks as SM

#: Fake corpus: normalised shelfmark -> (canonical ids, pgpids).
KNOWN: Dict[str, Any] = {"T-S 8J34.8": (["C_TS_8J34_8"], ["101"]), "T-S 16.22": (["C_TS_16_22"], ["202"]),
                         "T-S 10J5.17": (["C_TS_10J5_17"], ["203"]), "T-S 13J20.4": (["C_TS_13J20_4"], ["304"]),
                         "T-S Misc.28.79": ([f"C{i}" for i in range(9)], [str(500 + i) for i in range(9)])}


class FakeCorpus:
    """Stand-in corpus with the one map the resolver reads directly."""

    pgp_to_cids = {"777": {"C_HINT"}}


def fake_match(raw: str, corpus: Any) -> Dict[str, Any]:
    """Fake :func:`shelfmarks.match_shelfmark`.

    :param raw: Shelfmark.
    :param corpus: Ignored.
    :returns: Match dict.
    """
    cids, pg = KNOWN.get(raw, ([], []))
    return {"norm": raw, "core": raw, "how": "exact" if cids else "unmatched", "cids": list(cids), "pgpids": list(pg)}


class FakeHoldouts:
    """Hold-out set holding pgpid 202 (benchmark) and documents named ``T-S 99.1`` in a text."""

    def match_any(self, pgpids: Sequence[str] = (), canonicals: Sequence[str] = (), shelfmarks: Sequence[str] = (),
                  include_eval: bool = True) -> Optional[str]:
        """Best reason for any id.

        :param pgpids: PGP ids.
        :param canonicals: Canonical ids.
        :param shelfmarks: Shelfmarks.
        :param include_eval: Ignored.
        :returns: Reason or None.
        """
        return "pgp131" if "202" in set(pgpids) else None


def fake_mention(texts: Sequence[str], holdouts: Any) -> Optional[str]:
    """Fake :func:`contamination.text_mention_reason`.

    :param texts: Texts.
    :param holdouts: Ignored.
    :returns: Reason when a text names ``T-S 99.1``.
    """
    return "key:test" if any("T-S 99.1" in t for t in texts) else None


class FakeIx:
    """PGP index stand-in: pgpid 101 is trained by v0_1 and has an edition."""

    docs = {"101": {"edition_text": "בשם רחמן\nאלשיך אבו עלי", "edition_source_slug": "ed-slug"}}

    def v01_split(self, pid: str) -> Optional[str]:
        """v0_1 split of a PGP id.

        :param pid: PGP id.
        :returns: ``train`` for 101.
        """
        return "train" if pid == "101" else None


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> L.Resolver:
    """Resolver over the fakes.

    :param monkeypatch: Pytest monkeypatch.
    :returns: Resolver.
    """
    monkeypatch.setattr(SM, "match_shelfmark", fake_match)
    return L.Resolver(FakeCorpus(), FakeHoldouts(), FakeIx(), {}, mention_reason=fake_mention)


def test_neubauer_crosswalk() -> None:
    """Catalogue numbers printed next to MS Heb. volumes are collected; conflicts are dropped."""
    cw = L.neubauer_crosswalk(["Bodl. MS Heb. b 12 (Cat. 2875), f. 10", "MS Heb. d 66 (Cat. 2878)",
                               "MS Heb. c 28 (Cat. 2876)", "MS Heb. e 98 (Cat. 2876)"])
    assert cw == {"2875": "MS Heb. b 12", "2878": "MS Heb. d 66"}


def test_resolve_variants_joins_hints(resolver: L.Resolver) -> None:
    """Legacy spellings, joins and agent hints resolve; a whole box is broad."""
    r = resolver.resolve(["TS 8 J 34, f. 8"])
    assert r["pgpids"] == ["101"] and r["matched"][0]["variant"] == "T-S 8J34.8" and not r["broad"]
    j = resolver.resolve(["TS 16.22 and TS 10 J 5, f. 17"])
    assert j["pgpids"] == ["202", "203"]
    h = resolver.resolve(["Some Gallery XVI"], pgpid_hints=["777"])
    assert h["via_hint"] and h["canonical_ids"] == ["C_HINT"]
    assert resolver.resolve(["TS Misc. Box 28, f. 79"])["broad"]


def test_annotate_holdouts_and_splits(resolver: L.Resolver) -> None:
    """Document hold-outs, mention hold-outs, v0_1 train placement and hash splits."""
    a = resolver.annotate({}, ["TS 16.22"], ["text"], "k1")
    assert a["held_out"] and a["held_out_reason"] == "document:pgp131" and a["split"] == "heldout"
    b = resolver.annotate({}, ["TS 8 J 34, f. 8"], ["see T-S 99.1 here"], "k2")
    assert b["held_out"] and b["held_out_reason"].startswith("span_mention") and b["split_reason"] == "span_mention"
    c = resolver.annotate({}, ["TS 8 J 34, f. 8"], ["clean"], "k3")
    assert c["split"] == "train" and c["split_reason"] == "v0_1_train_group"
    d = resolver.annotate({}, ["Unknown 1.2"], ["clean"], "k4")
    assert d["split"] in ("train", "dev", "test_indist") and d["split_reason"] == "hash"
    assert resolver.pgp_edition(["101"])[2] == "101"


def test_strip_note_digits(resolver: L.Resolver) -> None:
    """A footnote number glued to a shelfmark is removed only when the page has that note and the rest resolves."""
    assert L.strip_note_digits("T-S 13J20.457", {"57"}, resolver) == ("T-S 13J20.4", "footnote_digits_stripped:57")
    assert L.strip_note_digits("T-S 13J20.457", {"12"}, resolver) == ("T-S 13J20.457", None)
    assert L.strip_note_digits("T-S 16.22", {"22"}, resolver) == ("T-S 16.22", None)


def test_quality_flags_edition() -> None:
    """Short, reversed, debris-heavy, fragmented and line-count-mismatched editions lose score."""
    good = ["בשם רחמן רחים אלשיך אבו עלי חסן בן דאוד"] * 12
    q, f = L.quality_flags_edition(good, {"raw": 12, "kept": 12, "short": 0}, lines_stated=12)
    assert q == 1.0 and f == []
    q2, f2 = L.quality_flags_edition(good, {"raw": 40, "kept": 12, "numeric": 28, "short": 0}, lines_stated=6)
    assert q2 < 0.6 and any(x.startswith("line_count_ratio") for x in f2) and any(x.startswith("ocr_debris") for x in f2)
    rev = [" ".join(w[::-1] for w in ["שלום", "אדם", "ארץ", "כנען", "מלך"] * 3)] * 3
    q3, f3 = L.quality_flags_edition(rev, {"raw": 3, "kept": 3})
    assert any(x.startswith("reversed_hebrew") for x in f3) and q3 <= 0.4


def test_strip_byline_and_passage() -> None:
    """Title and byline lines go; a passage widens to neighbouring paragraphs when short."""
    text, n = L.strip_byline("A Popular Calendar\n\nJane Doe\n\nThe fragment T-S 1.1 is a calendar.", "A Popular Calendar")
    assert n == 2 and text.startswith("The fragment")
    body = "Para one is here.\n\nPara two mentions T-S 5.5 in detail.\n\nPara three follows."
    a, b = L.passage_around(body, body.index("T-S 5.5"))
    assert "Para two" in body[a:b] and "Para one" in body[a:b]


def test_render_rows() -> None:
    """Describe / analyse rows use the rows schema with run-2 meta."""
    rec = {"doc_id": "lit:x:1", "split": "train", "shelfmark": "T-S 1.1", "source": "lit:x", "target": "A deed of gift.",
           "resolution": {"pgpids": ["101"], "broad": False}, "provenance": {"licence": "c", "licence_class": "copyright",
                                                                              "page_keys": ["w/p/page_001.json"]}}
    row = L.render_describe_row(rec, "בשם רחמן", "book_edition:x")
    assert row["meta"]["family"] == "describe_pgp" and row["meta"]["run2"] and row["meta"]["pgpid"] == "101"
    assert "<document>\nבשם רחמן\n</document>" in row["messages"][1]["content"]
    assert row["messages"][2]["content"] == "A deed of gift."
    row2 = L.render_analyse_row(rec, "בשם רחמן", "pgp")
    assert row2["meta"]["family"] == "analyse_document"
    assert R.validate_row(row2, require_v01=False) == [] and R.validate_row(row, require_v01=False) == []


def test_gate() -> None:
    """VLM candidates need score, images and a resolution; catalogue / commentary need a row."""
    base = {"held_out": False, "quality": {"score": 0.9}, "resolution": {"resolved": True, "broad": False}}
    assert B.gate("vlm_pages_hebrew_ja", {**base, "images": {"with_images": 1}})[1]
    assert not B.gate("vlm_pages_hebrew_ja", {**base, "images": {"with_images": 0}})[1]
    assert B.gate("catalogue_descriptions", {**base, "row": {"m": 1}})[1]
    ho, ok, why = B.gate("catalogue_descriptions", {**base, "held_out": True, "row": {"m": 1}})
    assert not ho and not ok and "held_out" in why
    row = {"meta": {"condition": "edition", "split": "train",
                    "run2": {"held_out": False, "quality": {"score": 0.9, "flags": []}}}}
    assert B.gate("translate_en", row)[1]
    row["meta"]["run2"]["quality"]["flags"] = ["dup_existing_t27b:x"]
    assert not B.gate("translate_en", row)[1]


def test_licence_class_of() -> None:
    """Licence strings map to classes."""
    assert B.licence_class_of("Public domain in the US") == "pd"
    assert B.licence_class_of("CC BY-NC 4.0") == "cc-by-nc"
    assert B.licence_class_of("JSTOR terms; personal use") == "copyright"
    assert B.licence_class_of(None) == "unknown"


def test_anthology_source(resolver: L.Resolver) -> None:
    """Anthology sections become verbatim-piece spans with PGP hints and glued note digits removed."""

    class Page:
        """Minimal PageRec."""

        def __init__(self, key: str, text: str) -> None:
            """Set fields.

            :param key: Page key.
            :param text: OCR text.
            """
            self.page_key, self.ocr_text, self.data = key, text, {"footnotes": {"min_57": "x"}}
            self.page_no, self.printed_page_num = 1, 5

    text = "\n".join(['1. "Title"', "Intro text of the editor.", "T-S 13J20.457 (Judaeo-Arabic)", "Your slave writes."])
    pg = Page("w/p/page_001.json", text)
    secs = S.anthology_sections([S.PageInput(pg.page_key, 1, text)])
    src, recs = L.anthology_source({pg.page_key: pg}, secs, {"slug": "anth", "work": "w"}, resolver)
    d = src["docs"][0]
    assert d["shelfmarks"] == ["T-S 13J20.4"] and d["pgpids"] == ["304"]
    assert set(d["spans"]) == {"translation", "intro"}
    piece = d["spans"]["translation"]["pieces"][0]
    assert piece["lines"] == [3, 3] and piece["head"] == "Your slave writes."
    assert recs[0]["shelfmark_flag"] == "footnote_digits_stripped:57"


def test_image_index_cache_only(tmp_path: Path) -> None:
    """With look-ups disabled the index answers from its cache and summarises counts."""
    cache = tmp_path / "es.json"
    cache.write_text(json.dumps({"C1": {"found": True, "n_images": 2, "source": "x", "image_urls": ["u1", "u2"]},
                                 "C2": {"found": True, "n_images": 0, "source": None, "image_urls": []}}))
    ix = L.ImageIndex(None, cache, enabled=False)
    assert ix.fetch(["C1", "C3"]) == 0
    s = ix.summary(["C1", "C2", "C3"])
    assert s["looked_up"] == 2 and s["with_images"] == 1 and s["n_images"] == 2 and "C1" in s["per_id"]


def test_image_index_local_images(tmp_path: Path) -> None:
    """Local images fill ids without served images (source ``cudl_local``, paths under ``per_id``); ids that already
    have served images keep the served record."""
    cache = tmp_path / "es.json"
    cache.write_text(json.dumps({"C1": {"found": True, "n_images": 2, "source": "x", "image_urls": ["u1", "u2"]},
                                 "C2": {"found": True, "n_images": 0, "source": None, "image_urls": []}}))
    local = tmp_path / "local_images.json"
    local.write_text(json.dumps({
        "C1": [{"path": "/d/M1/01_1r.jpg", "canvas_label": "1r", "manifest_id": "M1"}],
        "C2": [{"path": "/d/M2/01_1r.jpg", "canvas_label": "1r", "manifest_id": "M2"},
               {"path": "/d/M2/02_1v.jpg", "canvas_label": "1v", "manifest_id": "M2"}],
        "C4": [{"path": "/d/M4/01_1r.jpg", "canvas_label": "1r", "manifest_id": "M4"}]}))
    assert L.ImageIndex.load_local(None) == {}
    ix = L.ImageIndex(None, cache, enabled=False, local_images=L.ImageIndex.load_local(local))
    s = ix.summary(["C1", "C2", "C3", "C4"])
    assert s["looked_up"] == 3 and s["found"] == 3 and s["with_images"] == 3 and s["n_images"] == 5
    assert s["sources"] == ["cudl_local", "x"]
    assert s["per_id"]["C1"] == {"n_images": 2, "source": "x", "image_urls": ["u1", "u2"]}
    assert s["per_id"]["C2"]["source"] == "cudl_local" and s["per_id"]["C2"]["n_images"] == 2
    assert [x["path"] for x in s["per_id"]["C2"]["local_images"]] == ["/d/M2/01_1r.jpg", "/d/M2/02_1v.jpg"]
    assert s["per_id"]["C4"]["source"] == "cudl_local" and "C3" not in s["per_id"]
    assert ix.n_images("C2") == 2 and ix.n_images("C3") == 0
    base = {"held_out": False, "quality": {"score": 0.9}, "resolution": {"resolved": True, "broad": False}}
    assert B.gate("vlm_pages_hebrew_ja", {**base, "images": ix.summary(["C2"])})[1]
    assert not B.gate("vlm_pages_hebrew_ja", {**base, "images": L.ImageIndex(None, cache, enabled=False).summary(["C2"])})[1]


def test_scrub_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prose fields are scrubbed by the shared scrubber; identifier fields are left alone."""
    from src.finetuning.text27b import textnorm as TN
    monkeypatch.setattr(TN, "strip_source_names", lambda t: (t.replace("BADNAME", ""), int("BADNAME" in t)))
    rec = {"target": "kept BADNAME text", "shelfmark": "BADNAME 1", "row": {"messages": [{"role": "user", "content": "x BADNAME"}],
                                                                          "meta": {"run2": {"note": "BADNAME"}}}}
    out = B.scrub_record(rec)
    assert out["target"] == "kept  text" and out["shelfmark"] == "BADNAME 1"
    assert out["row"]["messages"][0]["content"] == "x " and out["row"]["meta"]["run2"]["note"] == ""


class PieceCorpus(FakeCorpus):
    """Fake corpus with lettered pieces of one box, a historic shelfmark, PGP footnotes and fragment rows."""

    core_idx = {"T_S_NS_J185a": {"C_J185a"}, "T_S_NS_J185b": {"C_J185b"}, "T_S_NS_J303a": {"C_J303a"},
                "T_S_NS_J303b": {"C_J303b"}}
    merged = {"C_J185a": {"pgpids": ["601"]}, "C_J185b": {"pgpids": ["602"]}, "C_J303a": {"pgpids": ["701"]},
              "C_J303b": {"pgpids": ["702"]}}
    pgp_to_cids = {"777": {"C_HINT"}, "601": {"C_J185a"}, "602": {"C_J185b"}, "701": {"C_J303a"}, "702": {"C_J303b"}}
    fragments = [{"shelfmark": "T-S NS J401a", "pgpids": ["801"], "historic": "T-S NS J401.7"},
                 {"shelfmark": "ENA NS 77.36", "pgpids": ["901"], "historic": ""}]
    fn_rows = [{"pgpid": "601", "slug": "w-1970", "location": "Doc. 216"},
               {"pgpid": "602", "slug": "w-1970", "location": "Docs. 220, 221"},
               {"pgpid": "603", "slug": "w-1970", "location": "doc. 159-160"},
               {"pgpid": "604", "slug": "other", "location": "Doc. 216"},
               {"pgpid": "605", "slug": "w-1970", "location": "Doc. 152, pp. 283 and 260-261"}]


@pytest.fixture
def piece_resolver(monkeypatch: pytest.MonkeyPatch) -> L.Resolver:
    """Resolver over the piece corpus (fake matcher knows ``T-S NS J401a``).

    :param monkeypatch: Pytest monkeypatch.
    :returns: Resolver.
    """
    known = dict(KNOWN, **{"T-S NS J401a": (["C_J401a"], ["801"])})

    def match(raw: str, corpus: Any) -> Dict[str, Any]:
        """Fake matcher.

        :param raw: Shelfmark.
        :param corpus: Ignored.
        :returns: Match dict.
        """
        cids, pg = known.get(raw, ([], []))
        return {"norm": raw, "core": raw, "how": "exact" if cids else "unmatched", "cids": list(cids), "pgpids": list(pg)}

    monkeypatch.setattr(SM, "match_shelfmark", match)
    return L.Resolver(PieceCorpus(), FakeHoldouts(), FakeIx(), {}, mention_reason=fake_mention)


def test_piece_anchor_never_positional(piece_resolver: L.Resolver) -> None:
    """A numbered piece of a box with lettered pieces is anchored to all of them, unresolved, with its piece."""
    r = piece_resolver.resolve(["TS NS J 185, no. 1"])
    assert not r["resolved"] and r["canonical_ids"] == [] and r["anchors"][0]["piece"] == "no. 1"
    assert r["anchors"][0]["anchor_cids"] == ["C_J185a", "C_J185b"] and r["anchors"][0]["anchor_pgpids"] == ["601", "602"]
    m = r["matched"][0]
    assert m["how"] == "piece_anchor" and m["pieces"] == ["J185a", "J185b"]


def test_bare_box_pieces_ambiguous(piece_resolver: L.Resolver) -> None:
    """A bare mark whose pieces carry letters resolves to all pieces and is ambiguous across documents."""
    r = piece_resolver.resolve(["TS NS J 303"])
    assert r["resolved"] and r["ambiguous"] and r["pgpids"] == ["701", "702"]
    n = piece_resolver.resolve(["TS NS J 303"], pgpid_hints=["702"], hint_source="pgp_footnote_doc_number")
    assert n["pgpids"] == ["702"] and not n["ambiguous"] and n["via_hint"]


def test_historic_shelfmark_match(piece_resolver: L.Resolver) -> None:
    """A printed numbered piece matches PGP's historic spelling of the current lettered piece."""
    m = piece_resolver.match_one("TS NS J 401, no. 7")
    assert m["how"] == "historic:exact" and m["historic_current"] == "T-S NS J401a" and m["pgpids"] == ["801"]


def test_hints_anchor_and_disagreement(piece_resolver: L.Resolver) -> None:
    """Hints resolve an anchored piece (and say whether they fall inside the anchor); a hint contradicting a
    resolved shelfmark is recorded, not applied."""
    a = piece_resolver.resolve(["TS NS J 185, no. 4"], pgpid_hints=["602"], hint_source="pgp_footnote_doc_number")
    assert a["resolved"] and a["pgpids"] == ["602"] and a["hint_within_anchor"] and a["hint_source"] == "pgp_footnote_doc_number"
    d = piece_resolver.resolve(["TS NS J 401, no. 7"], pgpid_hints=["999"])
    assert d["pgpids"] == ["801"] and d["hint_disagrees"] and not d["via_hint"]


def test_doc_numbers_and_crosswalk() -> None:
    """``Doc. N`` locations expand to document numbers; only the work's slug counts; page numbers are ignored."""
    assert L.doc_numbers("Docs. 90, 91") == ["90", "91"]
    assert L.doc_numbers("doc. 159-160") == ["159", "160"]
    assert L.doc_numbers("docs. 67 and 136") == ["67", "136"]
    assert L.doc_numbers("Doc. 152, pp. 283 and 260-261 (discussion)") == ["152"]
    assert L.doc_numbers("p. 12") == []
    xw = L.pgp_doc_crosswalk(PieceCorpus.fn_rows, "w-1970")
    assert xw == {"216": ["601"], "220": ["602"], "221": ["602"], "159": ["603"], "160": ["603"], "152": ["605"]}


def test_pgp_fragment_cids() -> None:
    """PGP ids without a merged record get canonical ids synthesised from their fragment shelfmark."""
    ids = L.pgp_fragment_cids(PieceCorpus(), ["901"])
    assert len(ids) == 1 and ids[0].endswith("ENA_NS_77_36")
    assert L.pgp_fragment_cids(PieceCorpus(), ["000"]) == []


def test_strip_note_digits_partial(resolver: L.Resolver) -> None:
    """One digit of a two-digit note glued to a shelfmark is stripped when the remainder resolves."""
    assert L.strip_note_digits("T-S 13J20.46", {"66"}, resolver) == ("T-S 13J20.4", "footnote_digits_stripped_partial:6~66")
    assert L.strip_note_digits("T-S 13J20.46", {"12"}, resolver) == ("T-S 13J20.46", None)


def test_anthology_alternate_classmark(resolver: L.Resolver) -> None:
    """When the printed shelfmark does not resolve, the classmark printed in parentheses is used."""
    KNOWN["Mosseri II 127.1"] = (["C_MOSS"], ["2759"])

    class Page:
        """Minimal PageRec."""

        def __init__(self, key: str, text: str) -> None:
            """Set fields.

            :param key: Page key.
            :param text: OCR text.
            """
            self.page_key, self.ocr_text, self.data = key, text, {"footnotes": {}}
            self.page_no, self.printed_page_num = 1, 5

    try:
        text = "\n".join(['1. "Title"', "Intro.", "Mosseri L 129.1 (II 127.1) 36 (Judaeo-Arabic)", "To be paid."])
        pg = Page("w/p/page_001.json", text)
        secs = S.anthology_sections([S.PageInput(pg.page_key, 1, text)])
        src, recs = L.anthology_source({pg.page_key: pg}, secs, {"slug": "anth", "work": "w"}, resolver)
        assert src["docs"][0]["shelfmarks"] == ["Mosseri II 127.1"] and src["docs"][0]["pgpids"] == ["2759"]
        assert recs[0]["shelfmark_flag"] == "printed_alternate_classmark" and recs[0]["shelfmark_alt"] == "Mosseri II 127.1"
    finally:
        KNOWN.pop("Mosseri II 127.1", None)


def test_gate_piece_reasons() -> None:
    """Anchored-but-unresolved and ambiguous-piece candidates fail the VLM gate with their own reasons."""
    base = {"held_out": False, "quality": {"score": 0.9}, "images": {"with_images": 1}}
    anch = {**base, "resolution": {"resolved": False, "broad": False, "anchors": [{"piece": "no. 3"}]}}
    assert B.gate("vlm_pages_hebrew_ja", anch)[2] == ["piece_unresolved_anchored"]
    amb = {**base, "resolution": {"resolved": True, "broad": False, "ambiguous": True}}
    assert B.gate("vlm_pages_hebrew_ja", amb)[2] == ["piece_ambiguous"]
    assert B.gate("translate_en_no_edition", amb)[2] == ["piece_ambiguous"]
    assert B.gate("catalogue_descriptions", {**amb, "row": {"m": 1}})[1]
    synth = {"resolution": {"canonical_ids": [], "canonical_ids_synth": ["X_1"], "broad": False}}
    assert B._cids_of(synth) == ["X_1"]


def test_apply_text_layers(tmp_path: Path) -> None:
    """A configured text layer replaces a part's OCR text page by page; a missing layer keeps the current text."""
    from src.finetuning.text27b import pages as PG
    part = tmp_path / "w" / "part"
    old = part / "part_structured_gemini_gemini_3.5_flash"
    new = part / "part_structured_gemini_lines"
    old.mkdir(parents=True)
    new.mkdir()
    (new / "page_001_structured.json").write_text(json.dumps({"metadata": {"page_number": 1}, "full_main_text": "NEW 1"}))
    (new / "page_002_structured.json").write_text(json.dumps({"lines": [{"text": "a"}, "b"]}))

    def rec(n: int) -> Any:
        """PageRec of the old structuring.

        :param n: Page number.
        :returns: PageRec.
        """
        return PG.PageRec(work="w", part="part", page_file=f"page_{n:03d}_structured.json", page_no=n,
                          path=str(old / f"page_{n:03d}_structured.json"), structuring_dir=old.name,
                          structurer_model=None, structurer_provider=None, printed_page=None, full_main_text="",
                          page_type=None, shelf_marks=[], ocr_text=f"OLD {n}")

    pages = [rec(1), rec(2), rec(3)]
    info = L.apply_text_layers(pages, {"works": {"w/part": {"text_layer": "gemini_lines"}}})
    assert info["w/part"]["exists"] and info["w/part"]["pages_replaced"] == 2
    assert [p.ocr_text for p in pages] == ["NEW 1", "a\nb", "OLD 3"]
    assert pages[0].ocr_text_raw == "OLD 1" and pages[0].ocr_source == "text_layer:gemini_lines"
    keep = [rec(1)]
    info2 = L.apply_text_layers(keep, {"works": {"w/part": {"text_layer": "missing_layer"}}})
    assert not info2["w/part"]["exists"] and keep[0].ocr_text == "OLD 1"
    assert L.apply_text_layers(keep, {"works": {"w/part": {"lang": "EN"}}}) == {}
