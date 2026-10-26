# File name: test_text27b_web_pairs.py
# Date: 10/26/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``web_pairs``: web normalisers, item loading, PGP-edition line ranges, span specs, item-level
hold-outs, dedupe against PGP / L5 / earlier web items, row meta, outputs and the commentary count.

Everything runs on synthetic fixtures written to ``tmp_path``: no real corpus, holdings or machine paths.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import book_pairs as BP
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import web_pairs as W

JA_LINES = ["בשמ רחמ", "כתאבי אלי אלשיך אבו עלי חסן בן דאוד", "וצל כתאבך ופהמתה ואלחמד ללה"]
EN_LINES = ["In the name of God, the Merciful, the Compassionate.", "My letter to the elder Abū ʿAlī Ḥasan ibn Dāwūd.",
            "Your letter arrived and I understood it, praise be to God."]


class FakeTC:
    """Token counter stand-in (3 characters per token)."""

    def count(self, text: str) -> int:
        """Approximate token count.

        :param text: Text.
        :returns: Tokens.
        """
        return len(text or "") // 3


class FakeHoldouts:
    """Hold-out stand-in: benchmark / eval tiers by pgpid or exact shelfmark."""

    def __init__(self, bench: Optional[Dict[str, str]] = None, evals: Optional[Dict[str, str]] = None) -> None:
        """Create the stand-in.

        :param bench: pgpid or shelfmark → reason (benchmark tier).
        :param evals: pgpid or shelfmark → reason (eval tier).
        """
        self.bench, self.evals = bench or {}, evals or {}

    @staticmethod
    def _hit(table: Dict[str, str], pgpid: Any, shelfmark: Optional[str]) -> Optional[str]:
        """Look a document up in one tier.

        :param table: Tier table.
        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :returns: Reason or None.
        """
        if pgpid is not None and str(pgpid) in table:
            return table[str(pgpid)]
        return table.get(shelfmark) if shelfmark else None

    def is_benchmark(self, pgpid: Any = None, shelfmark: Optional[str] = None, canonical: Optional[str] = None
                     ) -> Optional[str]:
        """Benchmark tier lookup.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Unused.
        :returns: Reason or None.
        """
        return self._hit(self.bench, pgpid, shelfmark)

    def is_eval_doc(self, pgpid: Any = None, shelfmark: Optional[str] = None, canonical: Optional[str] = None
                    ) -> Optional[str]:
        """Eval tier lookup.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Unused.
        :returns: Reason or None.
        """
        return self._hit(self.evals, pgpid, shelfmark)


def fake_mention(texts: List[str], holdouts: Any) -> Optional[str]:
    """Text-mention stand-in: an item text containing ``HELD-DOC`` names a held document.

    :param texts: Texts.
    :param holdouts: Unused.
    :returns: Reason or None.
    """
    return "test" if any("HELD-DOC" in (t or "") for t in texts) else None


def item_text(extra: str = "") -> str:
    """A synthetic post: heading, transcription, translation with footnote refs, commentary.

    :param extra: Extra commentary line.
    :returns: Text.
    """
    return "\n".join(["Title of the post", "", "Transcription:", "", *JA_LINES, "", "Translation:", "",
                      EN_LINES[0] + "[1]", EN_LINES[1], EN_LINES[2], "", "Commentary on the letter. " + extra,
                      "", "[1] A footnote."])


def write_dir(root: Path, name: str, items: List[Dict[str, Any]], licence: str = "Public domain in the US") -> None:
    """Write a synthetic web-source directory.

    :param root: Literature root.
    :param name: Directory name.
    :param items: ``[{"slug", "text", "review", "meta": {...}, "item": {...}}]``.
    :param licence: Directory licence statement.
    """
    base = root / name
    (base / "text").mkdir(parents=True)
    (base / "metadata").mkdir()
    (base / "metadata.json").write_text(json.dumps({"licence": licence}), encoding="utf-8")
    lines = []
    for it in items:
        (base / "text" / f"{it['slug']}.txt").write_text(it["text"], encoding="utf-8")
        meta = {"title": f"T {it['slug']}", "authors": ["A. Author"], "year": 2020, "review": it.get("review", {}),
                **it.get("meta", {})}
        (base / "metadata" / f"{it['slug']}_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        lines.append(json.dumps({"slug": it["slug"], "title": f"T {it['slug']}", **it.get("item", {})}))
    (base / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pairs_index(tmp_path: Path, rows: List[Dict[str, Any]]) -> BP.PgpIndex:
    """A PGP index over a synthetic pairs file.

    :param tmp_path: Temporary directory.
    :param rows: Pairs rows.
    :returns: The index.
    """
    path = tmp_path / "pairs.jsonl"
    R.write_jsonl(path, rows)
    return BP.PgpIndex.build(path)


def paths_for(tmp_path: Path) -> P.Paths:
    """A synthetic paths config under ``tmp_path``.

    :param tmp_path: Temporary directory.
    :returns: Loaded paths.
    """
    entries = {k: str(tmp_path / k) for k in (
        "probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
        "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
        "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch")}
    entries["repo"] = str(tmp_path)
    cfg = tmp_path / "paths.json"
    cfg.write_text(json.dumps(entries), encoding="utf-8")
    return P.load_paths(cfg)


def span_spec(doc: str, text: str, key: str, input_mode: str = "span", **extra: Any) -> Dict[str, Any]:
    """Spans-file document for the synthetic post layout (edition lines 4–6, translation lines 10–12).

    :param doc: Doc label.
    :param text: Item text.
    :param key: Item key.
    :param input_mode: ``span`` or ``pgp_edition``.
    :param extra: Further spec fields.
    :returns: Spans-file document.
    """
    spec = {"doc": doc, "shelfmarks": extra.pop("shelfmarks", ["T-S 5.5"]), "target_lang": "en", "lang_class": "JA",
            "input": input_mode, "translation": [[10, 12]], "normalise": ["drop_bracket_refs"], **extra}
    if input_mode == "span":
        spec["edition"] = [[4, 6]]
    return W.spec_to_span_doc(spec, key, text, extra.get("ix"))


# ---------------------------------------------------------------------------------- normalisers
def test_bracket_refs_and_inline_numbers() -> None:
    """Footnote refs go, editorial brackets stay; inline (N) markers go only when two or more ascend."""
    assert W.drop_bracket_refs("drin[k from the cup][3] and [12] more") == ("drin[k from the cup] and more", 2)
    assert W.drop_inline_line_numbers("a (1) b (2) c (3) d")[0] == "a b c d"
    assert W.drop_inline_line_numbers("item (1) only")[1] == 0
    text, ops = W.web_normalise("one[1]\n\n\n\ntwo  three", ["drop_bracket_refs", "drop_blank_lines"])
    assert text == "one\ntwo three" and ops == ["W:drop_bracket_refs", "W:drop_blank_lines"]
    with pytest.raises(KeyError):
        W.web_normalise("x", ["no_such_op"])


# ---------------------------------------------------------------------------------- items
def test_load_items_skip_duplicates_and_licence(tmp_path: Path) -> None:
    """Items load with review, hold-out and licence fields; skipped and duplicate items are left out."""
    root = tmp_path / "lit"
    write_dir(root, "src_a", [
        {"slug": "p1", "text": item_text(), "review": {"translation_level": "full", "translators": ["X Y", "X Y"],
                                                       "has_commentary": True},
         "meta": {"shelfmarks_mentioned": ["T-S 5.5"], "licence_terms": "CC BY 4.0"}},
        {"slug": "p2", "text": "x", "item": {"duplicate_of": "p1"}},
        {"slug": "p3", "text": "y"}])
    items = W.load_items(root, ["src_a", "missing_dir"], {"src_a": ["p3"]})
    assert list(items) == ["src_a/text/p1.txt"]
    it = items["src_a/text/p1.txt"]
    assert it.level == "full" and it.has_commentary and it.licence_class == "cc-by" and it.text().startswith("Title")
    assert W.translators_of(it.review) == ["X Y"]
    assert W.licence_of(None, "Public domain in the US (pre-1931)")[1] == "pd"
    assert W.licence_of(None, None) == (None, "unknown")


# ---------------------------------------------------------------------------------- PGP edition ranges
def test_pgp_edition_lines_verified(tmp_path: Path) -> None:
    """A PGP edition line range is cut verbatim and verified; a tampered record fails; Latin-only lines are dropped."""
    ix = pairs_index(tmp_path, [{"pgpid": "10", "shelfmark": "T-S 5.5", "split": "train",
                                 "edition_text": "Recto\n" + "\n".join(JA_LINES), "editions": [], "translations": []}])
    ed, slug = W.pgp_edition_text(ix, "10")
    assert ed.split("\n")[1:] == JA_LINES and W.pgp_edition_text(ix, "99") is None
    rec = W.pgp_line_piece(ed, 2, 3)
    text, why = W.cut_pgp_lines(ed, rec)
    assert why == "ok" and text == "\n".join(JA_LINES[1:])
    assert W.cut_pgp_lines(ed, None) == (ed, "ok")
    assert W.cut_pgp_lines(ed, {**rec, "sha1": "0" * 40})[1] == "sha1_mismatch"
    assert W.cut_pgp_lines(ed, {**rec, "lines": [2, 9]})[1] == "range"


# ---------------------------------------------------------------------------------- specs → documents
def test_spec_round_trip_and_doc_from_spec(tmp_path: Path) -> None:
    """Agent line ranges become verified pieces; span and PGP-edition inputs build documents; bad specs drop."""
    root = tmp_path / "lit"
    text = item_text()
    write_dir(root, "src_a", [{"slug": "p1", "text": text, "review": {"translation_level": "full",
                                                                      "translators": ["Rev Translator"]}}])
    it = W.load_items(root, ["src_a"])["src_a/text/p1.txt"]
    ix = pairs_index(tmp_path, [{"pgpid": "10", "shelfmark": "T-S 5.5", "split": "train",
                                 "edition_text": "\n".join(JA_LINES), "editions": [], "translations": []}])
    d = span_spec("1", text, it.key)
    assert d["spans"]["translation"]["pieces"][0]["sha1"] and d["spans"]["edition"]["pieces"][0]["lines"] == [4, 6]
    doc, info = W.doc_from_spec(it, d, text, ix)
    assert doc is not None and doc.edition_source == "web" and doc.translation.split("\n")[0] == EN_LINES[0]
    assert doc.doc_id == "web:src_a:p1:1" and doc.translator == "Rev Translator" and doc.translator_source == "review"
    assert doc.input_source == "web_edition:src_a:p1" and doc.provenance["work_pages"]["edition"] == [it.key]
    d2 = span_spec("2", text, it.key, "pgp_edition", pgpid="10", translator="Named Person", ix=ix,
                   pgp_lines=[0, 2])
    doc2, _ = W.doc_from_spec(it, d2, text, ix)
    assert doc2.edition_source == "pgp" and doc2.pgp_edition_pgpid == "10" and doc2.translator == "Named Person"
    assert doc2.input_source.startswith("pgp_digital_edition") and "edition" not in doc2.provenance["work_pages"]
    assert W.doc_from_spec(it, {"doc": "3", "not_placed": "excerpt_only"}, text, ix)[1]["drop"] == "not_placed:excerpt_only"
    assert W.doc_from_spec(it, {**d, "target_lang": "fr"}, text, ix)[1]["drop"] == "target_lang_unsupported:fr"
    bad = json.loads(json.dumps(d))
    bad["spans"]["translation"]["pieces"][0]["sha1"] = "0" * 40
    assert W.doc_from_spec(it, bad, text, ix)[1]["drop"] == "span_cut_failed"
    nope = span_spec("4", text, it.key, "pgp_edition", shelfmarks=["T-S 77.7"])
    assert W.doc_from_spec(it, nope, text, ix)[1]["drop"] == "no_pgp_edition"
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(json.dumps({"items": [{"dir": "src_a", "slug": "p1", "commentary": True, "docs": [
        {"doc": "1", "translation": [[10, 12]], "edition": [[4, 6]], "target_lang": "en"},
        {"doc": "2", "translation": [[10, 99]], "target_lang": "en"}]}]}), encoding="utf-8")
    merged = W.fill_specs([spec_file], root, ix)
    assert len(merged["items"]) == 1 and len(merged["items"][0]["docs"]) == 1
    assert merged["items"][0]["errors"][0]["doc"] == "2" and merged["items"][0]["commentary"] is True


# ---------------------------------------------------------------------------------- hold-outs and splits
def test_item_holdout_reason_and_split(tmp_path: Path) -> None:
    """An item naming a held document holds out all its pairs (source level, vision flag off); else D14."""
    root = tmp_path / "lit"
    write_dir(root, "src_a", [{"slug": "p1", "text": item_text()},
                              {"slug": "p2", "text": item_text(), "meta": {"shelfmarks_mentioned": ["T-S 9.9"]}}])
    items = W.load_items(root, ["src_a"])
    h = FakeHoldouts(bench={"T-S 9.9": "pgp131"})
    assert W.item_holdout_reason(items["src_a/text/p1.txt"], "HELD-DOC here", h, fake_mention) == "text:test"
    assert W.item_holdout_reason(items["src_a/text/p1.txt"], "clean", h, fake_mention) is None
    assert W.item_holdout_reason(items["src_a/text/p2.txt"], "clean", h, fake_mention) == "shelfmark:pgp131"
    doc = W.WebDoc(source="src_a", doc="p1:1", shelfmarks=["T-S 5.5"], target_lang="en", edition="x", translation="y",
                   translation_raw="y", script="hebrew", lang_class="JA", input_source="s", licence=None,
                   licence_class="pd")
    ix = BP.PgpIndex()
    assert W.assign_web_split(doc, [], "text:test", h, ix) == ("heldout", W.ITEM_MENTION, "item_mention:text:test")
    assert W.assign_web_split(doc, [], None, h, ix)[1] == "hash"
    assert W.assign_web_split(doc, [], None, FakeHoldouts(evals={"T-S 5.5": "dev"}), ix)[:2] == \
        ("heldout", "eval_doc:shelfmark")


def test_book_pair_keys(tmp_path: Path) -> None:
    """L5 translate rows register their shelfmark keys and pgpids per family; other families do not."""
    path = tmp_path / "book_pairs.jsonl"
    R.write_jsonl(path, [{"messages": [], "meta": {"family": "translate_en", "shelfmark": "T-S 5.5", "pgpids": ["10"],
                                                   "doc_id": "book:b:1"}},
                         {"messages": [], "meta": {"family": "analyse_document", "shelfmark": "T-S 6.6",
                                                   "doc_id": "book:b:2"}}])
    keys = W.book_pair_keys(path)
    assert keys[("translate_en", "pgpid:10")] == "book:b:1"
    assert not any(k[0] == "analyse_document" for k in keys) and W.book_pair_keys(tmp_path / "none.jsonl") == {}


# ---------------------------------------------------------------------------------- end to end
def test_build_dedupe_holdouts_rows_and_outputs(tmp_path: Path) -> None:
    """End to end: PGP-translated, L5-translated and repeated documents drop; eval-only items go to heldout;
    rows carry the web meta; outputs, audit sample and the commentary count are written."""
    p = paths_for(tmp_path)
    root = p.academic_literature
    write_dir(root, "src_a", [
        {"slug": "p1", "text": item_text(), "review": {"translation_level": "full", "translators": ["T One"],
                                                       "has_commentary": True},
         "meta": {"shelfmarks_mentioned": ["T-S 5.5"]}},
        {"slug": "p2", "text": item_text("HELD-DOC"), "review": {"translation_level": "partial", "has_commentary": True},
         "meta": {"shelfmarks_mentioned": ["T-S 6.6"]}},
        {"slug": "p3", "text": item_text(), "review": {"translation_level": "full"}},
        {"slug": "p4", "text": item_text(), "review": {"translation_level": "full"}},
        {"slug": "p5", "text": item_text(), "review": {"translation_level": "full"}}])
    ix = pairs_index(tmp_path, [
        {"pgpid": "10", "shelfmark": "T-S 7.7", "split": "train", "translation_en": "An English translation.",
         "edition_text": "אלף בית גימל דלת הא וו זין חית טית יוד", "editions": [], "translations": []}])
    items = W.load_items(root, ["src_a"])
    texts = {k: it.text() for k, it in items.items()}
    spans = {"items": [
        {"dir": "src_a", "slug": "p1", "docs": [span_spec("1", texts["src_a/text/p1.txt"], "src_a/text/p1.txt")]},
        {"dir": "src_a", "slug": "p2", "docs": [span_spec("1", texts["src_a/text/p2.txt"], "src_a/text/p2.txt",
                                                          shelfmarks=["T-S 6.6"])]},
        {"dir": "src_a", "slug": "p3", "docs": [span_spec("1", texts["src_a/text/p3.txt"], "src_a/text/p3.txt",
                                                          shelfmarks=["T-S 7.7"])]},
        {"dir": "src_a", "slug": "p4", "docs": [span_spec("1", texts["src_a/text/p4.txt"], "src_a/text/p4.txt",
                                                          shelfmarks=["T-S 8.8"])]},
        {"dir": "src_a", "slug": "p5", "docs": [span_spec("1", texts["src_a/text/p5.txt"], "src_a/text/p5.txt")]},
        {"dir": "src_b", "slug": "zz", "docs": []}]}
    book = {("translate_en", k): "book:b:1" for k in BP.BookDoc(
        source="b", doc="1", shelfmarks=["T-S 8.8"], target_lang="en", edition="", translation="", translation_raw="",
        script="hebrew", lang_class="JA", input_source="", licence=None, licence_class=None).keys}
    res = W.build(p, {"dirs": ["src_a"]}, spans, holdouts=FakeHoldouts(), ix=ix, noiser=False, tc=FakeTC(),
                  items=items, texts=texts, independence=False, book_keys=book, mention_reason=fake_mention)
    recs = {d["item"]: d for d in res["docs"]}
    assert recs["p1"]["translate"]["rows"] >= 1 and recs["p1"]["split"] != "heldout"
    assert recs["p2"]["split"] == "heldout" and recs["p2"]["split_reason"] == W.ITEM_MENTION
    assert recs["p3"]["translate"]["drop"] == "dup_pgp_translation"
    assert recs["p4"]["translate"]["drop"] == "dup_book_pairs"
    assert recs["p5"]["translate"]["drop"] == "dup_within_web"
    assert res["report"]["drops"].get("item_not_configured") == 1
    row = next(r for r in res["rows"] if r["meta"]["web_doc"] == "p1:1" and r["meta"]["condition"] == "edition")
    m = row["meta"]
    assert m["source"] == "web:src_a:p1" and m["doc_id"] == "web:src_a:p1:1" and m["translator"] == "T One"
    assert m["edition_source"] == "web" and m["page_keys"] == ["src_a/text/p1.txt"] and m["licence_class"] == "pd"
    assert "[1]" not in row["messages"][2]["content"] and R.validate_row(row) == []
    held = next(r for r in res["rows"] if r["meta"]["web_doc"] == "p2:1")
    assert held["meta"]["split"] == "heldout" and held["meta"]["held_out_vision_benchmark"] is False
    assert held["meta"]["item_holdout_reason"] == "text:test"
    com = res["report"]["commentary"]
    assert com["items"] == 2 and com["eval_only"] == 1 and com["train_eligible"] == 1
    man = W.write_outputs(res, p, build_args={"test": True}, exact_tokens=False)
    assert man["checks"]["duplicate_row_ids"] == 0 and man["checks"]["heldout_rows_in_train"] == 0
    assert not man["checks"]["schema_problems"] and man["checks"]["target_not_verbatim"] == 0
    assert man["pairs_by_lang_split"]["en"].get("heldout") == 1 and man["translators"]
    assert R.load_jsonl(p.out("families", "web_pairs.jsonl")) and R.load_jsonl(p.out("audit", W.AUDIT_NAME))
    assert set(man["per_source"]) == {"src_a"}


def test_scrub_rows_drops_message_hits_and_scrubs_meta() -> None:
    """Rows whose messages name a scrubbed project are dropped; meta strings are scrubbed."""
    from src.finetuning.text27b import textnorm as tn
    name = "".join(chr(c) for c in (70, 114, 105, 101, 100, 98, 101, 114, 103))  # assembled at runtime (F12)
    assert tn.has_source_names(f"the {name} collection")
    ok = {"messages": R.make_messages("u", "a"), "meta": {"shelfmark": f"MSS {name} 1-001", "family": "translate_en"}}
    bad = {"messages": R.make_messages(f"u {name} collection", "a"), "meta": {}}
    kept, dropped = W.scrub_rows([ok, bad])
    assert dropped == 1 and len(kept) == 1 and not tn.has_source_names(json.dumps(kept[0]["meta"]))


def test_candidates_and_find_pgp(tmp_path: Path) -> None:
    """The reading list resolves translated shelfmarks with PGP flags; find_pgp looks up keys and edition phrases."""
    root = tmp_path / "lit"
    write_dir(root, "src_a", [
        {"slug": "p1", "text": item_text(), "review": {"translation_level": "full", "translation_lang": ["en"],
                                                       "shelfmarks_translated": ["T-S 5.5", "T-S 7.7"]}},
        {"slug": "p2", "text": "x", "review": {"translation_level": "excerpts"}}])
    ix = pairs_index(tmp_path, [
        {"pgpid": "10", "shelfmark": "T-S 7.7", "split": "train", "translation_en": "x",
         "edition_text": "\n".join(JA_LINES), "editions": [], "translations": []},
        {"pgpid": "11", "shelfmark": "T-S 5.5", "split": "train", "edition_text": "בשמ רחמ אלף",
         "editions": [], "translations": []}])
    items = W.load_items(root, ["src_a"])
    c = W.candidates(items, ix, FakeHoldouts(), {k: None for k in items})
    assert len(c) == 1 and [d["suggested_input"] for d in c[0]["docs"]] == ["pgp_edition", "skip:pgp_has_translation"]
    assert W.find_pgp("T-S 7.7", ix).startswith("10\t") and "translation_en=True" in W.find_pgp("T-S 7.7", ix)
    assert W.find_pgp("וצל כתאבך", ix, grep=True).startswith("10\t") and "no PGP" in W.find_pgp("T-S 99.9", ix)
    assert "edition" in W.show_pgp("10", ix) and "not in the PGP" in W.show_pgp("77", ix)
    assert W.show_item("src_a/p1", root).split("\n")[0].endswith("Title of the post")
