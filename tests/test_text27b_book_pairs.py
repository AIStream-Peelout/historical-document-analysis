# File name: test_text27b_book_pairs.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``book_pairs`` (L5): span cutting and verification, normalisers, entry parsing,
PGP matching / dedupe, D14 splits, the D15 pair check, row building and the end-to-end build.

Everything runs on synthetic fixtures (tiny entries, page texts and PGP pairs written to ``tmp_path``);
no real corpus, holdings or machine paths are used.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import book_pairs as BP
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R


# ---------------------------------------------------------------------------------- fixtures
class FakeTC:
    """Token counter stand-in (3 characters per token)."""

    def count(self, text: str) -> int:
        """Approximate token count.

        :param text: Text.
        :returns: Tokens.
        """
        return len(text or "") // 3


class FakeHoldouts:
    """Hold-out stand-in: benchmark / eval tiers by pgpid and by shelfmark substring."""

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
        if shelfmark and shelfmark in table:
            return table[shelfmark]
        return None

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


class FakeNoiser:
    """Noise stand-in: drops every third character."""

    def available(self, script: str) -> bool:
        """Every script is available.

        :param script: Script.
        :returns: True.
        """
        return True

    def apply(self, text: str, script: str, rng: Any) -> tuple:
        """Deterministic 'noise'.

        :param text: Edition.
        :param script: Script.
        :param rng: Unused RNG.
        :returns: ``(noised, target cer)``.
        """
        return "".join(ch for i, ch in enumerate(text) if i % 3), 0.2


AR_LINES = ["بسم الله الرحمن الرحيم", "كتابي الى الشيخ ابي علي حسن بن داود", "وصل كتابك وفهمته والحمد لله"]
EN_LINES = ["In the name of God, the Merciful, the Compassionate.", "My letter to the elder Abū ʿAlī Ḥasan ibn Dāwūd.",
            "Your letter arrived and I understood it, praise be to God."]
JA_LINES = ["בשמ רחמ", "כתאבי אלי אלשיך אבו עלי חסן בן דאוד", "וצל כתאבך ופהמתה ואלחמד ללה"]


def lines_entry(no: int, ed: List[str], tr: List[str], side: Optional[str] = None, shelfmark: str = "T-S 1.1",
                languages: Optional[List[str]] = None) -> Dict[str, Any]:
    """A synthetic ``lines``-layout entry.

    :param no: Entry number.
    :param ed: Edition lines.
    :param tr: Translation lines.
    :param side: Side of every line.
    :param shelfmark: Shelfmark.
    :param languages: Stated languages.
    :returns: Entry dict.
    """
    return {"entry_no": no, "shelfmark": shelfmark, "shelfmarks": [shelfmark], "languages": languages or [],
            "transcription_lines": [{"n": i + 1, "side": side, "part": None, "text": t} for i, t in enumerate(ed)],
            "translation_lines": [{"n": i + 1, "side": side, "part": None, "text": t} for i, t in enumerate(tr)],
            "discussion": "A discussion.", "page": 10, "pdf_page": 12}


def make_doc(source: str = "bk", doc: str = "1", edition: str = "\n".join(AR_LINES),
             translation: str = "\n".join(EN_LINES), lang: str = "en", script: str = "arabic",
             lang_class: str = "AR", located: bool = False, shelfmarks: Optional[List[str]] = None) -> BP.BookDoc:
    """A synthetic :class:`BookDoc`.

    :param source: Source slug.
    :param doc: Document label.
    :param edition: Edition text.
    :param translation: Translation text.
    :param lang: Target language.
    :param script: Edition script.
    :param lang_class: lang_class.
    :param located: Located-span document.
    :param shelfmarks: Shelfmarks.
    :returns: The document.
    """
    return BP.BookDoc(source=source, doc=doc, shelfmarks=shelfmarks if shelfmarks is not None else ["T-S 1.1"],
                      target_lang=lang, edition=edition, translation=translation, translation_raw=translation,
                      script=script, lang_class=lang_class, input_source=f"book_edition:{source}", licence="lic",
                      licence_class="cc-by", located=located)


def ctx(holdouts: Any = None, ix: Optional[BP.PgpIndex] = None, noiser: Any = None) -> BP.BuildContext:
    """A build context with stand-ins.

    :param holdouts: Hold-outs stand-in.
    :param ix: PGP index.
    :param noiser: Noiser.
    :returns: Context.
    """
    return BP.BuildContext(holdouts=holdouts or FakeHoldouts(), ix=ix or BP.PgpIndex(), noiser=noiser, tc=FakeTC())


def write_pairs(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write a synthetic pgp_parallel file.

    :param path: Destination.
    :param rows: Pairs rows.
    :returns: The path.
    """
    R.write_jsonl(path, rows)
    return path


# ---------------------------------------------------------------------------------- normalisers
def test_rejoin_indented_joins_continuations_and_keeps_labels() -> None:
    """Indented lines open lines, flush lines continue, labels stand alone, hyphen breaks re-join."""
    raw = "  first line of the\ntranslation\n§ verso\nVerso text contrib-\nuted here\n   second"
    out, n = BP.rejoin_indented(raw)
    assert out.split("\n") == ["first line of the translation", "§ verso", "Verso text contributed here", "second"]
    assert n == 2


def test_join_hyphenated_keeps_compound_numbers() -> None:
    """A hyphen between two dictionary words is kept when the joined form is not a word (if a word list exists)."""
    words = BP.tn.common_words()
    joined = BP.join_hyphenated("fifty-", "four dinars")
    if words and "fifty" in words and "four" in words:
        assert joined == "fifty-four dinars"
    assert BP.join_hyphenated("contrib-", "uted") == "contributed"


def test_rejoin_blank_separated_and_labels() -> None:
    """Rows without a blank row between them form one line; label rows stand alone."""
    out, n = BP.rejoin_blank_separated("אחד שתים\nשלוש\n   \nארבע\n§ verso\nחמש")
    assert out.split("\n") == ["אחד שתים שלוש", "ארבע", "§ verso", "חמש"]
    assert n == 1


def test_prose_paragraphs() -> None:
    """Wrapped prose becomes paragraphs; short sentence-final lines end a paragraph."""
    raw = ("This document is a long introduction that runs over several printed\n"
           "lines of the book and discusses the partnership agreement in its full\n"
           "detail, including the partners and the commodities of the partner-\n"
           "ship and the terms. Short end.\n"
           "A new paragraph starts here and runs for a while longer than half a line.")
    out, _ = BP.prose_paragraphs(raw)
    paras = out.split("\n\n")
    assert len(paras) == 2
    assert "partnership" in paras[0] and paras[0].endswith("Short end.")


@pytest.mark.parametrize("text,expected", [
    ("Fustāt.37 and", "Fustāt. and"), ("al-Karrām42 for", "al-Karrām for"), ("(unintelligible)36", "(unintelligible)"),
    ("TS 13 J 3.5 and", "TS 13 J 3.5 and"), ("Torah 3:2,", "Torah 3:2,"), ("l. 11)", "l. 11)"),
    ("11/2 dirhams", "11/2 dirhams"), ("וחפצי 6אנוס", "וחפצי אנוס"), ("year 1372 (of", "year 1372 (of")])
def test_drop_footnote_refs(text: str, expected: str) -> None:
    """Glued footnote numbers go; free-standing numbers and number parts stay.

    :param text: Input.
    :param expected: Output.
    """
    assert BP.drop_footnote_refs(text)[0] == expected


def test_drop_digit_remnants_keeps_line_start_marker() -> None:
    """A line-number remnant is replaced by an indent; digit-only lines go; amounts stay."""
    out, n = BP.drop_digit_remnants(" 0 they finish\n7\n 12 dinars were paid")
    assert out.split("\n") == [" they finish", " 12 dinars were paid"]
    assert n == 2


def test_bracket_headings_side_and_heading_labels() -> None:
    """Section headings are dropped (also when run into text); side / heading lines become labels."""
    out, n = BP.drop_bracket_headings("[א. פתיחה]\n(1) בשמ\n[ב. ענין אחר] טקסט")
    assert out == "(1) בשמ\nטקסט" and n == 2
    assert BP.side_labels("Recto\ntext\n[ע\"ב]")[0] == "§ recto\ntext\n§ verso"
    lab, _ = BP.heading_labels("Transcription of TS 1.2\nContinuation of Document on Verso\nText shifted 90 degrees, x")
    assert lab.split("\n")[:2] == ["§ TS 1.2", "§ Continuation of Document on Verso"]
    assert lab.split("\n")[2].startswith("§ Text shifted 90 degrees")
    assert BP.latin_labels("שורה עברית\nWritten in the margin:")[0] == "שורה עברית\n§ Written in the margin"


def test_ocr_residue_normalisers() -> None:
    """Latin sigla in brackets, OCR footnote residue and bare numbers go; line markers stay."""
    assert BP.drop_latin_brackets("ושימחתני}[3.2,f J 1 TS](1) הידיעה")[0] == "ושימחתני} (1) הידיעה"
    assert BP.drop_ocr_refs("ה'מחודש20′ באניית [61] (4) שלח [ג6א] ל(ג)")[0] == "ה'מחודש באניית (4) שלח ל(ג)"
    assert BP.drop_bare_numbers("קערות 'קצעה'. 7 (4) ייטיב בשנת 1135")[0] == "קערות 'קצעה'. (4) ייטיב בשנת 1135"


def test_split_at_markers() -> None:
    """Prose with inline markers becomes one line per marker; bracketed sides become labels."""
    out, n = BP.split_at_markers("(1) בשמ' רחמ'. (2) הגיע מכתב\nהדרתו (3) קראתיו [ע\"ב] (1) אל הדרת")
    assert out.split("\n") == ["בשמ' רחמ'.", "הגיע מכתב הדרתו", "קראתיו", "§ verso", "אל הדרת"]
    assert n == 4
    assert BP.split_at_markers("שורה אחת (1) בלבד")[1] == 0


def test_normalise_and_words_subset() -> None:
    """``normalise`` lists the ops that changed something; ``words_subset`` flags added words, not labels."""
    text, ops = BP.normalise("Recto\nword.12 more", ["side_labels", "drop_footnote_refs", "drop_blank_lines"])
    assert text == "§ recto\nword. more" and ops == ["L5:side_labels", "L5:drop_footnote_refs"]
    with pytest.raises(KeyError):
        BP.normalise("x", ["no_such_op"])
    assert BP.words_subset("§ verso\nthe partnership", "the partner-\nship")[0]
    ok, missing = BP.words_subset("the invented word", "the word")
    assert not ok and missing == ["invented"]


def test_strip_latin_runs() -> None:
    """Latin notes inside an RTL line go; short sigla stay; a pure-Latin line empties."""
    line, n = BP.strip_latin_runs("A note on the source MS A 1, fol. 2r يحكى ان")
    assert line == "يحكى ان" and n == 1
    assert BP.strip_latin_runs("p. 194 وظهره")[0] == "p. 194 وظهره"


def test_paragraphs_to_lines() -> None:
    """Numbered paragraphs, inline markers, continuations and editorial notes."""
    lines, ops = BP.paragraphs_to_lines(["[Written in the hand of a scribe]", "1. With the help of God.", "2. It should arrive",
                                         "delivered to him.", "(3) After greetings (4) we inform you (5) that"])
    assert lines == ["With the help of God.", "It should arrive delivered to him.", "After greetings", "we inform you", "that"]
    assert set(ops) == {"editorial_note_dropped", "numbered_paragraphs", "continuation_joined", "split_paren_markers"}


# ---------------------------------------------------------------------------------- spans
def test_cut_piece_verifies_verbatim() -> None:
    """A piece re-cuts only when head, tail and sha1 still match the page text."""
    page = ("running head\n12\nline one of the translation text, which is long\n"
            "middle line here\nline two of the translation text, also long\nfootnote 1")
    pc = BP.piece_from_lines("w/p/page_001.json", page, 2, 4)
    text, why = BP.cut_piece(pc, page)
    assert why == "ok" and text.startswith("line one") and text.endswith("also long") and text in page
    assert BP.cut_piece(pc, None)[1] == "page_missing"
    assert BP.cut_piece(pc, page.replace("line one", "line uno"))[1] == "head_mismatch"
    assert BP.cut_piece(pc, page.replace("also long", "also lonG"))[1] == "tail_mismatch"
    assert BP.cut_piece(pc, page.replace("middle line", "middle 1ine"))[1] == "sha1_mismatch"
    assert BP.cut_piece(BP.Piece(page="x", lines=(5, 9)), page)[1] == "range"
    cols = BP.piece_from_lines("k", page, 2, 2, cols=(5, 8))
    assert BP.cut_piece(cols, page)[0] == "one"
    assert BP.Piece.from_dict(cols.to_dict()) == cols


def test_cut_span_joins_and_fails_on_any_piece() -> None:
    """Pieces join with a newline; one bad piece fails the span."""
    pages = {"a": "x\nfirst", "b": "second\ny"}
    span = {"pieces": [BP.piece_from_lines("a", pages["a"], 1, 1).to_dict(),
                       BP.piece_from_lines("b", pages["b"], 0, 0).to_dict()]}
    raw, info = BP.cut_span(span, pages)
    assert raw == "first\nsecond" and info["pages"] == ["a", "b"]
    raw, info = BP.cut_span(span, {"a": pages["a"]})
    assert raw is None and info["fail"] == "page_missing"


def test_doc_from_spans_span_and_pgp_inputs() -> None:
    """Span documents: words-added guard, book transcription input, PGP edition input, missing edition."""
    page = "Appendix\nTranscription of X\n" + "\n".join(JA_LINES) + "\nTranslation of X\n" + \
           "\n".join("  " + t for t in EN_LINES)
    pages = {"w/p/page_001.json": page}
    spans = {"edition": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 2, 4).to_dict()], "normalise": []},
             "translation": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 6, 8).to_dict()],
                             "normalise": ["rejoin_indented"]}}
    src = {"slug": "bk", "target_lang": "en", "input": "span", "licence": "lic", "licence_class": "copyright"}
    doc, info = BP.doc_from_spans(src, {"doc": "1", "shelfmarks": ["T-S 1.1"], "spans": spans}, pages)
    assert doc is not None and doc.located and doc.edition == "\n".join(JA_LINES)
    assert doc.translation.split("\n") == EN_LINES and doc.script == "hebrew"
    src_pgp = {**src, "input": "pgp_edition"}
    doc2, _ = BP.doc_from_spans(src_pgp, {"doc": "2", "spans": {"translation": spans["translation"]}}, pages,
                                lambda h, s: ("\n".join(AR_LINES), "slug-x", "77"))
    assert doc2.input_source == "pgp_digital_edition:slug-x" and "77" in doc2.pgp_hints and doc2.script == "arabic"
    doc3, info3 = BP.doc_from_spans(src_pgp, {"doc": "3", "spans": {"translation": spans["translation"]}}, pages,
                                    lambda h, s: None)
    assert doc3 is None and info3["drop"] == "no_pgp_edition"
    bad = {"translation": {"pieces": spans["translation"]["pieces"], "normalise": []}}
    doc4, info4 = BP.doc_from_spans(src, {"doc": "4", "spans": bad}, pages)
    assert doc4 is None and info4["drop"] == "span_cut_failed" or info4.get("drop") == "no_edition"


# ---------------------------------------------------------------------------------- entries
def test_doc_from_lines_entry_blocks_and_realign() -> None:
    """Two sides render with labels; mislabelled translation parts are realigned when counts agree."""
    e = lines_entry(1, AR_LINES, EN_LINES, side="recto")
    e["transcription_lines"][2]["side"] = "verso"
    e["translation_lines"][2]["side"] = "verso"
    doc = BP.doc_from_lines_entry(e, {"slug": "bk", "target_lang": "en"})
    assert doc.edition.startswith("§ recto\n") and "\n\n§ verso\n" in doc.translation and not doc.flags
    e2 = lines_entry(2, AR_LINES, EN_LINES)
    e2["transcription_lines"][2]["part"] = "margin"
    for ln in e2["translation_lines"]:
        ln["part"] = "margin"
    doc2 = BP.doc_from_lines_entry(e2, {"slug": "bk"})
    assert doc2.flags == ["translation_parts_realigned"] and BP.rendered_labels(doc2.translation_blocks) == ["", "margin"]
    e3 = lines_entry(3, AR_LINES, EN_LINES[:2])
    e3["transcription_lines"][2]["part"] = "margin"
    assert BP.doc_from_lines_entry(e3, {"slug": "bk"}).flags == ["block_keys_differ"]
    assert BP.doc_from_lines_entry(lines_entry(4, JA_LINES, EN_LINES, languages=["Judaeo-Arabic"]),
                                   {"slug": "bk"}).lang_class == "JA"


def test_doc_from_paragraph_entry() -> None:
    """Script filter, Latin-run strip, heading block drop and paragraph lines."""
    e = {"entry_no": 5, "shelfmarks": [], "transcription_lines": [
        {"n": 1, "part": "a", "script": "arabic", "text": "Source note MS Orient. A 1 " + AR_LINES[0]},
        {"n": 2, "part": "b", "script": "arabic", "text": AR_LINES[1]},
        {"n": None, "part": "b", "script": "latin", "text": "A Latin transliteration row"}],
         "translation_paragraphs": [{"part": None, "text": "Accounts of a Journey"},
                                    {"part": "a", "text": "1. " + EN_LINES[0]}, {"part": "b", "text": EN_LINES[1]}]}
    doc = BP.doc_from_paragraph_entry(e, {"slug": "bk"})
    assert doc.edition == f"§ a\n{AR_LINES[0]}\n\n§ b\n{AR_LINES[1]}"
    assert doc.translation == f"§ a\n{EN_LINES[0]}\n\n§ b\n{EN_LINES[1]}"
    assert doc.flags == ["heading_block_dropped"] and doc.provenance["edition_latin_runs_removed"] == 1


# ---------------------------------------------------------------------------------- PGP / splits
def test_pgp_index_matching_and_dedupe(tmp_path: Path) -> None:
    """PGP routes: slug + number, shelfmark key, edition text; translation presence; v0_1 split."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", [
        {"pgpid": "10", "shelfmark": "T-S 1.1", "split": "train", "translation_en": "x", "edition_text": "\n".join(AR_LINES),
         "editions": [{"source_slug": "book-slug", "location": "doc. 7", "raw": "\n".join(AR_LINES)}], "translations": []},
        {"pgpid": "11", "shelfmark": "T-S 2.2", "split": "dev", "edition_text": "\n".join(JA_LINES), "editions": [],
         "translations": []}])
    ix = BP.PgpIndex.build(pairs, v01={"11": type("V", (), {"splits": {"dev": 3}, "tasks": {}})()})
    assert BP.PgpIndex.location_number("doc. 7") == "7" and BP.PgpIndex.location_number("III-10a") == "III-10a"
    assert ix.edition_matches("\n".join(AR_LINES)) == [("10", 1.0)]
    d = make_doc(shelfmarks=["T-S 2.2"])
    d.pgp_number = "7"
    found, how = BP.match_pgpids(d, ix, ["book-slug"])
    assert found == ["10", "11"] and how == ["pgp_slug_number", "shelfmark_key"]
    assert ix.has_translation("10", "en") and not ix.has_translation("11", "en")
    assert ix.v01_split("11") == "dev" and ix.v01_split("10") is None


def test_hash_split_and_assign_split() -> None:
    """D14: hold-out tiers → heldout; v0_1 train group → train; else a stable 80/10/10 hash."""
    splits = [BP.hash_split(f"k{i}") for i in range(2000)]
    assert BP.hash_split("k1") == BP.hash_split("k1")
    assert 0.75 < splits.count("train") / 2000 < 0.85 and splits.count("dev") > 120 and splits.count("test_indist") > 120
    ix = BP.PgpIndex(v01={"5": type("V", (), {"splits": {"train": 2}, "tasks": {}})()})
    hold = FakeHoldouts(bench={"9": "pgp131"}, evals={"T-S 3.3": "key:test"})
    assert BP.assign_split(make_doc(), ["9"], hold, ix)[:2] == ("heldout", "benchmark:pgpid")
    assert BP.assign_split(make_doc(shelfmarks=["T-S 3.3"]), [], hold, ix)[0] == "heldout"
    assert BP.assign_split(make_doc(), ["5"], hold, ix)[:2] == ("train", "v0_1_train_group")
    assert BP.assign_split(make_doc(), [], hold, ix)[1] == "hash"


# ---------------------------------------------------------------------------------- pair check
def test_pair_check_band_rank_and_enforcement() -> None:
    """Length band always decides; the overlap rank decides only for located spans."""
    he_ed = "שלמה בן יעקב אלעדני כתב אלי אבו זכרי סלאם\nאלמהדיה ופלפל וזנגביל"
    he_tr = "שלמה בן יעקב העדני כתב אל אבו זכרי שלום\nהמהדיה ופלפל וזנגביל"
    decoy_ed = "פלאן בן פלאן פי מצר ואלקאהרה מן גיר שי\nכתאב אכר"
    doc = make_doc(edition=he_ed, translation=he_tr, lang="he", script="hebrew", lang_class="JA", located=True)
    decoy = make_doc(doc="2", edition=decoy_ed, lang="he", script="hebrew", located=True)
    chk = BP.pair_check(doc, he_tr, [doc, decoy])
    assert chk["ok"] and chk["own"] > chk["best_other"] and chk["decoys"] == 1
    swapped = BP.pair_check(decoy, he_tr, [doc, decoy])
    assert not swapped["ok"] and swapped["reason"] in ("pair_check_rank", "pair_check_floor", "length_ratio_F1")
    entry_doc = make_doc(edition=decoy_ed, lang="he", script="hebrew", located=False)
    loose = BP.pair_check(entry_doc, he_tr, [entry_doc, doc])
    assert loose["enforced"] is False and (loose["ok"] or loose["reason"] == "length_ratio_F1")
    short = BP.pair_check(make_doc(), "x", [])
    assert not short["ok"] and short["reason"] == "length_ratio_F1"
    assert "אדוני" in BP.common_edition_words(["אדוני א", "אדוני ב", "אדוני ג", "ד"], share=0.5, min_docs=2)


# ---------------------------------------------------------------------------------- rows
def test_translate_rows_conditions() -> None:
    """Arabic documents: edition + B9 rescript, no noised row; Hebrew train documents get a noised row."""
    c = ctx(noiser=FakeNoiser())
    rows, info = BP.translate_rows(make_doc(), "train", ["10"], "hash", None, [], c)
    assert [r["meta"]["condition"] for r in rows] == ["edition", "edition_rescript"]
    resc = rows[1]
    assert "Hebrew script" in resc["messages"][1]["content"] and resc["meta"]["input_script"] == "hebrew"
    assert all(R.validate_row(r) == [] for r in rows)
    assert len({r["meta"]["row_id"] for r in rows}) == 2 and rows[0]["meta"]["source"] == "book:bk"
    ja = make_doc(edition="\n".join(JA_LINES), script="hebrew", lang_class="JA")
    rows2, _ = BP.translate_rows(ja, "train", [], "hash", None, [], c)
    assert [r["meta"]["condition"] for r in rows2] == ["edition", "noised_edition"]
    assert rows2[1]["meta"]["noise_class"] == "synthetic" and rows2[1]["meta"]["input_cer"] is not None
    rows3, _ = BP.translate_rows(ja, "dev", [], "hash", None, [], c)
    assert [r["meta"]["condition"] for r in rows3] == ["edition"]
    assert BP.translate_rows(make_doc(script="latin"), "train", [], "hash", None, [], c)[1]["drop"] == "script_not_supported"
    flagged = make_doc()
    flagged.flags = ["block_keys_differ"]
    assert BP.translate_rows(flagged, "train", [], "hash", None, [], c)[1]["drop"] == "block_labels_mismatch"


def test_translate_rows_block_split_when_over_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An over-long document is split per side when the edition and translation blocks align."""
    monkeypatch.setattr(BP, "MAX_ROW_TOKENS", 520)
    ed_r, ed_v = AR_LINES * 3, AR_LINES[::-1] * 3
    tr_r, tr_v = EN_LINES * 3, EN_LINES[::-1] * 3
    e = lines_entry(1, ed_r + ed_v, tr_r + tr_v, side="recto")
    for ln in e["transcription_lines"][len(ed_r):] + e["translation_lines"][len(tr_r):]:
        ln["side"] = "verso"
    doc = BP.doc_from_lines_entry(e, {"slug": "bk", "target_lang": "en"})
    rows, info = BP.translate_rows(doc, "dev", [], "hash", None, [], ctx())
    assert info.get("split_into_blocks") == 2
    assert sorted({r["meta"]["side"] for r in rows}) == ["recto", "verso"]


def test_analyse_rows_gate() -> None:
    """Grounded introductions pass tier A′; ungrounded ones and short ones are dropped."""
    doc = make_doc(edition="\n".join(JA_LINES), script="hebrew", lang_class="JA")
    good = ("This letter was written to the elder Abū ʿAlī Ḥasan b. Dāwūd, who received it and understood it. "
            "The writer reports that the letter arrived and thanks God; the document is a short business note "
            "and its formulae are typical of letters of the period. ") * 2
    rows, info = BP.analyse_rows(doc, good, good, "intro", "train", [], "hash", None, ctx())
    assert rows and info["grounding_tier"] in ("A_prime", "B") and rows[0]["meta"]["family"] == "analyse_document"
    assert rows[0]["meta"]["grounding_share"] >= BP.GROUNDING_A
    bad = ("This letter mentions Zebulon, Ishmael, Methuselah, Nebuchadnezzar and Belshazzar of Babylon, all "
           "otherwise unknown, in the year 1188 and again in 1192 during travels to Samarkand and Bukhara. ") * 2
    assert BP.analyse_rows(doc, bad, bad, "intro", "train", [], "hash", None, ctx())[1]["drop"] == "analysis_grounding_fail"
    assert BP.analyse_rows(doc, "Too short.", "Too short.", "intro", "train", [], "hash", None,
                           ctx())[1]["drop"] == "analysis_short_after_strip"


# ---------------------------------------------------------------------------------- end to end
def _paths(tmp_path: Path) -> P.Paths:
    """A synthetic paths config under ``tmp_path``.

    :param tmp_path: Pytest temporary directory.
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


def test_build_and_write_outputs(tmp_path: Path) -> None:
    """End to end on synthetic books: dedupe against PGP, hold-out split, span source, manifest and audit."""
    p = _paths(tmp_path)
    book = p.book_editions_nas / "bk"
    book.mkdir(parents=True)
    R.write_jsonl(book / "entries.jsonl", [
        lines_entry(1, AR_LINES, EN_LINES, shelfmark="T-S 1.1"),
        lines_entry(2, [AR_LINES[1], AR_LINES[0], AR_LINES[2]], [EN_LINES[1], EN_LINES[0], EN_LINES[2]],
                    shelfmark="T-S 4.4"),
        lines_entry(3, [AR_LINES[2], AR_LINES[1]], [EN_LINES[2], EN_LINES[1]], shelfmark="T-S 9.9")])
    pairs = write_pairs(tmp_path / "pairs.jsonl", [
        {"pgpid": "10", "shelfmark": "T-S 1.1", "split": "train", "translation_en": "x", "edition_text": "y",
         "editions": [], "translations": []}])
    ix = BP.PgpIndex.build(pairs)
    page = "Appendix\n" + "\n".join(JA_LINES) + "\n" + "\n".join("  " + t for t in EN_LINES)
    spans = {"sources": [{"slug": "sp", "work": "w", "target_lang": "en", "input": "span", "licence": "lic",
                          "licence_class": "copyright", "docs": [{"doc": "1", "shelfmarks": ["T-S 5.5"], "spans": {
                              "edition": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 1, 3).to_dict()]},
                              "translation": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 4, 6).to_dict()],
                                              "normalise": ["rejoin_indented"]}}}]}]}
    cfg = {"nas_books": [{"slug": "bk", "dir": "bk", "layout": "lines", "target_lang": "en", "licence": "lic",
                          "licence_class": "cc-by"}]}
    res = BP.build(p, cfg, spans, holdouts=FakeHoldouts(bench={"T-S 9.9": "pgp131"}), ix=ix, noiser=False, tc=FakeTC(),
                   page_texts={"w/p/page_001.json": page})
    recs = {(d["source"], d["doc"]): d for d in res["docs"]}
    assert recs[("bk", "1")]["translate"]["drop"] == "dup_pgp_translation"
    assert recs[("bk", "3")]["split"] == "heldout" and recs[("bk", "3")]["holdout_reason"] == "pgp131"
    assert recs[("sp", "1")]["translate"]["rows"] >= 1
    assert res["report"]["heldout_docs"] and res["report"]["heldout_docs"][0]["doc"] == "3"
    man = BP.write_outputs(res, p, build_args={"test": True}, exact_tokens=False)
    fam = p.out("families", "book_pairs.jsonl")
    rows = R.load_jsonl(fam)
    assert rows and man["checks"]["rows"] == len(rows) and man["checks"]["duplicate_row_ids"] == 0
    assert man["checks"]["heldout_rows_in_train"] == 0 and man["checks"]["ar_noised_rows"] == 0
    assert not man["checks"]["schema_problems"]
    assert set(man["per_source"]) == {"bk", "sp"} and man["counts"]["total"]["rows"] == len(rows)
    audit = R.load_jsonl(p.out("audit", BP.AUDIT_NAME))
    assert audit and {"source_text", "target", "pair_check"} <= set(audit[0])
    assert (p.out("families", BP.MANIFEST_NAME)).exists()


def test_row_page_keys_per_family() -> None:
    """F2: rows of a located-span document carry the literature pages they draw on in ``meta.page_keys`` (translate:
    translation + the book's edition pages; analyse: intro, edition, translation); a PGP-edition input drops the
    edition pages; parsed entries carry none.

    :returns: None.
    """
    d = make_doc()
    d.provenance = {"work_pages": {"intro": ["w/p/page_001_structured.json"], "edition": ["w/p/page_002_structured.json"],
                                   "translation": ["w/p/page_003_structured.json", "w/p/page_002_structured.json"]}}
    assert BP.row_page_keys(d, "translate_en") == ["w/p/page_002_structured.json", "w/p/page_003_structured.json"]
    assert BP.row_page_keys(d, "analyse_document") == [f"w/p/page_00{i}_structured.json" for i in (1, 2, 3)]
    d.input_source = "pgp_digital_edition:x"
    assert BP.row_page_keys(d, "translate_he") == ["w/p/page_002_structured.json", "w/p/page_003_structured.json"]
    assert BP.row_page_keys(d, "analyse_document") == ["w/p/page_001_structured.json", "w/p/page_002_structured.json",
                                                       "w/p/page_003_structured.json"]
    assert BP.row_page_keys(make_doc(), "translate_en") == []
    meta = BP._base_meta(d, "translate_he", "edition", "train", [], "hash", None)
    assert meta["page_keys"] == ["w/p/page_002_structured.json", "w/p/page_003_structured.json"]


# ---------------------------------------------------------------------------------- fix pass F11: entry pages
def _page(work: str, n: int) -> Any:
    """A minimal page record (``work``, ``page_no``, ``page_key``).

    :param work: Work directory.
    :param n: Page number.
    :returns: The record.
    """
    from types import SimpleNamespace
    return SimpleNamespace(work=work, page_no=n, page_key=f"{work}/part/page_{n:03d}_structured.json")


def test_entry_page_ranges() -> None:
    """F11: an entry runs from its PDF start page to the next entry's start page inclusive, widened to the end page it
    records (``pdf_pages.end``, printed ``pages`` moved by the printed → PDF offset); the last entry gets ``tail``
    pages unless it records more; ``pdf_page`` wins over the printed ``page``.

    :returns: None.
    """
    ents = [{"entry_no": 1, "page": 10, "pdf_page": 18, "pages": [10, 12]},
            {"entry_no": 2, "page": 14, "pdf_page": 22},
            {"entry_no": 3, "pdf_page": 30, "pdf_pages": {"text": [30], "translation": [33], "end": 35}},
            {"entry_no": 4, "pdf_page": 40},
            {"entry_no": 5}]
    assert BP.entry_start_page(ents[0]) == 18 and BP.entry_start_page({"page": "7"}) == 7
    assert BP.entry_start_page(ents[4]) is None and BP.entry_last_page(ents[0]) == 20
    r = BP.entry_page_ranges(ents)
    assert r == {"1": (18, 22), "2": (22, 30), "3": (30, 40), "4": (40, 42)}
    assert BP.entry_page_ranges([{"entry_no": 1, "pdf_page": 5, "pdf_pages": {"end": 9}}])["1"] == (5, 9)


def test_entry_pages_keys_and_unmapped() -> None:
    """F11: the configured work (else the unique work named like the slug) and the entry range give the page keys; a
    book without a work, an entry without a start page or a range without pages are unmapped with a reason.

    :returns: None.
    """
    pages = [_page("bk_work_2024", n) for n in range(1, 12)] + [_page("other", n) for n in range(1, 4)]
    ep = BP.EntryPages.from_pages({"bk": None, "cfg": "other", "lost": None, "bad": "missing_work"}, pages)
    assert ep.works == {"bk": "bk_work_2024", "cfg": "other", "lost": None, "bad": None}
    ep.add_book("bk", [{"entry_no": 1, "pdf_page": 2}, {"entry_no": 2, "pdf_page": 4}, {"entry_no": 3},
                       {"entry_no": 4, "pdf_page": 50}])
    keys, why = ep.keys("bk", 1)
    assert why is None and keys == [f"bk_work_2024/part/page_{n:03d}_structured.json" for n in (2, 3, 4)]
    assert ep.keys("bk", "3") == ([], "no_start_page") and ep.keys("bk", 4) == ([], "no_pages")
    assert ep.keys("lost", 1) == ([], "no_work") and ep.keys("bad", 1) == ([], "no_work")
    assert ep.describe("bk", 2) == {"work": "bk_work_2024", "first": 4, "last": 50}
    assert BP.EntryPages.work_for("bk", None, ["bk_a", "bk_b"]) is None


def test_nas_rows_carry_entry_page_keys(tmp_path: Path) -> None:
    """F11: every row built from a parsed entry carries its entry's page keys (``page_keys_source`` = nas_entry,
    ``entry_pages``), the manifest documents list them, and an unmapped book's rows are flagged
    ``page_keys_unmapped``; the mapper is loaded from the literature tree and the local book configuration.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    p = _paths(tmp_path)
    for slug, entries in (("bk", [lines_entry(1, AR_LINES, EN_LINES, shelfmark="T-S 1.1"),
                                  dict(lines_entry(2, AR_LINES[:2], EN_LINES[:2], shelfmark="T-S 4.4"), pdf_page=14)]),
                          ("lost", [lines_entry(1, AR_LINES, EN_LINES, shelfmark="T-S 7.7")])):
        (p.book_editions_nas / slug).mkdir(parents=True)
        R.write_jsonl(p.book_editions_nas / slug / "entries.jsonl", entries)
    d = p.academic_literature / "bk_book_work" / "part_structured"
    d.mkdir(parents=True)
    for n in range(10, 17):
        (d / f"page_{n:03d}_structured.json").write_text(json.dumps({"full_main_text": f"page {n}",
                                                                     "metadata": {"page_number": n}}))
    cfg = {"nas_books": [{"slug": "bk", "dir": "bk", "layout": "lines", "target_lang": "en", "licence": "lic",
                          "licence_class": "cc-by", "work": "bk_book_work"},
                         {"slug": "lost", "dir": "lost", "layout": "lines", "target_lang": "en", "licence": "lic",
                          "licence_class": "cc-by"}]}
    res = BP.build(p, cfg, {"sources": []}, holdouts=FakeHoldouts(), ix=BP.PgpIndex(), noiser=False, tc=FakeTC())
    rows = {(r["meta"]["doc_id"], r["meta"]["family"], r["meta"]["condition"]): r["meta"] for r in res["rows"]}
    assert rows, "no rows built"
    for (doc_id, _, _), m in rows.items():
        assert m["page_keys_source"] == BP.NAS_PAGE_SOURCE
        if doc_id == "book:bk:1":
            assert m["page_keys"] == [f"bk_book_work/part/page_{n:03d}_structured.json" for n in (12, 13, 14)]
            assert m["entry_pages"] == {"work": "bk_book_work", "first": 12, "last": 14} and "page_keys_unmapped" not in m
        elif doc_id == "book:bk:2":
            assert m["page_keys"] == [f"bk_book_work/part/page_{n:03d}_structured.json" for n in (14, 15, 16)]
        else:
            assert m["page_keys"] == [] and m["page_keys_unmapped"] == "no_work"
    docs = {d["doc_id"]: d for d in res["docs"]}
    assert docs["book:bk:1"]["page_keys_source"] == BP.NAS_PAGE_SOURCE and len(docs["book:bk:1"]["page_keys"]) == 3
    assert docs["book:lost:1"]["page_keys_unmapped"] == "no_work"
    man = BP.write_outputs(res, p, exact_tokens=False)
    assert man["checks"]["rows_unmapped_entry_pages"] == sum(1 for k in rows if k[0] == "book:lost:1")
    assert man["checks"]["rows_by_page_keys_source"] == {BP.NAS_PAGE_SOURCE: len(rows)}


# ---------------------------------------------------------------------------------- books-mining pass (2026-10-25)
def test_numbered_line_normalisers_and_superscripts() -> None:
    """rejoin_numbered joins wrapped rows to their numbered line; drop_line_numbers strips edition numbers."""
    text = "Recto\n1. In the name of God the\nMerciful and Compa-\nssionate\n2. My letter\n§ verso\n3. Peace"
    out, n = BP.rejoin_numbered(text)
    assert out.split("\n") == ["Recto", "1. In the name of God the Merciful and Compassionate", "2. My letter",
                               "§ verso", "3. Peace"]
    assert n == 2
    ed, k = BP.drop_line_numbers("1. בשמ רחמ\n2 כתאבי אלי\nכתאבך .3\n§ recto\nline 4 Latin")
    assert ed.split("\n") == ["בשמ רחמ", "כתאבי אלי", "כתאבך", "§ recto", "line 4 Latin"] and k == 3
    s, m = BP.drop_superscripts("the house² of Abū¹³ ʿAlī")
    assert s == "the house of Abū ʿAlī" and m == 2
    ok, missing = BP.words_subset(s, "the house² of Abū¹³ ʿAlī")
    assert ok, missing
    assert {"rejoin_numbered", "drop_line_numbers", "drop_superscripts"} <= set(BP.NORMALISERS)


def test_drop_ocr_refs_keeps_restored_letter_inside_word() -> None:
    """A bracketed letter inside a word is a restoration, not a siglum."""
    out, n = BP.drop_ocr_refs("החכם ש[א]סר את [ג] האשה")
    assert out == "החכם ש[א]סר את האשה" and n == 1


def test_doc_from_spans_not_placed_input_override_and_commentary() -> None:
    """Recorded candidates without spans drop with their reason; a doc can override the source input; a
    commentary span is used as the analysis text."""
    page = "Head\n" + "\n".join(JA_LINES) + "\nTranslation\n" + "\n".join(EN_LINES) + \
        "\nCommentary\nThe letter of Abū ʿAlī Ḥasan ibn Dāwūd was written in Fustat about the year 1100."
    pages = {"w/p/page_001.json": page}
    src = {"slug": "bk", "target_lang": "en", "input": "pgp_edition", "licence": "l", "licence_class": "copyright"}
    doc, info = BP.doc_from_spans(src, {"doc": "9", "not_placed": "no_translation"}, pages)
    assert doc is None and info["drop"] == "not_placed:no_translation"
    spans = {"edition": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 1, 3).to_dict()], "normalise": []},
             "translation": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 5, 7).to_dict()],
                             "normalise": []},
             "commentary": {"pieces": [BP.piece_from_lines("w/p/page_001.json", page, 9, 9).to_dict()],
                            "normalise": []}}
    doc, info = BP.doc_from_spans(src, {"doc": "1", "input": "span", "spans": spans}, pages, lambda h, s: None)
    assert doc is not None and doc.input_source == "book_edition:bk" and doc.edition == "\n".join(JA_LINES)
    assert doc.intro.startswith("The letter of") and info["analysis_kind"] == "commentary"
    assert doc.provenance["work_pages"]["intro"] == ["w/p/page_001.json"]


def test_independence_check_flags_near_copies(tmp_path: Path) -> None:
    """A book translation that copies a PGP digital translation fails; an independent one passes."""
    pgp_tr = ("In the name of God. My letter to the elder Abū ʿAlī Ḥasan ibn Dāwūd, may God prolong his life. "
              "Your letter arrived and I understood it, praise be to God, and I sent the pepper with the boat.")
    pairs = write_pairs(tmp_path / "pairs.jsonl", [
        {"pgpid": "10", "shelfmark": "T-S 1.1", "split": "train", "translation_en": pgp_tr, "edition_text": "x",
         "editions": [], "translations": [{"language": "en", "text": pgp_tr}]},
        {"pgpid": "11", "shelfmark": "T-S 2.2", "split": "train", "translation_he": "שלום לך אדוני", "edition_text": "y",
         "editions": [], "translations": []}])
    ix = BP.PgpIndex.build(pairs, with_grams=False)
    tr = BP.PgpTranslations.build(ix)
    assert set(tr.texts["en"]) == {"10"} and set(tr.texts["he"]) == {"11"}
    copy = pgp_tr.replace("Your letter", "(3) Your letter")
    bad = BP.independence_check(copy, "en", tr, same_doc=[])
    assert not bad["ok"] and bad["pgpid"] == "10" and bad["containment"] >= 0.5 and not bad["same_doc"]
    other = ("The judge of the court confirmed the sale of half a house in the Jewish quarter to the widow, "
             "and the witnesses signed the deed in the month of Nisan.")
    good = BP.independence_check(other, "en", tr, same_doc=["10"])
    assert good["ok"] and good["containment"] < 0.5 and good["ratio"] <= 0.9
    assert BP.independence_check(copy, "en", None)["ok"]
    assert BP.independence_tokens("(12) Peace, Abū ʿAlī") == ["peace", "abū", "ʿalī"]


def test_translate_rows_drop_on_independence(tmp_path: Path) -> None:
    """translate_rows drops a located pair whose target near-copies a PGP translation."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", [
        {"pgpid": "10", "shelfmark": "T-S 9.9", "split": "train", "translation_en": "\n".join(EN_LINES),
         "edition_text": "x", "editions": [], "translations": []}])
    ix = BP.PgpIndex.build(pairs, with_grams=False)
    c = ctx(ix=ix)
    c.translations = BP.PgpTranslations.build(ix)
    doc = make_doc(located=True)
    rows, info = BP.translate_rows(doc, "train", [], "hash", None, [doc], c)
    assert rows == [] and info["drop"] == "independence_dup_pgp_translation"
    c.translations = None
    rows, info = BP.translate_rows(doc, "train", [], "hash", None, [doc], c)
    assert rows and rows[0]["meta"]["independence"]["ok"]


def test_check_report_lines() -> None:
    """check_report gives one verdict per document and lists span-level drops."""
    res = {"docs": [{"source": "bk", "doc": "1", "pgpids": ["5"], "split": "train", "holdout_reason": None,
                     "translate": {"rows": 2, "pair_check": {"length_ratio": 1.1, "own": 0.5, "best_other": 0.1},
                                   "independence": {"containment": 0.0, "ratio": 0.3}}, "analyse": {"drop": "no_intro"}}],
           "report": {"span_docs": [{"source": "bk", "doc": "2", "drop": "not_placed:no_translation"}]}}
    out = BP.check_report(res)
    assert out[0]["translate"].startswith("rows=2") and out[0]["analyse"].startswith("DROP no_intro")
    assert out[1]["doc"] == "2" and out[1]["translate"].startswith("DROP not_placed:no_translation")


def test_load_carry_and_carry_rows(tmp_path: Path) -> None:
    """Parsed-entry rows are carried from an earlier build and screened again for independence."""
    def row(doc_id: str, fam: str, target: str, source: str = "book:nas1") -> Dict[str, Any]:
        """A minimal carried row.

        :param doc_id: Document id.
        :param fam: Family.
        :param target: Assistant text.
        :param source: Source tag.
        :returns: Row.
        """
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                             {"role": "assistant", "content": target}],
                "meta": {"doc_id": doc_id, "family": fam, "condition": "edition", "source": source, "pgpids": []}}
    copied = " ".join(EN_LINES)
    rows = [row("book:nas1:1", "translate_en", copied), row("book:nas1:2", "translate_en", "An unrelated text about a "
                                                                                              "house sale in the quarter."),
            row("book:other:1", "translate_en", "x", source="book:other")]
    d = tmp_path / "prev"
    d.mkdir()
    R.write_jsonl(d / "book_pairs.jsonl", rows)
    (d / "book_pairs_manifest.json").write_text(json.dumps({
        "documents": [{"source": "nas1", "doc": "1", "doc_id": "book:nas1:1", "keys": ["k1"], "split": "train",
                       "translate": {"rows": 1}},
                      {"source": "nas1", "doc": "2", "doc_id": "book:nas1:2", "keys": ["k2"], "split": "heldout",
                       "translate": {"rows": 1}},
                      {"source": "other", "doc": "1", "doc_id": "book:other:1", "keys": [], "split": "train"}],
        "report": {"entries_per_book": {"nas1": 2, "other": 1}}}), encoding="utf-8")
    carry = BP.load_carry(d, ["nas1"])
    assert len(carry["rows"]) == 2 and [x["doc"] for x in carry["docs"]] == ["1", "2"]
    assert carry["entries_per_book"] == {"nas1": 2} and carry["docs"][0]["translate"]["family"] == "translate_en"
    pairs = write_pairs(tmp_path / "pairs.jsonl", [
        {"pgpid": "10", "shelfmark": "T-S 9.9", "split": "train", "translation_en": copied, "edition_text": "x",
         "editions": [], "translations": []}])
    tr = BP.PgpTranslations.build(BP.PgpIndex.build(pairs, with_grams=False))
    import collections
    drops: collections.Counter = collections.Counter()
    kept, docs = BP.carry_rows(carry, tr, drops)
    assert [r["meta"]["doc_id"] for r in kept] == ["book:nas1:2"]
    assert docs[0]["translate"]["drop"] == "independence_dup_pgp_translation" and drops
    kept2, _ = BP.carry_rows(carry, None, collections.Counter())
    assert len(kept2) == 2


def test_independence_candidates_break_ties_by_pgpid() -> None:
    """Equal shared-n-gram counts are ordered by pgpid, so the recorded nearest PGP translation is stable across runs."""
    text = "the merchant sent the flax to his brother in the city of the west with the caravan"
    tr = BP.PgpTranslations()
    for pid in ("30", "4", "200", "17"):
        tr.texts.setdefault("en", {})[pid] = [text]
        for g in BP._grams(BP.independence_tokens(text)):
            tr.grams.setdefault("en", {}).setdefault(g, set()).add(pid)
    got = tr.candidates(text, "en", k=3)
    assert [p for p, _ in got] == ["17", "200", "30"] and all(c == 1.0 for _, c in got)
