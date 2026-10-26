# File name: test_text27b_canon_diff.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``canon_diff``: Hebrew normalisation and the skeleton rule, abbreviation expansion, fragment
tokenisation, the trigram index and anchor search, word alignment with typed variants (orthographic, substitution,
omission, addition, order, lacunae, torn margins), catalogue routing, hold-out splits, row building end to end on a
synthetic KTIV bundle + Sefaria cache, the audit sample and the public-domain edition helpers.

Synthetic Hebrew strings and files in ``tmp_path`` only (made-up system numbers and shelfmarks); a tiny word-level
tokenizer and a small chat template stand in for the real ones.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.text27b import canon_diff as C
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import sefaria as S

#: A small template with the Qwen3.8 thinking behaviour that the render contract checks.
TEMPLATE = (
    "{%- for m in messages %}"
    "{%- if m.role == 'system' %}{{- '<|im_start|>system\\n' + m.content|trim + '<|im_end|>\\n' }}"
    "{%- elif m.role == 'user' %}{{- '<|im_start|>user\\n' + m.content|trim + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content|trim + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}")

CANON = ("אמר רבי יוחנן משום רבי שמעון בן יוחי מנין שאין מעמידין מתורגמן לחכם פחות מחמשים שנה "
         "שנאמר שר חמשים ונשוא פנים ואמר רבי יוחנן אין נותנין שבעה שרביטין לתלמיד חכם שלא קרא "
         "ושנה ושימש תלמידי חכמים הרבה ואמר רבי אבהו אמר רבי יוחנן כל המלמד את בן חבירו תורה "
         "מעלה עליו הכתוב כאילו ילדו שנאמר ואלה תולדות אהרן ומשה")
OTHER = "תנו רבנן שלשה דברים מרחיבין דעתו של אדם דירה נאה ואשה נאה וכלים נאים ושלשה משיבין דעתו"
FRAG = ["אמ' ר' יוחנן משום ר' שמעון בן יוחאי מנין שאין מעמידין",
        "מתורגמן לחכם פחות מחמשים שנה שנ' שר חמשים",
        "ונשוא פנים ואמר ר' יוחנן אין נותנין [...] שרביטין",
        "לתלמיד חכם שלא קרא ושנה ושימש תלמידי חכמים הרבה",
        "ואמ' רבי אבהו אמר ר' יוחנן כל המלמד את בן חבירו תורה",
        "מעלה עליו הכתוב כאלו ילדו"]


def segments(title: str, he_title: str, section: str, text: str, size: int = 10) -> List[S.Segment]:
    """Cut a text into Sefaria-like segments of ``size`` words.

    :param title: Title.
    :param he_title: Hebrew title.
    :param section: Section address.
    :param text: Text.
    :param size: Words per segment.
    :returns: Segments.
    """
    w = text.split()
    return [S.Segment(title, he_title, section, i // size + 1, " ".join(w[i:i + size]), "יט ע״ב")
            for i in range(0, len(w), size)]


def make_index() -> C.CanonIndex:
    """Index over two synthetic tractates (the passage lives in the second).

    :returns: Index.
    """
    t1 = S.CanonText("Berakhot", "ברכות", "bavli", "V", "CC-BY-SA", segments("Berakhot", "ברכות", "57b", OTHER))
    t2 = S.CanonText("Sanhedrin", "סנהדרין", "bavli", "V", "CC-BY-SA", segments("Sanhedrin", "סנהדרין", "19b", CANON))
    return C.CanonIndex("bavli", [t1, t2])


# ----------------------------------------------------------------------------------- normalisation
def test_norm_word_and_skeleton() -> None:
    """Marks and punctuation go, finals fold; internal ו/י drop, the first / last letter stay, final א ≡ ה."""
    assert C.norm_word("שָׁלוֹם,") == "שלומ"
    assert C.norm_word("[...]") == ""
    assert C.skeleton(C.norm_word("כאילו")) == C.skeleton(C.norm_word("כאלו"))
    assert C.skeleton(C.norm_word("הכא")) == C.skeleton(C.norm_word("הכה"))
    assert C.skeleton(C.norm_word("ואמר")) != C.skeleton(C.norm_word("אמר"))  # the conjunction is not spelling
    assert C.skeleton(C.norm_word("אמרי")) != C.skeleton(C.norm_word("אמר"))  # a suffix is not spelling
    assert C.skeleton("מיכאנ") == C.skeleton("מיכנ") and C.skeleton("דאמי") == C.skeleton("דמי")
    assert C.skeleton("שאל") != C.skeleton("של")
    assert C.strip_marks_he("בְּרֵאשִׁית:") == "בראשית:"


def test_abbreviations_and_ortho_equal() -> None:
    """Geresh abbreviations, gershayim acronyms (with a glued prefix), word division; mixed marks do not match."""
    assert C.abbrev_expands("אמ'", ["אמר"]) and C.abbrev_expands("ר׳", ["רבי"])
    assert C.abbrev_expands('א"ר', ["אמר", "רבי"])
    assert C.abbrev_expands('ור"ל', ["וריש", "לקיש"])
    assert C.abbrev_expands("א'ע'פ'", ["אפ", "על", "פי"])
    assert not C.abbrev_expands("אמ'", ["אמר", "רבי"]) and not C.abbrev_expands("אמר", ["אמר"])
    assert C.ortho_equal(["על", "גבי"], ["עלגבי"])
    assert C.ortho_equal(["שנ'"], ["שנאמר"]) and C.ortho_equal(["שנאמר"], ["שנ'"])
    assert not C.ortho_equal(["ר'"], ['ר"מ'])  # a word abbreviation is not an acronym
    assert C.ortho_equal(["אמ'"], ["א'"])
    assert not C.ortho_equal(["להן"], ["להם"])  # final mem / nun differ: a substitution


def test_tokenize_fragment_gaps_dots_edges() -> None:
    """``[...]`` is a gap, dotted words are uncertain, neighbours of a gap are edges; letterless tokens drop."""
    assert [t.gap for t in C.tokenize_fragment(["[] ..... אב"])] == [True, True, False]
    toks = C.tokenize_fragment(["אמר ש..ש [...] רבי :", "יוחנן"])
    assert [t.raw for t in toks] == ["אמר", "ש..ש", "[...]", "רבי", "יוחנן"]
    assert [t.gap for t in toks] == [False, False, True, False, False]
    assert [t.uncertain for t in toks] == [False, True, False, False, False]
    assert [t.edge for t in toks] == [False, False, False, True, False]
    assert [t.pos for t in toks] == ["1:1", "1:2", "1:3", "1:4", "2:1"]
    assert C.tokenize_canon("שָׁלוֹם־עֲלֵיכֶם : וגו'") == ["שלום", "עליכם", "וגו'"]


# ----------------------------------------------------------------------------------- index + alignment
def test_index_anchor_search_finds_the_right_tractate() -> None:
    """Rare trigrams vote for the tractate and offset of the passage; titles do not share trigrams."""
    idx = make_index()
    toks = C.tokenize_fragment(FRAG)
    cands = C.anchor_candidates(toks, idx)
    assert cands and idx.texts[cands[0].text_index].title == "Sanhedrin"
    t0, _ = idx.bounds(cands[0].text_index)
    assert abs(cands[0].diag - t0) <= 3
    assert idx.text_of_pos(10 ** 9) == len(idx.texts) - 1  # clamped


def test_align_types_variants_and_lacunae() -> None:
    """Abbreviations / plene spellings are orthographic, a changed letter a substitution, a gap a lacuna."""
    idx = make_index()
    al, _ = C.locate(C.tokenize_fragment(FRAG), idx)
    assert al is not None and idx.texts[al.text_index].title == "Sanhedrin"
    ref, he_ref = idx.ref_range(al.start, al.end)
    assert ref.startswith("Sanhedrin 19b:1") and he_ref == "סנהדרין יט ע״ב"
    got = {(v.type, v.fragment, v.canon) for v in al.variants}
    assert ("orthographic", "אמ'", "אמר") in got  # the edge pair before the first exact match is kept
    assert ("orthographic", "שנ'", "שנאמר") in got and ("orthographic", "כאלו", "כאילו") in got
    assert ("orthographic", "יוחאי", "יוחי") in got  # internal aleph of a long word is a vowel letter
    assert al.lacuna_words == 1 and al.counts()["omission"] == 0 and al.counts()["substitution"] == 0
    assert al.similarity > 0.8 and 0.9 < al.agreement_ratio <= 1.0
    tgt = C.diff_target(ref, al)
    assert tgt["canonical_ref"] == ref and tgt["lacuna_words"] == 1
    assert "1 canonical word falls in lacunae" in tgt["summary"] and "substitution" not in tgt["summary"]


def test_omission_addition_and_order() -> None:
    """A dropped phrase is an omission, an extra word an addition, swapped names an order variant."""
    idx = make_index()
    frag = ["אמר רבי יוחנן משום רבי שמעון בן יוחי מנין שאין מעמידין מתורגמן לחכם",
            "שנה שנאמר שר חמשים ונשוא פנים ואמר רבי יוחנן אין נותנין שבעה גדולים",
            "שרביטין לתלמיד חכם שלא קרא ושנה ושימש תלמידי חכמים הרבה ואמר רבי",
            "יוחנן אמר רבי אבהו כל המלמד את בן חבירו תורה"]
    al, _ = C.locate(C.tokenize_fragment(frag), idx)
    types = [(v.type, v.fragment, v.canon) for v in al.variants]
    assert ("omission", "", "פחות מחמשים") in types
    assert ("addition", "גדולים", "") in types
    assert any(t == "order" for t, _, _ in types)


def test_trim_edges_and_extend_edges() -> None:
    """A stray one-word match far from the body is dropped; orthographic pairs next to the span are added."""
    eq = [("equal", 0, 1, 0, 1), ("equal", 1, 6, 20, 25), ("equal", 6, 10, 25, 29)]
    assert C.trim_edges(eq) == eq[1:]
    toks = C.tokenize_fragment(["ר' יוחנן אמר"])
    assert C.extend_edges(toks, ["רבי", "יוחנן", "אמר"], 1, 1, 3, 3) == (0, 0, 3, 3)


def test_torn_margin_reclasses_line_break_omissions() -> None:
    """When most line breaks of a column lose canonical words, those omissions are the torn margin (lacunae)."""
    idx = make_index()
    frag = ["אמר רבי יוחנן משום", "פחות מחמשים שנה שנאמר", "ואמר רבי יוחנן אין", "לתלמיד חכם שלא קרא",
            "הרבה ואמר רבי אבהו"]
    al, _ = C.locate(C.tokenize_fragment(frag), idx)
    assert al.torn_margin and al.counts()["omission"] == 0 and al.lacuna_words > 10


def test_shape_gate() -> None:
    """A 41-word omission or a span twice the fragment marks a layout jump."""
    def al(variants: List[C.Variant], canon: int = 100, lacuna: int = 0, legible: int = 90) -> C.Alignment:
        return C.Alignment(0, canon, 0.8, 0.9, lacuna, variants, 80, legible, 0, canon, 0)

    assert C.shape_gate(al([C.Variant("omission", "", "x", "1:1", (0, 0), (0, 41))])) == "long_block"
    assert C.shape_gate(al([C.Variant("addition", "x", "", "1:1", (0, 40), (0, 0))])) is None
    assert C.shape_gate(al([], canon=300, legible=100)) == "span_overrun"
    assert C.shape_gate(al([], canon=300, lacuna=150, legible=100)) is None


def test_merge_order_on_variant_lists() -> None:
    """An addition and an omission of the same words within the window merge into one order variant."""
    toks = C.tokenize_fragment(["א ב ג"])
    vs = [C.Variant("omission", "", "ב", "1:1", (0, 0), (0, 1)), C.Variant("addition", "ב", "", "1:2", (1, 2), (2, 2))]
    out = C.merge_order(vs, toks, ["ב", "א", "ג"])
    assert [v.type for v in out] == ["order"] and out[0].fragment == "א ב" and out[0].canon == "ב א"


def test_trim_punct_in_variant_json() -> None:
    """Sentence punctuation is trimmed from the variant strings."""
    assert C.Variant("substitution", "דמו.", "דמי:", "1:1").as_json() == {"type": "substitution", "fragment": "דמו",
                                                                         "canon": "דמי", "pos": "1:1"}


# ----------------------------------------------------------------------------------- catalogue
def test_work_routing_and_catalogue_ref() -> None:
    """Frames name the corpus; Rambam frames go to the Mishneh Torah; a Bavli frame gives tractate and daf range."""
    bav = C.CatRecord("1", frames=[{"title": "Talmud Bavli [Text]", "frame": "[Talmud Bavli]: Mo'ed Qatan 10 a – 11 b"}])
    ref = C.catalogue_ref(bav)
    assert ref.corpus == "bavli" and ref.title_hint == "Mo'ed Qatan" and (ref.daf_from, ref.daf_to) == (18, 21)
    assert C.match_title(ref.title_hint, ["Berakhot", "Moed Katan"]) == "Moed Katan"
    ram = C.CatRecord("2", frames=[{"title": "Mishnah [Text]", "frame": "[Moses b. Maimon, Rambam, Mishnah]: Ma'aser Sheni"}])
    assert C.work_of(ram) == "mishneh_torah"
    assert C.work_of(C.CatRecord("3", title="פיוט")) is None
    assert C.work_of(C.CatRecord("4", title="משנה תורה (ספר אהבה, הלכות קריאת שמע א)")) == "mishneh_torah"
    mi = C.catalogue_ref(C.CatRecord("5", frames=[{"title": "Mishnah [Text]", "frame": "[Mishnah]: Bikkurim 3:1"}]))
    assert mi.corpus == "mishnah" and C.match_title(mi.title_hint, ["Mishnah Berakhot", "Mishnah Bikkurim"]) == \
        "Mishnah Bikkurim"


def test_hebrew_title_prior() -> None:
    """Hebrew catalogue titles pick the Sefaria title: hilkhot words for the Mishneh Torah, tractate after '('."""
    class T:
        def __init__(self, title: str, he: str) -> None:
            self.title, self.he_title = title, he

    mt = [T("Mishneh Torah, Prayer", "משנה תורה, הלכות תפילה וברכת כהנים"),
          T("Mishneh Torah, Tefillin", "משנה תורה, הלכות תפילין ומזוזה וספר תורה"),
          T("Mishneh Torah, Shema", "משנה תורה, הלכות קריאת שמע")]
    assert C.hebrew_title_prior("משנה תורה (ספר אהבה, הלכות תפילה ב)", mt, "mishneh_torah") == "Mishneh Torah, Prayer"
    assert C.hebrew_title_prior("משנה תורה (ספר אהבה, הלכות קריאת שמע)", mt, "mishneh_torah") == "Mishneh Torah, Shema"
    bav = [T("Bava Kamma", "בבא קמא"), T("Sukkah", "סוכה")]
    assert C.hebrew_title_prior('תלמוד בבלי סדר מועד (סוכה טז ע"ב)', bav, "bavli") == "Sukkah"
    assert C.hebrew_title_prior("תלמוד בבלי", bav, "bavli") is None


def test_locate_prefers_catalogue_tractate_within_margin() -> None:
    """With the passage in two tractates, the catalogue's one wins when it scores within the margin."""
    t1 = S.CanonText("Shabbat", "שבת", "bavli", "V", "x", segments("Shabbat", "שבת", "30a", CANON))
    t2 = S.CanonText("Makkot", "מכות", "bavli", "V", "x", segments("Makkot", "מכות", "10a", CANON))
    idx = C.CanonIndex("bavli", [t1, t2])
    toks = C.tokenize_fragment(FRAG)
    a0, _ = C.locate(toks, idx, prior_text=0)
    a1, _ = C.locate(toks, idx, prior_text=1)
    assert a0.text_index == 0 and a1.text_index == 1


# ----------------------------------------------------------------------------------- hold-outs and splits
def test_split_of_is_deterministic_and_roughly_80_10_10() -> None:
    """sha1 buckets give a stable 80/10/10 split."""
    splits = [C.split_of(f"9900{i:014d}") for i in range(3000)]
    assert splits == [C.split_of(f"9900{i:014d}") for i in range(3000)]
    share = {s: splits.count(s) / len(splits) for s in ("train", "dev", "test_indist")}
    assert 0.76 < share["train"] < 0.84 and 0.07 < share["dev"] < 0.13 and 0.07 < share["test_indist"] < 0.13


def test_heldout_reason_and_shingles() -> None:
    """KTIV benchmark sys_nums win; else the hold-out set is asked by pgpid / canonical / shelfmark."""
    class H:
        def match_any(self, pgpids: Sequence[str] = (), canonicals: Sequence[str] = (),
                      shelfmarks: Sequence[str] = (), include_eval: bool = True) -> Optional[str]:
            return "key:pgp131" if "T-S 99.1" in shelfmarks else None

    assert C.heldout_reason(C.CatRecord("1"), {"1": "ktiv_benchmark:x"}, H()) == "ktiv_benchmark:x"
    assert C.heldout_reason(C.CatRecord("2", shelfmark="T-S 99.1"), {}, H()) == "key:pgp131"
    assert C.heldout_reason(C.CatRecord("3", shelfmark="T-S 1.1"), {}, H()) is None
    assert C.heldout_reason(C.CatRecord("3"), {}, None) is None
    sh = C.shingle_set([CANON], n=10)
    assert C.shingle_share(CANON, sh, n=10) == 1.0 and C.shingle_share(OTHER, sh, n=10) < 0.1
    assert C.shingle_share("קצר", sh, n=10) == 0.0


# ----------------------------------------------------------------------------------- chunks
def test_split_lines_caps_words() -> None:
    """A long column is cut into near-equal line runs under the cap."""
    lines = [" ".join(["מילה"] * 10)] * 9
    parts = C.split_lines(lines, max_words=40)
    assert len(parts) == 3 and all(sum(len(x.split()) for x in p) <= 40 for p in parts)
    assert C.split_lines(lines[:2], max_words=40) == [lines[:2]]


def annotation_items(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """KTIV-like word annotations with boxes laid out right to left, one row per line.

    :param lines: Text lines.
    :returns: AnnotationPage items.
    """
    items = []
    n = 0
    for li, line in enumerate(lines):
        x = 3000
        for w in line.split():
            w = "....." if w == "[...]" else w  # KTIV writes a lacuna as a dot run
            x1, x0 = x, x - 100
            y0, y1 = 100 + li * 80, 150 + li * 80
            items.append({"id": f"w{n}", "body": {"value": w},
                          "target": {"selector": {"value": f"<svg><path d=\"M{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} "
                                                           f"{x0},{y0}z\"/></svg>"}}})
            x -= 125
            n += 1
    return items


def test_page_chunks_from_bundle() -> None:
    """A synthetic API-shape bundle rebuilds into one column chunk with the original lines."""
    bundle = {"source": "nli_ktiv_viewer", "pages": [{"fl": "FL1", "annotation_page": {"items": annotation_items(FRAG)}},
                                                     {"fl": "FL2", "annotation_page": {"items": []}}]}
    (ch,) = C.page_chunks(bundle, "990000000000000001")
    assert ch.key == "990000000000000001:FL1:c1" and ch.lines == FRAG and ch.page == 1


# ----------------------------------------------------------------------------------- end to end
def build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> P.Paths:
    """Synthetic KTIV dir, merged catalogue, Sefaria cache, hold-out files, tokenizer and template.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch (registry sys_nums stubbed).
    :returns: Loaded paths.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    ktiv = tmp_path / "ktiv"
    ktiv.mkdir()
    (ktiv / "unrelated_personal_note.txt").write_text("never read", encoding="utf-8")
    sys_train, sys_bench, sys_none = "990000000000000011", "990000000000000022", "990000000000000033"
    for sn, lines in ((sys_train, FRAG), (sys_bench, FRAG[:4]), (sys_none, ["פיוט פיוט פיוט"] * 3)):
        (ktiv / f"ktiv_PNX_MANUSCRIPTS{sn}-1_transcription.json").write_text(json.dumps(
            {"source": "nli_ktiv_viewer", "doc_id": f"PNX_MANUSCRIPTS{sn}-1",
             "pages": [{"fl": f"FL{sn[-2:]}", "annotation_page": {"items": annotation_items(lines)}}]},
            ensure_ascii=False), encoding="utf-8")
    frame = {"title": "Talmud Bavli [Text]", "frame": "[Talmud Bavli]: Sanhedrin 19 b"}
    merged = [{"canonical_id": f"X_{sn}", "pgpids": [], "sources": {"ktiv": {
        "sys_num": sn, "shelf_mark": f"Synthetic Library Ms. {i}", "basic_catalog": {"title": "תלמוד בבלי"},
        "full_catalog": {}, "scholarly_entries": [{"subsections": {"writing_characteristics": frame}}]}}}
        for i, sn in enumerate((sys_train, sys_bench), 1)]
    merged.append({"canonical_id": "X_none", "sources": {"ktiv": {"sys_num": sys_none, "basic_catalog": {"title": "פיוט"}}}})
    R.write_jsonl(tmp_path / "merged.jsonl", merged)
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"benchmark": "religious_synth", "docs": [{"sys_num": sys_bench, "gt": "קצר"}]}),
                     encoding="utf-8")
    sroot = tmp_path / "sefaria"
    vpath = "json/Talmud/Bavli/Seder Nezikin/Sanhedrin/Hebrew/V.json"
    words = CANON.split()
    (sroot / vpath).parent.mkdir(parents=True)
    text = [[] for _ in range(37)] + [[" ".join(words[i:i + 10]) for i in range(0, len(words), 10)]]
    (sroot / vpath).write_text(json.dumps({"title": "Sanhedrin", "heTitle": "סנהדרין", "versionTitle": "V",
                                           "license": "CC-BY-SA", "sectionNames": ["Daf", "Line"], "text": text},
                                          ensure_ascii=False), encoding="utf-8")
    (sroot / S.MANIFEST_FILE).write_text(json.dumps({"repo_commit": "b" * 40, "texts": [
        {"corpus": "bavli", "title": "Sanhedrin", "path": vpath, "licence": "CC-BY-SA"}]}), encoding="utf-8")
    entries = {k: str(tmp_path / k) for k in (
        "probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
        "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
        "ai_reads", "book_editions_nas", "out_dir", "scratch", "pgp_raw")}
    entries.update({"repo": str(tmp_path), "tokenizer_json": str(tmp_path / "tokenizer.json"),
                    "chat_template": str(tmp_path / "chat_template.jinja"), "merged_shelfmarks": str(tmp_path / "merged.jsonl"),
                    "ktiv_dir": str(ktiv), "sefaria_dir": str(sroot), "ktiv_benchmarks": [str(bench)]})
    cfg = tmp_path / "paths.json"
    cfg.write_text(json.dumps(entries), encoding="utf-8")
    from src.datasets.evaluations import benchmark_registry
    monkeypatch.setattr(benchmark_registry, "registered_benchmark_manuscripts", lambda *a, **k: set())
    return P.load_paths(cfg)


def test_build_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two families, schema-valid rows, benchmark manuscript held out, manifest with counts and provenance."""
    p = build_fixture(tmp_path, monkeypatch)
    man = C.build(C.BuildConfig(floor=0.5), p)
    ident = R.load_jsonl(p.out("families", "canon_identify.jsonl"))
    diff = R.load_jsonl(p.out("families", "canon_diff.jsonl"))
    assert man["families"] == {"canon_identify": 2, "canon_diff": 2} and len(ident) == len(diff) == 2
    by_sys = {r["meta"]["sys_num"]: r for r in diff}
    assert by_sys["990000000000000022"]["meta"]["split"] == "heldout"
    assert by_sys["990000000000000022"]["meta"]["heldout_reason"] == "ktiv_benchmark:religious_synth"
    assert by_sys["990000000000000022"]["meta"]["held_out_vision_benchmark"] is True
    assert by_sys["990000000000000011"]["meta"]["split"] == C.split_of("990000000000000011")
    for r in ident + diff:
        problems = [x for x in R.validate_row(r) if not x.startswith("meta.family=")]
        assert problems == []
        assert r["meta"]["condition"] == "edition" and r["meta"]["self_label"] is False
        assert r["meta"]["tokens"] > 0 and r["meta"]["licence_class"] == "cc-by"
    t = json.loads(by_sys["990000000000000011"]["messages"][2]["content"])
    assert set(t) == {"canonical_ref", "agreement_ratio", "lacuna_words", "variants", "summary"}
    assert t["canonical_ref"].startswith("Sanhedrin 19b:1")
    assert "Canonical text (Sanhedrin 19b:" in by_sys["990000000000000011"]["messages"][1]["content"]
    it = json.loads([r for r in ident if r["meta"]["sys_num"] == "990000000000000011"][0]["messages"][2]["content"])
    assert it["work"] == "Talmud Bavli" and it["title"] == "Sanhedrin" and it["canonical_text"].startswith("אמר רבי")
    assert man["manuscripts"]["no_canonical_work"] == 1
    assert man["holdouts"]["benchmark_sysnums_in_non_heldout_rows"] == []
    assert man["counts"]["canon_diff"]["Talmud Bavli"]["heldout"]["rows"] == 1
    assert "deterministic" in man["self_labels"] and man["sefaria"]["repo_commit"] == "b" * 40
    pool = R.load_jsonl(p.out("audit", "canon_audit_pool.jsonl"))
    assert len(pool) == 2 and all("similarity" in x for x in pool)


def test_audit_sample_stratifies() -> None:
    """The audit draws across corpora and similarity bands, sorted by corpus then similarity."""
    pool = [{"corpus": c, "similarity": s, "key": f"{c}{s}"} for c in ("a", "b") for s in (0.1, 0.4, 0.5, 0.7, 0.9)]
    pool.append({"corpus": "a", "key": "no_alignment"})
    out = C.audit_sample(pool, n=6, seed=1)
    assert len(out) == 6 and {r["corpus"] for r in out} == {"a", "b"}
    assert out == sorted(out, key=lambda r: (r["corpus"], r["similarity"]))


def test_identify_and_diff_messages_read_condition() -> None:
    """The read twin reuses the target and swaps only the input note."""
    target = {"work": "W"}
    e = C.identify_messages("טקסט", target, "edition")
    r = C.identify_messages("טקסט", target, "read")
    assert e[2] == r[2] and C.NOTE_EDITION in e[1]["content"] and R.READ_NOTE_V2 in r[1]["content"]
    d = C.diff_messages("טקסט", "Ref 1a:1", "קנון", {"canonical_ref": "Ref 1a:1"})
    assert d[1]["content"].endswith("Canonical text (Ref 1a:1):\nקנון")


# ----------------------------------------------------------------------------------- PD editions
def test_djvu_pages_assess_and_containment(tmp_path: Path) -> None:
    """Page text from a djvu.xml layer; a Latin-only layer is flagged Hebrew-unusable; 4-gram containment."""
    xml = ("<DjVuXML><BODY>"
           "<OBJECT><HIDDENTEXT><LINE><WORD>The</WORD><WORD>wisdom</WORD></LINE><LINE><WORD>of</WORD></LINE>"
           "</HIDDENTEXT></OBJECT><OBJECT><HIDDENTEXT><LINE><WORD>second</WORD></LINE></HIDDENTEXT></OBJECT>"
           "</BODY></DjVuXML>")
    q = tmp_path / "b_djvu.xml"
    q.write_text(xml, encoding="utf-8")
    assert C.djvu_pages(q) == ["The wisdom\nof", "second"]
    pages = ["word " * 80, "ְאלהים " * 10 + "x" * 300]
    rep = C.assess_pages(pages, n=10)
    assert rep["hebrew_usable"] is False and len(rep["sampled"]) == 2
    folded = C._fold_en("Honour the physi-\ncian before thou hast need of him.")
    assert C.ocr_containment("honour the physician before thou hast need", folded) == 1.0
    assert C.ocr_containment("a completely different sentence here", folded) == 0.0


def test_pd_pairs_verification_and_band(tmp_path: Path) -> None:
    """Verse groups become translate_en rows when the English is in the text layer and the length ratio fits."""
    he = S.CanonText("Ben Sira", "בן סירא", "ben_sira", "H", "Public Domain",
                     [S.Segment("Ben Sira", "בן סירא", "38", i, f"רעה רופא לפני צרכו פסוק מספר {i}", "לח")
                      for i in (1, 2, 3)])
    en = S.CanonText("Ben Sira", "בן סירא", "ben_sira_en", "E", "Public Domain",
                     [S.Segment("Ben Sira", "", "38", i, f"Honour a physician before thou hast need of him verse {i}", "")
                      for i in (1, 2, 3)])
    en.segments[2] = S.Segment("Ben Sira", "", "38", 3, "An entirely unrelated modern sentence that no book prints", "")
    ocr = " ".join(f"Honour a physician before thou hast need of him verse {i}" for i in (1, 2))
    tpl = tmp_path / "t.jinja"
    tpl.write_text(TEMPLATE, encoding="utf-8")
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tok.json"))
    rows, drops = C.pd_pairs(he, en, ocr, "slug", "public domain", tmp_path / "tok.json", tpl, group=1)
    assert len(rows) == 2 and drops == {"unverified_in_text_layer": 1}
    m = rows[0]["meta"]
    assert m["family"] == "translate_en" and m["source"] == "book:slug" and m["lang_class"] == "HE"
    assert m["licence_class"] == "pd" and rows[0]["messages"][2]["content"].startswith("(1) Honour")
    rows2, drops2 = C.pd_pairs(he, en, ocr, "slug", "public domain", tmp_path / "tok.json", tpl, group=1,
                               band=(5.0, 6.0))
    assert rows2 == [] and drops2.get("length_band") == 2
    assert C._looks_numeral("יז") and C._looks_numeral("טו") and not C._looks_numeral("זרה")
