# File name: test_build_pgp_editions.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the evidence-gated PGP edition builder (``build_pgp_editions``).

Covers label parsing, notation cleaning and commentary removal, block/side parsing, the
block-presence rule on synthetic pages, the strict side rule including every DROP case, answer
assembly and line numbering, the two reader cross-checks, the split and decontamination keys.
"""
import json
import random
from typing import Dict, Iterable, List

import pytest

from src.finetuning.qwen_hebrew import build_pgp_editions as B
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES

# --------------------------------------------------------------------------- synthetic text

RECTO = [
    "כתאבי אליך יא מולאי אטאל אללה בקאך ואדאם עזך",
    "וצלני כתאבך אלכרים ופהמת מא דכרתה מן אמר אלבצאעה",
    "ואנא מנתטר לוצול אלמרכב אלדי פיה אלכתאן ואלשמע",
    "פאדא וצל אן שא אללה בעת אליך במא יחצל מן תמנה",
    "ואקרא עליך אפצל אלסלאם ועלי מן תחוטה ענאיתך",
]
VERSO = [
    "למולאי אלשיך אבו אלפרג יוסף בן יעקב נע",
    "מן עבדה שלמה בן נסים אלמערוף באבן אלעטאר",
    "ישוע רב וקרובה ישועתו בכל עת ובכל זמן",
]
UNRELATED = [
    "בשני בשבת בעשרים יום לירח אדר שנת אתתצט לשטרות",
    "הודה לנא פלוני בן פלוני הודאה גמורה בלא אונס",
    "וקבלנא מנה קנין שלם במנא דכשר למקניא ביה",
    "ודלא כאסמכתא ודלא כטופסי דשטרי שריר ובריר וקים",
    "ומא כתבת הדא אלא למא קד עלמת מן חאלך ואנת",
    "תעלם מא ענדי מן אלשוק אליך ואלי ראיתך",
    "קאל רבי יוחנן אין מעמידין אלא על דברי תורה",
    "ומנה אלגמלה תלתה דנאניר ונצף וקיראט",
]


def reader_of(lines: List[str]) -> B.Reader:
    """A Reader whose VLM read exactly ``lines`` (no Kraken rows).

    :param lines: Reader lines.
    :returns: Reader.
    """
    return B.Reader(vlm=list(lines), rows=[], grams=B.grams(lines))


def noisy(line: str, k: int = 2, seed: int = 0) -> str:
    """Replace ``k`` letters of a line (a mis-read).

    :param line: Text.
    :param k: Substitutions.
    :param seed: RNG seed.
    :returns: Noisy text.
    """
    rng = random.Random(seed)
    chars = list(line)
    idx = [i for i, c in enumerate(chars) if "א" <= c <= "ת"]
    for i in rng.sample(idx, min(k, len(idx))):
        chars[i] = "ש" if chars[i] != "ש" else "ת"
    return "".join(chars)


@pytest.fixture
def nulls() -> List[B.Reader]:
    """Null readers: unrelated documentary lines, shuffled into pages.

    :returns: List of readers.
    """
    rng = random.Random(1)
    out = []
    for _ in range(20):
        out.append(reader_of(rng.sample(UNRELATED, 4)))
    return out


LABELLED = "\n".join(["Recto", *RECTO, "", "Recto - right margin", "ואלסלאם עליך", "",
                      "Verso - address", *VERSO])

# --------------------------------------------------------------------------- labels


@pytest.mark.parametrize("line,side,kind,paging,bare", [
    ("Recto", "recto", "main", False, True),
    ("verso:", "verso", "main", False, True),
    ("(verso)", "verso", "main", False, True),
    ("Recto - right margin", "recto", "margin", False, False),
    ("Verso - address", "verso", "address", False, False),
    ("verso - bottom margin - address", "verso", "address", False, False),
    ("Right margin, perpendicular lines.", None, "margin", False, False),
    ("Upside-Down", None, "margin", False, False),
    ('ע"ב', "verso", "main", False, True),
    ("ע״א", "recto", "main", False, True),
    ("שוליים ימניים", None, "margin", False, False),
    ("page b", None, "main", True, False),
    ("verso (השלישי)", "verso", "main", True, False),
    (":Addendum", None, "other", False, False),
])
def test_parse_label(line, side, kind, paging, bare):
    """Side, region, paging and bare flags of common PGP labels."""
    lab = B.parse_label(line)
    assert lab is not None
    assert (lab.side, lab.kind, lab.paging, lab.bare) == (side, kind, paging, bare)


@pytest.mark.parametrize("line", ["Footnotes", "Notes:", "הערות", "Remarks הערות"])
def test_apparatus_headers_stop(line):
    """Apparatus headers end the edition."""
    assert B.parse_label(line).kind == "stop"


@pytest.mark.parametrize("line", ["Transcription from FGP Metadata", "in the same direction",
                                  "כתאבי אליך יא מולאי", "بسم الله الرحمن الرحيم", "[...]"])
def test_non_labels(line):
    """Prose, content and gap lines are not labels."""
    assert B.parse_label(line) is None


# --------------------------------------------------------------------------- cleaning


def test_clean_line_restorations_become_gaps():
    """Bracketed restorations render as the ``[...]`` gap token."""
    cl = B.clean_line("וצלני כתא[בך] אלכרים")
    assert cl.kind == "content" and "[...]" in cl.text and "[בך]" not in cl.text


@pytest.mark.parametrize("raw,expected", [
    ("מולי (!) וריסי", "מולי וריסי"),
    ("ל(!)ך [?] אבג", "לך אבג"),
    ("עלי יד אן [= אבן] אלפאעוס", "עלי יד אן אלפאעוס"),
    ("ולא //אקול\\\\ אנהא", "ולא אקול אנהא"),
    ("ואסב/א\\באה מן", "ואסבאבאה מן"),
    ("יעא/י/נהא ממן", "יעאינהא ממן"),
    ("ואלי י<ו>מי הדא", "ואלי ימי הדא"),
    ("זבאד (civet cats) ומעל", "זבאד ומעל"),
    ("… אודה וקלבנא", "[...] אודה וקלבנא"),
    ("לוחין א נגאר3/4 עמל", "לוחין א נגאר3/4 עמל"),
    ("ויהב(נא) לה", "ויהב לה"),                  # expansion dropped by clean_diplomatic
    ("וקד עלם אללה | ואנת תעלם", "וקד עלם אללה ואנת תעלם"),   # pipe -> whitespace, same
])
def test_notation_normalisation(raw, expected):
    """PGP notation is resolved to visible ink (here or in clean_diplomatic)."""
    cl = B.clean_line(raw)
    assert cl.kind == "content" and cl.text == expected and not cl.flags


@pytest.mark.parametrize("raw", [
    "ראו גויטיין, חברה, ב, עמ' 35",
    "כרך ב, עמ' 12",
    "The correct location of the following lines is in line 5",
    "Transcription from FGP Metadata",
])
def test_commentary_removed(raw):
    """Editor notes are commentary, not ink."""
    assert B.clean_line(raw).kind == "commentary"


@pytest.mark.parametrize("raw", ["ושכרך ירוברב ויותקף", "אני זוכרך תמיד", "שאהיה עמ'"])
def test_words_containing_reference_markers_are_content(raw):
    """``כרך`` inside ordinary words and the scribal ``עמ'`` are not edition references."""
    assert B.clean_line(raw).kind == "content"


@pytest.mark.parametrize("raw", ["torn off", "(no reading)", "[...]", "....", "½"])
def test_lost_ink_lines_are_dropped(raw):
    """Lines that stand for ink the edition does not render are ``dropped`` (not commentary)."""
    assert B.clean_line(raw).kind == "dropped"


@pytest.mark.parametrize("raw,flag", [
    ("אבג (12) דהו", "parentheses"),
    ("אלסלם* וקד אשגל", "asterisk"),
    ("ניחומי46 ופשט", "glued_digits"),
    ("אטלז ברווגי/כרווגי בוקגה", "slash"),
    ("ותבתת //במעשה פיה", "slash"),
    ("9ליצירה אקום לה", "glued_digits"),
    ("סער כ'ה' ו>' מנה", "angle_brackets"),
    ("מן אלרמ לה ?", "question_mark"),
    ("In a different hand and ink ביד ובדיו אחרים", "latin_gloss"),
])
def test_unresolved_notation_is_flagged(raw, flag):
    """Notation that cannot be rendered as ink is kept but flagged (the page is then dropped)."""
    assert flag in B.clean_line(raw).flags


def test_numbered_edition_strips_line_numbers():
    """Leading line numbers of a numbered edition are removed."""
    content = "\n".join(f"{i}. {ln}" for i, ln in enumerate(RECTO, 1))
    ed = B.parse_edition(content)
    assert ed.numbered and ed.blocks[0].lines == RECTO


def test_flattened_edition_detected():
    """Inline ``(1) ... (2) ...`` markers mean the line breaks were lost."""
    assert B.parse_edition("(1) " + RECTO[0] + " (2) " + RECTO[1]).flattened


# --------------------------------------------------------------------------- parsing


def test_parse_edition_sides_and_kinds():
    """Labels open blocks with side/region; unlabelled margins inherit the current side."""
    ed = B.parse_edition(LABELLED)
    assert ed.labelled and not ed.complex_sides
    assert [(b.side, b.kind) for b in ed.blocks] == [("recto", "main"), ("recto", "margin"), ("verso", "address")]
    assert ed.blocks[0].lines == RECTO and ed.blocks[2].lines == VERSO


def test_blocks_before_first_side_label_are_recto():
    """Text before the first side label belongs to the recto."""
    ed = B.parse_edition("\n".join([*RECTO, "", "Verso", *VERSO]))
    assert [b.side for b in ed.blocks] == ["recto", "verso"]


def test_unlabelled_edition_has_no_side():
    """An edition without side labels is treated as single-sided."""
    ed = B.parse_edition("\n".join(RECTO))
    assert not ed.labelled and ed.blocks[0].side is None


def test_footnotes_and_rules():
    """Apparatus headers stop parsing; drawn rules only end a block."""
    ed = B.parse_edition("\n".join([*RECTO[:2], "-------", *RECTO[2:], "Footnotes", "1 ראו גויטיין"]))
    assert [b.lines for b in ed.blocks] == [RECTO[:2], RECTO[2:]] and ed.n_tail == 1


def test_repeated_bare_side_label_is_complex():
    """A side opened twice by a bare label (several leaves) is a complex structure."""
    ed = B.parse_edition("\n".join(["Recto", RECTO[0], "Verso", VERSO[0], "Recto", RECTO[1]]))
    assert ed.complex_sides


def test_countable_stops_after_a_lost_line():
    """Lines below a lost (unrendered) line cannot be addressed by their number."""
    ed = B.parse_edition("\n".join([RECTO[0], RECTO[1], "[...]", RECTO[2], RECTO[3]]))
    assert ed.blocks[0].countable == [True, True, False, False]
    assert ed.blocks[0].n_dropped == 1


# --------------------------------------------------------------------------- readers


def test_kraken_rows_cluster_and_order():
    """Fragments cluster into rows by vertical centre; each row reads right to left."""
    frags = [{"text": "ב", "box": [100, 100, 200, 140]}, {"text": "א", "box": [300, 105, 400, 138]},
             {"text": "ג", "box": [300, 300, 400, 340]}]
    assert B.kraken_rows(frags) == ["אב", "ג"]


def test_reader_from_raw_recovers_unparsed_vlm_text():
    """When the VLM output did not parse, its ``text`` fields are recovered from ``vlm_raw``."""
    raw = {"vlm_lines": [], "vlm_raw": '[{"text": "\\u05d0\\u05d1\\u05d2\\u05d3", "bbox_2d": [1, 2, 3, 4]}, {"text": "הו',
           "frags": []}
    assert B.reader_from_raw(raw).vlm == ["אבגד"]


# --------------------------------------------------------------------------- block presence


def test_block_on_page_is_kept(nulls):
    """A block read (noisily) on the page beats its null and is anchored."""
    reader = reader_of([noisy(ln, 2, i) for i, ln in enumerate(RECTO)])
    t = B.test_block(RECTO, reader, nulls)
    assert t.kept and t.share > t.null_max and t.hits >= B.MIN_HITS


def test_block_of_other_side_is_not_kept(nulls):
    """The other side's block is not on this page's readers."""
    reader = reader_of(RECTO)
    assert not B.test_block(VERSO, reader, nulls).kept


def test_shared_vocabulary_without_a_line_is_not_kept(nulls):
    """N-gram overlap from shared words alone (no line read at >= 0.5) is not presence."""
    words = " ".join(RECTO).split()
    shuffled = [" ".join(words[i::5]) for i in range(5)]      # same words, different lines
    t = B.test_block(RECTO, reader_of(shuffled), nulls)
    assert t.ngram_ok and not t.anchored and not t.kept      # the n-gram test alone would keep it
    assert B.anchored(RECTO, RECTO) and not B.anchored(RECTO, UNRELATED)


def test_null_sample_is_deterministic():
    """The per-page null does not depend on pool order."""
    pool = [f"p{i}" for i in range(80)]
    a = B.null_sample("X__0", pool)
    b = B.null_sample("X__0", list(reversed(pool)))
    assert a == b and len(a[0]) == B.NULL_K and a[1] not in a[0]


# --------------------------------------------------------------------------- side rule


def _ed(content: str) -> B.ParsedEdition:
    """Parse an edition text.

    :param content: Edition text.
    :returns: Parsed edition.
    """
    return B.parse_edition(content)


def test_side_rule_includes_complete_side():
    """Recto main + margin kept, verso not: included on the recto."""
    ed = _ed(LABELLED)
    d = B.decide_page(ed, [True, True, False])
    assert d.included and d.side == "recto" and d.coverage == 1.0


def test_side_rule_drops_incomplete_side():
    """A kept side missing more than 10% of its letters is dropped."""
    ed = _ed("\n".join(["Recto", *RECTO[:2], "", *RECTO[2:], "", "Verso", *VERSO]))
    d = B.decide_page(ed, [True, False, False])
    assert not d.included and d.reason == "side_incomplete" and d.coverage < B.SIDE_COVERAGE


def test_side_rule_tolerates_small_unkept_block():
    """An unkept block below 10% of the side's letters does not drop the page."""
    ed = _ed(LABELLED)
    d = B.decide_page(ed, [True, False, False])
    assert d.included and B.SIDE_COVERAGE <= d.coverage < 1.0


def test_side_rule_drops_both_sides():
    """Blocks kept on both sides: the page is dropped."""
    d = B.decide_page(_ed(LABELLED), [True, False, True])
    assert not d.included and d.reason == "both_sides_kept"


def test_side_rule_unlabelled_requires_whole_edition():
    """Unlabelled editions must be >= 90% present on the image."""
    ed = _ed("\n".join([*RECTO, "", *VERSO]))
    assert B.decide_page(ed, [True, True]).included
    d = B.decide_page(ed, [True, False])
    assert not d.included and d.reason == "unlabelled_incomplete"


def test_side_rule_drops_complex_and_empty():
    """Complex side labels or no kept block: dropped."""
    ed = _ed("\n".join(["page a", *RECTO, "page b", *VERSO]))
    assert B.decide_page(ed, [True, False]).reason == "complex_side_labels"
    assert B.decide_page(_ed(LABELLED), [False, False, False]).reason == "no_block_kept"


def test_side_rule_counts_unverifiable_arabic():
    """An Arabic-script block on the kept side (never testable) cannot be silently omitted."""
    arabic = ["لسيدي ومولاي الشيخ ابو الفرج يوسف بن يعقوب ادام الله عزه وتاييده"] * 6
    ed = _ed("\n".join(["Recto", *RECTO, "", "Recto - right margin", *arabic]))
    d = B.decide_page(ed, [True, False])
    assert not d.included and d.reason == "side_arabic_unverified"


# --------------------------------------------------------------------------- answer + rows


def test_answer_puts_margins_after_main_text():
    """Main text first, then margin/address blocks, in edition order."""
    ed = _ed("\n".join(["Recto - top margin", "ואלסלאם עליך ורחמה", "", "Recto", *RECTO]))
    ans = B.assemble_answer(ed, [True, True], "recto")
    assert ans.lines == RECTO + ["ואלסלאם עליך ורחמה"] and ans.regions[-1] == "margin"
    assert all(ans.countable[:len(RECTO)]) and not ans.countable[-1]


def test_nothing_countable_when_first_main_block_not_kept():
    """If the top of the main text is not in the answer, no line can be addressed by number."""
    ed = _ed("\n".join(["Recto", RECTO[0], "", *RECTO[1:]]))
    ans = B.assemble_answer(ed, [False, True], "recto")
    assert not any(ans.countable)


def test_answer_gate():
    """Flagged notation, too few letters or lines reject the answer."""
    ed = _ed("\n".join(RECTO))
    ans = B.assemble_answer(ed, [True], None)
    assert B.answer_gate(ans) is None
    short = B.assemble_answer(_ed("\n".join(RECTO[:2])), [True], None)
    assert B.answer_gate(short) in ("answer_too_few_letters", "answer_too_few_lines")
    flagged = B.assemble_answer(_ed("\n".join([*RECTO, "אלסלם* וקד אשגל"])), [True], None)
    assert B.answer_gate(flagged) == "notation_asterisk"


def test_line_by_number_rows():
    """One-based line numbers, lines with >= 8 letters and no gap, countable only."""
    lines = [*RECTO[:2], "אבג [...] דהוז חטי כלמנ", *RECTO[2:]]
    ans = B.assemble_answer(_ed("\n".join(lines)), [True], None)
    items = B.line_by_number_items(ans, random.Random(0), k=10)
    assert [n for _, _, n in items] == [1, 2, 4, 5, 6]
    for q, a, n in items:
        assert q == B.LINE_BY_NUMBER_PROMPT.format(n=n) and a == ans.lines[n - 1]


def test_line_of_phrase_row():
    """The phrase is unique on the page and the answer quotes its whole line."""
    ans = B.assemble_answer(_ed("\n".join(RECTO)), [True], None)
    q, a, n = B.line_of_phrase_item(ans, random.Random(3))
    obj = json.loads(a)
    phrase = q.split("«")[1].split("»")[0]
    assert obj["line"] == n and obj["text"] == ans.lines[n - 1] and phrase in obj["text"]
    assert ans.text.count(phrase) == 1


def test_row_schema_matches_ktiv_features():
    """Rows carry exactly the KTIV FEATURES columns."""
    r = B.row(B.Path("x.jpg"), "q", " a ", "page", "pgp_page", "s", 10, 20)
    assert set(r) == set(FEATURES) and r["answer"] == "a" and r["label_source"] == "pgp_edition"
    assert r["task"] == "fragment_transcribe"


# --------------------------------------------------------------------------- reader cross-checks


def test_unsupported_window_flags_other_side_text(nulls):
    """An unlabelled block running into text not on the image fails a sliding window."""
    other = UNRELATED[:6]
    ed = _ed("\n".join(RECTO + other))
    reader = reader_of(RECTO)
    assert B.unsupported_windows(ed, [True], reader, nulls)
    assert not B.unsupported_windows(_ed("\n".join(RECTO + RECTO[:3])), [True], reader_of(RECTO + RECTO[:3]),
                                     nulls)


def test_missing_reader_lines_needs_both_readers():
    """A line both readers see but the answer lacks is reported; a VLM-only line is not."""
    extra = "שהדותא דהות באנפנא אנחנא שהדי דחתמות ידנא לתתא"
    both = B.Reader(vlm=RECTO + [extra], rows=[noisy(extra, 3)], grams=set())
    assert B.missing_reader_lines(RECTO, both) == [extra]
    vlm_only = B.Reader(vlm=RECTO + [extra], rows=[], grams=set())
    assert B.missing_reader_lines(RECTO, vlm_only) == []


# --------------------------------------------------------------------------- split + decontamination


def _docs(n: int, trained: Iterable[str] = ()) -> Dict[str, B.DocInfo]:
    """Synthetic eligible documents.

    :param n: Number of documents.
    :param trained: Ids of documents trained on before.
    :returns: ``pgpid -> DocInfo``.
    """
    return {str(i): B.DocInfo(str(i), f"C_{i}", ["u"], trained=str(i) in trained) for i in range(n)}


def test_split_deterministic_and_held_out():
    """Same input, same split; trained documents never go to val; ~5% of groups in val."""
    docs = _docs(2000, trained={str(i) for i in range(0, 2000, 3)})
    groups = {p: d.canonical_id for p, d in docs.items()}
    s1 = B.assign_splits(docs, groups)
    assert s1 == B.assign_splits(docs, groups)
    assert all(s1[p] == "train" for p in docs if docs[p].trained)
    share = sum(v == "val" for v in s1.values()) / len(s1)
    assert 0.03 < share < 0.07


def test_split_keeps_groups_together():
    """Documents of one group share their split."""
    docs = _docs(400)
    groups = {p: f"G{int(p) // 2}" for p in docs}
    s = B.assign_splits(docs, groups)
    assert all(s[str(i)] == s[str(i + 1)] for i in range(0, 400, 2))


def test_subpart_relation_is_not_box_level():
    """Sub-parts of one shelfmark relate; different fragments of one box do not."""
    pgpids_of = {"Cambridge_CUL_T_S_13J4_15": ["1"]}
    assert B.subpart_related("Cambridge_CUL_T_S_8J4_3", "Cambridge_CUL_T_S_8J4_3_1", {})
    assert B.subpart_related("Cambridge_CUL_T_S_13J4_15_1", "Cambridge_CUL_T_S_13J4_15_2", pgpids_of)
    assert not B.subpart_related("Cambridge_CUL_T_S_10J12_4", "Cambridge_CUL_T_S_10J12_22", {})


def test_fragment_key_matches_bodleian_aliases():
    """One Bodleian fragment keys the same under its three names."""
    gate = B.DecontamGate()
    keys = {B.fragment_key(x, gate) for x in ("Bodl. MS heb. d 79/36", "Oxford_Bodleian_MS_heb_d_79_36",
                                              "Oxford_Bodleian_Bodl_MS_heb_d_79_36")}
    assert len(keys) == 1 and None not in keys
    assert B.fragment_key("T-S 16.110", gate) == B.fragment_key("Cambridge_CUL_T_S_16_110", gate)


def test_benchmark_overlap_containment():
    """A text duplicate has high pair containment; a shared formula does not."""
    text = " ".join(RECTO * 2)
    bench = B.Benchmark(ids=set(), keys=set(), shingles={"b": B.shingle_set(text)}, all_shingles=B.shingle_set(text))
    total, cont, bid = B.benchmark_overlap(text, bench)
    assert cont == 1.0 and bid == "b"
    total, cont, _ = B.benchmark_overlap(" ".join(UNRELATED) + " " + RECTO[0], bench)
    assert 0 < total and cont < B.SHINGLE_CONTAINMENT
