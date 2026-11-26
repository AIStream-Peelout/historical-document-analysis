"""Tests of the gate v2 target-cleaning rules (``ja_ar_build.clean_v2``). Synthetic strings only, no edition text."""
import random

import pytest

from src.finetuning.qwen_hebrew.ja_ar_build import clean_v2 as c

# made-up lines of 20+ letters (Hebrew script, Judaeo-Arabic style, Arabic script)
HE_A = "האיש הלך אל הבית הגדול עם בנו הקטן"
HE_B = "כתבתי אליך את המכתב הזה ביום השלישי"
HE_C = "ושלחתי לך את הספרים עם השליח הנאמן"
HE_D = "והנה באו האנשים מן העיר הרחוקה אלינו"
HE_E = "ואמרו לנו כי הדרך קשה מאד בימים האלה"
JA_A = "וכאן אלשיך קד וצל אלי אלבלד פי אלשהר"
AR_A = "كتبت اليك هذا الكتاب من البيت الكبير"
FILLER = ["והבאתי את הכסף אל השוק בבקר", "ושאלתי את הסוחר על מחיר הבגדים", "ולא מצאתי את החבר שלי בעיר",
          "וחזרתי אל הבית בערב עם השמש", "ואכלנו לחם ושתינו מים מן הבאר", "וישבנו יחד עד חצות הלילה"]


def log_subs(result: c.CleanResult) -> list:
    """``rule:sub`` of every log entry.

    :param result: Clean result.
    :return: Codes in log order.
    """
    return [f"{e['rule']}:{e['sub']}" for e in result.clean_log]


# ------------------------------------------------------------------------------------------------------------ skeleton

def test_skel_folds_finals_and_drops_marks():
    assert c.skel("שָׁלוֹם [...] 12 abc") == "שלומ"
    assert c.letters("مِنْ أَيْنَ") == 5


def test_skel_expands_presentation_forms():
    assert c.skel("ﬠ") == "ע"


# ---------------------------------------------------------------------------------------------------------- duplicates

def test_long_block_later_copy_dropped():
    block = [HE_A, HE_B, HE_C, HE_D]
    lines = block + [HE_E] + block
    res = c.drop_duplicates(lines)
    assert res.lines == block + [HE_E]
    assert [e["sub"] for e in res.log] == ["block"] * 4
    assert [e["copy_of"] for e in res.log] == [0, 1, 2, 3]
    assert all(e["after"] is None for e in res.log)


def test_short_block_dropped_when_share_high():
    lines = [HE_A, HE_B, HE_A, HE_B]
    res = c.drop_duplicates(lines)
    assert res.lines == [HE_A, HE_B]
    assert res.kept == [0, 1]


def test_short_block_in_register_flagged_not_dropped():
    witness = ["חתם פלוני בן פלוני השמש", "עד שני"]
    lines = [HE_A] + witness + FILLER + [HE_B, HE_C] + witness + [HE_D, HE_E]
    res = c.drop_duplicates(lines)
    assert res.lines == lines
    assert [f["flag"] for f in res.flags] == ["repeated_block"]
    assert res.flags[0]["copy_of"] == 1 and res.flags[0]["share"] < c.MIN_DUPLICATE_SHARE


def test_exact_doubled_line_dropped_once():
    res = c.drop_duplicates([HE_A, HE_A, HE_B])
    assert res.lines == [HE_A, HE_B]
    assert res.log[0]["sub"] == "doubled_line" and res.log[0]["exact"] is True


def test_triple_line_keeps_one_copy():
    res = c.drop_duplicates([HE_A, HE_A, HE_A, HE_B])
    assert res.lines == [HE_A, HE_B]


def test_near_doubled_line_flagged_when_alone():
    near = HE_A.replace("הקטן", "הקטנה")
    res = c.drop_duplicates([near, HE_A, HE_B])
    assert res.lines == [near, HE_A, HE_B]
    assert res.flags[0]["flag"] == "repeated_line" and res.flags[0]["adjacent"] is True


def test_near_doubled_line_dropped_when_doubling_is_systematic():
    near = HE_C.replace("הנאמן", "הנאמנה")
    lines = [HE_A, HE_A, HE_B, HE_B, HE_C, near]
    res = c.drop_duplicates(lines)
    assert res.lines == [HE_A, HE_B, HE_C]


def test_formula_line_repeated_elsewhere_is_kept():
    lines = [HE_A, HE_B, HE_C, HE_A, HE_D]
    res = c.drop_duplicates(lines)
    assert res.lines == lines
    assert res.flags[0]["flag"] == "repeated_line" and res.flags[0]["copy_of"] == 0


def test_short_lines_never_match():
    assert c.drop_duplicates(["א ב", "א ב", "א ב"]).lines == ["א ב", "א ב", "א ב"]


def test_lines_match_near_and_exact():
    assert c.lines_match(c.skel(HE_A), c.skel(HE_A.replace(" ", "")))
    assert not c.lines_match(c.skel(HE_A), c.skel(HE_B))


# ----------------------------------------------------------------------------------------------------------- editorial

def test_latin_and_number_lines_dropped():
    res = c.drop_editorial([HE_A, "Recto", "12", "(10) 3.", HE_B])
    assert res.lines == [HE_A, HE_B]
    assert {e["sub"] for e in res.log} == {"latin_line", "bracket_note"}


def test_protected_lines_never_dropped():
    lines = [AR_A + " 12", "כ''ה ל''ח ק''ג", "[...]", "١٢ ٣٠", "½ 3"]
    res = c.drop_editorial(lines)
    assert res.lines == lines and not res.log
    assert [f["flag"] for f in res.flags] == ["numeral_line"]


def test_arabic_line_in_hebrew_side_and_hebrew_line_in_arabic_side_kept():
    side = [HE_A, AR_A, HE_B]
    assert c.clean_side(side, source="pgp", script="hebrew").lines == side
    side = [AR_A, HE_A, AR_A.replace("الكبير", "الصغير")]
    assert c.clean_side(side, source="literature", script="arabic").lines == side


def test_bracket_notes_dropped_or_gapped():
    line = "כתב (margin) הסופר [sic] בשנה (+1,975) הזאת (1) עם (!) אחיו (?) כאן"
    out, removed, _, _ = c.bracket_notes(line)
    assert out == "כתב הסופר בשנה הזאת עם אחיו כאן"
    assert len(removed) == 6
    assert c.bracket_notes("אלל(ה [...] nrot) אפצל")[0] == "אלל [...] אפצל"
    assert c.bracket_notes("כתב (torn) הסופר")[0] == "כתב [...] הסופר"
    assert c.bracket_notes("כתב (perhaps) הסופר")[0] == "כתב הסופר"
    assert c.bracket_notes("כתב (perhaps אבו) הסופר")[0] == "כתב [...] הסופר"
    assert c.bracket_notes("להום (97ב) אתר")[0] == "להום אתר"


def test_mixed_bracket_without_loss_word_flagged_and_kept():
    out, removed, protected, codes = c.bracket_notes("שם (seal: דוד) כתוב")
    assert out == "שם (seal: דוד) כתוב" and not removed
    assert codes == ["bracket_mixed_latin"] and protected


def test_gap_token_is_not_a_bracket_note():
    assert c.bracket_notes("כתב [...] הסופר")[1] == []


def test_inline_latin_runs():
    res = c.drop_editorial(["nigram: " + HE_A, HE_B + " your strength, when", "X " + HE_C, "גvץ " + HE_D,
                            HE_E + " Natan Amram"])
    assert res.lines[0] == HE_A
    assert res.lines[1] == HE_B
    assert res.lines[2] == "X " + HE_C
    assert res.lines[4] == HE_E + " Natan Amram"
    codes = sorted(f["flag"] for f in res.flags)
    assert codes == ["latin_glued", "latin_inline", "latin_inline"]


def test_folio_label_token_dropped():
    assert c.drop_editorial([HE_A + " 1r " + HE_B]).lines == [HE_A + " " + HE_B]


def test_footnote_anchors():
    assert c.footnote_anchors("בספר ישראל46 ועוד")[0] == "בספר ישראל ועוד"
    assert c.footnote_anchors("קבל קכז3/4 דינאר")[0] == "קבל קכז3/4 דינאר"
    assert c.footnote_anchors("ומן פ2 אלתמן") == ("ומן פ2 אלתמן", [], ["glued_digits"])
    out, removed, codes = c.footnote_anchors("הדה אלס6 דינאר")
    assert out == "הדה אלס6 דינאר" and not removed and "glued_digits" in codes
    assert c.footnote_anchors("[אנחנו]14 ונאמר")[0] == "[אנחנו] ונאמר"
    assert c.footnote_anchors("אל [...]3 דרהם")[0] == "אל [...]3 דרהם"


def test_line_numbers_chain_dropped_amounts_kept():
    lines = ["1 " + HE_A, "2 " + HE_B, "3 " + HE_C, "25 פצה דראהם", "40 " + HE_D]
    res = c.drop_editorial(lines)
    assert res.lines == [HE_A, HE_B, HE_C, "25 פצה דראהם", "40 " + HE_D]
    assert all(e["sub"] == "line_number" for e in res.log)


def test_inline_line_numbers_of_flattened_text_dropped():
    line = f"39 {HE_A} 40 {HE_B} 41 {HE_C}"
    assert c.drop_editorial([line]).lines == [f"{HE_A} {HE_B} {HE_C}"]


def test_every_fifth_line_numbers_dropped():
    lines = [FILLER[k % len(FILLER)] for k in range(16)]
    for k in (4, 9, 14):
        lines[k] = f"{k + 1} {HE_B}"
    res = c.drop_editorial(lines)
    assert res.lines[4] == res.lines[9] == res.lines[14] == HE_B
    assert [e["line_index"] for e in res.log] == [4, 9, 14]


def test_single_leading_number_kept():
    lines = [HE_A, "2 " + HE_B, HE_C]
    assert c.drop_editorial(lines).lines == lines


def test_apparatus_tail_and_footnote_line():
    res = c.drop_editorial([HE_A + " הערות שורה : ביום תוספת מעל השורה", "r20 " + HE_B, HE_C])
    assert res.lines == [HE_A, HE_C]
    assert [e["sub"] for e in res.log] == ["apparatus_tail", "footnote_line"]
    assert res.log[0]["letters_removed"] > 0


def test_rule_lines_removed():
    assert c.drop_editorial(["סך הכל__________ " + HE_A]).lines == ["סך הכל " + HE_A]


def test_hebrew_editorial_phrase_only_flagged():
    line = HE_A + " לכלף, הכוונה"
    res = c.drop_editorial([line])
    assert res.lines == [line]
    assert res.flags[0]["flag"] == "hebrew_editorial_candidate"


def test_unchanged_lines_keep_their_spacing():
    line = "  " + HE_A + "   "
    assert c.drop_editorial([line]).lines == [line]


# -------------------------------------------------------------------------------------------------------- restorations

def test_restorations_left_for_cleaned_sources():
    res = c.render_restorations(["ע[ושה] לנו את (הדבר) הזה"], source="pgp")
    assert res.lines == ["ע[ושה] לנו את (הדבר) הזה"]
    assert not res.log
    assert {f["flag"] for f in res.flags} == {"paren_unresolved"}


def test_leiden_square_restorations_become_gaps():
    res = c.render_restorations(["ע[ושה] לנו [..] את [ ] הדבר [...] הזה", "] מן הצד ["], source="literature")
    assert res.lines == ["ע[...] לנו [...] את [...] הדבר [...] הזה", "[...] מן הצד [...]"]
    assert res.log[0]["sub"] == "square" and res.log[0]["letters_removed"] == 3
    assert res.log[1]["sub"] == "edge" and res.log[1]["letters_removed"] == 0


def test_leiden_leaves_ambiguous_notation():
    res = c.render_restorations(["אל [[כתב]] ספר (רבי) ⟨א⟩ {ו} בר"], source="literature")
    assert res.lines == ["אל [[כתב]] ספר (רבי) ⟨א⟩ {ו} בר"]
    assert {f["flag"] for f in res.flags} == {"double_bracket_unresolved", "paren_unresolved", "angle_unresolved",
                                              "brace_unresolved"}


def test_dot_parens_are_gaps():
    assert c.render_restorations(["כתב (...) הספר"], source="literature").lines == ["כתב [...] הספר"]


def test_weiss_side_converts_parentheses_only():
    lines = ["( )אר אבנה אבו (ת )", "דלי(ת ) ועוד [שם] כתב", "ואכד לנא (ר"]
    res = c.render_restorations(lines, source="lit_run2")
    assert res.lines == ["[...]אר אבנה אבו [...]", "דלי[...] ועוד [שם] כתב", "ואכד לנא [...]"]
    assert "square_in_weiss_unresolved" in {f["flag"] for f in res.flags}
    assert res.log[2]["sub"] == "weiss_edge"


def test_weiss_long_open_run_left():
    tail = "פי יום אלאחד אלמדכור פלמא עלמנא בדלך כתבנא אליך"
    res = c.render_restorations(["( ) כתב", "אל(" + tail, "תפארת שרי הבית פ) ועוד"], source="literature")
    assert res.lines[1] == "אל(" + tail
    assert res.lines[2] == "תפארת שרי הבית פ) ועוד"


def test_weiss_edge_run_never_crosses_a_lacuna():
    res = c.render_restorations(["( ) אבג דהו ( ) בר ש)לה נן"], source="lit_run2")
    assert res.lines == ["[...] אבג דהו [...] בר ש)לה נן"]
    assert "bracket_unpaired" in {f["flag"] for f in res.flags}


def test_mirrored_side_not_converted():
    lines = ["ראש]ישיבת גאון[ יהי שמו", "ע[ושה] לנו"]
    res = c.render_restorations(lines, source="literature")
    assert res.lines == lines and not res.log
    assert res.flags[0]["flag"] == "brackets_mirrored"
    hinted = c.render_restorations(["ע[ושה] לנו"], source="literature", mirrored=True)
    assert hinted.lines == ["ע[ושה] לנו"]


def test_mirrored_works_detection():
    sides = [(f"lit:col:bookA:p{k}:x", "literature", ["ראש]ישיבת גאון[ יהי", "כיר]ח יכון[ עולם"]) for k in range(3)]
    sides += [(f"lit:col:bookB:p{k}:x", "literature", ["ע[ושה] לנו", "ראש]ישיבת גאון[ יהי"]) for k in range(1)]
    sides += [("pgp:1", "pgp", ["ראש]ישיבת גאון[ יהי"] * 9)]
    assert c.mirrored_works(sides) == {"literature:lit:col:bookA"}
    assert c.work_key("lit:col:bookA:p3:x", "literature") == "literature:lit:col:bookA"


def test_mirrored_pairs_ignores_edges_and_proper_pairs():
    assert c.mirrored_pairs("] מן הצד [") == 0
    assert c.mirrored_pairs("ע[ושה] לנו [ה]דבר") == 0
    assert c.mirrored_pairs("ראש]ישיבת גאון[ יהי") == 1


# ----------------------------------------------------------------------------------------------------------- flattened

def test_flattened_flags():
    long_line = " ".join([HE_A, HE_B, HE_C, HE_D, HE_E])
    assert c.letters(long_line) > c.FLAT_LINE_LETTERS
    res = c.flag_flattened([long_line])
    assert res.flags[0]["flag"] == "flattened" and res.flags[0]["side_flat"] is True
    res = c.flag_flattened([HE_A, long_line, HE_B])
    assert res.flags[0]["line_index"] == 1 and res.flags[0]["side_flat"] is False
    assert res.lines == [HE_A, long_line, HE_B]


def test_line_at_threshold_not_flattened():
    line = "א" * c.FLAT_LINE_LETTERS
    assert c.flag_flattened([line]).flags == []


# ----------------------------------------------------------------------------------------------------------- scrambled

WORDS = ("האיש הלך אל הבית הגדול עם בנו הקטן כתבתי אליך את המכתב הזה ביום השלישי ושלחתי לך הספרים השליח "
         "הנאמן והנה באו האנשים מן העיר הרחוקה אלינו ואמרו לנו כי הדרך קשה מאד בימים האלה").split()


def corpus(n: int = 600, seed: int = 7) -> list:
    """Synthetic training lines: random word sequences.

    :param n: Lines.
    :param seed: Seed.
    :return: ``(line, script)`` pairs.
    """
    rng = random.Random(seed)
    return [(" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 9))), "hebrew") for _ in range(n)]


def scramble(line: str, seed: int = 3) -> str:
    """Shuffle the letters inside every word.

    :param line: Line.
    :param seed: Seed.
    :return: Scrambled line.
    """
    rng = random.Random(seed)
    out = []
    for word in line.split():
        letters_ = list(word)
        rng.shuffle(letters_)
        out.append("".join(letters_))
    return " ".join(out)


@pytest.fixture(scope="module")
def models() -> dict:
    """Models trained on the synthetic corpus.

    :return: ``{script: model}``.
    """
    return c.train_models(corpus())


def test_model_scores_scrambled_line_lower(models):
    heb = models["hebrew"]
    good = "כתבתי אליך את המכתב הזה ביום השלישי"
    bad = scramble(good)
    assert heb.score(bad)[0] < heb.threshold < heb.score(good)[0]
    res = c.flag_scrambled([good, bad], models)
    assert [f["line_index"] for f in res.flags] == [1]
    assert res.flags[0]["basis"] == "line" and res.flags[0]["score"] < res.flags[0]["threshold"]


def test_reversed_line_marked(models):
    good = "והנה באו האנשים מן העיר הרחוקה אלינו"
    flagged = c.flag_scrambled([good[::-1]], models).flags
    assert flagged and flagged[0]["reversed"] is True


def test_window_flag_on_flattened_line(models):
    good = [" ".join(random.Random(k).choice(WORDS) for _ in range(8)) for k in range(6)]
    long_line = " ".join(good[:3] + [scramble(good[3])] + good[4:])
    assert c.letters(long_line) > c.FLAT_LINE_LETTERS
    flags = c.flag_scrambled([long_line], models).flags
    assert flags and flags[0]["basis"] in ("line", "window")


def test_short_line_not_scored(models):
    assert models["hebrew"].score("אב גד") is None


def test_model_roundtrip(models, tmp_path):
    paths = c.save_models(models, tmp_path / "models")
    assert paths["hebrew"].name == "char3_hebrew.json"
    loaded = c.load_models(tmp_path / "models")
    line = "האיש הלך אל הבית הגדול עם בנו"
    assert loaded["hebrew"].score(line) == pytest.approx(models["hebrew"].score(line))
    assert loaded["hebrew"].threshold == pytest.approx(models["hebrew"].threshold)


def test_skip_tokens_and_segments():
    assert c.skip_token("ק'") and c.skip_token("ש'מ") and c.skip_token("א") and c.skip_token("דינ'")
    assert not c.skip_token("אלמד'כורין") and not c.skip_token("הבית")
    assert c.lm_segments("כתבתי אלי [...] ך את הספר", "hebrew") == ["כתבתי", "את הספר"]
    assert c.lm_segments("ק'ג' דינ' הספר הזה", "hebrew") == ["הספר הזה"]
    assert c.lm_segments("הספר הזה كتاب البيت", "hebrew") == ["הספר"]
    assert c.lm_segments("كتبت اليك هذا", "arabic") == ["كتبت اليك هذا"]
    assert c.lm_segments("בית־דין הגדול/שלום ישראל46", "hebrew") == ["בית דין הגדול שלום ישראל"]


def test_line_script():
    assert c.line_script(AR_A) == "arabic"
    assert c.line_script(HE_A) == "hebrew"
    assert c.line_script("12", "arabic") == "arabic"


# ---------------------------------------------------------------------------------------------------------- clean_side

def test_clean_side_maps_indices_to_input(models):
    lines = ["Recto", HE_A, HE_B, HE_C, HE_D, HE_A, HE_B, HE_C, HE_D, "ע[ושה] לנו (margin) את הדבר"]
    res = c.clean_side(lines, source="literature", script="hebrew", model=models)
    assert res.lines == [HE_A, HE_B, HE_C, HE_D, "ע[...] לנו את הדבר"]
    assert res.origin == [1, 2, 3, 4, 9]
    assert log_subs(res) == ["duplicate:block"] * 4 + ["editorial:latin_line", "editorial:bracket_note",
                                                        "restoration:square"]
    assert [e["line_index"] for e in res.clean_log if e["rule"] == "duplicate"] == [5, 6, 7, 8]
    assert [e["copy_of"] for e in res.clean_log if e["rule"] == "duplicate"] == [1, 2, 3, 4]
    assert res.clean_log[-1]["line_index"] == 9
    assert res.letters_removed > 0


def test_clean_side_flags_carry_both_indices():
    long_line = " ".join([HE_A, HE_B, HE_C, HE_D, HE_E])
    res = c.clean_side(["Verso", HE_A, long_line], source="pgp", script="hebrew")
    flat = [f for f in res.flags if f["flag"] == "flattened"][0]
    assert flat["line_index"] == 2 and flat["out_index"] == 1
    assert res.flag_codes == ["flattened"]


def test_clean_side_without_model_has_no_scrambled_flags():
    res = c.clean_side([scramble(HE_A)], source="fjp", script="hebrew", model=None)
    assert not [f for f in res.flags if f["flag"] == "scrambled_candidate"]
    assert res.lines == [scramble(HE_A)]


def test_clean_side_no_change_on_clean_text():
    lines = [HE_A, HE_B, JA_A]
    res = c.clean_side(lines, source="fjp", script="hebrew")
    assert res.lines == lines and res.clean_log == [] and res.origin == [0, 1, 2]
