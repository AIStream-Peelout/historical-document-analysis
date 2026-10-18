# File name: test_build_vqa_parse.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the parse-then-answer VQA builder (``build_vqa_parse``).

Covers the prompts (QA wordings with the answer instruction replaced, field definitions cut from the
QA wording), the field mapping and the abstention nulls, the look-ups, the Kraken row clustering and
the line-box rule, the edition -> model-reading line map and the parse choice with its fallbacks, the
row invariants (no model text in a target, cited numbers exist in the parse shown, whole-token
answers), split hygiene, and an end-to-end build on synthetic pages that writes the DatasetDict and
its images-once export.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from PIL import Image as PILImage

from src.finetuning.qwen_hebrew import build_pgp_editions as eds
from src.finetuning.qwen_hebrew import build_pgp_qa as qa
from src.finetuning.qwen_hebrew import build_vqa_parse as V
from src.finetuning.qwen_hebrew.build_ktiv_dataset import FEATURES

LETTER = [
    "כתאבי אליך יא מולאי אטאל אללה בקאך",
    "וצלני כתאבך מע יוסף בן יעקב אלצירפי",
    "ואנא מנתטר לוצול אלמרכב מן אלאסכנדריה",
    "וכתב פי אלעשר אלאול מן שהר כסליו שנת אלפא וחמש מאה",
    "ואקרא עליך אפצל אלסלאם",
]
UNDATED = [
    "שלום רב לאדוני ולכל בני ביתו",
    "ידע אדוני כי הגיע הספר אשר שלחת",
    "ושמחתי בו מאד ואני מבקש ממך",
    "לשלוח לי את הבגדים עם הנושא הזה",
    "ושלומך יגדל ויפרח",
]
VAL_LINES = [
    "אלממלוך יקבל אלארץ בין ידי מולאה",
    "וקד וצל כתאבה אלכרים פי שהר תמוז",
    "ופרחת בה פרחא עטימא גדא",
    "ואלסלאם עליך ורחמת אללה",
]


def count_words(text: str) -> int:
    """A cheap stand-in token counter.

    :param text: Text.
    :returns: Whitespace tokens.
    """
    return len(text.split())


def page_rec(lines: Sequence[str], cid: str = "C", idx: int = 0, pgpid: str = "1", split: str = "train",
             image_path: str = "p.jpg", sha: str = "s") -> Dict[str, Any]:
    """A minimal editions-manifest record.

    :param lines: Edition lines.
    :param cid: Canonical id.
    :param idx: Image index.
    :param pgpid: PGP id.
    :param split: ``train`` or ``val``.
    :param image_path: Image path.
    :param sha: Image sha256.
    :returns: Record (``key`` set).
    """
    return {"pgpid": pgpid, "canonical_id": cid, "image_index": idx, "key": f"{cid}__{idx}", "split": split,
            "lines": list(lines), "regions": ["main"] * len(lines), "countable": [True] * len(lines),
            "image_url": f"https://x/{cid}_{idx}.jpg", "image_path": image_path, "image_sha256": sha,
            "image_width": 400, "image_height": 500}


def fact(stem: str, family: str, section: str, answer: Any, cid: str = "C", idx: int = 0, split: str = "train",
         variants: Sequence[int] = (0, 2), role: Optional[str] = None) -> Dict[str, Any]:
    """A ``pgp_qa`` manifest record.

    :param stem: Fact stem.
    :param family: QA family.
    :param section: QA section.
    :param answer: Answer object (serialised here).
    :param cid: Canonical id.
    :param idx: Image index.
    :param split: Split.
    :param variants: Wording indices (val: ``(0,)``).
    :param role: Template role for person / ketubba-name wordings.
    :returns: Record.
    """
    key = f"{family}:{section}" if f"{family}:{section}" in qa.PARAPHRASES else family
    table = qa.PARAPHRASES[key]
    questions = [table[v].format(role=role) if role else table[v] for v in variants]
    return {"stem": stem, "family": family, "section": section, "canonical_id": cid, "image_index": idx,
            "split": split, "question": questions[0], "questions": questions, "prompt_variants": list(variants),
            "answer": json.dumps(answer, ensure_ascii=False), "line": answer.get("line", 0)
            if isinstance(answer, dict) else answer[0]["line"]}


def frag(text: str, box: Sequence[float]) -> Dict[str, Any]:
    """A Kraken fragment.

    :param text: Text.
    :param box: 0-1000 box.
    :returns: Fragment.
    """
    return {"text": text, "conf": 0.9, "box": list(box)}


def stacked_frags(lines: Sequence[str], top: int = 100, step: int = 60, height: int = 45) -> List[Dict[str, Any]]:
    """One full-width fragment per line, stacked top to bottom.

    :param lines: Texts.
    :param top: First line's top.
    :param step: Line pitch.
    :param height: Fragment height.
    :returns: Fragments.
    """
    return [frag(t, [100, top + k * step, 900, top + k * step + height]) for k, t in enumerate(lines)]


# --------------------------------------------------------------------------- prompts


def every_wording() -> List[str]:
    """Every QA wording, templates formatted.

    :returns: Wordings.
    """
    return [w.format(role="sender") if "{role}" in w else w for table in qa.PARAPHRASES.values() for w in table]


def test_question_stem_cuts_every_qa_wording():
    """Every QA builder wording loses exactly its answer-format clause: no JSON, braces or abstention
    wording remain, the question text before the clause is kept, and the stem ends a sentence."""
    for w in every_wording():
        stem = V.question_stem(w)
        assert "{" not in stem and "JSON" not in stem and "not stated" not in stem, w
        assert stem.endswith(("?", "."))
        assert w.startswith(stem.rstrip(".?")), w


def test_question_request_replaces_the_answer_instruction():
    """The date question ends in the new instruction (the "not stated" clause is gone); a list fact gets
    the list instruction."""
    q = V.question_request(qa.DATE_PROMPT, False)
    assert q == "Quote the line that gives the date of this document. " + V.ANSWER_INSTRUCTION
    assert '{"answer": null}' in q and "not stated" not in q
    assert V.question_request(qa.WITNESS_LIST_PROMPT, True).endswith(V.LIST_ANSWER_INSTRUCTION)


def test_question_stem_rejects_a_wording_without_format_clause():
    """A wording the cut does not recognise raises instead of passing JSON wording through."""
    with pytest.raises(ValueError):
        V.question_stem("In which month was this document written?")


def test_field_mapping_covers_every_qa_family_and_section():
    """Every QA family maps to a field; every field has a definition cut from its canonical wording."""
    families = {fam for fam, _ in V.FIELD_OF}
    assert families == set(qa.FAMILIES)
    for role in qa.PERSON_ROLES:
        assert V.field_name({"family": "qa_person", "section": role.lower().replace(" ", "_")})
    for name in V.FIELD_ORDER:
        assert V.field_definition(name).endswith(("?", "."))
    assert V.field_definition("month") == V.question_stem(qa.MONTH_PROMPT)
    assert V.field_definition("judge").startswith("Who is the validating judge?")
    assert set(V.FIELD_OF.values()) == set(V.FIELD_ORDER)


def test_field_name_unknown_section_raises():
    """A QA family/section without a mapping fails loudly."""
    with pytest.raises(KeyError):
        V.field_name({"family": "qa_person", "section": "scribe"})


# --------------------------------------------------------------------------- fields


def test_page_fields_from_facts():
    """Facts fill their fields in the fixed order; the items are the edition text and line."""
    facts = [fact("m", "qa_date_month", "month", {"text": "כסליו", "line": 4}),
             fact("p", "qa_place", "written", {"text": "אלאסכנדריה", "line": 3})]
    fields = V.page_fields(LETTER, facts)
    assert [f.name for f in fields] == ["month", "place"]
    assert fields[0].items == [("כסליו", 3)] and fields[1].items == [("אלאסכנדריה", 2)]


def test_abstention_gives_null_date_month_and_year():
    """An abstention fact gives a null date, and null month and year when the page has neither."""
    fields = V.page_fields(UNDATED, [fact("a", "qa_abstain", "no_date", {"answer": "not stated"})])
    assert [(f.name, f.items) for f in fields] == [("date", None), ("month", None), ("year", None)]
    assert fields[1].source == "derived:a"


def test_abstention_year_null_needs_no_year_word():
    """A page with a year word (Judaeo-Arabic אלסנה) gets no derived null year."""
    lines = UNDATED[:-1] + ["פי הדה אלסנה"]
    names = [f.name for f in V.page_fields(lines, [fact("a", "qa_abstain", "no_date", {"answer": "not stated"})])]
    assert names == ["date", "month"]


def test_two_facts_for_one_field_raise():
    """Two facts may not fill the same field."""
    facts = [fact("p1", "qa_party", "party", {"text": LETTER[0], "line": 1}),
             fact("p2", "qa_party_formula", "party_formula", {"text": LETTER[1], "line": 2})]
    with pytest.raises(ValueError):
        V.page_fields(LETTER, facts)


def test_fields_request_and_target():
    """The request lists one definition per field and names the keys; the target has exactly those keys,
    text first, numbered by the parse shown; a list field is a list."""
    fields = [V.FieldValue("date", None, False, "a"),
              V.FieldValue("witnesses", [("יוסף בן יעקב", 1), ("אלצירפי", 1)], True, "w")]
    req = V.fields_request(fields)
    assert req.splitlines()[0] == V.FIELDS_HEAD
    assert req.splitlines()[1] == f"- date: {V.field_definition('date')}"
    assert '"date", "witnesses"' in req and 'The value of "witnesses" is a JSON list' in req
    target = json.loads(V.render_fields(fields, {1: 7}))
    assert list(target) == ["date", "witnesses"]
    assert target == {"date": None, "witnesses": [{"text": "יוסף בן יעקב", "line": 7},
                                                  {"text": "אלצירפי", "line": 7}]}
    assert list(target["witnesses"][0]) == ["text", "line"]


def test_render_question_shapes():
    """Null, single and list answers."""
    assert json.loads(V.render_question([], False, {})) == {"answer": None}
    assert json.loads(V.render_question([("כסליו", 3)], False, {3: 5})) == {"answer": "כסליו", "line": 5}
    assert json.loads(V.render_question([("א ב", 0)], True, {0: 1})) == {"answer": [{"text": "א ב", "line": 1}]}


# --------------------------------------------------------------------------- look-ups


def test_pick_lookups_unique_phrases_from_gap_free_lines():
    """Both look-ups quote 2-3 whole words found once on the page, from gap-free lines; the after-phrase
    answer is the rest of its line; the draw is stable."""
    lines = LETTER[:2] + ["[...] ואנא מנתטר לוצול [...]"] + LETTER[3:]
    first, second = V.pick_lookups(lines, "K")
    assert (first, second) == V.pick_lookups(lines, "K")
    for lk in (first, second):
        assert lk is not None and "[...]" not in lines[lk.line]
        assert 2 <= len(lk.phrase.split()) <= 3
        assert "\n".join(lines).count(lk.phrase) == 1
    assert first.text == lines[first.line]
    assert lines[second.line].endswith(" " + second.text) and second.phrase != first.phrase
    qa.check_whole_tokens(second.text, lines[second.line])
    assert second.line != first.line


def test_pick_lookups_none_without_candidates():
    """A page whose lines all carry gaps has no look-up."""
    assert V.pick_lookups(["[...] אבג [...]", "[...] דהו"], "K") == (None, None)


def test_rest_after():
    """The rest of a line after a whole-token phrase; a non-token phrase raises."""
    assert V.rest_after("א בג דה וז", "בג דה") == "וז"
    assert V.rest_after("א בג דה", "בג דה") == ""
    with pytest.raises(ValueError):
        V.rest_after("א בגד", "בג")


# --------------------------------------------------------------------------- parses


def test_parse_json_round_trip():
    """One element per line, numbered from 1, text verbatim (quotes escaped), one element per text line."""
    lines = ['שלום "חבר"', "שני"]
    text = V.parse_json(lines)
    assert json.loads(text) == [{"n": 1, "text": lines[0]}, {"n": 2, "text": "שני"}]
    assert text.count("\n") == len(lines) + 1
    boxed = json.loads(V.parse_boxes_json(lines, [(1, 2, 3, 4), (1, 5, 3, 9)]))
    assert boxed[1] == {"n": 2, "text": "שני", "bbox_2d": [1, 5, 3, 9]}


def test_context_prompt_uses_the_zero_shot_lead():
    """Lead, parse, blank line, request."""
    p = V.context_prompt(["א"], "Q?")
    assert p == 'Line-by-line reading of this page (JSON, in reading order):\n[\n{"n": 1, "text": "א"}\n]\n\nQ?'


def test_wants_model_parse_share_and_stability():
    """About 30% of pages, the same pages on every call."""
    keys = [f"P{i}__0" for i in range(4000)]
    picks = [V.wants_model_parse(k) for k in keys]
    assert 0.27 < sum(picks) / len(keys) < 0.33
    assert picks == [V.wants_model_parse(k) for k in keys]


def test_model_reading_normalises_lines():
    """The reading is the VLM lines, whitespace collapsed, empty lines dropped."""
    raw = {"vlm_lines": [{"text": "  א   ב "}, {"text": ""}, {"text": "ג"}], "frags": []}
    assert V.model_reading(raw) == ["א ב", "ג"]


def test_map_reading_mutual_unique():
    """An edition line maps to the only reading line at >= 0.8 that maps back; duplicates, a closer
    edition line and weak reads do not map."""
    edition = ["אבגדהוזחטי", "כלמנסעפצקר", "שתאבגדהוזח"]
    reading = ["אבגדהוזחטי", "כלמנסעפצקש", "כלמנסעפצקר", "תתתתתתתתתת"]
    m = V.map_reading(edition, reading)
    assert m.to_reading == {0: 0}
    assert m.reasons[1] == "answer_line_matches_several_reading_lines"
    assert m.reasons[2] == "answer_line_below_min_similarity"
    m2 = V.map_reading(["אבגדהוזחטי", "אבגדהוזחטכ"], ["אבגדהוזחטי"])
    assert m2.to_reading == {0: 0} and m2.reasons[1] == "reading_line_closer_to_another_line"
    assert V.map_reading(edition, []).reasons[0] == "no_model_reading"


# --------------------------------------------------------------------------- Kraken rows and boxes


def test_kraken_row_boxes_core_band_does_not_chain_lines():
    """A tall fragment that joins line 1 does not pull line 2 into its row (the editions builder's
    growing band does); bare brackets are ignored; a row reads right to left and its box is the union."""
    hebrew = [frag("אבג", [100, 100, 500, 140]), frag("דהו", [600, 100, 900, 175]), frag("זחט", [100, 150, 900, 190])]
    rows = V.kraken_row_boxes(hebrew + [frag("]", [0, 0, 1000, 1000])])
    assert [r.text for r in rows] == ["דהואבג", "זחט"]
    assert rows[0].box == (100, 100, 900, 175) and rows[1].box == (100, 150, 900, 190)
    assert eds.kraken_rows(hebrew) == ["דהוזחטאבג"]       # the growing band chains the two lines


def test_align_lines_monotonic():
    """Every line gets a distinct row in order at the threshold; noise rows are skipped; no assignment
    when a line has no row or the only matches cross."""
    sims = [[0.9, 0.1, 0.1], [0.1, 0.2, 0.8]]
    assert V.align_lines(sims, 0.5) == [0, 2]
    assert V.align_lines([[0.9, 0.1], [0.1, 0.3]], 0.5) is None
    assert V.align_lines([[0.1, 0.9], [0.9, 0.1]], 0.5) is None
    assert V.align_lines([[0.9], [0.9]], 0.5) is None
    assert V.align_lines([[0.6, 0.9], [0.0, 0.7]], 0.5) == [0, 1]


def test_edge_coverage():
    """A complete read covers its line even when noisy; a read that misses the first words does not."""
    line = "ולו קצר אלורקה כנת אסלם עלי כל ואחד מנכם באסמה"
    assert V.edge_coverage(line, "ולו קצר אלורקא כנת אסלם עלי כל ואחד מנכם באסמא") > 0.9
    assert V.edge_coverage(line, "אסלם עלי כל ואחד מנכם באסמה") < 0.75
    assert V.edge_coverage(line, "ששש") == 0.0


def test_line_boxes_qualifies_a_clean_page():
    """Every line read by its own row: one box per line, top to bottom, from Kraken geometry."""
    res = V.line_boxes(LETTER, stacked_frags(LETTER), 400)
    assert res.reason == "" and len(res.boxes) == len(LETTER)
    assert res.boxes[0] == (100, 100, 900, 145)
    assert all(a[1] < b[1] for a, b in zip(res.boxes, res.boxes[1:]))


@pytest.mark.parametrize("mutate, reason", [
    (lambda fr: fr[:2] + fr[3:], "line_without_row_at_min_similarity"),
    (lambda fr: fr[:1] + [frag(fr[1]["text"], [100, 120, 900, 205])] + fr[2:], "boxes_overlap"),
    (lambda fr: fr[:4] + [frag("אפצל אלסלאם", [100, 340, 400, 385])], "row_misses_part_of_a_line"),
    (lambda fr: [], "no_kraken_rows"),
])
def test_line_boxes_rejections(mutate, reason):
    """A missing row, overlapping boxes, a row that stops short of its line and an empty read reject
    the page."""
    assert V.line_boxes(LETTER, mutate(stacked_frags(LETTER)), 400).reason == reason


def test_line_boxes_rejects_a_narrow_box():
    """The documentary-grounding width check (60 px) applies."""
    assert V.line_boxes(LETTER, stacked_frags(LETTER), 60).reason == "box_too_narrow"


# --------------------------------------------------------------------------- parse choice and rows


def ctx_for(lines: Sequence[str], reading: Sequence[str], want_model: bool = True) -> V.PageContext:
    """A page context with a model reading.

    :param lines: Edition lines.
    :param reading: Model reading.
    :param want_model: Model-parse page.
    :returns: Context.
    """
    ctx = V.PageContext("K", list(lines), want_model, list(reading))
    ctx.reading_map = V.map_reading(lines, reading)
    ctx.reading_has_date = any(qa.has_date_indication(t) for t in reading)
    return ctx


READING = ["כתאבי אליך יא מולאי אטאל אללה בקאך", "ושלום", "וצלני כתאבך מע יוסף בן יעקב אלצירפו",
           "ואנא מנתטר לוצול אלמרכב מן אלאסכנדריה", "וכתב פי אלעשר אלאול מן שהר כסלו שנת אלפא וחמש מאה",
           "ואקרא עליך אפצל אלסלאם"]


def month_spec(lines: Sequence[str] = LETTER) -> V.ContextSpec:
    """The month question spec of the LETTER page.

    :param lines: Edition lines.
    :returns: Spec.
    """
    return V.question_specs([fact("m", "qa_date_month", "month", {"text": "כסליו", "line": 4})])[0]


def test_model_parse_row_cites_the_reading_number_and_quotes_the_edition():
    """On a model-parse page the prompt shows the reading, the line number is the reading's, and the
    answer text stays the edition's (the reading spells the month differently)."""
    ctx = ctx_for(LETTER, READING)
    d, fallback, skip = V.make_context_row(ctx, month_spec(), count_words, 4500, 4500)
    assert (fallback, skip, d.parse) == ("", "", "model")
    assert json.loads(d.answer) == {"answer": "כסליו", "line": 5}
    assert "כסלו שנת" in d.question and "כסליו" not in d.question
    V.check_row(d, LETTER, ctx.shown("model")[1])
    assert d.section == "qa_date_month|model"


def test_gold_parse_page_cites_the_edition_number():
    """A page not drawn for the model parse shows the edition lines."""
    d, fallback, _ = V.make_context_row(ctx_for(LETTER, READING, want_model=False), month_spec(), count_words,
                                        4500, 4500)
    assert d.parse == "gold" and fallback == "" and json.loads(d.answer) == {"answer": "כסליו", "line": 4}


def test_fallback_when_the_answer_line_is_misread():
    """An answer line the reading does not read at 0.8 falls back to the gold parse, with the reason."""
    reading = READING[:4] + ["ומכתב פי אלאכר מן שהר"] + READING[5:]
    d, fallback, _ = V.make_context_row(ctx_for(LETTER, reading), month_spec(), count_words, 4500, 4500)
    assert d.parse == "gold" and fallback == "answer_line_below_min_similarity"
    assert json.loads(d.answer)["line"] == 4


def test_abstention_never_shows_a_reading_with_a_date_word():
    """A row asserting no date falls back to gold when the reading carries a dating word."""
    spec = V.question_specs([fact("a", "qa_abstain", "no_date", {"answer": "not stated"})])[0]
    clean = V.make_context_row(ctx_for(UNDATED, UNDATED), spec, count_words, 4500, 4500)
    assert clean[0].parse == "model" and json.loads(clean[0].answer) == {"answer": None}
    dated = V.make_context_row(ctx_for(UNDATED, UNDATED + ["כתב פי שהר תמוז"]), spec, count_words, 4500, 4500)
    assert dated[0].parse == "gold" and dated[1] == "model_reading_has_a_date_word"


def test_token_cap_falls_back_then_skips():
    """A model-parse prompt over the cap falls back to gold; a gold prompt over the cap is skipped."""
    ctx = ctx_for(LETTER, READING + ["מלה " * 200])
    d, fallback, skip = V.make_context_row(ctx, month_spec(), count_words, 150, 4500)
    assert d.parse == "gold" and fallback == "model_parse_prompt_over_max_tokens" and skip == ""
    d, _, skip = V.make_context_row(ctx, month_spec(), count_words, 20, 4500)
    assert d is None and skip == "prompt_over_max_tokens"


def test_check_row_rejects_a_model_reading_as_target():
    """A target quoting the model reading instead of the edition fails the whole-token check."""
    ctx = ctx_for(LETTER, READING)
    d, _, _ = V.make_context_row(ctx, month_spec(), count_words, 4500, 4500)
    d.answer = json.dumps({"answer": "כסלו", "line": 5}, ensure_ascii=False)
    d.cites = [(5, 3, "כסלו")]
    with pytest.raises(qa.AnswerSpanError):
        V.check_row(d, LETTER, ctx.shown("model")[1])


def test_check_row_rejects_a_number_outside_the_parse_shown():
    """A cited number the prompt's parse does not have, or that stands for another line, fails."""
    ctx = ctx_for(LETTER, READING)
    d, _, _ = V.make_context_row(ctx, month_spec(), count_words, 4500, 4500)
    d.answer, d.cites = json.dumps({"answer": "כסליו", "line": 9}, ensure_ascii=False), [(9, 3, "כסליו")]
    with pytest.raises(AssertionError):
        V.check_row(d, LETTER, ctx.shown("model")[1])
    d.answer, d.cites = json.dumps({"answer": "כסליו", "line": 4}, ensure_ascii=False), [(4, 3, "כסליו")]
    with pytest.raises(AssertionError):
        V.check_row(d, LETTER, ctx.shown("model")[1])


def test_check_row_rejects_a_target_that_differs_from_its_cites():
    """The target JSON must say what the row's cites say."""
    ctx = ctx_for(LETTER, READING, want_model=False)
    d, _, _ = V.make_context_row(ctx, month_spec(), count_words, 4500, 4500)
    d.answer = json.dumps({"answer": "שנת", "line": 4}, ensure_ascii=False)
    with pytest.raises(AssertionError):
        V.check_row(d, LETTER, ctx.shown("gold")[1])


def test_check_row_parse_targets_are_the_edition():
    """A parse target that changes a line fails."""
    d = V.parse_draft(page_rec(LETTER), count_words)
    V.check_row(d, LETTER)
    d.answer = V.parse_json(LETTER[:-1] + ["ואקרא עליך"])
    with pytest.raises(AssertionError):
        V.check_row(d, LETTER)


def test_target_cites_shapes():
    """Each family's target is read back; a wrong shape raises."""
    assert V.target_cites("lookup_from_parse", '{"line": 2, "text": "א"}') == [("א", 2)]
    assert V.target_cites("fields_from_parse", '{"date": null, "month": {"text": "א", "line": 1}}') == [("א", 1)]
    with pytest.raises(ValueError):
        V.target_cites("question_from_parse", '{"answer": "א"}')
    with pytest.raises(ValueError):
        V.target_cites("fields_from_parse", '{"month": {"line": 1, "text": "א", "x": 1}}')


def test_check_splits():
    """A val page image in train, or a registered document in train, fails."""
    tr, va = page_rec(LETTER, "A", sha="1", image_path="a.jpg"), page_rec(VAL_LINES, "B", split="val", sha="2",
                                                                           image_path="b.jpg")
    rows = {"train_parse_lines": [{"stem": "a", "image": "a.jpg"}], "val": [{"stem": "b", "image": "b.jpg"}]}
    assert V.check_splits(rows, {"a": tr, "b": va}, (set(), set())) == {"val_images": 1, "train_images": 1}
    with pytest.raises(AssertionError):
        V.check_splits(rows, {"a": tr, "b": va}, ({"A"}, set()))
    clash = dict(tr, image_sha256="2")
    with pytest.raises(AssertionError):
        V.check_splits(rows, {"a": clash, "b": va}, (set(), set()))


def test_review_sample_mixes_parses_and_kinds():
    """Eight rows per family; context families take half from model parses and show every look-up kind
    in each half; the question family spreads over QA families."""
    manifest = []
    for i in range(40):
        for parse in ("gold", "model"):
            manifest.append({"stem": f"lk{i}{parse}", "family": "lookup_from_parse", "parse": parse,
                             "section": f"{('line_of_phrase', 'after_phrase')[i % 2]}|{parse}", "split": "train"})
            manifest.append({"stem": f"q{i}{parse}", "family": "question_from_parse", "parse": parse,
                             "qa_family": f"qa_f{i % 5}", "section": "x", "split": "train"})
        manifest.append({"stem": f"p{i}", "family": "parse_lines", "parse": "none", "section": "page",
                         "split": "val" if i == 7 else "train"})
    sample = V.review_sample(manifest)
    look = [m for m in sample if m["family"] == "lookup_from_parse"]
    assert len(look) == 8 and sum(m["parse"] == "model" for m in look) == 4
    for parse in ("gold", "model"):
        assert {V.review_kind(m) for m in look if m["parse"] == parse} == {"line_of_phrase", "after_phrase"}
    assert len({m["qa_family"] for m in sample if m["family"] == "question_from_parse"}) == 5
    parse_rows = [m for m in sample if m["family"] == "parse_lines"]
    assert len(parse_rows) == 8 and any(m["split"] == "val" for m in parse_rows)
    assert sample == V.review_sample(manifest)


# --------------------------------------------------------------------------- end to end


def write_fixture(tmp: Path) -> Tuple[Path, Path, Path]:
    """Editions + QA manifests, page images and reader records for three pages.

    Page A (train): a letter with a month and a place fact and a clean Kraken read (box family).
    Page B (train): an undated letter with an abstention fact. Page C (val): a month fact.

    :param tmp: Directory.
    :returns: ``(editions dir, qa dir, raw dir)``.
    """
    eds_dir, qa_dir, raw_dir, img_dir = tmp / "eds", tmp / "qa", tmp / "raw", tmp / "img"
    for d in (eds_dir, qa_dir, raw_dir, img_dir):
        d.mkdir()
    pages = [page_rec(LETTER, "Doc_A", 0, "11"), page_rec(UNDATED, "Doc_B", 0, "12"),
             page_rec(VAL_LINES, "Doc_C", 1, "13", split="val")]
    readings = {"Doc_A": READING, "Doc_B": UNDATED, "Doc_C": VAL_LINES}
    for k, p in enumerate(pages):
        path = img_dir / f"{p['canonical_id']}.jpg"
        PILImage.new("RGB", (400, 500), (40 * k, 90, 160)).save(path, "JPEG")
        p["image_path"], p["image_sha256"] = str(path), f"sha{k}"
        raw = {"image_url": p["image_url"], "sha256": p["image_sha256"],
               "vlm_lines": [{"text": t, "box": [0, 0, 1, 1]} for t in readings[p["canonical_id"]]],
               "frags": stacked_frags(p["lines"]) if p["canonical_id"] == "Doc_A" else []}
        eds.evidence_path(raw_dir, p["canonical_id"], p["image_index"]).write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        p.pop("key")
    (eds_dir / "manifest.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages),
                                            encoding="utf-8")
    facts = [fact("pgpqa_11_0_qa_date_month_0", "qa_date_month", "month", {"text": "כסליו", "line": 4}, "Doc_A"),
             fact("pgpqa_11_0_qa_place_1", "qa_place", "written", {"text": "אלאסכנדריה", "line": 3}, "Doc_A"),
             fact("pgpqa_12_0_qa_abstain_0", "qa_abstain", "no_date", {"answer": "not stated"}, "Doc_B"),
             fact("pgpqa_13_1_qa_date_month_0", "qa_date_month", "month", {"text": "תמוז", "line": 2}, "Doc_C", 1,
                  "val", (0,))]
    (qa_dir / "manifest.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts),
                                           encoding="utf-8")
    return eds_dir, qa_dir, raw_dir


def run_build(tmp: Path, share: float = 1.0, registered: Tuple[set, set] = (set(), set())) -> V.BuildResult:
    """Build the fixture.

    :param tmp: Directory with the fixture.
    :param share: Model-parse share.
    :param registered: Registered documents.
    :returns: Result.
    """
    return V.build(tmp / "eds", tmp / "qa", tmp / "raw", count=count_words, model_parse_share=share,
                   registered=registered, dataset_name="vqa_test")


def test_build_end_to_end(tmp_path):
    """Every family is built on the fixture, val rows come only from the val page, every target text is
    the edition's, and the outputs (DatasetDict, sidecars, images-once export) are written."""
    write_fixture(tmp_path)
    res = run_build(tmp_path)
    by_split = {k: len(v) for k, v in res.rows.items()}
    assert by_split == {"train_parse_lines": 2, "train_parse_lines_boxes": 1, "train_fields_from_parse": 2,
                        "train_question_from_parse": 6, "train_lookup_from_parse": 4, "val": 5}
    assert {r["task"] for r in res.rows["val"]} == {"parse_lines", "fields_from_parse", "question_from_parse",
                                                    "lookup_from_parse"}
    assert all(set(r) == set(FEATURES) for rs in res.rows.values() for r in rs)
    assert all(r["image"].endswith("Doc_C.jpg") for r in res.rows["val"])
    month = [m for m in res.manifest if m.get("qa_family") == "qa_date_month" and m["split"] != "val"]
    assert {m["parse"] for m in month} == {"model"} and month[0]["cites"][0]["shown_text"].endswith(
        "שהר כסלו שנת אלפא וחמש מאה")
    model_rows = [r for rs in res.rows.values() for r in rs if r["section"].endswith("|model")]
    assert any("כסלו שנת" in r["question"] for r in model_rows)          # the reading is shown in prompts ...
    for rs in res.rows.values():
        for r in rs:
            assert "כסלו שנת" not in r["answer"] and "אלצירפו" not in r["answer"]   # ... and never in a target
    fields_b = next(r for r in res.rows["train_fields_from_parse"] if "Doc_B" in r["image"])
    assert json.loads(fields_b["answer"]) == {"date": None, "month": None, "year": None}
    assert res.stats["pages"]["model_parse_pages_train"] == 2
    assert res.stats["boxes"]["page_outcomes"]["qualified"] == 1
    for fam in V.FAMILIES:
        assert f"## {fam}" in res.review
    out, io_dir = tmp_path / "out" / "vqa_test", tmp_path / "out" / "vqa_test_images_once"
    V.write_outputs(res, out, io_dir, dry_run=False)
    from datasets import load_from_disk

    dsd = load_from_disk(str(out))
    assert {k: dsd[k].num_rows for k in dsd} == by_split
    assert dsd["val"].features == FEATURES
    assert (out / "review_sample.md").exists() and (out / "manifest.jsonl").exists()
    assert json.loads((out / "stats.json").read_text())["rows_total"] == sum(by_split.values())
    manifest = json.loads((io_dir / "manifest.json").read_text())
    assert {k: v["rows"] for k, v in manifest["splits"].items()} == by_split
    assert manifest["splits"]["val"]["unique_images"] == 1


def test_build_gold_only_and_deterministic(tmp_path):
    """With no model-parse pages every context row shows the gold parse; two builds are identical."""
    write_fixture(tmp_path)
    a, b = run_build(tmp_path, share=0.0), run_build(tmp_path, share=0.0)
    assert a.rows == b.rows
    assert {m["parse"] for m in a.manifest if m["family"] in V.CONTEXT_FAMILIES} == {"gold"}


def test_build_drops_registered_train_pages(tmp_path):
    """A train page of a registered benchmark document yields no rows."""
    write_fixture(tmp_path)
    res = run_build(tmp_path, registered=({"Doc_A"}, set()))
    assert not any("Doc_A" in r["image"] for rs in res.rows.values() for r in rs)
    assert res.stats["skipped"]["page_registered_benchmark_document_in_train"] == 1


def test_build_rejects_a_fact_split_mismatch(tmp_path):
    """A val fact on a train page is a data error, not a skip."""
    write_fixture(tmp_path)
    lines = (tmp_path / "qa" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["split"] = "val"
    (tmp_path / "qa" / "manifest.jsonl").write_text("\n".join([json.dumps(bad, ensure_ascii=False)] + lines[1:]),
                                                    encoding="utf-8")
    with pytest.raises(AssertionError):
        run_build(tmp_path)


@pytest.mark.skipif(not V.DEFAULT_TOKENIZER.exists(), reason="v21b tokenizer not on this machine")
def test_real_tokenizer_counts_hebrew():
    """The v21b tokenizer counts a Hebrew parse."""
    count = V.tokenizer_counter(V.DEFAULT_TOKENIZER)
    assert 0 < count(V.parse_json(LETTER)) < 400
