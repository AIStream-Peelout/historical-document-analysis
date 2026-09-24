# File name: test_build_pgp_qa.py
# Date: 9/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Unit tests for the extractive PGP QA builder (``build_pgp_qa``).

Every answer must be a verbatim span of one page line (``quote`` asserts it); metadata only
validates. Covers the month table and date-line detection, the strict person-location rule,
ketubba name parsing, the party line, abstention eligibility, the per-page and abstention caps
and the review sample.
"""
import collections
import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.finetuning.qwen_hebrew import build_pgp_qa as Q


def page(lines: List[str], pgpid: str = "1", key: str = "C__0",
         countable: Optional[List[bool]] = None) -> Dict[str, Any]:
    """A minimal editions-manifest record.

    :param lines: Page lines.
    :param pgpid: PGP id.
    :param key: Page key.
    :param countable: Per-line countable flags.
    :returns: Record.
    """
    return {"pgpid": pgpid, "canonical_id": "C", "image_index": 0, "key": key, "lines": list(lines),
            "regions": ["main"] * len(lines), "countable": countable or [True] * len(lines),
            "image_url": "u", "image_path": "p.jpg", "image_width": 10, "image_height": 10, "split": "train"}


def ctx(relations: Optional[Dict[str, List[Dict[str, str]]]] = None,
        pgp_docs: Optional[Dict[str, Dict[str, str]]] = None,
        edition_lines: Optional[Dict[str, List[str]]] = None) -> Q.QAContext:
    """A QA context without people.csv (names come from the relation rows).

    :param relations: ``pgpid -> relation rows``.
    :param pgp_docs: ``pgpid -> documents.csv row``.
    :param edition_lines: ``pgpid -> edition lines``.
    :returns: Context.
    """
    return Q.QAContext(pgp_docs or {}, relations or {}, {}, {}, edition_lines or {})


def rel(name: str, relation: str, pgpid: str = "1") -> Dict[str, str]:
    """A PGP relation row.

    :param name: Romanized person name.
    :param relation: Relation label.
    :param pgpid: Document.
    :returns: Row.
    """
    return {"person_name": name, "person_slug": name.lower().replace(" ", "-"), "relation": relation, "pgpid": pgpid}


LETTER = [
    "כתאבי אליך יא מולאי אטאל אללה בקאך",
    "וצלני כתאבך מע יוסף בן יעקב אלצירפי",
    "ואנא מנתטר לוצול אלמרכב מן אלאסכנדריה",
    "ואקרא עליך אפצל אלסלאם",
]

# --------------------------------------------------------------------------- answers


def test_quote_accepts_a_span_of_the_line():
    """The answer JSON carries the one-based line and the verbatim span."""
    assert json.loads(Q.quote(LETTER, 2, "יוסף בן יעקב")) == {"line": 2, "text": "יוסף בן יעקב"}


@pytest.mark.parametrize("n,text", [(2, "יוסף בר יעקב"), (1, "יוסף בן יעקב"), (9, "יוסף"), (2, "")])
def test_quote_rejects_non_spans(n, text):
    """Text not on that line (or an empty/out-of-range answer) violates the extractive contract."""
    with pytest.raises(AssertionError):
        Q.quote(LETTER, n, text)


# --------------------------------------------------------------------------- months / dates


def test_month_table_covers_the_twelve_months():
    """The table maps every month to the spec's Hebrew spelling."""
    expected = {"tishrei": "תשרי", "marheshvan": "חשון", "kislev": "כסלו", "tevet": "טבת", "shevat": "שבט",
                "adar": "אדר", "nisan": "ניסן", "iyyar": "אייר", "sivan": "סיון", "tammuz": "תמוז", "av": "אב",
                "elul": "אלול"}
    for key, spelling in expected.items():
        assert spelling in Q.MONTHS[key]
    assert "מרחשון" in Q.MONTHS["marheshvan"]


@pytest.mark.parametrize("raw,month", [
    ("19 Adar 1427", "adar"), ("First decade of Ḥeshvan 1442", "marheshvan"), ("Marḥeshvan 1429", "marheshvan"),
    ("Wednesday, 15 Kislev 1500", "kislev"), ("Ṭevet 1548", "tevet"), ("20 Shevaṭ 1564", "shevat"),
    ("3 Adar II 4845", "adar"), ("10 Nisan 4716", "nisan"), ("Iyyar 1438", "iyyar"), ("12 Sivan 4795", "sivan"),
    ("Tammuz 1288", "tammuz"), ("15 Av 1346", "av"), ("Elul 1428", "elul"), ("17 Tishri 1183", "tishrei"),
    ("Tishrei–Ṭevet 1495", None), ("[Tammuz] 4831", None), ("Ḥannuka 1548", None), ("1570", None), ("", None),
])
def test_pgp_month(raw, month):
    """Exactly one visible month; ranges, editor-restored months and no month give None."""
    assert Q.pgp_month(raw) == month


def test_month_tokens_allow_prefixes():
    """Month names match as whole tokens with ב/ל/ד/ו prefixes, not inside other words."""
    assert Q.month_tokens("בעשרים יום לירח באדר שנת") == {"אדר"}
    assert Q.month_tokens("ותמוז ואלול") == {"תמוז", "אלול"}
    assert Q.month_tokens("אדרבה") == set()


@pytest.mark.parametrize("line,is_date", [
    ("בשני בשבת בעשרים יום לירח אדר שנת אתתצט לשטרות", True),
    ("אלעשר אלאכיר מן חדש טבת שנת אתקלא לשטרות", True),
    ("וסאפר פי אדר אלי מצר", False),            # a month without a date word
    ("שנת אתתצט לשטרות", False),               # a year without a month
])
def test_is_date_line(line, is_date):
    """A date-formula line needs a month name AND a date word."""
    assert Q.is_date_line(line) is is_date


def test_date_row_uses_the_pgp_month_line():
    """The single date line carrying the PGP month is the answer, quoted whole."""
    lines = LETTER + ["בשני בשבת בעשרים יום לירח אדר שנת אתתצט לשטרות"]
    r = Q.date_row(page(lines), {"doc_date_original": "20 Adar 1499"}, collections.Counter())
    assert r and json.loads(r.answer) == {"line": 5, "text": lines[4]} and "not stated" in r.question


@pytest.mark.parametrize("extra,meta", [
    (["בשני בשבת בעשרים יום לירח אדר שנת אתתצט לשטרות"], {"doc_date_original": "20 Nisan 1499"}),
    (["בשני בשבת בעשרים יום לירח אדר שנת [...] לשטרות"], {"doc_date_original": "20 Adar 1499"}),
    (["בעשרים יום לירח אדר שנת אתתצט", "ויפרע לה בירח אדר שנת אתתק"], {"doc_date_original": "Adar 1499"}),
    (["בשני בשבת בעשרים יום לירח אדר שנת אתתצט לשטרות"], {"doc_date_original": "Tishrei–Ṭevet 1495"}),
])
def test_date_row_skips(extra, meta):
    """Month not on the line, a gap in the line, several candidate lines, or no single PGP month."""
    assert Q.date_row(page(LETTER + extra), meta, collections.Counter()) is None


@pytest.mark.parametrize("line,ok", [
    # rejected: the review sample's row 12 (Paris AIU VII D 48) ends in "שנת", the year is on the next line
    ("עד תשלום שנה תמימה והיה זה בשליש ראשון לחדש תשרי שנת", False),
    # rejected: row 10 (Paris AIU VII A 42), "כסלו הנז״ל" refers back to a year given earlier
    ("ששה ועשרים יום לחדש כסלו הנז׳׳ל פה מצרים ולראית", False),
    # rejected: a Judaeo-Arabic year word needs a following token too
    ("וכתב פי אלעשר מן כסליו סנה", False),
    # accepted: rows 5-9, 11, 13-19 of the review sample
    ("תאלת מחדש כסליו שנת אתרצא", True),
    ("הרשב׳׳א זלה׳׳ה - והיה זה בר׳׳ח אדר הא׳ ונתאחרה הכתיבה עד היום שולהי סיון שנת התקע׳׳ח ליצירה פה מצרים", True),
    ("ראשון לחודש אלול משנתינו הת׳ק׳פו ליצירה פה מצרים יע׳׳א והכל שריר ובריר", True),
    ("בארבעא בשבא דהוא שיתא יומין לחדש תמוז שנת תרין אלפין ומאתן ועסר שנין", True),
    ("שטר זה בחודש תשרי במועד חג הסוכות שנת ארבע מאות", True),
    ("היום ה׳ לחודש שבט משנתינו התקס׳׳ה", True),
    ("מעידים אנו חתומי מטה מה שהיה בפנינו באחד בשבת יום אחד ועשרים מחדש סיון בשנת אתקמד לשטרות", True),
    ("ביה וכאן דלך אלעשר אלאול מן שהר טבת שנת אלפא וארבע מאה ותלתין", True),
    ("אלעשר אלאכיר מן חדש טבת שנת אתקלא לשטרות במניה זפתי", True),
    ("סלאם סדר כד אלול קצז ליצירה", True),
    ("עשר מן סיון שנת אתפו בעיר מניה זפתא ליכון", True),
    ("דהוא יט יומין בירח סיון שנת אתיח", True),
    ("וכאן דלך פי אלעשר אלאכיר מן חדש אייר דשנת אלפא", True),
    # accepted: row 20 (NLI 577.3/3), Judaeo-Arabic "סנה" = year; and its other forms
    ("כסליו סנה ארבעת אלפים ושמונה", True),
    ("פי שהר ניסן סנת אתמג", True),
    ("מן שהר אלול פי אלסנה אלמאציה", True),
    ("כתב פי תמוז בסנה אלמדכורה", True),
    ("ואלעשר מן אב וסנה תמאן", True),
])
def test_has_year(line, ok):
    """The date answer line must carry the year: a שנת-word followed by a token, or an era word."""
    assert Q.has_year(line) is ok


def test_date_row_requires_the_year_on_the_line():
    """A date line with the PGP month but no year (row 12) is rejected; with the year it is kept."""
    stats = collections.Counter()
    no_year = LETTER + ["עד תשלום שנה תמימה והיה זה בשליש ראשון לחדש תשרי שנת"]
    assert Q.date_row(page(no_year), {"doc_date_original": "Tishrei 5534"}, stats) is None
    assert stats["date_skip_no_year_on_line"] == 1
    with_year = LETTER + ["והיה זה בשליש ראשון לחדש תשרי שנת התקל׳׳ד ליצירה"]
    r = Q.date_row(page(with_year), {"doc_date_original": "Tishrei 5534"}, stats)
    assert r and json.loads(r.answer) == {"line": 5, "text": with_year[4]}


def test_abstention_eligibility():
    """Only pages without any date indication, in an edition without one, with no PGP date."""
    c = ctx(edition_lines={"1": LETTER})
    assert Q.abstain_eligible(page(LETTER), {"description": "Letter from X to Y."}, c)
    assert not Q.abstain_eligible(page(LETTER), {"description": "Letter. Dated 1101 CE."}, c)
    assert not Q.abstain_eligible(page(LETTER), {"inferred_date_display": "ca. 1100"}, c)
    dated = ctx(edition_lines={"1": LETTER + ["[כתב פי ניסן]"]})
    assert not Q.abstain_eligible(page(LETTER), {"description": "Letter."}, dated)


# --------------------------------------------------------------------------- persons


def test_locate_exact_given_plus_father():
    """A tier-1 (given + father) exact match is located; the span is as written."""
    idx = Q.page_index(LETTER)
    hit = Q.locate_exact(["Yosef b. Yaʿaqov"], [], "Recipient", idx, LETTER)
    assert hit and hit.line == 2 and hit.span == "יוסף בן יעקב" and hit.tier == 1


@pytest.mark.parametrize("name,lines", [
    ("Abū l-Faraj", ["וסלם עלי אבו אלפרג ואולאדה ואהלה"]),        # kunya only: never
    ("Yosef b. Yaʿaqov", ["וצלני כתאבך מע יוסף בן יעקף אלצירפי"]),   # near miss: never fuzzy
    ("Yosef", ["וצלני כתאבך מע יוסף אלצירפי ואלסלאם"]),              # given name alone: no strict query
])
def test_locate_exact_rejects_weak_matches(name, lines):
    """Kunya-only, family-only, given-only and fuzzy hits never locate a person."""
    assert Q.locate_exact([name], [], "Mentioned", Q.page_index(lines), lines) is None


def test_person_rows_role_uniqueness_and_answer():
    """A unique Sender gives a row; a role PGP lists twice is skipped."""
    c = ctx(relations={"1": [rel("Yosef b. Yaʿaqov", "Sender"), rel("Shelomo b. Nissim", "Witness"),
                             rel("Avraham b. Yiṣḥaq", "Witness")]})
    lines = LETTER + ["שלמה בן נסים שהד", "אברהם בן יצחק שהד"]
    rows, located = Q.person_rows(page(lines), c, Q.page_index(lines), collections.Counter())
    assert [r.section for r in rows] == ["sender"]
    assert json.loads(rows[0].answer) == {"line": 2, "text": "יוסף בן יעקב"}
    assert "sender" in rows[0].question and len(located) == 3


def test_person_rows_skip_uncertain_and_gapped_lines():
    """Uncertain relations and answer lines with ``[...]`` give no row."""
    lines = ["וצלני כתאבך מע יוסף בן יעקב [...] אלצירפי", *LETTER[2:]]
    c = ctx(relations={"1": [rel("Yosef b. Yaʿaqov", "Sender")]})
    rows, _ = Q.person_rows(page(lines), c, Q.page_index(lines), collections.Counter())
    assert rows == []
    c2 = ctx(relations={"1": [rel("Yosef b. Yaʿaqov", "Sender (uncertain)")]})
    rows, _ = Q.person_rows(page(LETTER), c2, Q.page_index(LETTER), collections.Counter())
    assert rows == []


@pytest.mark.parametrize("relation", ["Scribe", "Mentioned", "Mentioned (deceased)"])
def test_scribe_and_mentioned_are_never_asked(relation):
    """Scribes and mentioned people are not question roles, even when located exactly and unique."""
    c = ctx(relations={"1": [rel("Yosef b. Yaʿaqov", relation)]})
    rows, _ = Q.person_rows(page(LETTER), c, Q.page_index(LETTER), collections.Counter())
    assert rows == []
    assert "Mentioned" not in Q.PERSON_ROLES and "Mentioned" not in Q.ROLE_PRIORITY


# --------------------------------------------------------------------------- ketubba / party


@pytest.mark.parametrize("desc,couple", [
    ("Marriage contract (ketubba). Groom: Peraḥya b. Tiqva ha-Levi. Bride: Sitt al-Thanāʾ bt. Avraham. Signed.",
     {"groom": "Peraḥya b. Tiqva ha-Levi", "bride": "Sitt al-Thanāʾ bt. Avraham"}),
    ("Betrothal document. Fiance: Yosef ha-Levi b. Berakhot. Fiancee: Sitt al-Kull bt. Yefet. Wedding later.",
     {"groom": "Yosef ha-Levi b. Berakhot", "bride": "Sitt al-Kull bt. Yefet"}),
    ("Ketubba of Yefet b. Nissim and Sitt al-Dār bt. Yiṣḥaq.", {"groom": "Yefet b. Nissim", "bride": "Sitt al-Dār bt. Yiṣḥaq"}),
    ("Ketubba fragment. The groom is Yefet b. Avraham.", {"groom": "Yefet b. Avraham"}),
    ("Ketubba. Bride: Mubāraka bt. [...].", {}),
    ("Betrothal register between Biqa (?) b. Moshe and Mimuna bt. Ḥasan.", {}),
])
def test_parse_couple(desc, couple):
    """Name grammar keeps b./bt. chains, stops at punctuation, rejects uncertain names."""
    assert Q.parse_couple(desc) == couple


def test_ketubah_row_answers_the_formula_line():
    """The formula line on which the groom is located is quoted whole."""
    lines = ["בשני בשבת בעשרים יום לירח אדר", "איך יפת בר נסים אמר לה לסת אלדאר", "בת יצחק הוי לי לאנתו כדת משה"]
    meta = {"description": "Ketubba of Yefet b. Nissim and Sitt al-Dār bt. Yiṣḥaq."}
    r = Q.ketubah_row(page(lines), meta, Q.page_index(lines), collections.Counter())
    assert r and json.loads(r.answer) == {"line": 2, "text": lines[1]}
    assert Q.ketubah_row(page(lines), {"description": "Letter."}, Q.page_index(lines), collections.Counter()) is None


def test_party_row_requires_a_located_party_on_the_line():
    """An acknowledgment line counts only with a Party/Witness located on it."""
    lines = ["בפנינו אנו החתומים", "אנא סעדיה בר ישועה מודה אני בפניכם", "וקנינא מנה קנין גמור"]
    c = ctx(relations={"1": [rel("Saʿadya b. Yeshuʿa", "Party")]})
    p = page(lines)
    _, located = Q.person_rows(p, c, Q.page_index(lines), collections.Counter())
    r = Q.party_row(p, {"type": "Legal document"}, located, collections.Counter())
    assert r and json.loads(r.answer) == {"line": 2, "text": lines[1]}
    assert Q.party_row(p, {"type": "Letter"}, located, collections.Counter()) is None
    assert Q.party_row(p, {"type": "Legal document"}, [], collections.Counter()) is None


# --------------------------------------------------------------------------- caps + sample


def _qa(family: str, line: int = 1, priority: int = 5) -> Q.QARow:
    """A synthetic QA row.

    :param family: Row family.
    :param line: Answer line.
    :param priority: Cap priority.
    :returns: Row.
    """
    return Q.QARow(family, "s", "q", Q.quote(["אבגד"], 1, "אבגד") if family != "qa_abstain" else Q.ABSTAIN_ANSWER,
                   line, priority=priority)


def test_caps_rows_per_page_and_abstention_shares():
    """<= 3 rows per page; abstain rows <= 10% of all QA rows and on <= 20% of pages."""
    pages = [page(LETTER, key=f"k{i}") for i in range(100)]
    cands = {f"k{i}": [_qa("qa_person"), _qa("qa_date"), _qa("qa_party"), _qa("qa_person")] for i in range(40)}
    eligible = {f"k{i}": True for i in range(40, 100)}
    kept = Q.apply_caps(pages, cands, eligible)
    assert all(len(v) <= Q.MAX_ROWS_PER_PAGE for v in kept.values())
    rows = [r for v in kept.values() for r in v]
    n_abstain = sum(r.family == "qa_abstain" for r in rows)
    assert 0 < n_abstain <= Q.ABSTAIN_MAX_ROW_SHARE * len(rows)
    assert sum(any(r.family == "qa_abstain" for r in v) for v in kept.values()) <= Q.ABSTAIN_MAX_PAGE_SHARE * len(pages)


def test_abstention_page_share_binds():
    """With few other rows the page share (20%) still bounds abstention."""
    pages = [page(LETTER, key=f"k{i}") for i in range(50)]
    cands = {f"k{i}": [_qa("qa_date")] * 3 for i in range(40)}
    kept = Q.apply_caps(pages, cands, {f"k{i}": True for i in range(40, 50)})
    assert sum(r.family == "qa_abstain" for v in kept.values() for r in v) <= min(10, int(0.2 * 50))


def test_review_sample_is_stratified():
    """The sample covers every family and never exceeds its size."""
    rows = [{"family": f, "stem": f"{f}_{i}"} for f, n in (("qa_person", 300), ("qa_date", 150), ("qa_abstain", 5))
            for i in range(n)]
    s = Q.review_sample(rows, 200)
    fam = collections.Counter(r["family"] for r in s)
    assert len(s) == 200 and fam["qa_abstain"] == 5 and fam["qa_person"] >= 90 and fam["qa_date"] >= 90
    assert len(Q.review_sample(rows[:10], 200)) == 10


def test_page_candidates_answers_are_spans():
    """Every candidate on a synthetic legal page quotes a span of its own line."""
    lines = ["בפנינו אנו החתומים בעשרים יום לירח אדר שנת אתתצט לשטרות",
             "אנא סעדיה בר ישועה מודה אני בפניכם", "וקנינא מנה קנין גמור ושריר וקים"]
    c = ctx(relations={"1": [rel("Saʿadya b. Yeshuʿa", "Party")]},
            pgp_docs={"1": {"type": "Legal document", "doc_date_original": "20 Adar 1499", "description": "Deed."}})
    rows, eligible = Q.page_candidates(page(lines), c, collections.Counter())
    assert {r.family for r in rows} == {"qa_person", "qa_date", "qa_party", "qa_date_month", "qa_date_year"}
    assert not eligible
    for r in rows:
        for it in Q.answer_items(r.answer):
            assert it["text"] in lines[it["line"] - 1]


# --------------------------------------------------------------------------- set-valued roles

DEED = [
    "בפנינו אנו החתומים מטה הודה סעדיה בר ישועה",      # a name in the body, above the signature window
    "הודאה גמורה בלא אונס ובלא הכרח",
    "ומחל ופטר כל תביעה ודין ודברים",
    "מן יומא דנן ולעלם",
    "וקנינא מנה קנין שלם במנא דכשר",
    "למקניא ביה על כל מה דכתיב",
    "ומפורש לעילא",
    "ושריר וקים",
    "יוסף בר יעקב עד",
    "שלמה בר נסים עד",
]


def _witness_ctx(names: List[str]) -> Q.QAContext:
    """A context whose document lists ``names`` as witnesses.

    :param names: Romanized witness names.
    :returns: Context.
    """
    return ctx(relations={"1": [rel(n, "Witness") for n in names]})


def _set_rows(lines: List[str], c: Q.QAContext, role: str = "Witness") -> Tuple[List[Q.QARow], Dict[str, bool]]:
    """Run the set-valued rule on a synthetic page.

    :param lines: Page lines.
    :param c: Context.
    :param role: Relation role.
    :returns: ``(rows, flags)``.
    """
    p = page(lines)
    _, located = Q.person_rows(p, c, Q.page_index(lines), collections.Counter())
    return Q.set_valued_rows(p, c, located, role, collections.Counter())


def test_witness_list_when_every_signature_is_a_located_witness():
    """Completeness check passes: both signature lines hold a located witness -> one list row."""
    rows, flags = _set_rows(DEED, _witness_ctx(["Yosef b. Yaʿaqov", "Shelomo b. Nissim"]))
    assert flags == {"has_role": True, "all_located": True, "complete": True}
    assert [r.family for r in rows] == ["qa_witnesses_list"] and rows[0].section == "witness_all"
    assert json.loads(rows[0].answer) == [{"line": 9, "text": "יוסף בר יעקב"}, {"line": 10, "text": "שלמה בר נסים"}]


def test_witness_list_refused_when_a_signature_is_unaccounted():
    """Completeness check fails: a third signature PGP does not list -> no list, fallback line only."""
    short = DEED[:1] + DEED[-2:]      # the party's body line falls inside the window: also incomplete
    assert not _set_rows(short, _witness_ctx(["Yosef b. Yaʿaqov", "Shelomo b. Nissim"]))[1]["complete"]
    lines = DEED + ["אברהם בר יצחק עד"]
    rows, flags = _set_rows(lines, _witness_ctx(["Yosef b. Yaʿaqov", "Shelomo b. Nissim"]))
    assert flags["all_located"] and not flags["complete"]
    assert [r.family for r in rows] == ["qa_witness_line"] and rows[0].section == "witness_any"
    assert json.loads(rows[0].answer) == {"line": 9, "text": "יוסף בר יעקב עד"}


def test_witness_list_refused_when_a_holder_is_not_located():
    """A PGP witness who cannot be located exactly blocks the list; located ones give the fallback."""
    rows, flags = _set_rows(DEED, _witness_ctx(["Yosef b. Yaʿaqov", "Shelomo b. Nissim", "Moshe b. Levi"]))
    assert not flags["all_located"] and [r.family for r in rows] == ["qa_witness_line"]


def test_single_holder_roles_stay_with_qa_person():
    """A role PGP gives one person is not set-valued."""
    rows, flags = _set_rows(DEED, _witness_ctx(["Yosef b. Yaʿaqov"]))
    assert rows == [] and not flags["has_role"]


def test_party_list_and_completeness():
    """All parties located and every name on the page accounted for -> list; else fallback."""
    lines = ["אנא סעדיה בר ישועה ואנא יוסף בר יעקב מודים", "וקנינא מנהון קנין שלם", "ושריר וקים"]
    c = ctx(relations={"1": [rel("Saʿadya b. Yeshuʿa", "Party"), rel("Yosef b. Yaʿaqov", "Party")]})
    rows, flags = _set_rows(lines, c, "Party")
    assert flags["complete"] and [r.family for r in rows] == ["qa_parties_list"]
    assert [it["text"] for it in json.loads(rows[0].answer)] == ["סעדיה בר ישועה", "יוסף בר יעקב"]
    # a third person on the SAME line as a listed party (the declarant) makes the list incomplete
    lines2 = ["יקול שלמה בר נסים אני ואקפת סעדיה בר ישועה ויוסף בר יעקב", *lines[1:]]
    rows, flags = _set_rows(lines2, c, "Party")
    assert not flags["complete"] and [r.family for r in rows] == ["qa_party_line"]
    assert rows[0].section == "party_any"


def test_witness_completeness_is_per_name():
    """A line holding a listed witness AND an unlisted signature fails (a line-level check would pass)."""
    witnesses = _witness_ctx(["Yosef b. Yaʿaqov", "Shelomo b. Nissim"])
    lines = DEED[:-2] + ["יוסף בר יעקב עד אברהם בר יצחק עד", "שלמה בר נסים עד"]
    rows, flags = _set_rows(lines, witnesses)
    assert flags["all_located"] and not flags["complete"] and [r.family for r in rows] == ["qa_witness_line"]
    ok_rows, ok_flags = _set_rows(DEED[:-2] + ["יוסף בר יעקב עד שלמה בר נסים עד"], witnesses)
    assert ok_flags["complete"] and len(json.loads(ok_rows[0].answer)) == 2


def test_list_row_counts_as_one_under_the_cap():
    """A set-valued row takes one of the three page slots."""
    lst = Q.QARow("qa_witnesses_list", "witness_all", "q", Q.quote_list(["אב גד", "הו זח"], [(1, "אב"), (2, "הו")]),
                  1, priority=Q.PRIORITY["qa_witnesses_list"])
    others = [_qa("qa_date", priority=2), _qa("qa_person", priority=5), _qa("qa_person", priority=6)]
    kept = Q.apply_caps([page(LETTER, key="k")], {"k": sorted([lst] + others, key=lambda r: r.priority)}, {})
    assert len(kept["k"]) == 3 and kept["k"][1].family == "qa_witnesses_list"


def test_quote_list_rejects_non_spans():
    """Every list item must be a span of its own line."""
    with pytest.raises(AssertionError):
        Q.quote_list(["אב גד", "הו זח"], [(1, "אב"), (2, "אב")])


# --------------------------------------------------------------------------- month / year spans


def test_month_and_year_with_a_gap_elsewhere_on_the_line():
    """A gap elsewhere blocks the whole-line date row but not the month and year spans."""
    lines = LETTER + ["[...] בעשרים יום לירח אדר שנת אתתצט לשטרות"]
    meta = {"doc_date_original": "20 Adar 1499"}
    stats = collections.Counter()
    assert Q.date_row(page(lines), meta, stats) is None and stats["date_skip_gap_in_line"] == 1
    m = Q.month_row(page(lines), meta, stats)
    y = Q.year_row(page(lines), meta, stats)
    assert json.loads(m.answer) == {"line": 5, "text": "אדר"}
    assert json.loads(y.answer) == {"line": 5, "text": "אתתצט לשטרות"}


def test_year_on_the_next_line():
    """"... לחדש תשרי שנת" ends the line; the year and era word open the next one."""
    lines = LETTER + ["עד תשלום שנה תמימה והיה זה בשליש ראשון לחדש תשרי שנת", "התקל׳׳ד ליצירה פה רשיד יע׳׳א"]
    y = Q.year_row(page(lines), {"doc_date_original": "Tishrei 5534"}, collections.Counter())
    assert json.loads(y.answer) == {"line": 6, "text": "התקל׳׳ד ליצירה"}


@pytest.mark.parametrize("lines,expected", [
    (["היום ה׳ לחודש שבט משנתינו התקס׳׳ה"], (0, "התקס׳׳ה")),                            # marked numeral, no era
    (["כסליו סנה ארבעת אלפים ושמונה שנין"], (0, "ארבעת אלפים ושמונה")),                 # number words + terminator
    (["כסליו סנה ארבעת אלפים ושמונה"], None),                                          # number words, nothing closes
    (["סיון דשנת אלפא וארבע מאה ותרתי"], None),                                        # unknown number word: cut
    (["בירח כסליו דשנת אלפא וארבע מאהוחמש"], None),                                    # glued typo: cut
    (["ומן שהר כסלו שנת אלפא וחמש מאה", "לשטרות בפסטאט"], (0, "אלפא וחמש מאה")),        # era word on next line
    (["לחדש כסלו שנת חמשת אלפים ושע מאות ושבע עשרה ליצירה"], (0, "חמשת אלפים ושע מאות ושבע עשרה ליצירה")),
    (["תאלת מחדש כסליו שנת אתרצא"], None),                                             # unmarked, no era -> skip
    (["לחדש כסלו הנז׳׳ל פה מצרים ולראית"], None),                                      # no year at all
    (["כסליו סנה ארבעת אלפים ושמונה", "מאות וששים וארבעה שנים למנינא"], None),        # year runs on -> skip
    (["לחדש אייר שנת חמשת אל [...] ושלש מאות"], None),                                 # gap cuts the year
    (["חדש אדר שנת [...] לשטרות"], None),                                              # span with a gap
    (["לחדש אייר שנת במעשה ידיך ארנן פה"], None),                                      # no numeral
    (["לחדש אייר שנת הנז׳ ליצירה"], None),                                             # "the aforementioned"
])
def test_year_span_extraction(lines, expected):
    """Era span, 1-3 numerals, and every skip case of the year rule."""
    assert Q.year_span(lines, 0) == expected


def test_year_row_requires_a_pgp_date():
    """No PGP date at all: no year row (the value itself is never compared)."""
    lines = ["בעשרים יום לירח אדר שנת אתתצט לשטרות"]
    assert Q.year_row(page(lines), {"doc_date_original": ""}, collections.Counter()) is None
    assert Q.year_row(page(lines), {"doc_date_original": "1499"}, collections.Counter())


def test_month_row_quotes_the_standalone_token():
    """Prefixed month tokens are skipped; an Adar qualifier is part of the month."""
    stats = collections.Counter()
    assert Q.month_row(page(["בעשרים יום באדר שנת אתתצט"]), {"doc_date_original": "Adar 1499"}, stats) is None
    r = Q.month_row(page(["בעשרים יום לחדש אדר שני שנת אתתצט"]), {"doc_date_original": "Adar II 1499"}, stats)
    assert json.loads(r.answer) == {"line": 1, "text": "אדר שני"}


@pytest.mark.parametrize("tok,ok", [("התקע׳׳ח", True), ("ד׳תתי״א", True), ("אלפא", True), ("ושלש", True),
                                    ("הנז׳", False), ("יצ׳׳ו", False), ("ה׳", False), ("אתתצט", False)])
def test_is_numeral(tok, ok):
    """Marked letter numerals and number words count; abbreviations and unmarked letters do not."""
    assert Q.is_numeral(tok) is ok


# --------------------------------------------------------------------------- ketubba names


def test_ketubah_names_on_any_line():
    """Groom and bride located exactly on non-formula lines give name rows."""
    lines = ["בשני בשבת בעשרים יום לירח אדר", "הוי לי לאנתו כדת משה וישראל", "והודה יפת בר נסים חתנא דנן",
             "ואקנית סת אלדאר בת יצחק כלתא"]
    meta = {"description": "Ketubba of Yefet b. Nissim and Sitt al-Dār bt. Yiṣḥaq."}
    rows = Q.ketubah_name_rows(page(lines), meta, Q.page_index(lines), collections.Counter())
    got = {r.family: json.loads(r.answer) for r in rows}
    assert got == {"qa_ketubah_groom": {"line": 3, "text": "יפת בר נסים"},
                   "qa_ketubah_bride": {"line": 4, "text": "סת אלדאר בת יצחק"}}
    assert "Who is the groom?" in rows[0].question


def test_alhatan_line_is_a_formula_line():
    """A line opening with אלחתן / אלכלה counts as a formula line for qa_ketubah_parties."""
    lines = ["בשני בשבת בעשרים יום לירח אדר", "אלחתן יפת בר נסים אלמערוף באבן אלעטאר", "ואלכלה סת אלדאר"]
    meta = {"description": "Ketubba of Yefet b. Nissim and Sitt al-Dār bt. Yiṣḥaq."}
    r = Q.ketubah_row(page(lines), meta, Q.page_index(lines), collections.Counter())
    assert r and json.loads(r.answer) == {"line": 2, "text": lines[1]}


# --------------------------------------------------------------------------- party formula (wording only)


@pytest.mark.parametrize("line,ok", [
    ("מותבה אנא אברהם הכהן בר אהרן הכהן", True),                          # deed opening, title in apposition
    ("אנא יפת בר יוסף צביתי ברעות נפשי כד", True),
    ("כאן עלי אנא טוביה הלוי בר סהל כמסה דנאניר לכניסת", True),
    ("ומודה אני שמואל בר יעקב שקבלתי ממנו", False),                        # "ומודה" is not the verb token
    ("מודה אני שמואל בר יעקב שקבלתי ממנו כל", True),
    ("חצר אלינא אנן חתומי מטה אבו אלעלא בן בו סהל אלגבילי", False),       # court/witness "we": third person
    ("שהדותא דהות באנפנא אנן שהדי דחתמות ידנא לתחתא כן הוה חצר מ הבה בר מ משה", False),
    ("חצרת אנא נתן ביר שמואל החבר זל וקד אסתופא", False),                  # the clerk attending
    ("אנא ועמאר בר פראח אלאטראבלסי", False),                              # "I and ʿAmmār ..."
    ("עיקר ואחריות חוב דנן עלאי אנא יוסף בר יצחק", False),               # pronoun after the 4th token
    ("אנא יפת בר יוסף [...] ברעות נפשי", False),                          # gap
    ("אנא דן בר גד", False),                                               # < 12 letters
])
def test_party_formula_line(line, ok):
    """Singular אנא (or מודה/מודים) with the name in apposition; court and witness formulas excluded."""
    assert Q.party_formula_line(line) is ok


def test_party_formula_row_rules():
    """Legal documents only, exactly one formula line per page, answered with the whole line."""
    lines = ["בפסטאט מצרים דעל נילוס נהרא", "מותבה אנא אברהם הכהן בר אהרן הכהן", "צביתי ברעות נפשי ובלא אונס"]
    r = Q.party_formula_row(page(lines), {"type": "Legal document"}, collections.Counter())
    assert r and r.family == "qa_party_formula" and json.loads(r.answer) == {"line": 2, "text": lines[1]}
    assert r.question == Q.PARTY_PROMPT
    assert Q.party_formula_row(page(lines), {"type": "Letter"}, collections.Counter()) is None
    two = lines + ["אנא יפת בר יוסף צביתי ברעות נפשי כד"]
    assert Q.party_formula_row(page(two), {"type": "Legal document"}, collections.Counter()) is None


def test_party_formula_skipped_when_qa_party_exists():
    """A page with a located-party qa_party row gets no formula row (same question)."""
    lines = ["בפנינו אנו החתומים", "אנא סעדיה בר ישועה מודה אני בפניכם", "וקנינא מנה קנין שלם ושריר וקים"]
    c = ctx(relations={"1": [rel("Saʿadya b. Yeshuʿa", "Party")]}, pgp_docs={"1": {"type": "Legal document"}})
    rows, _ = Q.page_candidates(page(lines), c, collections.Counter())
    fams = [r.family for r in rows]
    assert "qa_party" in fams and "qa_party_formula" not in fams


# --------------------------------------------------------------------------- places

PLACES = {"Fustat": {"name_variants": "Fusṭāṭ, פסטאט, فسطاط"}, "Alexandria": {"name_variants": "אלאסכנדריה, נא אמון"},
          "Aden": {"name_variants": "ʿAdan, עדן, عدن"}, "Cairo": {"name_variants": "al-Qāhira"},
          "Tyre": {"name_variants": "צור, صور"}}


def _places(lines: List[str], meta: Dict[str, str]) -> List[Q.QARow]:
    """Place rows of a synthetic page.

    :param lines: Page lines.
    :param meta: PGP document row (origin / destination / location).
    :returns: Rows.
    """
    return Q.place_rows(page(lines), meta, PLACES, collections.Counter())


def test_place_written_and_sent():
    """Location -> "written", destination -> "sent to"; the answer is the name without its prefix."""
    lines = ["וכאן דלך בפסטאט מצרים דעל נילוס נהרא", "ואנפדתה אלי אלאסכנדריה מע אלרסול", "ושריר וקים"]
    rows = _places(lines, {"location": "Fustat", "destination": "Alexandria"})
    got = {r.section: (json.loads(r.answer), r.question.split("?")[0]) for r in rows}
    assert got["written"] == ({"line": 1, "text": "פסטאט"}, "Where was this document written")
    assert got["sent"] == ({"line": 2, "text": "אלאסכנדריה"}, "To where was this document sent")


@pytest.mark.parametrize("lines,meta", [
    (["כתבת מן אלאסכנדריה אלי מצרים", "ושלום"], {"origin": "Alexandria", "destination": "Fustat"}),  # one line
    (["פי פסטאט אולא", "ואנא פי פסטאט אלאן"], {"origin": "Fustat"}),                               # two lines
    (["וכאן דלך בפסטאט [...] נילוס", "ושלום"], {"origin": "Fustat"}),                              # gap in line
    (["ואבוה נוחו עדן ושלום"], {"origin": "Aden"}),                                                # "rest in Eden"
    (["ישתבח צור ישראל ואלסלאם"], {"origin": "Tyre"}),                                             # ordinary word
    (["כתבת מן אלאסכנדריה"], {"origin": "Alexandria, Fustat"}),                                    # two PGP places
])
def test_place_skips(lines, meta):
    """Ambiguous or unsafe place matches give no row."""
    assert _places(lines, meta) == []


def test_place_article_written_apart():
    """"אל קאהרה" (article separated) is quoted whole."""
    rows = _places(["הכא בעיר אל קאהרה הסמוכה"], {"location": "Cairo"})
    assert json.loads(rows[0].answer) == {"line": 1, "text": "אל קאהרה"}


def test_status_text():
    """QA is part of the v22 mixture; the review sample stays."""
    assert "included in the v22 mixture" in Q.review_markdown([], {"status": "included in the v22 mixture; review "
                                                                    "sample available at qa_review_sample.md",
                                                                    "generated_at": "t", "editions_manifest": "m",
                                                                    "rows_by_family": {}})

