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
from typing import Any, Dict, List, Optional

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
    assert {r.family for r in rows} == {"qa_person", "qa_date", "qa_party"} and not eligible
    for r in rows:
        obj = json.loads(r.answer)
        assert obj["text"] in lines[obj["line"] - 1]
