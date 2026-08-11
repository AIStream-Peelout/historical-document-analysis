"""Regression tests for relationship_extractor.py's deterministic compaction rules.

Covers the citation-evidence detector (English + Hebrew) and the
shelfmark-mistyped-as-Institution guard added during the 2026-07-17 KG audit.
"""

import pytest

from src.models.llm.academic.relationship_extractor import (
    _compact_relations,
    _is_citation_evidence,
)


# ── citation-evidence detector ────────────────────────────────────────────────

@pytest.mark.parametrize("evidence", [
    # English
    "ed. Franklin, Margariti, Rustow",
    "Cohen, M. (1980). Some Title. Princeton: Princeton University Press.",
    # Hebrew: explicit "editors" credit
    "בתוך יוסף הקר, בנימין זאב קדר ויוסף קפלן (עורכים), ראשונים ואחרונים",
    "גלית חזן־רוקם ויום טוב עסיס (עורכים), חברה ותרבות, יהודי ספרד לאחר הגירוש",
    # Hebrew: page/siman reference abbreviation
    "אלכסנדר שייבר ומאיר בניהו, \"פניית חכמי מצרים\", ספונות ו (תשכ\"ב), עמ קכח",
    "מסעוד בן שמעון ומשה רובינשטיין, ירושלים תשמ\"א, סי' קט",
    # Hebrew: parenthetical "(City: Publisher..." shape, including the
    # truncated-mid-citation case seen live (no closing paren, no trailing
    # gershayim on the year)
    "כהן ואלישבע סימון־פיקאלי, יהודים בבית המשפט המוסלמי (ירושלים: יד יצחק בן־צבי, תשנ",
])
def test_citation_evidence_detected(evidence):
    assert _is_citation_evidence(evidence) is True


@pytest.mark.parametrize("evidence", [
    "",
    "The two merchants traveled together from Fustat to Alexandria.",
    # Hebrew narrative content, not a citation
    "הסוחר \"אברהם ארפול\" היושב בקהיר במחצית השנייה של המאה השש־עשרה לשותפו בעסקיו",
    "שלמה בן יהודה נסע ממצרים לירושלים ופגש שם את אחיו",
])
def test_non_citation_evidence_not_flagged(evidence):
    assert _is_citation_evidence(evidence) is False


# ── shelfmark mistyped as Institution ─────────────────────────────────────────

def _relation(subject="Book", subject_type="Fragment", relation="HELD_AT",
              obj="Institution", object_type="Institution", evidence="x"):
    return {
        "subject": subject, "subject_type": subject_type,
        "relation": relation,
        "object": obj, "object_type": object_type,
        "evidence": evidence, "confidence": "high",
    }


@pytest.mark.parametrize("bad_institution_name", [
    "ENA 3902.5 verso",
    "Bodl. MS. Heb. b. 12",
    "CUL Or. 1080 J80",
    "BL Or. 4305",
])
def test_shelfmark_shaped_institution_rejected(bad_institution_name):
    rel = _relation(obj=bad_institution_name)
    accepted, rejected = _compact_relations([rel], {})
    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0]["reject_reason"] == "institution_is_shelfmark"


@pytest.mark.parametrize("real_institution_name", [
    "Jewish Theological Seminary",
    "John Rylands Library, University of Manchester",
    # Real institutions that trip classify_shelfmark's looser "berlin"
    # bucket (bare substring/prefix match) — must NOT be rejected by the
    # institution_is_shelfmark guard, which only uses the strict "standard"
    # classification.
    "Berlin, Staatsbibliothek zu Berlin",
    "Jüdische Gemeindebibliothek (Berlin)",
])
def test_real_institution_not_rejected_as_shelfmark(real_institution_name):
    rel = _relation(obj=real_institution_name)
    accepted, rejected = _compact_relations([rel], {})
    assert len(accepted) == 1
    assert rejected == []
