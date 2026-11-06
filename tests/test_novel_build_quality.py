"""Tests of stage 3 (quality) of the novel-rows build: every repair and gate on crafted synthetic texts.

Texts are random words over the Hebrew and Arabic alphabets (not texts), plus the few editorial words the detectors look
for. The tokenizer is a fake (one token per character); no model is loaded.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.qwen_hebrew.novel_build import paths, quality as q
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, ImageRef, QualityResult, SideDecision, TargetSide,
                                                           read_jsonl, write_jsonl)

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"                  # 22 Hebrew letters (an alphabet, not a text)
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"              # 28 Arabic letters (an alphabet, not a text)
FATHA = "\u064e"


class FakeTokenizer:
    """One token per character, called like a ``transformers`` tokenizer."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """Tokenise.

        :param text: Text.
        :param add_special_tokens: Ignored (no special tokens).
        :return: ``{"input_ids": [...]}``.
        """
        return {"input_ids": [ord(ch) for ch in text]}


def words(alphabet: str, letters: int, per_line: int = 6, seed: int = 0) -> str:
    """Random words over an alphabet, about ``letters`` letters, ``per_line`` words per line.

    :param alphabet: Letters.
    :param letters: Approximate letter count.
    :param per_line: Words per line.
    :param seed: Random seed.
    :return: The text.
    """
    rng = random.Random(seed)
    out: List[str] = []
    while sum(len(w) for w in out) < letters:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(2, 6))))
    return "\n".join(" ".join(out[i:i + per_line]) for i in range(0, len(out), per_line))


def make(text: str, *, source: str = "pgp", width: Optional[int] = 2000, height: Optional[int] = 3000,
         coverage: Optional[Dict[str, Any]] = None, notes: Optional[Dict[str, Any]] = None) -> tuple:
    """A candidate with one target and its accepted decision.

    :param text: Target text.
    :param source: Candidate source.
    :param width: Photograph width.
    :param height: Photograph height.
    :param coverage: Coverage signals of the decision (default: a plain 1:1 read).
    :param notes: Candidate notes.
    :return: ``(candidate, target, decision)``.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    target = TargetSide(side_key="recto", text=text, raw_text=text, lines=lines, pointer={"source": "synthetic"})
    image = ImageRef(image_id="iiif_cudl:0", origin="iiif_cudl", sha256="sha_q", status="present", width=width,
                     height=height, path=None)
    cand = Candidate(row_id="row:1", source=source, document_group="group:1", canonical_ids=[], pgpids=[], ktiv_id=None,
                     shelfmark="", script="hebrew", languages_stated=None, verdict="novel", witnesses=[], unit="document",
                     n_letters=0, targets=[target], images=[image], notes=notes or {})
    cover = coverage if coverage is not None else {"letters_ratio": 1.0, "kraken_lines": len(lines), "target_lines": len(lines)}
    decision = SideDecision(row_id="row:1", side_key="recto", outcome="accepted", reasons=[], image_sha256="sha_q",
                            basis="p99", reader="kraken:test", recall=0.8, null_p95=0.01, null_p99=0.02,
                            competitors=[{"role": "chosen", "sha256": "sha_q", "coverage": cover}])
    return cand, target, decision


def check(text: str, **kwargs: Any) -> QualityResult:
    """Gate one crafted side with the fake tokenizer.

    :param text: Target text.
    :param kwargs: Passed to :func:`make`.
    :return: The result.
    """
    cand, target, decision = make(text, **kwargs)
    return q.check_side(cand, target, decision, tokenizer=FakeTokenizer())


def test_clean_hebrew_side_passes() -> None:
    """A clean Hebrew side passes every gate and its tokens are question plus answer."""
    result = check(words(HEBREW, 200))
    assert result.ok and result.reasons == [] and result.script == "hebrew"
    assert result.tokens == len(q.SCRIPT_NEUTRAL_PROMPT) + len(result.text)
    assert result.letters_hebrew >= 200 and result.letters_arabic == 0


@pytest.mark.parametrize("char", list(q.HYGIENE_CHARS))
def test_hygiene_characters_fail(char: str) -> None:
    """Each hygiene character fails the side.

    :param char: The character under test.
    """
    result = check(words(HEBREW, 200) + f" אב{char}גד")
    assert not result.ok and q.HYGIENE in result.reasons and q.NON_TEXT not in result.reasons


@pytest.mark.parametrize("char", ["@", "*", "\u200f", "\t", "+"])
def test_non_text_characters_fail(char: str) -> None:
    """Characters outside the allowed set fail the side.

    :param char: The character under test.
    """
    result = check(words(HEBREW, 200) + f" אב{char}גד")
    assert not result.ok and q.NON_TEXT in result.reasons


def test_brackets_digits_latin_and_punctuation_are_text() -> None:
    """Gap tokens, brackets, digits, Latin letters and common punctuation are allowed."""
    result = check(words(HEBREW, 200) + "\n[...] (אב) 12, גד. ef׳ ״ ־ ،؟")
    assert result.ok, result.reasons


def test_presentation_form_finals_are_repaired_with_a_flag() -> None:
    """Hebrew presentation-form finals are repaired by NFKC and only flagged."""
    text = words(HEBREW, 200) + "\nשל\ufb26 ו\ufb4f"
    result = check(text)
    assert result.ok and q.MOJIBAKE in result.reasons
    assert "שלם ואל" in result.text and not q.HEBREW_PRESENTATION.search(result.text)


def test_a_presentation_form_left_unrepaired_fails() -> None:
    """A presentation form that NFKC cannot repair fails ``mojibake_final``."""
    result = check(words(HEBREW, 200) + " א\ufb1e")
    assert not result.ok and q.MOJIBAKE in result.reasons


def test_harakat_are_stripped_with_a_flag() -> None:
    """Harakat are stripped from the final text and flagged, not failed."""
    text = words(ARABIC, 200)
    voweled = text.replace(" ", FATHA + " ", 5)
    result = check(voweled)
    assert result.ok and result.reasons == [q.HARAKAT_STRIPPED]
    assert FATHA not in result.text and result.text == q.strip_harakat(voweled)


def test_arabic_presentation_forms_and_persian_letters_are_folded() -> None:
    """Arabic presentation forms (the lam-alef ligature) and Persian yeh / keheh take the pool's spelling, flagged."""
    result = check(words(ARABIC, 200) + " \ufefb \u06cc\u06a9")
    assert result.ok and result.reasons == [q.HARAKAT_STRIPPED]
    assert "\u0644\u0627 \u064a\u0643" in result.text and not q.ARABIC_PRESENTATION.search(result.text)


def test_letter_and_line_minimums() -> None:
    """Under 40 letters fails; under 3 lines fails only when the lines are known."""
    assert q.UNDER_LETTERS in check(words(HEBREW, 30)).reasons
    two_lines = words(HEBREW, 120, per_line=20)
    assert len(two_lines.splitlines()) == 2
    assert q.UNDER_LINES in check(two_lines).reasons
    flat = words(HEBREW, 200, per_line=1000)
    assert check(flat, source="fjp").ok
    assert q.UNDER_LINES not in check(flat, source="pgp").reasons


def test_editorial_terms_and_edition_apparatus() -> None:
    """The legacy-store detectors and upstream cleaner marks fail editorial prose and apparatus."""
    assert q.EDITORIAL in check(words(HEBREW, 200) + "\nאב מעל השורה גד").reasons
    apparatus = check(words(HEBREW, 200) + "\nמקרא סימני אב השלמת המהדיר גד")
    assert q.APPARATUS in apparatus.reasons
    assert q.APPARATUS in check(words(HEBREW, 200), notes={"family": "edition_apparatus"}).reasons
    assert q.EDITORIAL in check(words(HEBREW, 200), notes={"flags": {"editorial_terms": 2}}).reasons


def test_arabic_ratio_and_minimum() -> None:
    """Arabic sides need 40 Arabic letters and 9 times as many Arabic as Hebrew letters."""
    arabic = words(ARABIC, 100, seed=1)
    ratio = check(arabic + "\n" + "אבגדה אבגדה אבגד")
    assert ratio.script == "arabic" and q.ARABIC_RATIO in ratio.reasons and q.ARABIC_UNDER not in ratio.reasons
    short = check(words(ARABIC, 34, seed=2) + "\n" + "אבג")
    assert short.script == "arabic" and q.ARABIC_UNDER in short.reasons
    assert check(words(ARABIC, 120, seed=3)).ok


def test_arabic_density_uses_the_chosen_photograph() -> None:
    """Arabic letters per megapixel of the chosen photograph above 1,000 fail; an unknown size is flagged."""
    text = words(ARABIC, 400, seed=4)
    assert q.ARABIC_DENSE in check(text, width=500, height=500).reasons
    assert check(text, width=2000, height=3000).ok
    unknown = check(text, width=None, height=None)
    assert unknown.ok and q.SIZE_UNKNOWN in unknown.reasons


@pytest.mark.parametrize("ratio, kraken_factor, ok, reasons", [
    (1.0, 1.0, True, []),
    (1.49, 1.0, True, []),
    (1.5, 1.0, False, [q.READ_OVER]),
    (2.3, 1.0, False, [q.READ_OVER]),
    (0.41, 1.0, True, []),
    (0.39, 1.0, False, [q.READ_UNDER]),
    (1.0, 1.4, True, [q.KRAKEN_LINES]),
    (1.0, 1.5, False, [q.KRAKEN_LINES_OVER]),
    (1.0, 2.0, False, [q.KRAKEN_LINES_OVER])])
def test_coverage_signals(ratio: float, kraken_factor: float, ok: bool, reasons: List[str]) -> None:
    """Read letters at 1.5 times the target's or more, or under 0.4 times, fail; Kraken lines at 1.5 times the target's
    fail and between 1.3 and 1.5 times are flagged.

    :param ratio: Read letters over target letters.
    :param kraken_factor: Kraken lines over target lines.
    :param ok: Expected pass.
    :param reasons: Expected reasons.
    """
    text = words(HEBREW, 200)
    lines = len(text.splitlines())
    result = check(text, coverage={"letters_ratio": ratio, "kraken_lines": round(kraken_factor * lines),
                                   "target_lines": lines})
    assert result.ok is ok and result.reasons == reasons


def test_a_dead_first_or_last_third_is_a_partial_image() -> None:
    """A first or last third of at least 100 letters without one matched n-gram fails; shorter thirds do not."""
    text = words(HEBREW, 400)
    lines = len(text.splitlines())
    base = {"letters_ratio": 1.0, "kraken_lines": lines, "target_lines": lines}
    dead = check(text, coverage={**base, "thirds": {"letters": [130, 130, 130], "matched": [40, 35, 0]}})
    assert not dead.ok and dead.reasons == [q.PARTIAL_IMAGE]
    alive = check(text, coverage={**base, "thirds": {"letters": [130, 130, 130], "matched": [40, 0, 2]}})
    assert alive.ok
    short = check(text, coverage={**base, "thirds": {"letters": [90, 90, 90], "matched": [0, 30, 0]}})
    assert short.ok


def test_token_budget_with_a_fake_tokenizer() -> None:
    """Question plus answer over the budget fails; without a tokenizer the budget is flagged unmeasured."""
    long_text = words(HEBREW, q.TOKEN_BUDGET, per_line=8)
    result = check(long_text)
    assert result.tokens > q.TOKEN_BUDGET and q.TOKENS_OVER in result.reasons and not result.ok
    cand, target, decision = make(words(HEBREW, 200))
    unmeasured = q.check_side(cand, target, decision, tokenizer=None)
    assert unmeasured.ok and unmeasured.tokens is None and unmeasured.reasons == [q.TOKENS_UNMEASURED]


def test_mixed_script_is_measured_from_the_final_text() -> None:
    """The script is measured from the final text (both scripts give ``mixed``)."""
    result = check(words(HEBREW, 150, seed=5) + "\n" + words(ARABIC, 80, seed=6))
    assert result.script == "mixed"


def test_main_with_no_tokens(tmp_path: Path) -> None:
    """The CLI gates only accepted sides and ``--no-tokens`` flags the budget as unmeasured.

    :param tmp_path: Pytest temporary directory.
    """
    cand, target, decision = make(words(HEBREW, 200) + " אב&")
    review = SideDecision(row_id="row:1", side_key="verso", outcome="needs_review", reasons=["n3_below_p95"])
    write_jsonl(paths.stage_path(tmp_path, "images"), [cand])
    write_jsonl(paths.stage_path(tmp_path, "sides"), [decision, review])
    q.main(["--work-dir", str(tmp_path), "--no-tokens"])
    out = list(read_jsonl(paths.stage_path(tmp_path, "quality"), QualityResult))
    assert len(out) == 1 and out[0].side_key == "recto" and out[0].tokens is None
    assert not out[0].ok and out[0].reasons == [q.HYGIENE, q.TOKENS_UNMEASURED]


def test_missing_coverage_is_recomputed_from_reads() -> None:
    """A decision without coverage signals gets them from the reads, else a flag."""
    from src.finetuning.qwen_hebrew.novel_build.schema import ReadRecord

    text = words(HEBREW, 200)
    cand, target, decision = make(text)
    decision.competitors = []
    page = "\n".join([text, text, text])
    lines = [line for line in page.splitlines() if line.strip()]
    reads = {"sha_q": [ReadRecord("sha_q", "kraken:test", page, lines, len(lines), 600, 0, "ok", 1.0, "2026-11-03T00:00:00Z")]}
    results = q.run_quality({cand.row_id: cand}, [decision], tokenizer=FakeTokenizer(), reads=reads)
    assert q.READ_OVER in results[0].reasons
    assert q.COVERAGE_UNKNOWN in q.run_quality({cand.row_id: cand}, [decision], tokenizer=FakeTokenizer())[0].reasons


# --- target flags ----------------------------------------------------------------------------------------------------

def test_target_flags_gate_and_record() -> None:
    """Legacy-store ``unresolved_finals`` and PGP ``complex_side_labels`` fail; ``not_body_text`` (spans kept as ink)
    and the notation flags are recorded only; a failing flag of another source's kind does not apply."""
    text = words(HEBREW, 200)
    for source, flags, expected in (("fjp", ["unresolved_finals", "wrap_reversed"], ["unresolved_finals"]),
                                    ("fjp_scrape", ["not_body_text"], ["not_body_text"]),
                                    ("fjp", ["not_body_text", "unresolved_finals"], ["unresolved_finals",
                                                                                     "not_body_text"]),
                                    ("pgp", ["complex_side_labels", "notation_latin"], ["complex_side_labels",
                                                                                        "notation_latin"]),
                                    ("pgp", ["notation_glued_digits", "unrendered_lines"], ["notation_glued_digits"]),
                                    ("pgp", ["unresolved_finals"], [])):
        result = check(text, source=source, notes={"target_flags": {"recto": flags, "verso": ["not_body_text"]}})
        assert result.reasons == expected
        assert result.ok is all(r in q.FLAG_ONLY for r in expected)


def test_gap_density_over_30_percent_fails_legacy_store_sides() -> None:
    """More than 30 gap tokens per 100 letters fails a legacy-store side (the October build's rule), not a PGP one."""
    gappy = "\n".join(f"{line} [...] [...]" for line in words(HEBREW, 60, per_line=1).splitlines())
    assert q.gap_density(gappy) > q.MAX_GAP_DENSITY
    assert q.GAP_DENSITY in check(gappy, source="fjp").reasons
    assert q.GAP_DENSITY not in check(gappy, source="pgp").reasons


# --- sign normalisations and allowed characters ----------------------------------------------------------------------

def test_backtick_becomes_a_geresh_with_a_flag() -> None:
    """An ASCII backtick is a geresh: replaced and flagged, not failed."""
    result = check(words(HEBREW, 200) + "\nאב` גד")
    assert result.ok and result.reasons == [q.BACKTICK_GERESH] and "אב' גד" in result.text


def test_reversed_supplied_span_is_dropped_for_pgp_only() -> None:
    """A reversed ``>x<`` in a PGP target is dropped like ``<x>`` (letters supplied by the editor); elsewhere it
    fails."""
    text = words(HEBREW, 200) + "\nכל אשר >לו< אך"
    pgp = check(text)
    assert pgp.ok and pgp.reasons == [q.REVERSED_SUPPLIED_DROPPED] and "כל אשר  אך" in pgp.text
    other = check(text, source="literature")
    assert not other.ok and q.HYGIENE in other.reasons


def test_lone_fraction_signs_after_a_numeral_or_at_a_line_end_are_dropped() -> None:
    """A lone ``>`` / ``<`` after a digit, ending a token after a Hebrew letter numeral, or ending a line is dropped
    and flagged; one inside a token stays and fails."""
    dropped = check(words(HEBREW, 200) + "\nאב 5> גד\nבן יז> ז< דאר\nהו זח <")
    assert dropped.ok and dropped.reasons == [q.FRACTION_SIGN_DROPPED]
    assert "אב 5 גד" in dropped.text and "בן יז ז דאר" in dropped.text and "<" not in dropped.text
    kept = check(words(HEBREW, 200) + "\nאב ז>גד הו")
    assert not kept.ok and kept.reasons == [q.HYGIENE]


def test_combining_marks_and_fractions_are_text() -> None:
    """Combining marks (a dotted letter of Judaeo-Arabic) and the fractions are allowed characters."""
    result = check(words(HEBREW, 200) + "\nג̇ צ̇ 5½ 3¼ ¾ ⅛ 1⁄2")
    assert result.ok, result.reasons


def test_papyrus_numerals_are_allowed_on_arabic_sides_only() -> None:
    """An asterisk, Greek letters and the modifier prime pass on an Arabic-script side and fail on a Hebrew one."""
    numerals = " * ιβʹ χ"
    assert check(words(ARABIC, 200) + numerals).ok
    hebrew = check(words(HEBREW, 200) + numerals)
    assert not hebrew.ok and hebrew.reasons == [q.NON_TEXT]


# --- coverage on the read that covers most of the target -------------------------------------------------------------

def test_a_low_coverage_on_kraken_reads_only_asks_for_a_second_read() -> None:
    """A low coverage computed on Kraken reads only fails with ``coverage_second_read_wanted``; when a page-reader
    read of the photograph exists and coverage still fails, no second read is asked for."""
    text = words(HEBREW, 300)
    alone = check(text, coverage={"letters_ratio": 0.3, "coverage_reader": "kraken:test", "usable_readers": ["kraken"]})
    assert not alone.ok and alone.reasons == [q.READ_UNDER, q.SECOND_READ_WANTED]
    both = check(text, coverage={"letters_ratio": 0.3, "coverage_reader": "vlm:test",
                                 "usable_readers": ["kraken", "vlm"]})
    assert not both.ok and both.reasons == [q.READ_UNDER]
    legacy = check(text, coverage={"letters_ratio": 0.3, "reader": "kraken:test"})
    assert legacy.reasons == [q.READ_UNDER, q.SECOND_READ_WANTED]


def test_a_page_reader_read_clears_a_kraken_under_read() -> None:
    """End to end: a Kraken read of 30 % of the target places the side but its coverage fails (second read wanted);
    once a page-reader read of 80 % exists, the coverage is computed on it and the side passes."""
    from src.finetuning.qwen_hebrew.novel_build import side_assign as sa
    from src.finetuning.qwen_hebrew.novel_build.schema import ReadRecord

    def read_of(sha: str, text: str, reader: str) -> ReadRecord:
        """A read record.

        :param sha: Photograph hash.
        :param text: Read text.
        :param reader: Reader id.
        :return: The record.
        """
        lines = [line for line in text.splitlines() if line.strip()]
        return ReadRecord(sha, reader, text, lines, len(lines), sum(ch in HEBREW for ch in text), 0, "ok", 1.0,
                          "2026-11-03T00:00:00Z")

    page = words(HEBREW, 300, per_line=4, seed=9)
    lines = page.splitlines()
    cand, target, _ = make(page)
    nulls = [read_of(f"null_{reader}_{i}", words(HEBREW, 200, seed=100 + i), f"{reader}:test")
             for reader in ("kraken", "vlm") for i in range(60)]
    kraken = read_of("sha_q", "\n".join(lines[: int(len(lines) * 0.3)]), "kraken:test")
    vlm = read_of("sha_q", "\n".join(lines[: int(len(lines) * 0.8)]), "vlm:test")
    first = sa.assign_sides([cand], nulls + [kraken], n_boot=20, seed=3)[0]
    assert first.outcome == "accepted" and first.reader == "kraken:test"
    under = q.run_quality({cand.row_id: cand}, [first], tokenizer=FakeTokenizer())[0]
    assert not under.ok and q.READ_UNDER in under.reasons and q.SECOND_READ_WANTED in under.reasons
    second = sa.assign_sides([cand], nulls + [kraken, vlm], n_boot=20, seed=3)[0]
    assert second.outcome == "accepted" and second.competitors[0]["coverage"]["coverage_reader"] == "vlm:test"
    cleared = q.run_quality({cand.row_id: cand}, [second], tokenizer=FakeTokenizer())[0]
    assert cleared.ok, cleared.reasons
