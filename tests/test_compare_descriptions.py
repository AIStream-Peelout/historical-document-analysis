# File name: test_compare_descriptions.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Comparison of generated-description runs: inputs, per-label and paired metrics, bootstrap, report and CLI.

Toy ``describe_from_reads`` rows only; no model and no network.
"""
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets.consensus import compare_descriptions as cd

BASE = "Fragment of a piyyut for the Day of Atonement with an alphabetical acrostic."
FAR = "Legal deed recording the sale of a house in Fustat before the court of three."
MID = "Fragment of a piyyut for the New Year with an alphabetical acrostic."


# --------------------------------------------------------------------------------------------- fixtures


def desc_row(doc_id: str, genre: str = "liturgical", confidence: Optional[float] = 0.5, eligible: bool = True,
             reason: Optional[str] = None, summary: str = BASE, kept: Optional[Dict[str, List[str]]] = None,
             dropped: int = 0, capped: bool = False, work: Optional[str] = None, n_lines: int = 20,
             n_agreed: int = 2, generated_at: str = "2026-10-09T00:00:00+00:00",
             readers: Sequence[str] = ("qwen3-vl-8b-heb-v21b-step1200",), **extra: Any) -> Dict[str, Any]:
    """A ``describe_from_reads`` ``--out`` row with the fields the comparison reads.

    :param doc_id: Document.
    :param genre: Genre.
    :param confidence: Grounded confidence.
    :param eligible: ``embed_eligible``.
    :param reason: ``embed_reason`` (default from ``eligible``).
    :param summary: Summary.
    :param kept: Grounded spans per kind (default one person).
    :param dropped: Dropped spans.
    :param capped: ``confidence_capped``.
    :param work: Identified work title, or None.
    :param n_lines: Transcription lines.
    :param n_agreed: Agreed lines.
    :param generated_at: Row timestamp.
    :param readers: VLM readers.
    :param extra: Top-level keys to override.
    :return: Row.
    :rtype: Dict[str, Any]
    """
    kept = {"people": ["משה"]} if kept is None else kept
    entities = {kind: list(kept.get(kind, [])) for kind in cd.ENTITY_KINDS}
    identified = None if work is None else {"title": work, "section": None, "confidence": 0.5, "verified": False,
                                            "verification": "unverified"}
    reply = {"genre": genre, "document_type": "Piyyut", "language": ["Hebrew"], "script": "Hebrew",
             "summary": summary, "identified_work": identified, "entities": entities, "themes": [],
             "confidence": confidence, "notes": ""}
    row = {"doc_id": doc_id, "n_pages": 1, "pages_read": 1, "n_lines": n_lines, "n_agreed": n_agreed,
           "agreed_lines_pct": round(100.0 * n_agreed / n_lines, 1) if n_lines else 0.0,
           "readers": {"vlm": list(readers), "htr": ["MiDRASH_Gen_01"]}, "source_files": ["reads.jsonl"],
           "model": "qwen/qwen3.8-27b", "prompt_version": "gen-desc-v1-20261008", "reasoning": "off",
           "attempts": 1, "generated_at": generated_at, "json": reply,
           "grounding": {"kept": entities, "dropped": [{"kind": "people", "span": "x", "reason": "absent"}] * dropped,
                         "summary_spans_absent": [], "confidence_capped": capped,
                         "identified_work": "none" if work is None else "unverified", "confidence": confidence},
           "description_text": f"Description: {summary}", "embed_eligible": eligible,
           "embed_reason": reason or ("eligible" if eligible else "low_confidence"),
           "description_source": "generated_from_consensus", "machine_derived": True}
    row.update(extra)
    return row


def write_rows(path: Path, rows: Sequence[Dict[str, Any]], tail: str = "") -> Path:
    """Write rows as JSONL, then an optional raw tail (a bad or partial line).

    :param path: File.
    :param rows: Rows.
    :param tail: Raw text appended after the rows.
    :return: The path.
    :rtype: Path
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + tail, encoding="utf-8")
    return path


def docs_of(rows: Sequence[Dict[str, Any]]) -> Dict[str, cd.Description]:
    """Descriptions by document from rows.

    :param rows: Rows.
    :return: Descriptions.
    :rtype: Dict[str, cd.Description]
    """
    return {r["doc_id"]: cd.description_from_row(r, f"toy.jsonl:{i}") for i, r in enumerate(rows, 1)}


# --------------------------------------------------------------------------------------------- inputs


def test_input_specs_and_errors(tmp_path: Path) -> None:
    """``LABEL=PATH[+PATH...]`` parsing; malformed, repeated, missing and single-label inputs are named."""
    f = write_rows(tmp_path / "a.jsonl", [desc_row("D1")])
    assert cd.split_input_spec(f"single={f}+{f}") == ("single", [str(f), str(f)])
    assert cd.split_input_spec("no-label") == ("", [])
    assert cd.input_errors([f"a={f}", f"b={f}"]) == []
    errors = cd.input_errors([f"a={f}", f"a={f}", f"b={tmp_path / 'missing.jsonl'}", "junk"])
    assert any("given twice" in e for e in errors) and any("not found" in e for e in errors)
    assert any("'junk'" in e for e in errors)
    assert cd.input_errors([f"a={f}"]) == ["give at least two labels"]


def test_load_label_keeps_the_newest_row_and_lists_skipped_lines(tmp_path: Path) -> None:
    """Files of a label are concatenated; a rewritten document keeps its newest row; bad lines are listed."""
    a = write_rows(tmp_path / "a.jsonl", [desc_row("D1", genre="liturgical", generated_at="2026-10-09T01:00:00+00:00"),
                                          desc_row("D2")])
    b = write_rows(tmp_path / "b.jsonl", [desc_row("D1", genre="rabbinic", generated_at="2026-10-09T02:00:00+00:00"),
                                          {"doc_id": "D9", "stage": "lm_studio", "error": "HTTP 500"}],
                   tail='not json\n{"doc_id": "D3", ')
    run = cd.load_label("single", [a, b])
    assert set(run.docs) == {"D1", "D2"} and run.docs["D1"].genre == "rabbinic"
    assert (run.rows, run.duplicates) == (3, 1)
    assert [p.split(": ", 1)[1].split(" ")[0] for p in run.problems] == ["no", "not", "partial"]
    assert cd.load_label("reversed", [b, a]).docs["D1"].genre == "rabbinic"   # an older row never replaces a newer


# --------------------------------------------------------------------------------------------- per label


def test_label_metrics() -> None:
    """Eligibility, confidence median and tiers, genres, works, entity and dropped counts, agreement, lengths."""
    rows = [desc_row("D1", confidence=0.1, eligible=False, genre="unclear", kept={}, n_lines=10, n_agreed=0),
            desc_row("D2", confidence=0.5, capped=True, dropped=2,
                     kept={"people": ["משה", "אהרן"], "places": ["מצרים"]}, n_lines=20, n_agreed=5, work="Mahzor"),
            desc_row("D3", confidence=0.6, genre="rabbinic", n_lines=30, n_agreed=3, summary="Short."),
            desc_row("D4", confidence=0.9, genre="rabbinic", n_lines=40, n_agreed=12)]
    m = cd.label_metrics(list(docs_of(rows).values()))
    assert (m["documents"], m["eligible"]) == (4, 3) and m["reasons"] == {"eligible": 3, "low_confidence": 1}
    assert m["confidence_median"] == pytest.approx(0.55) and m["confidence_mean"] == pytest.approx(0.525)
    assert [m["tiers"][t] for t in cd.TIERS] == [1, 1, 1, 1, 0]
    assert (m["capped"], m["identified_work"]) == (1, 1)
    assert m["genres"] == {"unclear": 1, "liturgical": 1, "rabbinic": 2}
    assert m["kept"] == {"people": 4, "places": 1, "dates": 0, "commodities": 0, "institutions": 0}
    assert (m["docs_with_kept"], m["dropped"], m["docs_with_dropped"]) == (3, 2, 1)
    assert m["dropped_reasons"] == {"absent": 2}
    assert m["agreed_pct_median"] == pytest.approx(17.5) and m["agreed_pct_pooled"] == pytest.approx(20.0)
    assert (m["lines_median"], m["agreed_lines_median"]) == (25.0, 4.0)
    assert m["summary_chars_median"] == statistics.median([len(BASE)] * 3 + [len("Short.")])
    assert m["provenance"]["readers"] == {"v21b-step1200": 4}


def test_confidence_tier_edges() -> None:
    """Tier edges follow the generator's thresholds: below 0.25 not embedded, up to 0.5 the cap, 0.75 and up."""
    assert cd.confidence_tier(0.2499) == cd.TIERS[0]
    assert cd.confidence_tier(0.25) == cd.confidence_tier(0.5) == cd.TIERS[1]
    assert cd.confidence_tier(0.55) == cd.TIERS[2]
    assert cd.confidence_tier(0.75) == cd.TIERS[3]
    assert cd.confidence_tier(None) == cd.TIERS[4]


# --------------------------------------------------------------------------------------------- paired


def test_eligibility_differences_both_ways() -> None:
    """Documents whose embedding decision flips are listed per direction, with the reasons that changed."""
    a = docs_of([desc_row("D1"), desc_row("D2"), desc_row("D3", eligible=False), desc_row("D4", eligible=False)])
    b = docs_of([desc_row("D1", eligible=False, reason="unclear_content"), desc_row("D2"), desc_row("D3"),
                 desc_row("D4", eligible=False)])
    p = cd.pair_metrics(a, b, resamples=200)
    assert p["eligible_a_only"] == ["D1"] and p["eligible_b_only"] == ["D3"] and p["same_decision"] == 2
    assert p["reason_changes"] == {("eligible", "unclear_content"): 1, ("low_confidence", "eligible"): 1}


def test_confidence_deltas_and_bootstrap_are_deterministic_with_a_seed() -> None:
    """Paired deltas are later minus earlier; the same seed gives the same interval, which holds the estimate."""
    values = [0.0, 0.05, -0.1, 0.2, 0.15, 0.0, -0.05, 0.3, 0.1, 0.0]
    first = cd.bootstrap_ci(values, resamples=10_000, seed=7)
    assert first == cd.bootstrap_ci(values, resamples=10_000, seed=7)
    lo, hi = first["mean"]
    assert lo < statistics.fmean(values) < hi
    lo, hi = first["median"]
    assert lo <= statistics.median(values) <= hi
    assert cd.bootstrap_ci([0.1] * 5, resamples=500, seed=0)["median"] == pytest.approx((0.1, 0.1))
    assert cd.bootstrap_ci([], resamples=10) == {"median": None, "mean": None}

    a = docs_of([desc_row("D1", confidence=0.3), desc_row("D2", confidence=0.5), desc_row("D3", confidence=0.4),
                 desc_row("D4", confidence=None)])
    b = docs_of([desc_row("D1", confidence=0.35), desc_row("D2", confidence=0.5), desc_row("D3", confidence=0.2),
                 desc_row("D4", confidence=0.4)])
    p = cd.pair_metrics(a, b, resamples=1000, seed=1)
    assert p["deltas"] == {"D1": 0.05, "D2": 0.0, "D3": -0.2}            # rounded, D4 has no earlier confidence
    assert (p["higher"], p["equal"], p["lower"]) == (1, 1, 1) and p["delta_median"] == 0.0
    assert p["delta_ci"] == cd.pair_metrics(a, b, resamples=1000, seed=1)["delta_ci"]


def test_genre_agreement_rate() -> None:
    """Genre agreement counts shared documents with the same genre; changes are counted per direction."""
    genres_a = ["liturgical", "rabbinic", "unclear", "biblical"]
    genres_b = ["liturgical", "liturgical", "unclear", "rabbinic"]
    a = docs_of([desc_row(f"D{i}", genre=g) for i, g in enumerate(genres_a)])
    b = docs_of([desc_row(f"D{i}", genre=g) for i, g in enumerate(genres_b)])
    p = cd.pair_metrics(a, b, resamples=100)
    assert p["genre_agree"] == 2
    assert p["genre_changes"] == {("rabbinic", "liturgical"): 1, ("biblical", "rabbinic"): 1}


def test_summary_similarity() -> None:
    """Character-level ratio: whitespace-insensitive, 1 only for identical text."""
    assert cd.summary_similarity(BASE, "  " + BASE.replace(" ", "\n ") + " ") == 1.0
    assert cd.summary_similarity("", "") == 1.0
    assert cd.summary_similarity(BASE, MID) > cd.summary_similarity(BASE, FAR)
    assert 0.0 < cd.summary_similarity(BASE, FAR) < 1.0


def test_divergence_sample_ordering() -> None:
    """Lowest summary similarity first; ties by genre change, decision change, confidence change, then doc_id."""
    a = docs_of([desc_row("D_same"), desc_row("D_far_a"), desc_row("D_far_b"), desc_row("D_far_c"),
                 desc_row("D_mid"), desc_row("D_genre"), desc_row("D_conf", confidence=0.3),
                 desc_row("D_conf_small", confidence=0.3)])
    b = docs_of([desc_row("D_same"), desc_row("D_far_c", summary=FAR), desc_row("D_far_b", summary=FAR,
                                                                               genre="documentary"),
                 desc_row("D_far_a", summary=FAR), desc_row("D_mid", summary=MID),
                 desc_row("D_genre", genre="rabbinic"), desc_row("D_conf", confidence=0.6),
                 desc_row("D_conf_small", confidence=0.35)])
    expected = ["D_far_b", "D_far_a", "D_far_c", "D_mid", "D_genre", "D_conf", "D_conf_small"]
    p = cd.pair_metrics(a, b, resamples=100, sample=3)
    assert cd.divergent_documents(a, b, p["similarity"]) == expected
    assert p["sample"] == expected[:3] and p["n_divergent"] == len(expected)


# --------------------------------------------------------------------------------------------- report and CLI


def test_missing_documents_are_listed_not_fatal(tmp_path: Path) -> None:
    """A document in one label only is listed; paired figures use the shared documents; summaries sit side by side."""
    a = write_rows(tmp_path / "a.jsonl", [desc_row("D1"), desc_row("D2"), desc_row("Only_A")])
    b = write_rows(tmp_path / "b.jsonl", [desc_row("D1"), desc_row("D2", summary="Letter | about a debt of 5 dinars."),
                                          desc_row("Only_B")])
    out = tmp_path / "report.md"
    cd.main([f"single={a}", f"voted={b}", "--out", str(out), "--resamples", "200"])
    text = out.read_text(encoding="utf-8")
    assert "- only in single: `Only_A`" in text and "- only in voted: `Only_B`" in text
    assert "## Per label, shared documents only (2)" in text and "Documents in both: 2." in text
    assert f"| summary | {BASE} | Letter \\| about a debt of 5 dinars. |" in text
    pair = cd.compare_runs([cd.load_label("single", [a]), cd.load_label("voted", [b])], resamples=200)["pairs"][0]
    assert (pair["shared"], pair["only_a"], pair["only_b"]) == (["D1", "D2"], ["Only_A"], ["Only_B"])
    assert pair["sample"] == ["D2"]


def test_self_comparison_is_identical_with_no_divergence(tmp_path: Path) -> None:
    """A file compared with itself: equal per-label metrics, zero deltas, full agreement, no divergent document."""
    f = write_rows(tmp_path / "gen.jsonl", [desc_row("D1", confidence=0.3), desc_row("D2", confidence=0.7,
                                                                                   genre="rabbinic"),
                                            desc_row("D3", confidence=0.1, eligible=False)])
    runs = [cd.load_label("a", [f]), cd.load_label("b", [f])]
    result = cd.compare_runs(runs, resamples=1000, seed=3)
    assert result["per_label"]["a"] == result["per_label"]["b"] and result["per_label_shared"] is None
    p = result["pairs"][0]
    assert (p["n_divergent"], p["sample"], p["eligible_a_only"], p["eligible_b_only"]) == (0, [], [], [])
    assert p["delta_median"] == 0.0 and p["delta_ci"] == {"median": (0.0, 0.0), "mean": (0.0, 0.0)}
    assert p["genre_agree"] == 3 and set(p["similarity"].values()) == {1.0}
    text = cd.render_markdown(runs, result, 1000, 3, built_at="T")
    assert "No document differs in summary, genre, embedding decision or confidence." in text
    assert "| median | +0.000 | +0.000 to +0.000 |" in text and "Same genre: 3 (100.0 %)." in text
    assert "Warning" not in text


def test_setup_differences_are_flagged(tmp_path: Path) -> None:
    """Different prompt versions between labels put a warning in the report."""
    a = write_rows(tmp_path / "a.jsonl", [desc_row("D1")])
    b = write_rows(tmp_path / "b.jsonl", [desc_row("D1", prompt_version="gen-desc-v2")])
    runs = [cd.load_label("a", [a]), cd.load_label("b", [b])]
    text = cd.render_markdown(runs, cd.compare_runs(runs, resamples=100), 100, 0, built_at="T")
    assert "**Warning:** generator, prompt version or reasoning mode differ" in text


def test_dry_run_validates_only(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``--dry-run`` reports the inputs and writes nothing; bad arguments and empty labels stop it."""
    f = write_rows(tmp_path / "a.jsonl", [desc_row("D1")], tail='{"doc_id": "D2", ')
    out = tmp_path / "report.md"
    cd.main([f"a={f}", f"b={f}", "--out", str(out), "--dry-run"])
    printed = capsys.readouterr().out
    assert not out.exists()
    assert "dry run: inputs valid; nothing written" in printed and "partial last line skipped" in printed
    assert "a vs b: shared 1; only in a 0; only in b 0" in printed
    with pytest.raises(SystemExit):
        cd.main([f"a={f}", f"b={tmp_path / 'missing.jsonl'}", "--dry-run"])
    with pytest.raises(SystemExit):
        cd.main([f"a={f}", "--dry-run"])
    failures = write_rows(tmp_path / "gen.failures.jsonl", [{"doc_id": "D1", "stage": "lm_studio", "error": "x"}])
    with pytest.raises(SystemExit):
        cd.main([f"a={f}", f"b={failures}", "--dry-run"])
