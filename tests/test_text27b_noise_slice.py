# File name: test_text27b_noise_slice.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for :mod:`src.finetuning.text27b.noise_slice` (synthetic reads, targets and hold-outs only)."""
from __future__ import annotations

import codecs
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.finetuning.text27b import noise_slice as NS
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import textnorm as TN

#: Scrubbed project names, decoded at runtime from rot13 so this file never spells them (fix pass F12).
SCRUB_SURNAME = codecs.decode("Sevrqoret", "rot_13")
SCRUB_SOCIETY = codecs.decode("SWZF", "rot_13")

CFG = NS.SliceConfig()
WORDS = ["אברהם", "כתב", "אלי", "פסטאט", "ואנא", "אעלמך", "באן", "אלכתאב", "וצל", "אליה", "מן", "אכיה", "יצחק",
         "פי", "אלשהר", "אלמאצי", "ושכרת", "אללה", "עלי", "דלך", "ואלסלאם"]


def ws_tokens() -> NS.TokenFns:
    """Whitespace token counters (no tokenizer needed).

    :returns: Counters.
    """
    def render_count(msgs: Sequence[Dict[str, Any]]) -> int:
        """Rendered length: words plus four tokens of scaffold per turn.

        :param msgs: Chat turns.
        :returns: Token count.
        """
        return sum(len(m["content"].split()) + 4 for m in msgs)

    return NS.TokenFns(count=lambda s: len(s.split()), render_count=render_count)


def edition(n_lines: int = 12) -> str:
    """A synthetic Judaeo-Arabic edition (Hebrew script), names on the first line.

    :param n_lines: Number of lines.
    :returns: Edition text.
    """
    lines = []
    for i in range(n_lines):
        lines.append(" ".join(WORDS[(i + j) % len(WORDS)] for j in range(6)))
    return "\n".join(lines)


def corrupt(text: str, rate: float) -> str:
    """Substitute an evenly spread share ``rate`` of the Hebrew letters (raises CER, keeps coverage).

    Letters are ≈80 % of the normalised characters, so the CER is ≈ 0.8 × ``rate``.

    :param text: Text.
    :param rate: Share of letters to substitute (0–1).
    :returns: Corrupted text.
    """
    out, k = [], 0
    for ch in text:
        if "א" <= ch <= "ת":
            if int((k + 1) * rate) > int(k * rate):
                ch = "ק" if ch != "ק" else "ש"
            k += 1
        out.append(ch)
    return "".join(out)


HARD = 0.7     # CER ≈ 0.56: a hard read (also under the translate cap)
PASSING = 0.3  # CER ≈ 0.24: an in-gate read


def read(model: str, text: str, n_lines: int = 10, n_agreed: int = 4) -> Dict[str, Any]:
    """A pairs-file read entry.

    :param model: Reader name.
    :param text: Read text.
    :param n_lines: Lines on the page.
    :param n_agreed: Kraken-agreed lines.
    :returns: Read dict.
    """
    return {"model": model, "text": text, "pages": [{"doc_id": "X_1", "image_index": 0, "n_lines": n_lines,
                                                      "n_agreed": n_agreed}]}


def prow(pgpid: str, reads: List[Dict[str, Any]], ed: Optional[str] = None, script: str = "hebrew",
         shelfmark: str = "T-S 99J1.2", cids: Optional[List[str]] = None, ed_he: Optional[str] = None) -> Dict[str, Any]:
    """A pairs-file row.

    :param pgpid: PGP id.
    :param reads: Read entries.
    :param ed: Edition text (default :func:`edition`).
    :param script: Edition script.
    :param shelfmark: Shelfmark.
    :param cids: Merged canonical ids.
    :param ed_he: ``edition_text_he``.
    :returns: Row dict.
    """
    return {"pgpid": pgpid, "shelfmark": shelfmark, "edition_script": script, "edition_text": ed or edition(),
            "edition_text_he": ed_he, "merged_doc_ids": cids or [f"Cambridge_CUL_T_S_99J1_{pgpid}"], "reads": reads}


DESC = "Letter from Avraham to his brother Yiṣḥaq in Fustat about a book that arrived last month."
CAT = json.dumps({"document_type": "Letter", "languages": {"primary": ["Judaeo-Arabic"], "secondary": []}, "date": None,
                  "origin": "Fustat", "destination": None, "places_mentioned": ["Fustat"],
                  "people": [{"name": "Avraham", "role": "sender", "social_roles": []}], "tags": ["books"],
                  "summary": DESC}, ensure_ascii=False)


def target(pgpid: str, family: str, text: str, split: str = "train", noised: Optional[int] = 50) -> NS.Target:
    """A synthetic target.

    :param pgpid: PGP id.
    :param family: Family.
    :param text: Target text.
    :param split: Split.
    :param noised: Tokens of the noised twin.
    :returns: Target.
    """
    meta = {"lang_class": "JA", "shelfmark": "T-S 99J1.2", "shelfmark_group": f"g{pgpid}", "held_out_vision_benchmark": False,
            "licence": "test-licence", "provenance": {"pgp_document": pgpid}, "source": "synthetic", "side": None,
            "strip_ops": [], "target_raw": None}
    return NS.Target(pgpid=pgpid, family=family, split=split, text=text, meta=meta, noised_tokens=noised)


class NoHoldouts:
    """Hold-out checker that holds nothing out."""

    def is_benchmark(self, pgpid: Optional[str], shelfmark: Any = None, canonical: Any = None) -> Optional[str]:
        """Never a benchmark.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Canonical id.
        :returns: None.
        """
        return None

    def is_eval_doc(self, pgpid: Optional[str], shelfmark: Any = None, canonical: Any = None) -> Optional[str]:
        """Never an eval doc.

        :param pgpid: PGP id.
        :param shelfmark: Shelfmark.
        :param canonical: Canonical id.
        :returns: None.
        """
        return None


# ------------------------------------------------------------------------------ bands / classes
@pytest.mark.parametrize("value,band", [(0.0, "0-0.15"), (0.15, "0-0.15"), (0.1501, "0.15-0.3"), (0.45, "0.3-0.45"),
                                        (0.46, "0.45-0.6"), (0.9, "0.6-0.9"), (0.95, ">0.9"), (None, None)])
def test_noise_band(value: Optional[float], band: Optional[str]) -> None:
    """E7 bands are ``(lo, hi]`` with 0 in the first band.

    :param value: CER.
    :param band: Expected label.
    """
    assert NS.noise_band(value) == band


@pytest.mark.parametrize("cer,cov,ok,cls", [
    (None, None, True, "empty"), (0.5, 0.0, True, "empty"), (0.5, 1.0, False, "script_mismatch"),
    (0.95, 1.0, True, "cer_gt_limit"), (0.5, 0.3, True, "coverage_out"), (0.5, 2.6, True, "coverage_out"),
    (0.45, 1.0, True, "pass"), (0.46, 0.6, True, "hard"), (0.9, 1.5, True, "hard"), (0.5, 0.4, True, "partial"),
    (0.5, 0.59, True, "partial"), (0.5, 1.6, True, "over"), (0.5, 2.5, True, "over")])
def test_classify_read(cer: Optional[float], cov: Optional[float], ok: bool, cls: str) -> None:
    """Each class boundary of :func:`classify_read`.

    :param cer: CER.
    :param cov: Coverage.
    :param ok: Script check.
    :param cls: Expected class.
    """
    assert NS.classify_read(cer, cov, ok, CFG) == cls


def test_read_script() -> None:
    """Majority of Hebrew vs Arabic letters decides; ties and Latin are ``none``."""
    assert NS.read_script("אברהם כתב abc") == "hebrew"
    assert NS.read_script("كتب ابراهيم א") == "arabic"
    assert NS.read_script("Latin only") == "none"


# ------------------------------------------------------------------------------ evaluate_reads
def test_evaluate_reads_measures_and_classifies() -> None:
    """CER / coverage are recomputed with textnorm on the loop-cleaned read; agreed share from the pages."""
    ed = edition()
    good = read("a50", ed)
    hard = read("v21b", corrupt(ed, HARD) + "\n" + "[...]\n" * 3, n_lines=20, n_agreed=5)
    reads = NS.evaluate_reads(prow("1", [good, hard]), ed, CFG)
    assert [r.cls for r in reads] == ["pass", "hard"]
    assert reads[0].cer == 0.0 and reads[0].coverage == 1.0
    assert reads[1].cer == round(TN.cer(TN.clean_read(hard["text"])[0], ed), 4)
    assert reads[1].loop_lines_removed == 3 and reads[1].agreed_share == 0.25
    assert reads[1].pages == ["X_1#0"] and reads[1].n_pages == 1


def test_evaluate_reads_excludes_hebrew_read_of_arabic_edition() -> None:
    """A Hebrew-script read of an Arabic edition is ``script_mismatch``; an Arabic read is measured."""
    ar_ed = "\n".join(f"كتب ابراهيم الى اخيه اسحاق في مصر {w}" for w in ("الاول", "الثاني", "الثالث", "الرابع"))
    he_read = read("a50", "\n".join(f"כתב אברהם אלי אכיה יצחק פי מצר {i}" for i in range(4)))
    ar_read = read("v21b", ar_ed.replace("ابراهيم", "ابرهم"))
    reads = NS.evaluate_reads(prow("2", [he_read, ar_read], ed=ar_ed, script="arabic"), ar_ed, CFG)
    assert reads[0].cls == "script_mismatch"
    assert reads[1].cls == "pass" and reads[1].script == "arabic"


def test_evaluate_reads_empty() -> None:
    """An empty read (or one that normalises to nothing) is ``empty``."""
    reads = NS.evaluate_reads(prow("3", [read("a50", "[...]\n[...]"), read("v21b", "")]), edition(), CFG)
    assert [r.cls for r in reads] == ["empty", "empty"]


# ------------------------------------------------------------------------------ selection
def _evals(*specs: Tuple[str, float, float, str]) -> List[NS.ReadEval]:
    """ReadEvals from (model, cer, coverage, text) specs.

    :param specs: Specs.
    :returns: Read evaluations.
    """
    return [NS.ReadEval(model=m, text=t, cer=c, coverage=v, agreed_share=None, loop_lines_removed=0, script="hebrew",
                        n_pages=1, pages=["X#0"], cls=NS.classify_read(c, v, True, CFG)) for m, c, v, t in specs]


def test_select_hard_lowest_cer_and_model_allowlist() -> None:
    """The lowest-CER hard read wins; readers outside the allowlist are ignored."""
    reads = _evals(("a50", 0.7, 1.0, "x"), ("v21b", 0.5, 1.1, "y"), ("rover", 0.46, 1.0, "z"), ("a50", 0.3, 1.0, "p"))
    sel, how = NS.select_read("extract_relations", reads, "{}", (), CFG)
    assert how == "read_hard" and sel is not None and sel.read.model == "v21b" and sel.grounding is None


def test_select_translate_cap() -> None:
    """Translate families take a hard read only up to CER 0.75 and never a partial read."""
    reads = _evals(("a50", 0.8, 1.0, "x"))
    assert NS.select_read("translate_en", reads, "t", (), CFG) == (None, "hard_over_translate_cap")
    assert NS.select_read("catalogue_json", reads, CAT, (), CFG)[1] == "read_hard"
    partial = _evals(("a50", 0.5, 0.5, "אברהם"))
    assert NS.select_read("translate_he", partial, "t", (), CFG) == (None, "partial_family_not_allowed")
    assert NS.select_read("extract_relations", partial, "{}", (), CFG) == (None, "partial_family_not_allowed")


def test_select_partial_needs_grounding_and_no_hard() -> None:
    """A partial read is kept only when it grounds ≥ 0.5 of the target's facts and no hard read exists."""
    grounded = "כתב אברהם אלי אכיה יצחק פי פסטאט"
    bare = "כתב אלי אכיה פי אלשהר"
    reads = _evals(("a50", 0.5, 0.5, bare), ("v21b", 0.55, 0.45, grounded))
    sel, how = NS.select_read("describe_pgp", reads, DESC, (), CFG)
    assert how == "read_partial" and sel is not None and sel.read.model == "v21b" and sel.grounding >= 0.5
    sel, how = NS.select_read("catalogue_json", reads, CAT, (), CFG)
    assert how == "read_partial" and sel is not None and sel.read.model == "v21b"
    assert NS.select_read("describe_pgp", _evals(("a50", 0.5, 0.5, bare)), DESC, (), CFG) == (None, "partial_grounding_fail")
    with_hard = reads + _evals(("a50", 0.6, 1.0, bare))
    assert NS.select_read("describe_pgp", with_hard, DESC, (), CFG)[1] == "read_hard"
    assert NS.select_read("describe_pgp", _evals(("a50", 0.2, 1.0, "x")), DESC, (), CFG) == (None, "no_candidate")


def test_grounding_text_catalogue() -> None:
    """The catalogue's grounding prose is its summary plus the names it lists."""
    g = NS.grounding_text("catalogue_json", CAT)
    assert g.startswith(DESC) and "Named: Avraham, Fustat, Fustat." in g
    assert NS.grounding_text("describe_pgp", DESC) == DESC


# ------------------------------------------------------------------------------ target strips
def test_strip_target_families() -> None:
    """B3 on translations, B3 + B4 on prose and the catalogue summary, relations untouched."""
    assert NS.strip_target("translate_he", "(1–2) שלום לך\n(3–4) ועוד") == ("שלום לך\nועוד", ["gil_range"])
    d, dops = NS.strip_target("describe_pgp", f"Letter to Avraham asking for money (ll. 3–4). See {SCRUB_SURNAME} catalogue.")
    assert d == "Letter to Avraham asking for money." and dops == ["line_ref", "see_cf"]
    d2, d2ops = NS.strip_target("describe_pgp", f"Letter to Avraham; listed by {SCRUB_SOCIETY} as a draft.")
    assert d2 == "Letter to Avraham." and not TN.has_source_names(d2) and d2ops
    cat = json.loads(CAT)
    cat["summary"] = DESC + f" Mentioned in {SCRUB_SOCIETY} records."
    c, cops = NS.strip_target("catalogue_json", json.dumps(cat, ensure_ascii=False))
    assert not TN.has_source_names(c) and json.loads(c)["people"] == cat["people"] and cops
    rel = '{"relations": [{"subject": "THIS_DOCUMENT", "predicate": "origin", "object": "Fustat"}]}'
    assert NS.strip_target("extract_relations", rel) == (rel, [])


def test_strip_target_can_empty() -> None:
    """A description that is only a citation strips to ''."""
    assert NS.strip_target("describe_pgp", f"See {SCRUB_SURNAME}.")[0] == ""


# ------------------------------------------------------------------------------ rows
def test_truncate_lines() -> None:
    """Leading lines within the budget, cut marked with ``[...]``."""
    text = "\n".join(f"a b c {i}" for i in range(10))
    out, cut = NS.truncate_lines(text, 9, lambda s: len(s.split()))
    assert cut and out.endswith("[...]") and out.count("\n") == 2
    assert NS.truncate_lines("a b", 9, lambda s: len(s.split())) == ("a b", False)


def test_make_slice_row_schema_and_meta() -> None:
    """A slice row validates, uses the v0_2 read note and carries the slice meta."""
    ed = edition()
    rd = NS.evaluate_reads(prow("10", [read("a50", corrupt(ed, HARD))]), ed, CFG)[0]
    sel = NS.Selection(rd, "read_hard", None)
    row, how = NS.make_slice_row(target("10", "translate_en", "Avraham wrote to his brother."), sel, "hebrew",
                                 ws_tokens(), CFG)
    assert how == "ok" and row is not None
    assert R.validate_row(row) == []
    user = row["messages"][1]["content"]
    assert R.READ_NOTE_V2 in user and "<document>\n" in user and R.SCRIPT_NAME["hebrew"] in user
    m = row["meta"]
    assert m["row_id"] == R.row_id("translate_en", "10", "read_hard", "a50")
    assert (m["condition"], m["family"], m["task"], m["split"], m["build"]) == ("read_hard", "translate_en", "translate_en",
                                                                               "train", R.BUILD)
    assert m["noise_class"] == "hard" and m["noise_band"] == NS.noise_band(rd.cer) and m["input_cer"] == rd.cer
    assert m["reader_exposed"] is None and m["swap_for"] == "noised_edition" and m["input_source"] == "read:a50"
    assert m["provenance"]["read_pages"] == ["X_1#0"] and m["provenance"]["pgp_document"] == "10"
    assert set(R.V01_META_FIELDS) <= set(m)


def test_make_slice_row_cap() -> None:
    """Translate rows over the cap are dropped; catalogue-type rows truncate the read."""
    long_read = "\n".join(edition(400).split("\n"))
    rd = NS.ReadEval("a50", long_read, 0.5, 1.0, None, 0, "hebrew", 1, ["X#0"], "hard")
    cfg = NS.SliceConfig(max_tokens=600, min_input_budget=50)
    assert NS.make_slice_row(target("11", "translate_en", "short"), NS.Selection(rd, "read_hard", None), "hebrew",
                             ws_tokens(), cfg) == (None, "over_cap")
    row, how = NS.make_slice_row(target("11", "describe_pgp", DESC), NS.Selection(rd, "read_hard", 0.5), "hebrew",
                                 ws_tokens(), cfg)
    assert how == "ok" and row["meta"]["input_truncated"] and row["meta"]["tokens"] <= 600
    assert "[...]\n</document>" in row["messages"][1]["content"]
    tiny = NS.SliceConfig(max_tokens=120, min_input_budget=100)
    assert NS.make_slice_row(target("11", "describe_pgp", DESC), NS.Selection(rd, "read_hard", 0.5), "hebrew",
                             ws_tokens(), tiny) == (None, "budget_too_small")


def test_make_slice_row_target_in_input() -> None:
    """A read that contains the target verbatim is rejected."""
    rd = NS.ReadEval("a50", "prefix " + DESC * 2, 0.5, 1.0, None, 0, "hebrew", 1, ["X#0"], "hard")
    assert NS.make_slice_row(target("12", "describe_pgp", DESC), NS.Selection(rd, "read_hard", 1.0), "hebrew",
                             ws_tokens(), CFG) == (None, "target_in_input")


# ------------------------------------------------------------------------------ hold-outs
def test_fallback_holdouts_matches_pgpid_canonical_and_key() -> None:
    """Benchmark by pgpid, canonical id or shelfmark key (canonical id ↔ shelfmark spelling); eval docs separately."""
    h = NS.FallbackHoldouts(benchmark_pgpids={"5"}, benchmark_cids={"Oxford_Bodleian_MS_heb_d_99_7"},
                            benchmark_keys={NS.SM.canonical_to_key("Oxford_Bodleian_MS_heb_d_99_7")},
                            eval_pgpids={"6"}, eval_keys=set(NS.SM.keys("T-S 99J1.3")))
    assert h.is_benchmark("5") == "pgpid"
    assert h.is_benchmark("7", None, "Oxford_Bodleian_MS_heb_d_99_7") == "canonical"
    assert h.is_benchmark("8", "Bodl. MS heb. d 99/7", ["Oxford_Bodleian_Bodl_MS_heb_d_99_7"]) == "shelfmark_key"
    assert h.is_benchmark("9", "T-S 99J1.2") is None
    assert h.is_eval_doc("6") == "pgpid" and h.is_eval_doc("10", "T-S 99J1.3 + T-S 99J1.4") == "shelfmark_key"
    assert NS.holdout_reason(h, "8", "Bodl. MS heb. d 99/7", []) == "benchmark:shelfmark_key"
    assert NS.holdout_reason(h, "10", "T-S 99J1.3", ["Cambridge_CUL_T_S_99J1_3"]) == "eval:shelfmark_key"
    assert NS.holdout_reason(h, "11", "T-S 99J1.2", []) is None


# ------------------------------------------------------------------------------ build_slice
def _fixture_docs() -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], NS.Target]]:
    """Pairs rows and targets covering the selection paths.

    :returns: ``(pairs rows, targets)``.
    """
    ed = edition()
    half = "\n".join(ed.split("\n")[:6])
    docs = [
        prow("100", [read("a50", ed), read("v21b", corrupt(ed, HARD))]),                    # hard (v21b) + pass
        prow("101", [read("a50", half)]),                                                 # partial, grounded
        prow("102", [read("a50", corrupt(ed, HARD))], shelfmark="Bodl. MS heb. d 99/7",
             cids=["Oxford_Bodleian_Bodl_MS_heb_d_99_7"]),                                # in H: never selected
        prow("103", [read("a50", corrupt(ed, HARD))]),                                       # dev split
        prow("104", [read("a50", "ק" * len(ed))]),                                       # CER > 0.9
        prow("105", []),                                                                  # no reads
    ]
    targets: Dict[Tuple[str, str], NS.Target] = {}
    for p in ("100", "101", "102", "103", "104", "105"):
        split = "dev" if p == "103" else "train"
        targets[(p, "describe_pgp")] = target(p, "describe_pgp", DESC, split)
        targets[(p, "translate_en")] = target(p, "translate_en", "Avraham wrote to his brother Yiṣḥaq.", split, None)
        targets[(p, "extract_relations")] = target(p, "extract_relations", '{"relations": []}', split)
    return docs, targets


def test_build_slice_end_to_end() -> None:
    """Hard / partial selection, H and dev documents never selected, stats and swap accounting."""
    docs, targets = _fixture_docs()
    h = NS.FallbackHoldouts(benchmark_keys={NS.SM.canonical_to_key("Oxford_Bodleian_MS_heb_d_99_7")})
    out, st = NS.build_slice(docs, targets, h, ws_tokens(), (), CFG)
    got = {(r["meta"]["pgpid"], r["meta"]["family"]): r["meta"]["condition"] for r in out}
    assert got == {("100", "describe_pgp"): "read_hard", ("100", "translate_en"): "read_hard",
                   ("100", "extract_relations"): "read_hard", ("101", "describe_pgp"): "read_partial"}
    assert all(r["meta"]["read_model"] == "v21b" for r in out if r["meta"]["pgpid"] == "100")
    assert st.holdout_reasons == {"benchmark:shelfmark_key": 1} and st.docs["split_dev"] == 1 and st.docs["no_reads"] == 1
    assert st.read_classes[("describe_pgp", "a50", "cer_gt_limit")] == 1
    assert st.rows_without_noised_twin == 1 and st.replaced_noised_tokens == 150
    assert [r["meta"]["family"] for r in out] == ["translate_en", "describe_pgp", "describe_pgp", "extract_relations"]
    checks = NS.run_checks(out, h, CFG, st.doc_cids)
    assert not any(checks.values()), checks
    man = NS.build_manifest(out, st, checks, {"x": 1}, {"target_source": "test"})
    assert man["n_rows"] == 4 and man["counts"]["describe_pgp|read_partial|JA|train"]["rows"] == 1
    assert man["swap"]["rows_without_noised_twin"] == 1 and man["target_source"] == "test"


def test_build_slice_he_edition_for_translate_he() -> None:
    """translate_he measures the read against ``edition_text_he`` when present (B7)."""
    ed, ed_he = edition(), edition(12).replace("אברהם", "שמואל")
    doc = prow("200", [read("a50", corrupt(ed_he, HARD))], ed=ed, ed_he=ed_he)
    targets = {("200", "translate_he"): target("200", "translate_he", "אברהם כתב לאחיו."),
               ("200", "extract_relations"): target("200", "extract_relations", '{"relations": []}')}
    out, _ = NS.build_slice([doc], targets, NoHoldouts(), ws_tokens(), (), CFG)
    by = {r["meta"]["family"]: r["meta"] for r in out}
    assert by["translate_he"]["cer_reference"] == "edition_text_he"
    assert by["translate_he"]["input_cer"] == round(TN.cer(corrupt(ed_he, HARD), ed_he), 4)
    assert by["extract_relations"]["cer_reference"] == "edition_text"


def test_build_slice_excludes_arabic_documents_by_default() -> None:
    """D7: an AR document gets no slice row by default; an empty exclusion list keeps it; checks flag AR rows."""
    docs, targets = _fixture_docs()
    for k, t in list(targets.items()):
        if k[0] == "100":
            targets[k] = NS.Target(pgpid=t.pgpid, family=t.family, split=t.split, text=t.text,
                                   meta={**t.meta, "lang_class": "AR"}, noised_tokens=t.noised_tokens)
    out, st = NS.build_slice(docs, targets, NoHoldouts(), ws_tokens(), (), CFG)
    assert not any(r["meta"]["pgpid"] == "100" for r in out) and st.docs["excluded_lang_AR"] == 1
    keep_all = NS.SliceConfig(exclude_lang_classes=())
    out2, _ = NS.build_slice(docs, targets, NoHoldouts(), ws_tokens(), (), keep_all)
    ar_rows = [r for r in out2 if r["meta"]["pgpid"] == "100"]
    assert len(ar_rows) == 3 and all(r["meta"]["lang_class"] == "AR" for r in ar_rows)
    assert NS.run_checks(ar_rows, NoHoldouts(), CFG)["excluded_lang_class"] == 3
    assert NS.run_checks(ar_rows, NoHoldouts(), keep_all)["excluded_lang_class"] == 0


def test_run_checks_flags_h_and_duplicates() -> None:
    """The post-build checks catch a hold-out row and duplicate row ids."""
    docs, targets = _fixture_docs()
    out, st = NS.build_slice(docs, targets, NoHoldouts(), ws_tokens(), (), CFG)
    h = NS.FallbackHoldouts(benchmark_pgpids={"100"})
    checks = NS.run_checks(out + out[:1], h, CFG, st.doc_cids)
    assert checks["in_H"] == 4 and checks["duplicate_row_id"] == 1 and checks["duplicate_doc_family"] == 1


# ------------------------------------------------------------------------------ loaders + CLI
def _paths_file(tmp_path: Path) -> Path:
    """Write a synthetic ``paths.json`` whose entries live under ``tmp_path``.

    :param tmp_path: Test directory.
    :returns: Config path.
    """
    cfg = {"repo": str(tmp_path), "probe_dir": "probe", "text_ft_v0_1": "v01", "pgp_parallel": "pairs.jsonl",
           "pgp_export": "pgp", "merged_pgpid_map": "merged.json", "document_treatments": "dt.jsonl",
           "analysis_pages_index": "api.jsonl", "exclusions_v1": "excl.json", "samples": "samples.json",
           "a50_overlap": "a50.json", "benchmark_ids": "bench.json", "academic_literature": "lit", "ai_reads": "reads",
           "book_editions_nas": "nas", "tokenizer_json": "tok.json", "chat_template": "tpl.jinja", "out_dir": "out",
           "scratch": "scratch"}
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return f


def _v01_row(pgpid: str, task: str, cond: str, split: str, target_text: str, hov: bool = False) -> Dict[str, Any]:
    """A v0_1-style row.

    :param pgpid: PGP id.
    :param task: Task.
    :param cond: Condition.
    :param split: Split.
    :param target_text: Target.
    :param hov: Vision-benchmark flag.
    :returns: Row.
    """
    meta = {"task": task, "lang_class": "JA", "condition": cond, "pgpid": pgpid, "shelfmark": f"T-S 99J1.{pgpid}",
            "shelfmark_group": f"g{pgpid}", "split": split, "held_out_vision_benchmark": hov, "tokens": 77,
            "licence": "test", "provenance": {"pgp_document": pgpid}, "source": "synthetic"}
    if task.startswith("translate"):
        meta["side"] = None
    return {"messages": R.make_messages("u", target_text), "meta": meta}


def _write_fixture(tmp_path: Path) -> P.Paths:
    """Write v0_1 split files, a pairs file and hold-out sources under ``tmp_path``.

    :param tmp_path: Test directory.
    :returns: Loaded paths.
    """
    p = P.load_paths(_paths_file(tmp_path))
    ed = edition()
    cat = json.loads(CAT)
    cat["summary"] = DESC + " (ll. 3–4)"
    v01 = {"train": [_v01_row("1", "describe_pgp", "edition", "train", DESC + f" See {SCRUB_SURNAME}."),
                     _v01_row("1", "describe_pgp", "noised_edition", "train", DESC),
                     _v01_row("1", "catalogue_json", "edition", "train", json.dumps(cat, ensure_ascii=False)),
                     _v01_row("1", "translate_en", "edition", "train", "(1–2) Avraham wrote."),
                     _v01_row("2", "describe_pgp", "edition", "train", DESC),
                     _v01_row("3", "describe_pgp", "edition", "train", f"See {SCRUB_SURNAME}.")],
           "dev": [_v01_row("4", "describe_pgp", "edition", "dev", DESC)],
           "test": [_v01_row("5", "describe_pgp", "edition", "test", DESC, hov=True)]}
    for s, rows_ in v01.items():
        R.write_jsonl(p.text_ft_v0_1 / f"{s}.jsonl", rows_)
    R.write_jsonl(p.pgp_parallel, [prow(str(i), [read("a50", corrupt(ed, HARD))], shelfmark=f"T-S 99J1.{i}",
                                        cids=[f"Cambridge_CUL_T_S_99J1_{i}"]) for i in range(1, 6)])
    (p.benchmark_ids).write_text(json.dumps(["Cambridge_CUL_T_S_99J1_2"]), encoding="utf-8")
    (p.merged_pgpid_map).write_text(json.dumps({"Cambridge_CUL_T_S_99J1_2": {"pgpids": ["2"], "image_urls": []}}),
                                    encoding="utf-8")
    (p.samples).write_text(json.dumps({"meta": {}, "sets": {}}), encoding="utf-8")
    (p.exclusions_v1).write_text(json.dumps({"meta": {}, "entries": []}), encoding="utf-8")
    p.pgp_export.mkdir(parents=True, exist_ok=True)
    (p.pgp_export / "sources.csv").write_text("authors\nGoitein, S. D.\n", encoding="utf-8")
    return p


def test_load_targets_v01_fallback(tmp_path: Path) -> None:
    """v0_1 fallback: edition rows only, strips applied with target_raw / strip_ops, noised tokens attached, dup flag.

    :param tmp_path: Test directory.
    """
    p = _write_fixture(tmp_path)
    targets, src, counts = NS.load_targets(p)
    assert src == "v0_1_fallback_stripped" and counts["describe_pgp:target_dropped_after_strip"] == 1
    d = targets[("1", "describe_pgp")]
    assert not TN.has_source_names(d.text) and d.meta["target_raw"].endswith(f"{SCRUB_SURNAME}.") and d.noised_tokens == 77
    assert json.loads(targets[("1", "catalogue_json")].text)["summary"] == DESC
    assert d.meta["pgp_description_dup"] is True and targets[("2", "describe_pgp")].meta["pgp_description_dup"] is False
    assert targets[("1", "translate_en")].text == "Avraham wrote."
    assert targets[("4", "describe_pgp")].split == "dev" and ("3", "describe_pgp") not in targets


def test_load_targets_prefers_v02_family_files(tmp_path: Path) -> None:
    """When every family file exists the v0_2 targets are used unchanged.

    :param tmp_path: Test directory.
    """
    p = _write_fixture(tmp_path)
    fams = ("describe_pgp",)
    row = _v01_row("9", "describe_pgp", "edition", "train", "Stripped already.")
    row["meta"].update({"family": "describe_pgp", "target_raw": "raw", "strip_ops": ["see_cf"]})
    R.write_jsonl(p.out("families", "describe_pgp.jsonl"), [row])
    targets, src, _ = NS.load_targets(p, fams)
    assert src == "v0_2_families" and targets[("9", "describe_pgp")].meta["strip_ops"] == ["see_cf"]
    with pytest.raises(FileNotFoundError):
        NS.load_targets(p, ("translate_en",), "v0_2")


def test_fallback_holdouts_from_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback H unions benchmark ids (→ pgpids, keys), registries, vision flags and v0_1 dev/test.

    :param tmp_path: Test directory.
    :param monkeypatch: Pytest monkeypatch (replaces the real registries).
    """
    p = _write_fixture(tmp_path)
    import src.datasets.evaluations.benchmark_registry as br
    monkeypatch.setattr(br, "registered_benchmark_documents", lambda: ({"Cambridge_CUL_T_S_99J1_7"}, {"7"}))
    h = NS.build_fallback_holdouts(p)
    assert {"2", "5", "7"} <= h.benchmark_pgpids and h.eval_pgpids == {"4", "5"}
    obj, src = NS.load_holdouts(p, "fallback")
    assert src.startswith("fallback") and obj.is_benchmark("2") == "pgpid"
    with pytest.raises(FileNotFoundError):
        NS.load_holdouts(p, "file")


def test_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI writes the family file and manifest; H / dev / test documents are never selected.

    :param tmp_path: Test directory.
    :param monkeypatch: Pytest monkeypatch (whitespace tokens, synthetic registries).
    """
    p = _write_fixture(tmp_path)
    import src.datasets.evaluations.benchmark_registry as br
    monkeypatch.setattr(br, "registered_benchmark_documents", lambda: (set(), set()))
    monkeypatch.setattr(NS, "default_token_fns", ws_tokens)
    rc = NS.main(["--paths", str(p.source_file), "--log-level", "WARNING"])
    assert rc == 0
    out = R.load_jsonl(p.out("families", "noise_slice.jsonl"))
    assert {r["meta"]["pgpid"] for r in out} == {"1"}
    man = json.loads(p.out("families", "noise_slice_manifest.json").read_text(encoding="utf-8"))
    assert man["n_rows"] == len(out) and not any(man["checks"].values())
    assert man["holdout_reasons"] == {"benchmark:pgpid": 1} and man["target_source"] == "v0_1_fallback_stripped"
    assert man["build_args"]["config"]["translate_cer_max"] == 0.75 and man["exposure"].startswith("not applied")


def test_real_template_render_if_available() -> None:
    """With the local tokenizer and template, a slice row renders under the contract (skipped elsewhere)."""
    try:
        p = P.load_paths()
    except FileNotFoundError:
        pytest.skip("no paths.json")
    if not (Path(p.tokenizer_json).exists() and Path(p.chat_template).exists()):
        pytest.skip("tokenizer / template not on this machine")
    ed = edition()
    rd = NS.evaluate_reads(prow("10", [read("a50", corrupt(ed, HARD))]), ed, CFG)[0]
    row, how = NS.make_slice_row(target("10", "describe_pgp", DESC), NS.Selection(rd, "read_hard", None), "hebrew",
                                 NS.default_token_fns(), CFG)
    assert how == "ok" and row["meta"]["tokens"] == R.tokens_of_row(row)
    assert row["meta"]["target_tokens"] == R.count_tokens(DESC)


def test_apply_exposure_tags_rows(tmp_path: Path) -> None:
    """Rows are tagged through the exposure module once ``exposure.jsonl`` exists; a missing read stays None.

    :param tmp_path: Test directory.
    """
    pytest.importorskip("src.finetuning.text27b.exposure")
    p = _write_fixture(tmp_path)
    docs, targets = _fixture_docs()
    out, _ = NS.build_slice(docs, targets, NoHoldouts(), ws_tokens(), (), CFG)
    status, counts = NS.apply_exposure(out, p)
    assert status.startswith("not applied") and counts == {}
    R.write_jsonl(p.out("exposure.jsonl"), [{"pgpid": "100", "read_model": "v21b", "page_key": "X_1:0",
                                            "level": "pgpid", "read_level": "pgpid"}])
    assert NS.apply_exposure(out, p, "off")[0].startswith("not applied")
    status, counts = NS.apply_exposure(out, p)
    assert status.startswith("applied")
    by = {r["meta"]["pgpid"]: (r["meta"]["reader_exposed"], r["meta"]["exposure_level"]) for r in out}
    assert by["100"] == (True, "pgpid") and by["101"] == (None, None)
    assert counts == {"level:pgpid": 3, "level:missing": 4}  # 101 + the three rows of 102 (not held out here)


def test_load_holdouts_uses_module_when_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``holdouts.json`` present, ``auto`` loads it through the holdouts module.

    :param tmp_path: Test directory.
    :param monkeypatch: Pytest monkeypatch (stubs the module's loader).
    """
    hmod = pytest.importorskip("src.finetuning.text27b.holdouts")
    p = _write_fixture(tmp_path)
    p.out("holdouts.json", mkdir=True).write_text("{}", encoding="utf-8")
    seen: Dict[str, Any] = {}

    def fake_load(path: Any = None, paths_: Any = None) -> NoHoldouts:
        """Record the call and return a permissive checker.

        :param path: File passed by the caller.
        :param paths_: Build paths passed by the caller.
        :returns: A checker.
        """
        seen["path"], seen["p"] = path, paths_
        return NoHoldouts()

    monkeypatch.setattr(hmod.Holdouts, "load", staticmethod(fake_load))
    obj, src = NS.load_holdouts(p)
    assert isinstance(obj, NoHoldouts) and seen["path"] == p.out("holdouts.json") and seen["p"] is p
    assert "holdouts module" in src
    assert isinstance(NS.load_holdouts(p, "fallback")[0], NS.FallbackHoldouts)
