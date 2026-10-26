# File name: test_text27b_scorer.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``scorer`` (score_v2): cleaning, flags, bootstrap CIs, catalogue schema map,
grounding, relations, v0_2 row scoring and the ``rescore`` CLI on a synthetic probe directory.

Everything runs on synthetic strings and files in ``tmp_path``; no real outputs or references.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.finetuning.text27b import paths as P
from src.finetuning.text27b import scorer as S

EN_REF = ("In the name of God. I write to you from Alexandria on Friday; the ship from Sicily has arrived "
          "with the flax and the pepper, and the account is settled with the elder.")
HE_SRC = "כתאבי אליך מן אלאסכנדריה יום אלג׳מעה ווצל אלמרכב מן צקליה באלכתאן"
HE_TR = "מכתבי אליך מאלכסנדריה ביום שישי והגיעה הספינה מסיציליה עם הפשתן"


# ------------------------------------------------------------------------------ translation
def test_clean_translation_v1_vs_v2() -> None:
    """Fences and § lines are dropped in both modes; markers only in v2."""
    text = "```\n§ recto\n(1–2) In the name of God.\n\n(3−4) Greetings.\n```"
    assert S.clean_translation(text) == "In the name of God.\nGreetings."
    assert S.clean_translation(text, strip_markers=False) == "(1–2) In the name of God.\n(3−4) Greetings."
    assert S.clean_translation(None) == ""


def test_translation_metrics_identity_and_markers() -> None:
    """Identical text scores 100; a marker-only difference costs v1 but not v2."""
    m = S.translation_metrics(EN_REF, EN_REF, "English")
    assert m["chrfpp"] == pytest.approx(100.0) and not m["degenerate"] and m["len_ratio"] == pytest.approx(1.0)
    ref = "(1–3) " + EN_REF
    m2 = S.translation_metrics(EN_REF, ref, "en")
    assert m2["chrfpp"] == pytest.approx(100.0)
    assert m2["chrfpp_v1"] < 100.0


def test_translation_flags() -> None:
    """Empty, wrong script, length, repetition and copy-through flags."""
    assert S.translation_metrics("", EN_REF, "English")["empty"]
    ws = S.translation_metrics(HE_TR, EN_REF, "English")
    assert ws["wrong_script"] and ws["degenerate"]
    assert S.translation_metrics(EN_REF * 3, EN_REF, "English")["len_gt2"]
    assert S.translation_metrics("In the name of God.", EN_REF, "English")["len_lt05"]
    loop = EN_REF + " " + "of the month " * 60
    assert S.translation_metrics(loop, EN_REF, "English")["repetition"]
    copy = S.translation_metrics(HE_SRC, HE_TR, "Modern Hebrew", source=HE_SRC)
    assert copy["copy_through"] and copy["copy_share"] == pytest.approx(1.0) and not copy["wrong_script"]
    legit = S.translation_metrics(HE_TR, HE_TR, "Modern Hebrew", source=HE_SRC)
    assert not legit["copy_through"] and legit["copy_share"] < 0.5
    with pytest.raises(ValueError):
        S.target_lang("French")


def test_repetition_legit_repeat_is_not_a_loop() -> None:
    """A passage written twice (a letter copied on both sides) is not a loop; a long exact run is."""
    twice = EN_REF + "\n" + EN_REF
    assert not S.is_repetitive(S.repetition_stats(twice))
    assert S.is_repetitive(S.repetition_stats("... " * 200))
    assert S.repetition_stats("one two")["distinct4"] is None


def test_source_copy_share() -> None:
    """Share of hypothesis tokens copied from the source."""
    assert S.source_copy_share("a b", None) is None
    assert S.source_copy_share("", HE_SRC) is None
    assert S.source_copy_share("כתאבי אליך שלום", HE_SRC) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------- bootstrap
def test_segment_stats_sum_to_corpus() -> None:
    """Summed per-document statistics reproduce sacrebleu's corpus chrF++."""
    hyps = ["the cat sat on the mat", "a dog barked at night"]
    refs = ["the cat is on the mat", "the dog barked loudly at night"]
    st = S.segment_stats(hyps, refs)
    assert S._score_from_stats(S.CHRFPP, st.sum(axis=0)) == pytest.approx(S.CHRFPP.corpus_score(hyps, [refs]).score)
    assert S._score_from_stats(S.CHRFPP, st[0]) == pytest.approx(S.CHRFPP.sentence_score(hyps[0], [refs[0]]).score)


def test_bootstrap_ci_properties() -> None:
    """Perfect hypotheses give a degenerate CI at 100; the CI brackets the score and is seeded."""
    refs = [f"document number {i} about the flax trade in Alexandria" for i in range(12)]
    perfect = S.bootstrap_ci(refs, refs, n_boot=50)
    assert perfect["score"] == pytest.approx(100) and perfect["lo"] == pytest.approx(100)
    hyps = [r if i % 2 else r.replace("flax", "silk") for i, r in enumerate(refs)]
    a, b = S.bootstrap_ci(hyps, refs, n_boot=200, seed=1), S.bootstrap_ci(hyps, refs, n_boot=200, seed=1)
    assert a == b and a["lo"] <= a["score"] <= a["hi"] and a["n"] == 12
    assert S.bootstrap_ci([], [], n_boot=10)["score"] is None


def test_paired_bootstrap() -> None:
    """A better system gets a positive delta whose CI excludes 0; cached statistics give the same answer."""
    refs = [f"the ship from Sicily number {i} arrived with flax and pepper" for i in range(15)]
    good = list(refs)
    bad = [r.replace("Sicily", "nowhere").replace("pepper", "stones") for r in refs]
    pb = S.paired_bootstrap(good, bad, refs, n_boot=200)
    assert pb["delta"] > 0 and pb["excludes_0"] and pb["p_le0"] == 0.0 and pb["n"] == 15
    via_stats = S.paired_bootstrap_stats(S.segment_stats(good, refs), S.segment_stats(bad, refs), n_boot=200)
    assert via_stats == pb
    with pytest.raises(ValueError):
        S.paired_bootstrap(good, bad[:-1], refs)
    assert S.paired_bootstrap([], [], [], n_boot=5)["delta"] is None


# --------------------------------------------------------------------------------- catalogue
TRAIN_REC = {"document_type": "Letter", "languages": {"primary": ["Judaeo-Arabic"], "secondary": ["Hebrew"]},
             "date": {"original": "Tishri 1360 Sel.", "calendar": "Seleucid", "standard": "1048-09-20",
                      "year_ce_start": 1048, "year_ce_end": 1049, "inferred": True},
             "origin": "Fusṭāṭ", "destination": None, "places_mentioned": ["Alexandria", "Sicily"],
             "people": [{"name": "Nahray b. Nissim", "role": "recipient", "social_roles": []}],
             "tags": [], "summary": "Letter to Nahray b. Nissim about flax."}


def test_json_parsing() -> None:
    """Strict vs lenient parsing, fences, prose and a corrected second object."""
    obj = {"document_type": "Letter", "summary": "x"}
    raw = json.dumps(obj)
    assert S.strict_json(raw) == obj and S.strict_json("```json\n" + raw + "\n```") is None
    assert S.parse_json_lenient("```json\n" + raw + "\n```") == obj
    assert S.parse_json_lenient("Here it is: " + raw + " done") == obj
    two = '{"document_type": "Letter"} then corrected {"document_type": "Legal document"}'
    assert S.parse_json_lenient(two, "document_type") == {"document_type": "Legal document"}
    pred, strict = S.parse_catalogue("prose " + raw)
    assert pred == obj and not strict
    assert S.parse_catalogue("nothing", {"document_type": "Letter"}) == ({"document_type": "Letter"}, False)


def test_normalise_catalogue_schema_map() -> None:
    """Training schema maps onto the probe schema; probe records pass through."""
    n = S.normalise_catalogue(TRAIN_REC)
    assert (n["date_year_ce"], n["date_year_ce_max"], n["date_text"]) == (1048, 1049, "Tishri 1360 Sel.")
    assert n["origin_place"] == "Fusṭāṭ" and n["destination_place"] is None
    assert n["places"] == ["Alexandria", "Sicily"] and n["languages"] == ["Judaeo-Arabic", "Hebrew"]
    probe = {"document_type": "Letter", "date_year_ce": 1100, "origin_place": "Aden", "places": ["Aden"]}
    assert S.normalise_catalogue(probe)["date_year_ce"] == 1100
    assert S.normalise_catalogue(None)["people"] == []


def test_reference_from_gold_and_dates() -> None:
    """Gold dates become the reference interval with their stated / inferred source."""
    ref = S.reference_from_gold(TRAIN_REC)
    assert S.reference_date(ref) == {"lo": 1048, "hi": 1049, "source": "inferred"}
    assert ref["origin"] == "Fusṭāṭ" and ref["people"] == ["Nahray b. Nissim"]
    ok = S.date_metrics({"date_year_ce": 1050}, ref)
    assert ok["date_ok2"] and ok["date_err"] == pytest.approx(1.0)
    far = S.date_metrics({"date_year_ce": 1100, "date_year_ce_max": 1120}, ref)
    assert not far["date_ok10"] and far["pred_point"] == 1110
    assert S.date_metrics({}, ref)["date_abstained"]
    assert S.reference_date({"doc_date_standard": "1094-09-14/1095-09-02"}) == {"lo": 1094, "hi": 1095, "source": "doc"}
    assert S.reference_date({"doc_date_standard": "", "inferred_date_standard": ""}) is None


def test_places() -> None:
    """Aliases, regions, abstention and stated-without-reference outcomes."""
    assert S.place_metrics("Fusṭāṭ (Old Cairo), Egypt", "Fustat") == "correct"
    assert S.place_metrics("Egypt", "Fustat") == "region"
    assert S.place_metrics("Aden", "Fustat") == "wrong"
    assert S.place_metrics(None, "Fustat") == "abstained"
    assert S.place_metrics("Aden", "") == "noref_stated" and S.place_metrics("null", "") == "noref_null"


def test_people_recall_and_dictionary(tmp_path: Path) -> None:
    """Name normalisation drops particles and articles; the dictionary keeps longest matches.

    :param tmp_path: Pytest temporary directory.
    """
    assert S.name_norm("Abū l-Faraj b. Yosef") == S.name_norm("Abu al-Faraj ibn Yosef")
    pr = S.people_recall(["Abu al-Faraj ibn Yosef"], ["Abū l-Faraj b. Yosef", "Halfon b. Netanel"])
    assert pr["n_found"] == 1 and pr["recall"] == pytest.approx(0.5) and pr["missed"] == ["Halfon b. Netanel"]
    csv = tmp_path / "people.csv"
    csv.write_text("name\nHalfon b. Netanel\nHalfon\nNetanel ha-Levi b. Halfon\n", encoding="utf-8")
    d = S.load_people_dictionary(csv)
    assert "halfon" not in d
    assert S.dictionary_people("A letter of Halfon b. Netanel to his brother.", d) == ["Halfon b. Netanel"]


def test_name_grounding() -> None:
    """Short names match by 2-consonant skeletons; titles alone are kept; absent names fail."""
    idx = S.SourceIndex.of("מן אבו סעיד אלי נהראי בן נסים ואלריס")
    assert S.name_grounded("Abū Saʿīd", idx)
    assert S.name_grounded("Nahray b. Nissim", idx)
    assert S.name_grounded("al-Rayyis", idx)
    assert not S.name_grounded("Zakkur Barhun", idx)
    g = S.people_grounding(["Abū Saʿīd", "Zakkur Barhun"], "מן אבו סעיד")
    assert g["n"] == 2 and g["grounded"] == 1 and g["ungrounded"] == ["Zakkur Barhun"]
    assert S.people_grounding([], "x")["share"] is None


def test_candidate_grounded_matches_textnorm() -> None:
    """The per-candidate check gives the same share as textnorm.grounding."""
    from src.finetuning.text27b import textnorm as tn

    target = "A letter from Barhun to Zakkur about Qwertzan and the year 1048 CE."
    source = "ברהון כתב אלי זכור 1048"
    share, n, _ = tn.grounding(target, source)
    idx = S.SourceIndex.of(source)
    cands = tn.entity_candidates(target)
    mine = sum(1 for c in cands if S.candidate_grounded(c, idx)) / len(cands)
    assert n == len(cands) and mine == pytest.approx(share)


def test_specifics_split() -> None:
    """Candidates split into grounded / only in the reference / in neither."""
    sp = S.specifics_split("A letter from Barhun to Zakkur about Qwertzan.", "ברהון כתב",
                           "Zakkur is the addressee.")
    assert sp["grounded"] == 1 and sp["in_ref_only"] == 1 and sp["in_neither"] == 1
    assert sp["in_neither_list"] == ["qwertzan"] and sp["grounded_share"] == pytest.approx(1 / 3)
    assert S.specifics_split("nothing specific here.", "x")["grounded_share"] == 1.0


def test_catalogue_metrics_training_schema() -> None:
    """A training-schema prediction scored against a pairs-style reference."""
    ref = {"type": "Letter", "doc_date_standard": "", "inferred_date_standard": "1048", "origin": "Fustat",
           "destination": "", "location": "", "mentioned": ["Sicily", "Palermo"],
           "description": "Letter to Nahray b. Nissim about flax from Sicily."}
    m = S.catalogue_metrics(TRAIN_REC, ref, "כתאב אלי נהראי בן נסים", {"dict": ["Nahray b. Nissim"], "judge": None})
    assert m["parse_ok"] and m["type_ok"] and m["date_ok2"] and m["ref_date"]["source"] == "inferred"
    assert m["origin_outcome"] == "correct" and m["destination_outcome"] == "noref_null"
    assert m["mentioned_places_n"] == 2 and m["mentioned_places_found"] == 1
    assert m["people_dict"]["recall"] == 1.0 and m["people_judge"] is None
    assert m["people_grounding"]["share"] == 1.0
    bad = S.catalogue_metrics(None, ref)
    assert not bad["parse_ok"] and not bad["type_ok"] and bad["date_abstained"] and bad["people_grounding"] is None


# --------------------------------------------------------------------------------- relations
def test_relations_prf() -> None:
    """Greedy matching with name / place normalisation; unparseable answers count as no predictions."""
    gold = [{"subject": "THIS_DOCUMENT", "predicate": "sender", "object": "Abū l-Faraj b. Yosef"},
            {"subject": "THIS_DOCUMENT", "predicate": "origin", "object": "Fustat"},
            {"subject": "Abū l-Faraj b. Yosef", "predicate": "has_social_role", "object": "merchant"}]
    pred_text = json.dumps({"relations": [
        {"subject": "THIS_DOCUMENT", "predicate": "sender", "object": "Abu al-Faraj ibn Yosef"},
        {"subject": "THIS_DOCUMENT", "predicate": "origin", "object": "Fusṭāṭ (Old Cairo)"},
        {"subject": "THIS_DOCUMENT", "predicate": "recipient", "object": "Nahray"}]})
    r = S.relations_prf(S.parse_relations(pred_text), gold)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1) and r["f1"] == pytest.approx(2 / 3)
    none = S.relations_prf(S.parse_relations("not json"), gold)
    assert not none["parse_ok"] and none["recall"] == 0.0 and none["precision"] is None
    micro = S.micro_prf([r, none])
    assert micro["tp"] == 2 and micro["fn"] == 4 and micro["docs"] == 2 and micro["parse_ok"] == 1
    assert S.parse_relations('[{"subject": "a", "predicate": "b", "object": "c"}]') == [
        {"subject": "a", "predicate": "b", "object": "c"}]


# ---------------------------------------------------------------------------------- v0_2 rows
def make_row(family: str, gold: str, doc: str = HE_SRC, **meta: Any) -> Dict[str, Any]:
    """A synthetic v0_2 family row.

    :param family: Family name.
    :param gold: Assistant turn.
    :param doc: Document text inside the user turn.
    :param meta: Extra meta fields.
    :returns: The row.
    """
    m = {"family": family, "task": family, "row_id": f"{family}-1", "pgpid": "1", "condition": "edition",
         "lang_class": "JA", "split": "test_indist"}
    m.update(meta)
    return {"messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": f"Instruction.\n\nNote.\n\n<document>\n{doc}\n</document>"},
                         {"role": "assistant", "content": gold}], "meta": m}


def test_score_output_families() -> None:
    """Each family dispatches to its scorer; unknown families raise."""
    row = make_row("translate_en", EN_REF)
    assert S.document_of(row) == HE_SRC and S.gold_of(row) == EN_REF
    t = S.score_output(row, EN_REF)
    assert t["chrfpp"] == pytest.approx(100) and t["family"] == "translate_en" and t["row_id"] == "translate_en-1"
    he = S.score_output(make_row("translate_he", HE_TR), HE_SRC)
    assert he["copy_through"]
    cat = S.score_output(make_row("catalogue_json", json.dumps(TRAIN_REC)), "```json\n" + json.dumps(TRAIN_REC) + "\n```")
    assert cat["parse_ok"] and not cat["json_strict"] and cat["type_ok"] and cat["people_gold"]["recall"] == 1.0
    d = S.score_output(make_row("describe_pgp", "Letter to Nahray about flax."), "Letter to Nahray about flax.")
    assert d["chrfpp"] == pytest.approx(100) and not d["degenerate"]
    rel_gold = json.dumps({"relations": [{"subject": "THIS_DOCUMENT", "predicate": "origin", "object": "Fustat"}]})
    rel = S.score_output(make_row("extract_relations", rel_gold), rel_gold)
    assert rel["f1"] == 1.0
    with pytest.raises(ValueError):
        S.score_output(make_row("domain_reading", "x"), "x")


def test_aggregate_scored_cells() -> None:
    """Cells per (family, condition, lang_class) carry family-specific aggregates."""
    scored = [S.score_output(make_row("translate_en", EN_REF, row_id=f"t{i}"), EN_REF) for i in range(3)]
    scored += [S.score_output(make_row("extract_relations", json.dumps({"relations": []}), row_id="r1"), "{}")]
    scored += [S.score_output(make_row("catalogue_json", json.dumps(TRAIN_REC), row_id="c1"), json.dumps(TRAIN_REC))]
    cells = {c["family"]: c for c in S.aggregate_scored(scored, n_boot=20)}
    assert cells["translate_en"]["n"] == 3 and cells["translate_en"]["corpus_chrfpp"] == pytest.approx(100)
    assert cells["translate_en"]["degenerate_rate"] == 0.0
    assert cells["extract_relations"]["docs"] == 1
    assert cells["catalogue_json"]["json_strict"] == 1.0 and cells["catalogue_json"]["date_acc2"] == 1.0


# ------------------------------------------------------------------------------ probe rescore
REQUIRED = ["probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
            "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
            "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch"]


def write_probe_fixture(tmp_path: Path) -> P.Paths:
    """Write a tiny probe directory (outputs, samples, exclusions, pairs, PGP csvs) and its paths.json.

    :param tmp_path: Pytest temporary directory (used as the repo root).
    :returns: Loaded :class:`Paths`.
    """
    cfg: Dict[str, Any] = {"repo": str(tmp_path)}
    cfg.update({k: f"data/{k}" for k in REQUIRED})
    cfg.update({"probe_dir": "probe", "exclusions_v1": "probe/exclusions_v1.json", "samples": "probe/samples.json",
                "out_dir": "out/v0_2", "pgp_raw": "pgp"})
    probe = tmp_path / "probe"
    (probe / "outputs").mkdir(parents=True)
    (probe / "scores").mkdir()
    (tmp_path / "pgp").mkdir()
    (tmp_path / "pgp" / "people.csv").write_text("name\nNahray b. Nissim\n", encoding="utf-8")
    (tmp_path / "pgp" / "sources.csv").write_text("authors\n\"Scholar, A. B.\"\n", encoding="utf-8")
    pairs: List[Dict[str, Any]] = []
    for i in range(1, 7):
        pairs.append({"pgpid": str(i), "lang_class": "JA", "type": "Letter", "edition_text": HE_SRC + f" {i}",
                      "translation_en": f"(1–2) {EN_REF} Document {i}.", "translation_he": None,
                      "description": "Letter to Nahray b. Nissim about flax.", "doc_date_standard": "1048",
                      "inferred_date_standard": "", "origin": "Fustat", "destination": "", "location": "",
                      "mentioned": ["Sicily"], "reads": [{"model": "a50", "text": HE_SRC, "cer_vs_edition": 0.2,
                                                          "coverage": 0.9}]})
    (probe / "pgp_parallel_v1.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pairs) + "\n",
                                                 encoding="utf-8")
    samples = {"sets": {"S1_ja_en": {"task": "translate", "target": "English",
                                     "entries": [{"pgpid": str(i), "read_model": "a50"} for i in range(1, 7)]},
                        "C1_catalogue": {"task": "catalogue", "target": None,
                                         "entries": [{"pgpid": str(i)} for i in range(1, 7)]}}}
    (probe / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    (probe / "exclusions_v1.json").write_text(json.dumps({"entries": [
        {"sample_set": "S1_ja_en", "pgpid": "6", "reasons": ["F1_partial_reference"]}]}), encoding="utf-8")
    cat = {"document_type": "Letter", "date_year_ce": 1048, "date_year_ce_max": None, "origin_place": "Fusṭāṭ",
           "destination_place": None, "people": [{"name": "Nahray b. Nissim", "role": "recipient"}],
           "places": ["Sicily"], "summary": "Letter to Nahray b. Nissim about flax."}
    for system, good in (("stock", False), ("frontier", True)):
        rows = []
        for i in range(1, 7):
            text = f"{EN_REF} Document {i}." if good else f"In the name of God. Something else {i}."
            rows.append({"sample_set": "S1_ja_en", "pgpid": str(i), "task": "translate", "condition": "edition",
                         "system": system, "output_text": text, "output_json": None, "error": None})
            rows.append({"sample_set": "C1_catalogue", "pgpid": str(i), "task": "catalogue", "condition": "edition",
                         "system": system, "output_text": json.dumps(cat), "output_json": cat, "error": None})
        rows.append({"sample_set": "S1_ja_en", "pgpid": "1", "task": "translate", "condition": "edition",
                     "system": system, "output_text": None, "error": "timeout"})
        (probe / "outputs" / f"{system}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    desc = "Letter to Nahray b. Nissim about flax."
    fp = hashlib.sha1(f"{S.JUDGE_VERSION_V1}|{desc}".encode("utf-8")).hexdigest()
    (probe / "scores" / "judge_cache.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"kind": "ref_people", "pgpid": "1", "fingerprint": fp, "result": {"people": ["Nahray b. Nissim"]}},
        {"kind": "ref_people", "pgpid": "2", "fingerprint": "stale", "result": {"people": ["X"]}}]) + "\n",
        encoding="utf-8")
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return P.load_paths(f)


def test_rescore_end_to_end(tmp_path: Path) -> None:
    """The rescore CLI writes per-system scores, a summary with CIs and contrasts, and the baselines page.

    :param tmp_path: Pytest temporary directory.
    """
    p = write_probe_fixture(tmp_path)
    s = S.rescore(paths=p, base_system="stock", n_boot=50)
    out = p.out("scores_v2")
    assert (out / "stock.json").exists() and (out / "frontier.json").exists() and (out / "summary_v2.json").exists()
    md = p.out("BASELINES_v2.md").read_text(encoding="utf-8")
    assert "S1_ja_en" in md and "Paired contrasts" in md
    cells = {(c["system"], c["sample_set"]): c for c in s["clean"]["translation"]}
    assert cells[("frontier", "S1_ja_en")]["n"] == 5 and cells[("frontier", "S1_ja_en")]["corpus_chrfpp"] == pytest.approx(100)
    assert cells[("frontier", "S1_ja_en")]["corpus_chrfpp_v1"] < 100
    full = {(c["system"], c["sample_set"]): c for c in s["full"]["translation"]}
    assert full[("frontier", "S1_ja_en")]["n"] == 6
    contrast = s["clean"]["system_contrasts"][0]
    assert contrast["system"] == "frontier" and contrast["delta"] > 0 and contrast["excludes_0"]
    assert s["bars_rebased"][0]["bar_v2"] == pytest.approx(cells[("stock", "S1_ja_en")]["corpus_chrfpp"] + 5)
    rows = json.loads((out / "stock.json").read_text(encoding="utf-8"))
    assert not any(k.startswith("_") for r in rows for k in r)
    assert sum(1 for r in rows if r["exclusion_reasons"]) == 1
    cat = {c["system"]: c for c in s["clean"]["catalogue"]}["stock"]
    assert cat["people_judge"]["docs"] == 1 and cat["type_acc"] == 1.0 and cat["json_strict"] == 1.0


def test_load_ref_people_cache_fingerprint(tmp_path: Path) -> None:
    """Only cache records whose description fingerprint still matches are used.

    :param tmp_path: Pytest temporary directory.
    """
    p = write_probe_fixture(tmp_path)
    refs = S.load_references(p.probe_dir / "pgp_parallel_v1.jsonl", {"1", "2"})
    got = S.load_ref_people_cache(p.probe_dir / "scores" / "judge_cache.jsonl", refs)
    assert got == {"1": ["Nahray b. Nissim"]}
    assert S.load_ref_people_cache(None, refs) == {}


def test_repo_relative(tmp_path: Path) -> None:
    """Paths inside the repo become relative; outside paths keep only their name.

    :param tmp_path: Pytest temporary directory.
    """
    assert S.repo_relative(tmp_path / "a" / "b.json", tmp_path) == str(Path("a") / "b.json")
    assert S.repo_relative(Path("/elsewhere/c.json"), tmp_path) == "c.json"


def test_rebased_bars_and_contrasts_shape() -> None:
    """Bars are offsets from the base system's v2 score; missing cells are skipped."""
    cells = [{"system": "stock", "sample_set": "S1_ja_en", "condition": "edition", "corpus_chrfpp": 40.0,
              "corpus_chrfpp_v1": 39.5, "n": 10}]
    bars = S.rebased_bars(cells, "stock")
    assert len(bars) == 1 and bars[0]["bar_v2"] == 45.0 and bars[0]["stock_v1_recomputed"] == 39.5
    assert np.isclose(S._score_from_stats(S.CHRFPP, S.segment_stats(["x"], ["x"])[0]), 100.0)


def test_token_found_short_names_and_possessives() -> None:
    """Short skeletons, possessives and generic calendar vocabulary."""
    idx = S.SourceIndex.of("כתב יוסף ואברהם אלי יעקב")
    assert S.token_found("yosef", idx) and S.token_found("avrahams", idx) and S.token_found("yaqub", idx)
    assert not S.token_found("zakkur", idx) and not S.token_found("1048", idx)
    assert S.is_generic("judaeo arabic") and S.is_generic("iyyar") and not S.is_generic("iyyar yosef")
    sp = S.specifics_split("It is dated Thursday in Iyyar by Yosef.", "כתב יוסף")
    assert sp["generic_skipped"] >= 1 and sp["in_neither"] == 0


# ------------------------------------------------------------------------------ canon (E8)
def test_parse_canon_ref_and_same_section() -> None:
    """Bavli daf-amud, chapter and whole-section ranges parse; same section means same title and overlapping range."""
    assert S.parse_canon_ref("Niddah 27b:16-28a:7") == {"title": "niddah", "start": (27, 1), "end": (28, 0)}
    assert S.parse_canon_ref("Yevamot 110b:6-12")["end"] == (110, 1)
    assert S.parse_canon_ref("Mishneh Torah, Foundations of the Torah 4:11")["title"] == "mishneh torah, foundations of the torah"
    assert S.parse_canon_ref("Pirkei Avot 1:14–2:2")["end"] == (2, 0)  # en dash
    assert S.parse_canon_ref("Berakhot 4a-5b") == {"title": "berakhot", "start": (4, 0), "end": (5, 1)}
    assert S.parse_canon_ref("nonsense") is None and S.parse_canon_ref(None) is None
    assert S.same_section("Niddah 28a:1-3", "Niddah 27b:16-28a:7")
    assert not S.same_section("Niddah 29a:1", "Niddah 27b:16-28a:7")
    assert not S.same_section("Sotah 28a:1", "Niddah 27b:16-28a:7")
    assert S.norm_ref(" Niddah  27b:16 – 28a:7 ") == "niddah 27b:16-28a:7"


def test_canon_identify_metrics() -> None:
    """Exact reference, same section, anchor F1 and over-quoting of a recited span."""
    gold = {"work": "Talmud Bavli", "title": "Niddah", "he_title": "נדה", "ref": "Niddah 27b:16-28a:7",
            "he_ref": "נדה כז ע״ב", "anchor": "היה וטמאו לו פתחים גדולים וטהרו"}
    exact = S.canon_identify_metrics(json.dumps(gold, ensure_ascii=False), gold)
    assert exact["json_ok"] and exact["ref_exact"] and exact["same_section"] and exact["anchor_f1"] == 1.0
    near = dict(gold, ref="Niddah 28a:2-9", anchor="הָיָה וטמאו לו " + "מלה " * 20)  # points ignored, 23 words
    m = S.canon_identify_metrics("```json\n" + json.dumps(near, ensure_ascii=False) + "\n```", gold)
    assert not m["ref_exact"] and m["same_section"] and m["title_ok"] and m["over_quoted"]
    assert 0 < m["anchor_f1"] < 1
    bad = S.canon_identify_metrics("I think it is the Talmud.", gold)
    assert not bad["json_ok"] and not bad["same_section"] and bad["anchor_f1"] == 0.0
    cells = S.aggregate_canon_identify([exact, m, bad])
    assert cells["n"] == 3 and cells["ref_exact"] == pytest.approx(1 / 3) and cells["same_section"] == pytest.approx(2 / 3)


def test_canon_diff_metrics_prf() -> None:
    """Variant multiset P / R / F1 against the apparatus: typed, untyped (type ignored), strict (pos too), by type."""
    gold = {"canonical_ref": "Niddah 27b:16-28a:7", "agreement_ratio": 0.8, "lacuna_words": 2, "variants": [
        {"type": "orthographic", "fragment": "היא", "canon": "היה", "pos": "1:1"},
        {"type": "substitution", "fragment": "קטנים", "canon": "גדולים", "pos": "1:5"},
        {"type": "omission", "fragment": "", "canon": "וקא", "pos": "2:1"},
        {"type": "orthographic", "fragment": "היא", "canon": "היה", "pos": "3:1"}]}
    perfect = S.canon_diff_metrics(json.dumps(gold, ensure_ascii=False), gold)
    assert perfect["typed"]["f1"] == 1.0 and perfect["strict"]["f1"] == 1.0 and perfect["ref_exact"]
    assert perfect["agreement_abs_err"] == 0.0 and perfect["lacuna_abs_err"] == 0.0
    pred = {"canonical_ref": "Niddah 27b:16-28a:7", "agreement_ratio": 0.7, "lacuna_words": 2, "variants": [
        {"type": "orthographic", "fragment": "הִיא", "canon": "היה", "pos": "1:1"},          # points ignored
        {"type": "orthographic", "fragment": "קטנים", "canon": "גדולים", "pos": "1:5"},     # wrong type
        {"type": "addition", "fragment": "גם", "canon": "", "pos": "2:4"}]}                  # spurious
    m = S.canon_diff_metrics(json.dumps(pred, ensure_ascii=False), gold)
    assert m["typed"] == {"tp": 1, "fp": 2, "fn": 3, "precision": 1 / 3, "recall": 0.25, "f1": pytest.approx(2 / 7)}
    assert m["untyped"]["tp"] == 2 and m["strict"]["tp"] == 1
    assert m["by_type"]["substitution"]["fn"] == 1 and m["by_type"]["addition"]["fp"] == 1
    assert m["agreement_abs_err"] == pytest.approx(0.1) and m["n_pred"] == 3 and m["n_gold"] == 4
    none = S.canon_diff_metrics("no json", gold)
    assert not none["json_ok"] and none["typed"]["recall"] == 0.0 and none["typed"]["precision"] is None
    empty = S.canon_diff_metrics(json.dumps({"variants": []}), {"variants": []})
    assert empty["typed"]["f1"] == 1.0
    agg = S.aggregate_canon_diff([perfect, m, none])
    assert agg["n"] == 3 and agg["typed"]["tp"] == 5 and agg["typed"]["fn"] == 7
    with pytest.raises(ValueError):
        S.variant_keys([], "fuzzy")


def test_score_output_routes_canon_families() -> None:
    """score_output picks the canon metrics by family and aggregate_scored groups them."""
    gold_i = {"work": "Mishnah", "title": "Pirkei Avot", "he_title": "אבות", "ref": "Pirkei Avot 2:9-13",
              "he_ref": "אבות ב", "anchor": "אמר להם צאו וראו"}
    gold_d = {"canonical_ref": "Pirkei Avot 2:9-13", "agreement_ratio": 1.0, "lacuna_words": 0, "variants": []}
    rows = [{"messages": [{"role": "user", "content": "<document>\nאמר\n</document>"},
                          {"role": "assistant", "content": json.dumps(g, ensure_ascii=False)}],
             "meta": {"family": fam, "row_id": fam, "condition": "edition", "lang_class": "HE", "split": "heldout",
                      "work": "Mishnah"}} for fam, g in (("canon_identify", gold_i), ("canon_diff", gold_d))]
    scored = [S.score_output(r, r["messages"][1]["content"]) for r in rows]
    assert scored[0]["ref_exact"] and scored[0]["work"] == "Mishnah" and scored[1]["typed"]["f1"] == 1.0
    cells = {c["family"]: c for c in S.aggregate_scored(scored)}
    assert cells["canon_identify"]["ref_exact"] == 1.0 and cells["canon_diff"]["typed"]["f1"] is None


def test_canon_type_alias_and_renamed_type() -> None:
    """Prep pass 2026-10-26: ``orthographic_or_minor`` is the type; a legacy ``orthographic`` label scores as it."""
    assert "orthographic_or_minor" in S.CANON_VARIANT_TYPES and "orthographic" not in S.CANON_VARIANT_TYPES
    gold = {"variants": [{"type": "orthographic_or_minor", "fragment": "היא", "canon": "היה", "pos": "1:1"}]}
    for label in ("orthographic", "Orthographic_or_minor", "orthographic or minor"):
        pred = {"variants": [{"type": label, "fragment": "היא", "canon": "היה", "pos": "1:1"}]}
        m = S.canon_diff_metrics(json.dumps(pred, ensure_ascii=False), gold)
        assert m["typed"]["f1"] == 1.0 and m["by_type"]["orthographic_or_minor"]["tp"] == 1, label
    assert S.CANON_SCORER_VERSION.startswith("canon_v2")
