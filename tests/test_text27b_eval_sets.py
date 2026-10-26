# File name: test_text27b_eval_sets.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``eval_sets``: helpers, the E1–E7 set builders, thresholds and the manifest CLI.

Everything runs on synthetic family rows, holdouts, treatments, reads and plans written to ``tmp_path``.
"""
import codecs
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.finetuning.text27b import eval_sets as E
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import scorer as S

EDITION = "כתאבי אליך מן אלאסכנדריה יום אלג׳מעה ווצל אלמרכב מן צקליה באלכתאן ואלפלפל"


def row(family: str, pgpid: str, split: str, condition: str = "edition", lang: str = "JA",
        read_model: Optional[str] = None, exposed: Optional[bool] = None, **meta: Any) -> Dict[str, Any]:
    """A synthetic v0_2 family row (texts are irrelevant to the set builders).

    :param family: Family.
    :param pgpid: Document id.
    :param split: Split.
    :param condition: Condition.
    :param lang: lang_class.
    :param read_model: Reader of a read row.
    :param exposed: ``reader_exposed``.
    :param meta: Extra meta fields.
    :returns: The row.
    """
    m = {"family": family, "task": family, "pgpid": pgpid, "split": split, "condition": condition, "lang_class": lang,
         "read_model": read_model, "reader_exposed": exposed, "build": R.BUILD, "tokens": 10,
         "row_id": R.row_id(family, pgpid, condition, read_model)}
    m.update(meta)
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"x {pgpid} {condition}"},
                         {"role": "assistant", "content": f"y {family} {pgpid}"}], "meta": m}


def refs_of(rows: List[Dict[str, Any]]) -> Dict[str, List[E.RowRef]]:
    """Group :class:`RowRef` by family.

    :param rows: Rows.
    :returns: ``{family: [RowRef]}``.
    """
    out: Dict[str, List[E.RowRef]] = {}
    for r in rows:
        ref = E.ref_from_row(r)
        out.setdefault(ref.family, []).append(ref)
    return out


# ------------------------------------------------------------------------------------- helpers
def test_helpers(tmp_path: Path) -> None:
    """Language cells, bands, seeded ranks, prompt version and file signatures.

    :param tmp_path: Pytest temporary directory.
    """
    assert E.base_lang("JA?") == "JA" and E.base_lang(None) == "UNKNOWN"
    assert E.band_of(0.0) == "0-0.15" and E.band_of(0.15) == "0-0.15" and E.band_of(0.151) == "0.15-0.3"
    assert E.band_of(0.9) == "0.6-0.9" and E.band_of(0.95) is None and E.band_of(None) is None
    assert E.seeded_rank(1, "a", 2) == E.seeded_rank(1, "a", 2) != E.seeded_rank(2, "a", 2)
    v = E.train_prompt_version()
    assert v.startswith("t27-prompt-") and len(v) == len("t27-prompt-") + 12 and v == E.train_prompt_version()
    f = tmp_path / "d" / "x.json"
    f.parent.mkdir()
    f.write_text("{}", encoding="utf-8")
    sig = E.file_sig(f, tmp_path)
    assert sig["path"] == str(Path("d") / "x.json") and sig["bytes"] == 2 and len(sig["sha256"]) == 64
    assert E.file_sig(tmp_path / "missing.json", tmp_path) == {"path": "missing.json", "exists": False}


def test_ref_from_row_and_loaders(tmp_path: Path) -> None:
    """v0_2 rows keep their row_id; v0_1 rows get one; v0_1 test splits map by the vision flag.

    :param tmp_path: Pytest temporary directory.
    """
    r = E.ref_from_row(row("translate_en", "5", "dev", "read", read_model="a50", exposed=True, input_cer=0.2))
    assert r.family == "translate_en" and r.reader_exposed is True and r.input_cer == 0.2 and r.read_model == "a50"
    v01 = {"messages": [], "meta": {"task": "catalogue_json", "pgpid": "7", "split": "test", "condition": "edition",
                                    "lang_class": "HE", "held_out_vision_benchmark": True}}
    ref = E.ref_from_row(v01, "catalogue_json")
    assert ref.row_id == R.row_id("catalogue_json", "7", "edition", None, None) and ref.held_out_vision
    fam = tmp_path / "families"
    fam.mkdir()
    R.write_jsonl(fam / "translate_en.jsonl", [row("translate_en", "1", "dev")])
    got = E.load_family_refs(fam)
    assert list(got) == ["translate_en"] and got["translate_en"][0].pgpid == "1"
    v01_dir = tmp_path / "v01"
    v01_dir.mkdir()
    R.write_jsonl(v01_dir / "test.jsonl", [v01, {"messages": [], "meta": dict(v01["meta"], pgpid="8",
                                                                              held_out_vision_benchmark=False)}])
    refs = E.load_v01_refs(v01_dir)
    assert sorted(x.split for x in refs["catalogue_json"]) == ["heldout_vision", "test_indist"]
    assert E.v01_split("dev", True) == "dev"


# ------------------------------------------------------------------------------------------ E1
def test_build_e1() -> None:
    """Exclusions drop (set, pgpid) rows; counts and reasons are kept."""
    samples = {"sets": {"S1_ja_en": {"task": "translate", "target": "English", "conditions": ["edition"],
                                     "entries": [{"pgpid": "1"}, {"pgpid": "2"}, {"pgpid": "3"}]}}}
    e1 = E.build_e1(samples, {("S1_ja_en", "2"): ["B1_header_leak"]})
    assert e1["counts"]["S1_ja_en"] == {"kept": 2, "excluded": 1}
    assert e1["items"]["S1_ja_en"]["pgpids"] == ["1", "3"]
    assert e1["items"]["S1_ja_en"]["excluded"] == {"2": ["B1_header_leak"]}


# ------------------------------------------------------------------------------------------ E2
def test_build_e2_top_up_twins_and_determinism() -> None:
    """Cells fill from test_indist, top up from heldout_vision, carry twins and are seeded."""
    rows = [row("translate_en", str(i), "test_indist", lang="AR") for i in range(3)]
    rows += [row("translate_en", str(i), "test_indist", "edition_rescript", lang="AR") for i in range(3)]
    rows += [row("translate_en", str(10 + i), "heldout_vision", lang="AR?") for i in range(5)]
    rows += [row("translate_en", "99", "train", lang="AR")]
    rows += [row("catalogue_json", str(20 + i), "test_indist", lang="HE") for i in range(4)]
    rows += [row("describe_pgp", str(20 + i), "test_indist", lang="HE") for i in range(4)]
    refs = refs_of(rows)
    targets = {("translate_en", "AR"): 5, ("catalogue_json", "HE"): 2, ("extract_relations", "ALL"): 3}
    e2 = E.build_e2(refs, targets, seed=7)
    ar = e2["cells"]["translate_en:AR"]
    assert ar["docs"] == 5 and ar["by_split"] == {"test_indist": 3, "heldout_vision": 2} and ar["shortfall"] == 0
    assert ar["available"] == {"test_indist": 3, "heldout_vision": 5} and ar["twin_rows"] == 3
    assert all("twin_row_ids" in it for it in e2["items"]["translate_en:AR"] if it["split"] == "test_indist")
    cat = e2["cells"]["catalogue_json:HE"]
    assert cat["docs"] == 2 and cat["twin_rows"] == 2
    assert e2["cells"]["extract_relations:ALL"]["docs"] == 0 and e2["cells"]["extract_relations:ALL"]["shortfall"] == 3
    assert E.build_e2(refs, targets, seed=7) == e2
    strict = E.build_e2(refs, targets, seed=7, top_up=())
    assert strict["cells"]["translate_en:AR"]["docs"] == 3 and strict["cells"]["translate_en:AR"]["shortfall"] == 2


def test_build_e2_read() -> None:
    """Vision-holdout read rows (heldout_vision, or heldout with the flag), exposed ones dropped."""
    refs = refs_of([row("translate_en", "1", "heldout_vision", "read", read_model="a50", input_cer=0.2),
                    row("translate_en", "2", "heldout_vision", "read", read_model="a50", exposed=True),
                    row("translate_en", "3", "test_indist", "read", read_model="a50"),
                    row("translate_en", "4", "heldout_vision", "noised_edition"),
                    row("translate_en", "5", "heldout", "read", read_model="a50", held_out_vision_benchmark=True),
                    row("translate_en", "6", "heldout", "read", read_model="a50")])
    e = E.build_e2_read(refs)
    assert [i["pgpid"] for i in e["items"]["translate_en"]] == ["1", "5"] and e["items"]["translate_en"][0]["band"] == "0.15-0.3"
    assert e["exposed_dropped"]["translate_en"] == 1 and e["counts"]["translate_en"] == {"JA": 2}
    assert e["by_split"]["translate_en"] == {"heldout_vision": 1, "heldout": 1}
    assert [i["split"] for i in e["items"]["translate_en"]] == ["heldout_vision", "heldout"]


def test_split_overrides_new_books_and_loaders(tmp_path: Path) -> None:
    """split_conflict reasons move rows; L5 rows group by family; book ids are hashed; analyse reads the L2 file.

    :param tmp_path: Pytest temporary directory.
    """
    ex = tmp_path / "excluded_rows.json"
    ex.write_text(json.dumps({"r1": "split_conflict:test_indist:book:bookx:17", "r2": "ngram_target:x#w8",
                              "r3": ["mention:y", "split_conflict:heldout:pgpid:9"], "r4": "split_conflict:bogus:z"}),
                  encoding="utf-8")
    assert E.split_overrides(ex) == {"r1": "test_indist", "r3": "heldout"}
    assert E.split_overrides(None) == {} and E.split_overrides(tmp_path / "missing.json") == {}
    fam = tmp_path / "families"
    fam.mkdir()
    book = [row("translate_en", "", "train", doc_id="book:bookx:17", row_id="r1"),
            row("translate_en", "", "train", "edition_rescript", doc_id="book:bookx:17", row_id="r1b"),
            row("translate_en", "", "test_indist", lang="AR", doc_id="book:bookx:19", row_id="r5"),
            row("translate_he", "", "dev", doc_id="book:bookx:20", row_id="r6"),
            row("translate_he", "", "heldout", doc_id="book:bookx:21", row_id="r7"),
            row("analyse_document", "33", "heldout", doc_id="book:bookx:22", row_id="r8")]
    R.write_jsonl(fam / "book_pairs.jsonl", book)
    nb = E.load_new_book_refs(fam, {"r1": "test_indist"})
    assert set(nb) == {"translate_en", "translate_he", "analyse_document"}
    moved = next(r for r in nb["translate_en"] if r.row_id == "r1")
    assert moved.split == "test_indist" and moved.split_file == "train" and moved.origin == "book_pairs"
    assert moved.doc == "book:bookx:17"
    e2nb = E.build_e2_new_books(nb)
    cell = e2nb["cells"]["translate_en:new_books"]
    assert cell["docs"] == 2 and cell["rows"] == 2 and cell["split_overridden_docs"] == 1
    assert e2nb["cells"]["translate_he:new_books"] == {"docs": 0, "rows": 0, "by_split": {}, "by_lang": {},
                                                        "split_overridden_docs": 0, "heldout_available": 1}
    assert "bookx" not in json.dumps(e2nb)
    assert E.doc_hash("123") == "123" and E.doc_hash("book:bookx:17").startswith("d") and len(E.doc_hash("x")) == 16
    assert E.load_new_book_refs(tmp_path / "nowhere") == {}
    # analyse_document comes from the L2 file when it exists
    R.write_jsonl(fam / "analyse_document.jsonl", [row("analyse_document", "1", "train")])
    assert E.family_file(fam, "analyse_document").name == "analyse_document.jsonl"
    R.write_jsonl(fam / "analyse_document_v2.jsonl", [row("analyse_document", "2", "dev"), row("analyse_document", "3", "test_indist")])
    assert E.family_file(fam, "analyse_document").name == "analyse_document_v2.jsonl"
    refs = E.load_family_refs(fam, ("analyse_document", "catalogue_json"))
    assert list(refs) == ["analyse_document"] and [r.pgpid for r in refs["analyse_document"]] == ["2", "3"]
    refs_all = [r for rs in list(refs.values()) + list(nb.values()) for r in rs]
    assert E.strongest_split(refs_all)["33"] == "heldout"
    assert E.strongest_split(refs_of([row("x", "5", "train"), row("x", "5", "dev"), row("x", "5", "train")])["x"])["5"] == "dev"


# ------------------------------------------------------------------------------------------ E3
def test_build_e3() -> None:
    """All dev translate rows, seeded samples of the rest, exposed reads removed, unknown exposure flagged."""
    rows = [row("translate_en", "1", "dev"), row("translate_en", "1", "dev", "read", read_model="a50", exposed=True),
            row("translate_he", "2", "dev", "read", read_model="v21b", exposed=False),
            row("translate_en", "3", "train")]
    rows += [row("catalogue_json", str(10 + i), "dev") for i in range(6)]
    rows += [row("extract_relations", str(20 + i), "dev", "read", read_model="a50") for i in range(3)]
    e3 = E.build_e3(refs_of(rows), seed=1, samples={"catalogue_json": 4, "extract_relations": 2})
    assert e3["counts"] == {"translate_en": 1, "translate_he": 1, "catalogue_json": 4, "extract_relations": 2}
    assert e3["exposed_removed"] == {"translate_en": 1}
    assert e3["exposure_unknown_read_rows"] == {"extract_relations": 3} and e3["status"] == "provisional"
    clean = E.build_e3(refs_of(rows[:3]), seed=1, samples={})
    assert clean["status"] == "ready" and clean["rows"] == 2


def test_build_e3_new_books_length_and_overrides() -> None:
    """L5 dev translate rows join; over-long rows and overridden rows are skipped and counted."""
    rows = [row("translate_en", "1", "dev"), row("translate_en", "2", "dev", tokens=7000),
            row("translate_en", "3", "train", row_id="moved")]
    refs = {"translate_en": [E.ref_from_row(r, origin="translate_en",
                                            split_override={"moved": "dev"}.get(r["meta"]["row_id"]))
                             for r in rows]}
    assert refs["translate_en"][2].split == "dev" and refs["translate_en"][2].split_file == "train"
    nb = {"translate_he": [E.ref_from_row(row("translate_he", "", "dev", doc_id="book:b:1", row_id="nb1"),
                                          origin="book_pairs"),
                           E.ref_from_row(row("translate_he", "", "test_indist", doc_id="book:b:2", row_id="nb2"),
                                          origin="book_pairs")]}
    e3 = E.build_e3(refs, seed=1, samples={}, new_book_refs=nb)
    assert e3["row_ids"]["translate_en"] == [rows[0]["meta"]["row_id"]] and e3["row_ids"]["translate_he"] == ["nb1"]
    assert e3["over_max_tokens"] == {"translate_en": 1} and e3["split_overridden"] == {"translate_en": 1}
    assert e3["origins"] == {"translate_en": {"translate_en": 1}, "translate_he": {"book_pairs": 1}}
    assert E.build_e3(refs, seed=1, samples={}, max_tokens=None)["counts"]["translate_en"] == 2


# ------------------------------------------------------------------------------------------ E4
def holdouts_dict() -> Dict[str, Any]:
    """A synthetic parsed holdouts.json.

    :returns: The dict.
    """
    return {"benchmark": {"11": ["pgp131"], "12": ["key:pgp131"], "13": ["group:pgp131"], "14": ["registry"],
                          "15": ["probe"]},
            "eval_docs": {"21": ["test"], "22": ["dev"], "23": ["group:test"]},
            "canonical": {"Inst_A_1": ["pgp131"]},
            "meta": {"pgp131_unmapped_resolution": [{"canonical_id": "Inst_B_d_6_2", "pgpids": ["12"]}]}}


def test_build_e4() -> None:
    """Canonical ids resolve through the pgpid map or the holdouts resolution; group members count."""
    merged = {"Inst_A_1": {"pgpids": ["11", "99"]}}
    e4 = E.build_e4(["Inst_A_1", "Inst_B_d_6_2", "Inst_C_9"], holdouts_dict(), merged, described={"11", "13"})
    items = {i["canonical_id"]: i for i in e4["items"]}
    assert items["Inst_A_1"]["pgpids"] == ["11"] and items["Inst_A_1"]["via"] == "pgpid_map"
    assert items["Inst_B_d_6_2"]["pgpids"] == ["12"] and items["Inst_B_d_6_2"]["via"] == "key_or_corpus"
    assert e4["canonical_without_pgpid"] == ["Inst_C_9"] and e4["pgpids"] == ["11", "12", "13"]
    assert e4["counts"]["pgpids_with_description"] == 2 and items["Inst_A_1"]["described_pgpids"] == ["11"]


# ------------------------------------------------------------------------------------------ E5
def test_build_e5_status_and_ids() -> None:
    """Targets need a clean span, a pgpid and 300 characters; status comes from the holdouts tiers."""
    base = {"book": "bookX", "part": "p", "boundary_clean": True, "intro_chars": 200, "notes_chars": 150, "pages": ["a"]}
    recs = [dict(base, doc_label="1", pgpids=["11"]), dict(base, doc_label="2", pgpids=["21"]),
            dict(base, doc_label="3", pgpids=["22"]), dict(base, doc_label="4", pgpids=["30"]),
            dict(base, doc_label="5", pgpids=[], merged_doc_ids=["M"]),
            dict(base, doc_label="6", pgpids=["21"], boundary_clean=False),
            dict(base, doc_label="7", pgpids=["21"], intro_chars=10, notes_chars=10),
            dict(base, doc_label="8", pgpids=["30"], merged_doc_ids=["M"])]
    e5 = E.build_e5(recs, {"M": {"pgpids": ["14"]}}, holdouts_dict())
    assert e5["counts"]["targets_by_status"] == {"heldout": 2, "test": 1, "dev": 1, "train": 1}
    assert e5["counts"]["excluded"] == 4
    st = {tuple(i["pgpids"]): i for i in e5["items"]}
    assert st[("14", "30")]["status"] == "heldout" and st[("22",)]["monitor_only"]
    assert all("bookX" not in json.dumps(i) for i in e5["items"]) and len(e5["items"][0]["id"]) == 16
    assert E.treatment_status(["30"], holdouts_dict(), {"30": "test_indist"}) == "test"
    assert E.treatment_status(["30"], holdouts_dict(), {}) == "not_in_build"
    assert E.treatment_status(["23"], holdouts_dict()) == "test"
    assert "analysis_rows" not in e5


def test_build_e5_attaches_analysis_rows() -> None:
    """Non-train analysis rows join the items sharing a pgpid; the rest are listed with hashed ids."""
    base = {"book": "bookX", "part": "p", "boundary_clean": True, "intro_chars": 400, "notes_chars": 0, "pages": ["a"]}
    recs = [dict(base, doc_label="1", pgpids=["11"]), dict(base, doc_label="2", pgpids=["22"])]
    analysis = [E.ref_from_row(row("analyse_document", "11", "heldout"), origin="analyse_document_v2"),
                E.ref_from_row(row("analyse_document", "11", "heldout", "read", read_model="a50", exposed=False),
                               origin="analyse_document_v2"),
                E.ref_from_row(row("analyse_document", "40", "train"), origin="analyse_document_v2"),
                E.ref_from_row(row("analyse_document", "", "test_indist", doc_id="book:bookX:3", row_id="b3"),
                               origin="book_pairs")]
    e5 = E.build_e5(recs, {}, holdouts_dict(), analysis_refs=analysis)
    item = next(i for i in e5["items"] if i["pgpids"] == ["11"])
    assert len(item["row_ids"]) == 2 and "row_ids" not in next(i for i in e5["items"] if i["pgpids"] == ["22"])
    assert e5["counts"]["items_with_rows"] == 1 and e5["counts"]["analysis_docs_outside_items"] == 1
    assert e5["counts"]["analysis_rows_by_split"] == {"heldout": 2, "test_indist": 1}
    outside = [d for d in e5["analysis_rows"] if not d["in_item"]]
    assert outside[0]["row_ids"] == ["b3"] and outside[0]["pgpid"] is None and "bookX" not in json.dumps(e5)


# ------------------------------------------------------------------------------------------ E6
def test_build_e6() -> None:
    """Only documents with at least one leak-safe page count, per family."""
    rag = {"leak_max": 0.05, "docs": [
        {"pgpid": "1", "held_out_vision_benchmark": True, "leak_safe_pages": {"describe_pgp": ["p1", "p2"],
                                                                            "translate_en": []}},
        {"pgpid": "2", "leak_safe_pages": {"describe_pgp": []}}]}
    e6 = E.build_e6(rag)
    assert e6["counts"] == {"docs": 1, "docs_per_family": {"describe_pgp": 1}}
    assert e6["items"] == [{"pgpid": "1", "held_out_vision_benchmark": True, "leak_safe_pages": {"describe_pgp": 2}}]


# ------------------------------------------------------------------------------------------ E7
def write_pairs(path: Path) -> None:
    """Pairs rows with reads of known quality.

    :param path: Destination.
    :returns: None
    """
    noisy = EDITION.replace("א", "ע").replace("ל", "ר")
    rows = [{"pgpid": "1", "edition_text": EDITION, "reads": [
                {"model": "a50", "text": EDITION}, {"model": "v21b", "text": noisy}]},
            {"pgpid": "2", "edition_text": EDITION, "reads": [{"model": "a50", "text": noisy}]},
            {"pgpid": "3", "edition_text": EDITION, "reads": [{"model": "a50", "text": EDITION[:10]}]},
            {"pgpid": "4", "edition_text": EDITION, "reads": [{"model": "a50", "text": ""}]}]
    R.write_jsonl(path, rows)


def test_exposure_reads_and_e7(tmp_path: Path) -> None:
    """Exposure picks the strongest level; exposed, empty and low-coverage reads are skipped; bands per family.

    :param tmp_path: Pytest temporary directory.
    """
    exp = tmp_path / "exposure.jsonl"
    R.write_jsonl(exp, [{"pgpid": "1", "read_model": "a50", "level": "none", "read_level": "none"},
                        {"pgpid": "1", "read_model": "a50", "level": "page", "read_level": "page"},
                        {"pgpid": "2", "read_model": "a50", "level": "none"}])
    lookup = E.load_exposure(exp)
    assert lookup[("1", "a50")] == "page" and lookup[("2", "a50")] == "none"
    assert E.load_exposure(None) is None
    pairs = tmp_path / "pairs.jsonl"
    write_pairs(pairs)
    reads = E.doc_reads(pairs, {"1", "2", "3", "4"}, lookup)
    r1 = {r["read_model"]: r for r in reads["1"]}
    assert r1["a50"]["cer"] == 0.0 and r1["a50"]["exposed"] and r1["v21b"]["exposed"] is False
    assert reads["4"][0]["empty"] and reads["3"][0]["coverage"] < 0.6
    best = E.best_read(reads["1"])
    assert best["read_model"] == "v21b"
    assert E.best_read(reads["3"]) is None and E.best_read(reads["4"]) is None
    refs = refs_of([row("translate_en", p, "heldout_vision") for p in ("1", "2", "3", "4")]
                   + [row("translate_en", "2", "heldout_vision", "read", read_model="a50"),
                      row("translate_en", "5", "test_indist"),
                      row("translate_en", "6", "heldout")])
    e7 = E.build_e7(refs, reads, exposure_known=True, families=("translate_en",))
    fam = e7["families"]["translate_en"]
    assert e7["status"] == "ready" and fam["docs_with_edition"] == 4 and fam["docs_banded"] == 2
    assert sum(fam["bands"].values()) == 2 and fam["plan_estimate"]["0-0.15"] == 12
    item2 = next(i for i in fam["items"] if i["pgpid"] == "2")
    assert item2["read_row_id"] == R.row_id("translate_en", "2", "read", "a50") and item2["split"] == "heldout_vision"
    assert fam["items_without_read_row"] == 1 and fam["by_split"] == {"heldout_vision": 2}
    # a flagged heldout document joins the pool (the PGP-131 documents), an unflagged one does not
    reads2 = dict(reads, **{"6": reads["2"], "7": reads["2"]})
    refs2 = refs_of([row("translate_en", "6", "heldout", held_out_vision_benchmark=True), row("translate_en", "7", "heldout")])
    fam2 = E.build_e7(refs2, reads2, exposure_known=True, families=("translate_en",))["families"]["translate_en"]
    assert [i["pgpid"] for i in fam2["items"]] == ["6"] and fam2["items"][0]["split"] == "heldout"
    assert E.build_e7(refs, reads, exposure_known=False, families=("translate_en",))["status"] == "provisional"


# ----------------------------------------------------------------------------------- thresholds
PLAN = """# Plan

Metrics: chrF++.
Bars (amendable): t27a success = S1 ≥ +5 (≥42.4,
target 45). Optional baseline arm: something.

**Eval E7.** text. Success: t27b's hard-band Δ smaller
than stock's.

- **E6 RAG:** text
- Promotion adds: verbatim continuation of ≥200 chars.
"""


def test_thresholds_verbatim_and_rebased(tmp_path: Path) -> None:
    """Verbatim paragraphs come from the plan; re-based bars from the scores summary.

    :param tmp_path: Pytest temporary directory.
    """
    v = E.extract_verbatim(PLAN)
    assert v["bars_s5"] == "Bars (amendable): t27a success = S1 ≥ +5 (≥42.4, target 45)."
    assert v["e7_success_s9_1"] == "Success: t27b's hard-band Δ smaller than stock's."
    assert v["promotion_s9_5"] == "Promotion adds: verbatim continuation of ≥200 chars."
    assert E.extract_verbatim("nothing")["bars_s5"] is None
    plan = tmp_path / "PLAN_2026-01-01.md"
    plan.write_text(PLAN, encoding="utf-8")
    summ = tmp_path / "summary_v2.json"
    summ.write_text(json.dumps({"scorer_version": "s", "base_system": "stock", "bars_rebased": [
        {"bar": "S1", "sample_set": "S1_ja_en", "condition": "edition", "stock_v2": 37.5, "bar_v2": 42.5, "n": 50,
         "offset": 5}]}), encoding="utf-8")
    t = E.thresholds(plan, summ)
    assert t["t27a"]["S1_ja_en_min_gain_over_stock"] == 5.0 and t["verbatim_source"] == plan.name
    assert t["rebased_v2"]["bars"][0] == {"bar": "S1", "sample_set": "S1_ja_en", "condition": "edition",
                                          "stock_v2": 37.5, "bar_v2": 42.5, "n": 50}
    assert "verbatim" not in E.thresholds(None, None)
    assert E.latest_plan(tmp_path) == plan and E.latest_plan(tmp_path / "x") is None


# ------------------------------------------------------------------------------------- manifest
REQUIRED = ["probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
            "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
            "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch"]


def write_fixture(tmp_path: Path, with_families: bool) -> Path:
    """Write a synthetic build tree and its paths.json.

    :param tmp_path: Pytest temporary directory (the repo root).
    :param with_families: Also write the family files, holdouts and exposure.
    :returns: The paths.json path.
    """
    cfg: Dict[str, Any] = {"repo": str(tmp_path)}
    cfg.update({k: f"data/{k}" for k in REQUIRED})
    cfg.update({"out_dir": "out/v0_2", "samples": "data/samples.json", "exclusions_v1": "data/excl.json",
                "pgp_parallel": "data/pairs.jsonl", "benchmark_ids": "data/bench.json",
                "merged_pgpid_map": "data/map.json", "document_treatments": "data/treat.jsonl"})
    d = tmp_path / "data"
    d.mkdir(parents=True)
    (d / "samples.json").write_text(json.dumps({"sets": {"S1_ja_en": {"task": "translate", "target": "English",
                                                                     "entries": [{"pgpid": "1"}]}}}), encoding="utf-8")
    (d / "excl.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    write_pairs(d / "pairs.jsonl")
    (d / "bench.json").write_text(json.dumps(["Inst_A_1"]), encoding="utf-8")
    (d / "map.json").write_text(json.dumps({"Inst_A_1": {"pgpids": ["11"]}}), encoding="utf-8")
    R.write_jsonl(d / "treat.jsonl", [{"book": "b", "part": "p", "doc_label": "1", "pgpids": ["21"], "pages": [],
                                       "boundary_clean": True, "intro_chars": 400, "notes_chars": 0}])
    out = tmp_path / "out" / "v0_2"
    out.mkdir(parents=True)
    (tmp_path / "out" / "PLAN_2026-01-01.md").write_text(PLAN, encoding="utf-8")
    (out / "rag_test_docs.json").write_text(json.dumps({"docs": [{"pgpid": "21", "leak_safe_pages": {
        "describe_pgp": ["x"]}}]}), encoding="utf-8")
    if with_families:
        fam = out / "families"
        fam.mkdir()
        for f in ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations"):
            R.write_jsonl(fam / f"{f}.jsonl", [row(f, "1", "heldout_vision"), row(f, "2", "heldout_vision"),
                                               row(f, "30", "test_indist"), row(f, "40", "dev"),
                                               row(f, "2", "heldout_vision", "read", read_model="a50", exposed=False)])
        (out / "holdouts.json").write_text(json.dumps(holdouts_dict()), encoding="utf-8")
        R.write_jsonl(out / "exposure.jsonl", [{"pgpid": "1", "read_model": "a50", "level": "page"}])
        write_canon(fam)
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return f


def canon_row(fam: str, sys_num: str, column: int, split: str, work: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """A canon-family row (one chunk of a KTIV manuscript).

    :param fam: ``canon_identify`` / ``canon_diff``.
    :param sys_num: Manuscript system number.
    :param column: Page column (chunk key part).
    :param split: Split.
    :param work: Canonical work.
    :param reason: Hold-out reason.
    :returns: The row.
    """
    meta = R.new_meta(fam, "edition", split, f"{sys_num}:{column}", work=work, sys_num=sys_num, fl="FL1",
                      column=column, part=1, heldout_reason=reason, tokens=50)
    return {"messages": R.make_messages("<document>\nטקסט\n</document>", "{}"), "meta": meta}


def write_canon(fam_dir: Path) -> None:
    """Canon family files: three held-out chunks (two Bavli, one Mishnah) and one train chunk.

    :param fam_dir: ``families`` directory.
    :returns: None.
    """
    spec = [("9901", 1, "heldout", "Talmud Bavli", "ktiv_benchmark:religious"),
            ("9901", 2, "heldout", "Talmud Bavli", "ktiv_benchmark:religious"),
            ("9902", 1, "heldout", "Mishnah", "gt_overlap"), ("9903", 1, "train", "Mishneh Torah", None)]
    for fam in ("canon_identify", "canon_diff"):
        R.write_jsonl(fam_dir / f"{fam}.jsonl", [canon_row(fam, *x) for x in spec])


def test_build_e8_canon(tmp_path: Path) -> None:
    """E8: every heldout canon row per family, work / reason counts, a paired stratified sample, no train overlap."""
    write_canon(tmp_path)
    e8 = E.build_e8(tmp_path, sample=2)
    assert e8["status"] == "ready" and e8["counts"]["rows"] == {"canon_identify": 3, "canon_diff": 3}
    assert e8["counts"]["manuscripts"] == 2 and e8["train_manuscripts_overlap"] == 0
    fi = e8["families"]["canon_identify"]
    assert fi["by_work"] == {"Mishnah": 1, "Talmud Bavli": 2} and fi["by_reason"] == {"gt_overlap": 1, "ktiv_benchmark": 2}
    assert fi["in_mixtures"] == [] and e8["families"]["canon_diff"]["in_mixtures"] == ["t27b"]
    samp = e8["sample"]
    assert samp["chunks"] == 3 and samp["by_work"] == {"Talmud Bavli": 2, "Mishnah": 1}  # ≥ 2 per work when available
    assert len(samp["row_ids"]["canon_identify"]) == len(samp["row_ids"]["canon_diff"]) == 3
    assert e8["scorer_version"] == S.CANON_SCORER_VERSION and e8["prompt_version"].startswith("canon-prompt-")
    rows = R.load_jsonl(tmp_path / "canon_diff.jsonl")
    rows[3]["meta"]["sys_num"] = "9901"  # a train chunk of a held manuscript is reported
    R.write_jsonl(tmp_path / "canon_diff.jsonl", rows)
    assert E.build_e8(tmp_path)["train_manuscripts_overlap"] == 1


def test_build_manifest_pending_then_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without family files E2/E3/E7 and E4/E5 are pending; with them every set is built.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    cfg = write_fixture(tmp_path, with_families=False)
    p = P.load_paths(cfg)
    m = E.build_manifest(p)
    assert m["status"]["E1_probe_clean"] == "ready" and m["status"]["E6_rag"] == "ready"
    assert m["status"]["E2_test_stratified"] == "pending" and m["status"]["E4_pgp131"] == "pending"
    assert m["sets"]["E4_pgp131"]["requires"] == [str(Path("out") / "v0_2" / "holdouts.json")]
    assert m["thresholds"]["verbatim"]["bars_s5"].startswith("Bars (amendable)")
    assert str(tmp_path) not in json.dumps(m)
    cfg2 = write_fixture(tmp_path / "full", with_families=True)
    p2 = P.load_paths(cfg2)
    m2 = E.build_manifest(p2)
    assert set(m2["status"].values()) == {"ready"}, m2["status"]
    e7 = m2["sets"]["E7_noise_bands"]["families"]["translate_en"]
    assert e7["docs_banded"] == 2 and {i["pgpid"]: i["read_model"] for i in e7["items"]} == {"1": "v21b", "2": "a50"}
    assert m2["sets"]["E4_pgp131"]["pgpids"] == ["11", "12", "13"]
    assert m2["sets"]["E5_excluded_treatments"]["counts"]["excluded"] == 1
    assert m2["sets"]["E2_read"]["counts"]["translate_en"] == {"JA": 1}
    assert all(s["exists"] for s in m2["sets"]["E2_test_stratified"]["source_files"])
    out = E.write_manifest(m2, p2.out("eval_v1.json"))
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "eval_v1"
    assert any(line.startswith("E7_noise_bands") for line in E.summary_lines(m2))
    assert m["status"]["E8_canon"] == "pending" and m2["sets"]["E8_canon"]["counts"]["rows"]["canon_diff"] == 3
    assert any(line.startswith("E8_canon") for line in E.summary_lines(m2))
    mat = E.materialise_dev_monitor(m2, p2)
    assert mat["rows"] == 4 and mat["path"] == str(Path("out") / "v0_2" / "dev" / "dev_monitor.jsonl")
    assert m2["sets"]["E3_dev_monitor"]["materialised"]["sha256"] == E.sha256_file(p2.out(E.DEV_MONITOR_FILE))
    monkeypatch.setenv(P.ENV_VAR, str(cfg2))
    E.main(["build", "--out", str(tmp_path / "cli.json")])
    cli = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert cli["status"]["E3_dev_monitor"] == "ready" and cli["sets"]["E3_dev_monitor"]["materialised"]["rows"] == 4
    assert all(r["meta"]["split"] == "dev" for r in R.read_jsonl(p2.out(E.DEV_MONITOR_FILE)))


# ---------------------------------------------------------------------------------- dev monitor
def test_dev_monitor_rows_and_problems(tmp_path: Path) -> None:
    """Rows come back in E3 order (translate first); every contract breach is reported.

    :param tmp_path: Pytest temporary directory.
    """
    fam = tmp_path / "families"
    fam.mkdir()
    R.write_jsonl(fam / "translate_en.jsonl", [row("translate_en", "1", "dev"), row("translate_en", "2", "train")])
    R.write_jsonl(fam / "catalogue_json.jsonl", [row("catalogue_json", "3", "dev")])
    R.write_jsonl(fam / "book_pairs.jsonl", [row("translate_en", "", "dev", doc_id="book:b:1", row_id="nb1")])
    rid = lambda f, p: R.row_id(f, p, "edition", None)  # noqa: E731
    e3 = {"row_ids": {"catalogue_json": [rid("catalogue_json", "3")], "translate_en": [rid("translate_en", "1"), "nb1"]}}
    rows = E.dev_monitor_rows(e3, fam)
    assert [r["meta"]["row_id"] for r in rows] == [rid("translate_en", "1"), "nb1", rid("catalogue_json", "3")]
    assert E.dev_monitor_problems(rows) == []
    with pytest.raises(ValueError):
        E.select_rows([fam / "translate_en.jsonl"], ["nope"])
    bad = [row("translate_en", "1", "train"), row("translate_en", "2", "dev", "read", read_model="a50", exposed=True),
           row("translate_en", "3", "dev", "read", read_model="a50"), row("translate_en", "4", "dev", tokens=9000),
           row("translate_en", "5", "dev", build="v0_1"), row("other", "6", "dev")]
    del bad[-1]["meta"]["reader_exposed"]
    bad.append(dict(bad[0], meta=dict(bad[0]["meta"], split="dev")))
    probs = " | ".join(E.dev_monitor_problems(bad))
    for needle in ("split 'train'", "reader-exposed read", "read row with reader_exposed=None", "tokens 9000",
                   "build 'v0_1'", "family 'other'", "no reader_exposed tag", "duplicate row_id"):
        assert needle in probs, needle
    # fix pass F3 / F4: NFC message contents and no project-name string anywhere in a dev row
    nfc_bad = row("translate_en", "7", "dev")
    nfc_bad["messages"][2]["content"] = "Fust\u0323a\u0304t\u0323"
    scrub_bad = row("translate_en", "8", "dev")
    scrub_bad["meta"]["target_raw"] = "A letter; images at the " + codecs.decode("Sevrqoret", "rot_13") + " site."
    probs = " | ".join(E.dev_monitor_problems([nfc_bad, scrub_bad]))
    assert "message content not NFC (assistant)" in probs and "project-name string in the row" in probs


# ------------------------------------------------------------------------------------------ Hub
def write_mixtures(out: Path) -> Dict[str, Any]:
    """Two synthetic materialised epochs and their mixture manifest under ``out/mixtures``.

    :param out: ``out_dir``.
    :returns: The manifest.
    """
    mix = out / "mixtures"
    mix.mkdir(parents=True)
    arms = {"t27a": [row("translate_en", "1", "train", mixture="t27a", target_raw="Pre-strip target, T-S 999.1"),
                     row("translate_he", "2", "train", mixture="t27a"),
                     row("translate_en", "", "train", source="book:secret_book_title", doc_id="book:secret_book_title:4",
                         row_id="nbt")],
            "t27b": [row("catalogue_json", "1", "train", target_raw='{"shelfmark": "T-S 999.1"}'),
                     row("domain_reading", "", "train", "literature", row_id="lit1"),
                     row("translate_en", "2", "train")]}
    manifest: Dict[str, Any] = {}
    for arm, rows_ in arms.items():
        f = mix / f"{arm}_epoch1.jsonl"
        R.write_jsonl(f, rows_)
        counts = Counter(f"{r['meta']['family']}|{r['meta']['condition']}|JA|train" for r in rows_)
        manifest[arm] = {"mixture": arm, "output": f"mixtures/{arm}_epoch1.jsonl", "output_sha256": E.sha256_file(f),
                         "fingerprint": R.fingerprint(rows_), "counts": {"rows": dict(counts), "tokens": {}},
                         "totals": {"rows": len(rows_), "tokens": 30}, "projected_steps": {"unpacked_steps": 1, "packed_steps": 1},
                         "components": {"translate_en": {"rows": 2, "tokens": 20, "share": 0.66}},
                         "noise_slice": {"share": 0.0}}
    (mix / "mixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_build_hub_layout_contract_and_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub folder has the harness layout, copies without ``meta.target_raw`` that keep the mixtures' text
    fingerprint and row-id order, its own sha256s, a clean card and verifiable checksums.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    from src.finetuning.qwen_hebrew.colab import derive_t27 as D
    cfg = write_fixture(tmp_path, with_families=True)
    # synthetic machine locations in the config: none may reach the card or the Hub manifests (rule 3)
    fake_mount, fake_models = tmp_path / "fake_mount", tmp_path / "home" / ".fake-model-store" / "fake-27b"
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    conf.update({"book_editions_nas": str(fake_mount / "book_editions"), "ai_reads": str(fake_mount / "reads"),
                 "tokenizer_json": str(fake_models / "tokenizer.json"),
                 "chat_template": str(fake_models / "chat_template.jinja")})
    cfg.write_text(json.dumps(conf), encoding="utf-8")
    p = P.load_paths(cfg)
    m = E.build_manifest(p)
    E.materialise_dev_monitor(m, p)
    dev_src = p.out(E.DEV_MONITOR_FILE)
    dev_rows = [dict(r, meta=dict(r["meta"], target_raw=f"pre-strip {i}")) for i, r in enumerate(R.load_jsonl(dev_src))]
    R.write_jsonl(dev_src, dev_rows)                     # the build's dev monitor keeps target_raw; the Hub copy does not
    write_mixtures(p.out_dir)
    summary = E.build_hub(p, m)
    hub = p.out("hub", E.HUB_NAME)
    assert summary["problems"] == [] and summary["hub"] == str(Path("out") / "v0_2" / "hub" / E.HUB_NAME)
    # the harness data cell downloads exactly these files
    assert (hub / D.DEV_FILE).exists() and E.DEV_MONITOR_FILE == D.DEV_FILE and E.HUB_NAME == D.LOCAL_LEAF
    sums = dict(reversed(line.split("  ", 1)) for line in (hub / E.SHA256_FILE).read_text(encoding="utf-8").splitlines())
    for arm in ("t27a", "t27b"):
        train, mix_file = hub / f"{arm}/train.jsonl", p.out("mixtures", f"{arm}_epoch1.jsonl")
        rows_, mix_rows = D.read_jsonl(train), R.load_jsonl(mix_file)
        assert any("target_raw" in r["meta"] for r in mix_rows)                          # positive control
        assert not any("target_raw" in r["meta"] for r in rows_)
        assert rows_ == [E.hub_row(r) for r in mix_rows] and train.read_bytes() != mix_file.read_bytes()
        assert [r["meta"]["row_id"] for r in rows_] == [r["meta"]["row_id"] for r in mix_rows]
        man = json.loads((hub / f"{arm}/mixture_manifest.json").read_text(encoding="utf-8"))
        assert D.text_fingerprint(rows_) == man["hub"]["text_fingerprint_sha256"] == man["fingerprint"]
        assert man["hub"]["sha256"] == E.sha256_file(train) == sums[f"{arm}/train.jsonl"] != man["output_sha256"]
        assert man["hub"]["source_sha256"] == man["output_sha256"] == E.sha256_file(mix_file)
        assert man["hub"]["row_ids_sha256"] == E.row_ids_digest(r["meta"]["row_id"] for r in mix_rows)
        assert man["hub"]["rows"] == len(mix_rows) and man["hub"]["dropped_meta"] == ["target_raw"]
        assert all(not D.row_problems(r, "train", D.ARMS[arm].families, R.CONDITIONS, R.BUILD) for r in rows_)
    dev = D.read_jsonl(hub / D.DEV_FILE)
    assert dev and all(not D.row_problems(r, "dev", R.FAMILIES, R.CONDITIONS, R.BUILD) for r in dev)
    assert dev == [E.hub_row(r) for r in dev_rows] and not any("target_raw" in r["meta"] for r in dev)
    assert summary["dev"]["sha256"] == sums[E.DEV_MONITOR_FILE] != E.sha256_file(dev_src)
    readme = (hub / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: other") and "t27a/train.jsonl" in readme
    hub_text = readme + "".join((hub / f"{a}/mixture_manifest.json").read_text(encoding="utf-8") for a in ("t27a", "t27b"))
    for forbidden in ("secret_book_title", str(tmp_path), "fake_mount", ".fake-model-store", "fake-27b", "localhost"):
        assert forbidden not in hub_text, forbidden
    assert sorted(sums) == sorted(
        ["README.md", "dev/dev_monitor.jsonl", "t27a/mixture_manifest.json", "t27a/train.jsonl",
         "t27b/mixture_manifest.json", "t27b/train.jsonl"])
    # a copy that carries target_raw again (same texts, same row ids) is caught
    leaked = [dict(r, meta=dict(r["meta"], target_raw="pre-strip")) for r in D.read_jsonl(hub / "t27a/train.jsonl")]
    R.write_jsonl(hub / "t27a/train.jsonl", leaked)
    probs = " | ".join(E.hub_problems(hub))
    assert "t27a: 3 rows with bad target_raw" in probs and "t27a: sha256 differs from its manifest" in probs
    assert "t27a: fingerprint / rows / row-id order differ" not in probs
    # a changed row-id order with the same texts is caught
    rows_b = D.read_jsonl(hub / "t27b/train.jsonl")
    rows_b[0]["meta"]["row_id"], rows_b[2]["meta"]["row_id"] = rows_b[2]["meta"]["row_id"], rows_b[0]["meta"]["row_id"]
    R.write_jsonl(hub / "t27b/train.jsonl", rows_b)
    assert "t27b: fingerprint / rows / row-id order differ from its manifest" in " | ".join(E.hub_problems(hub))
    # appended rows are caught
    with open(hub / "t27b/train.jsonl", "ab") as fh:
        fh.write(orjson_line(row("translate_en", "40", "train")))
    probs = " | ".join(E.hub_problems(hub))
    assert "checksum mismatch: t27b/train.jsonl" in probs and "t27b: fingerprint / rows / row-id order differ" in probs
    assert "t27b: 1 documents in train and the dev monitor" in probs
    # a dev copy that carries target_raw is caught
    R.write_jsonl(hub / E.DEV_MONITOR_FILE, dev_rows)
    assert f"dev: {len(dev_rows)} rows carry target_raw" in " | ".join(E.hub_problems(hub))
    # an epoch file that is not the one its manifest describes is refused
    mix = json.loads(p.out("mixtures", "mixture_manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        E.hub_arm("t27a", dict(mix["t27a"], fingerprint="0" * 64), p.out("mixtures"), tmp_path / "h2")
    with pytest.raises(ValueError):
        E.hub_arm("t27a", dict(mix["t27a"], output_sha256="0" * 64), p.out("mixtures"), tmp_path / "h2")
    monkeypatch.setenv(P.ENV_VAR, str(cfg))
    E.write_manifest(m, p.out(E.EVAL_FILE))
    E.main(["hub", "--out", str(tmp_path / "h3"), "--arms", "t27a"])
    assert (tmp_path / "h3" / "t27a" / "train.jsonl").exists() and not (tmp_path / "h3" / "t27b").exists()
    assert not any("target_raw" in r["meta"] for r in D.read_jsonl(tmp_path / "h3" / "t27a" / "train.jsonl"))


def test_hub_row_and_identity(tmp_path: Path) -> None:
    """``hub_row`` drops only ``meta.target_raw`` (input untouched); ``stream_identity`` sees texts, order and drops.

    :param tmp_path: Pytest temporary directory.
    """
    r = row("translate_en", "1", "train", target_raw="raw", strip_ops=["marker"])
    h = E.hub_row(r)
    assert "target_raw" not in h["meta"] and r["meta"]["target_raw"] == "raw" and h["meta"]["strip_ops"] == ["marker"]
    assert list(h["meta"]) == [k for k in r["meta"] if k != "target_raw"] and h["messages"] is r["messages"]
    assert E.hub_row({"messages": []}) == {"messages": []}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    R.write_jsonl(a, [r, row("translate_en", "2", "train")])
    assert E.write_hub_rows(a, b) == 2
    ia, ib = E.stream_identity(a), E.stream_identity(b)
    assert ia["rows_with_dropped"] == 1 and ib["rows_with_dropped"] == 0
    assert ia["text_fingerprint_sha256"] == ib["text_fingerprint_sha256"] and ia["row_ids"] == ib["row_ids"]
    assert ia["row_ids_sha256"] == E.row_ids_digest(ia["row_ids"]) != E.row_ids_digest(ia["row_ids"][::-1])
    assert E.sha256_file(a) != E.sha256_file(b)


def orjson_line(r: Dict[str, Any]) -> bytes:
    """One JSONL line.

    :param r: Row.
    :returns: Encoded line.
    """
    return (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")


def test_load_id_list(tmp_path: Path) -> None:
    """Both benchmark file shapes are read; anything else raises.

    :param tmp_path: Pytest temporary directory.
    """
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(["x", 2]), encoding="utf-8")
    b.write_text(json.dumps({"ids": ["y"]}), encoding="utf-8")
    c.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert E.load_id_list(a) == ["x", "2"] and E.load_id_list(b) == ["y"]
    with pytest.raises(ValueError):
        E.load_id_list(c)
