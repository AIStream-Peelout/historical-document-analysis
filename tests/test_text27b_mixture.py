# File name: test_text27b_mixture.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for :mod:`src.finetuning.text27b.mixture` (t27a / t27b epoch materialisation).

Every test builds synthetic family files in ``tmp_path`` (a tiny word-level tokenizer and a small
Jinja template with the Qwen3.8 thinking behaviour stand in for the real ones). One test also renders
through the real local Qwen3.8 template / tokenizer named in ``paths.json`` and is skipped elsewhere.
"""
import codecs
import ast
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.text27b import mixture as M
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R

#: Same thinking behaviour as the Qwen3.8 template: effort sentence unless thinking is off; one empty think block.
GOOD_TEMPLATE = (
    "{%- if enable_thinking is undefined or enable_thinking is true %}{%- set effort = 'Reasoning effort is set to xhigh.' %}"
    "{%- else %}{%- set effort = '' %}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role == 'system' %}{{- '<|im_start|>system\\n' + (effort + '\\n\\n' if effort else '') + m.content|trim + '<|im_end|>\\n' }}"
    "{%- elif m.role == 'user' %}{{- '<|im_start|>user\\n' + m.content|trim + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content|trim + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}")
#: Injects the effort sentence even with thinking off (must fail the render check).
EFFORT_TEMPLATE = GOOD_TEMPLATE.replace("{%- set effort = '' %}", "{%- set effort = 'Reasoning effort is set to low.' %}")
CONFIG_KEYS = ["probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
               "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
               "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch"]


# -------------------------------------------------------------------------------------- fixtures
def mk(family: str, pgpid: Any, condition: str, split: str = "train", lang: str = "JA",
       read_model: Optional[str] = None, target: Optional[str] = None, tokens: Optional[int] = 100,
       **extra: Any) -> Dict[str, Any]:
    """Build one synthetic v0_2 row.

    :param family: Row family.
    :param pgpid: Document id (page key for literature rows).
    :param condition: Input condition.
    :param split: Split.
    :param lang: ``lang_class``.
    :param read_model: Reader for read-like rows.
    :param target: Assistant text (default: one per family and document, shared across conditions).
    :param tokens: ``meta.tokens`` (None leaves it out so the mixture counts it).
    :param extra: Further meta fields.
    :returns: The row.
    """
    target = target if target is not None else f"target of {family} doc {pgpid}"
    user = f"prompt {family} {pgpid} {condition} {read_model or 'none'}"
    fields: Dict[str, Any] = {"lang_class": lang, "pgpid": None if family == "domain_reading" else str(pgpid),
                              "shelfmark": f"T-S X{pgpid}", "shelfmark_group": f"g{pgpid}",
                              "held_out_vision_benchmark": False, "target_chars": len(target), "target_tokens": 10,
                              "input_cer": None, "input_source": condition, "input_truncated": False,
                              "licence": "synthetic", "provenance": {}, "source": "synthetic"}
    if tokens is not None:
        fields["tokens"] = tokens
    fields.update(extra)
    meta = R.new_meta(family, condition, split, pgpid, read_model, **fields)
    if tokens is None:
        meta.pop("tokens", None)
    return {"messages": R.make_messages(user, target, system="sys"), "meta": meta}


def doc_rows(family: str, pgpid: Any, conditions: Sequence[str] = ("edition", "noised_edition", "read"),
             **kw: Any) -> List[Dict[str, Any]]:
    """Rows of one document in several conditions (reads get reader ``a50``).

    :param family: Row family.
    :param pgpid: Document id.
    :param conditions: Conditions to emit.
    :param kw: Passed to :func:`mk`.
    :returns: Rows.
    """
    return [mk(family, pgpid, c, read_model="a50" if c in R.READ_LIKE_CONDITIONS and c != "noised_edition" else None,
               **kw) for c in conditions]


def write_families(root: Path, files: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Write family files ``<root>/families/<stem>.jsonl``.

    :param root: Output root (``out_dir``).
    :param files: ``{stem: rows}``.
    :returns: The families directory.
    """
    fdir = root / "families"
    fdir.mkdir(parents=True, exist_ok=True)
    for stem, rows in files.items():
        R.write_jsonl(fdir / f"{stem}.jsonl", rows)
    return fdir


def make_tokenizer(tmp_path: Path) -> Path:
    """Save a tiny word-level tokenizer (one token per whitespace-separated word).

    :param tmp_path: Pytest temporary directory.
    :returns: ``tokenizer.json`` path.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "sys": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    out = tmp_path / "tokenizer.json"
    tok.save(str(out))
    return out


def make_template(tmp_path: Path, text: str = GOOD_TEMPLATE, name: str = "chat_template.jinja") -> Path:
    """Write a chat template.

    :param tmp_path: Pytest temporary directory.
    :param text: Template source.
    :param name: File name.
    :returns: Template path.
    """
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def base_t27b_files() -> Dict[str, List[Dict[str, Any]]]:
    """Minimal t27b family set: the five required PGP families with a few train documents.

    :returns: ``{stem: rows}``.
    """
    files: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in range(1, 4):
        files["translate_en"] += doc_rows("translate_en", d)
        files["translate_he"] += doc_rows("translate_he", 100 + d)
    for d in range(10, 20):
        for fam in ("catalogue_json", "describe_pgp", "extract_relations"):
            files[fam] += doc_rows(fam, d)
    return dict(files)


def build(tmp_path: Path, name: str, files: Dict[str, List[Dict[str, Any]]], render: bool = True,
          template: str = GOOD_TEMPLATE, **overrides: Any) -> Dict[str, Any]:
    """Write families, build one mixture and return its manifest section plus the written rows.

    :param tmp_path: Pytest temporary directory.
    :param name: Mixture name.
    :param files: Family files.
    :param render: Run the render check.
    :param template: Template source.
    :param overrides: :func:`mixture.default_config` overrides.
    :returns: ``{"section", "rows", "out"}``.
    """
    out = tmp_path / "out"
    fdir = write_families(out, files)
    tpl = make_template(tmp_path, template)
    tok = make_tokenizer(tmp_path)
    cfg = M.default_config(name, **overrides)
    section = M.build_mixture(cfg, fdir, out, tpl, tok, run_render_check=render)
    rows = R.load_jsonl(out / "mixtures" / f"{name}_epoch{cfg.epoch}.jsonl")
    return {"section": section, "rows": rows, "out": out}


def item(row: Dict[str, Any], src: Optional[str] = None) -> M.Item:
    """Wrap a row as a pool item.

    :param row: Row.
    :param src: Source file stem (default: the row's family).
    :returns: The item.
    """
    return M.Item(row=row, src=src or row["meta"]["family"], uid=row["meta"]["row_id"] + ":" + row["messages"][1]["content"],
                  tokens=row["meta"].get("tokens"))


# ----------------------------------------------------------------------------------------- basics
def test_unit_hash_and_pick_weighted() -> None:
    """Hash uniforms are deterministic and in [0, 1); weighted picks follow cumulative weights."""
    u = M.unit_hash(3407, "a", 1)
    assert u == M.unit_hash(3407, "a", 1) and 0.0 <= u < 1.0
    assert u != M.unit_hash(3408, "a", 1)
    opts = [("edition", 0.4), ("read", 0.4), ("noised", 0.2)]
    assert M.pick_weighted(0.0, opts) == "edition"
    assert M.pick_weighted(0.39, opts) == "edition"
    assert M.pick_weighted(0.41, opts) == "read"
    assert M.pick_weighted(0.85, opts) == "noised"
    assert M.pick_weighted(0.7, [("edition", 0.4), ("noised", 0.2)]) == "noised"  # renormalised: 0.4/0.6 cut
    assert M.pick_weighted(0.6, [("edition", 0.4), ("noised", 0.2)]) == "edition"
    with pytest.raises(ValueError):
        M.pick_weighted(0.5, [("edition", 0.0)])


def test_doc_key_and_cond_class() -> None:
    """Document keys prefer pgpid; conditions map onto the three classes."""
    assert M.doc_key({"pgpid": "7", "shelfmark": "x"}) == "pgpid:7"
    assert M.doc_key({"pgpid": None, "doc_id": "n1"}) == "doc_id:n1"
    assert M.doc_key({"shelfmark": "T-S 1"}) == "shelfmark:T-S 1"
    assert M.doc_key({"row_id": "ab"}) == "row:ab"
    assert M.cond_class("edition_rescript") == "edition"
    assert M.cond_class("read_hard") == "noised" and M.cond_class("read_partial") == "noised"
    assert M.cond_class("literature") is None


def test_default_config_values_and_validation() -> None:
    """Plan defaults: t27a translate ×2 without slice; t27b weights 0.4/0.4/0.2, relations 0.4; bad input rejected."""
    a = M.default_config("t27a")
    assert a.repeats == {"translate_en": 2, "translate_he": 2, "book_pairs": 2} and not a.include_slice and a.one_per_doc == ()
    assert M.BOOK_PAIRS_FILE in a.files and M.BOOK_PAIRS_FILE not in a.required
    b = M.default_config("t27b", repeats={"analyse_document": 3})
    assert b.repeats["analyse_document"] == 3 and b.repeats["translate_en"] == 2  # merged, not replaced
    assert b.class_weights == {"edition": 0.4, "read": 0.4, "noised": 0.2}
    assert b.subsample == {"extract_relations": 0.4} and b.seed == 3407 and b.slice_policy == "single"
    with pytest.raises(ValueError):
        M.default_config("t27c")
    with pytest.raises(ValueError):
        M.default_config("t27b", slice_policy="sometimes")
    with pytest.raises(ValueError):
        M.default_config("t27b", class_weights={"bogus": 1.0})


# ------------------------------------------------------------------------------------------- t27a
def test_t27a_every_translate_row_twice_in_separate_halves(tmp_path: Path) -> None:
    """t27a: every train translate row (incl. edition_rescript) exactly twice, copy 0 in the first half."""
    files = {"translate_en": doc_rows("translate_en", 1) + doc_rows("translate_en", 2)
             + [mk("translate_en", 3, "edition", lang="AR"), mk("translate_en", 3, "edition_rescript", lang="AR")]
             + doc_rows("translate_en", 9, split="dev"),
             "translate_he": doc_rows("translate_he", 4) + doc_rows("translate_he", 8, split="test_indist"),
             "catalogue_json": doc_rows("catalogue_json", 5)}
    res = build(tmp_path, "t27a", files)
    rows, sec = res["rows"], res["section"]
    ids = Counter(r["meta"]["row_id"] for r in rows)
    train_ids = {r["meta"]["row_id"] for stem in ("translate_en", "translate_he") for r in files[stem]
                 if r["meta"]["split"] == "train"}
    assert set(ids) == train_ids and set(ids.values()) == {2}
    assert any(r["meta"]["condition"] == "edition_rescript" for r in rows)
    assert all(r["meta"]["family"] in ("translate_en", "translate_he") for r in rows)
    half = len(rows) // 2
    assert {r["meta"]["row_id"] for r in rows[:half]} == train_ids
    assert all(r["meta"]["mix_copy"] == 0 for r in rows[:half]) and all(r["meta"]["mix_copy"] == 1 for r in rows[half:])
    # a translate document with edition + noised + read shares one target → 6 occurrences
    assert sec["repeats_per_target"]["histogram"] == {"4": 1, "6": 3}
    assert sec["gates"]["ok"], sec["gates"]
    assert sec["load"]["dropped_split"] == {"dev": 3, "test_indist": 3}


def test_t27a_takes_l5_translate_pairs_twice(tmp_path: Path) -> None:
    """t27a: L5 translate pairs (train) enter ×2 as ``book_pairs``; their analysis and dev rows do not; the
    file stays optional (absent → listed as missing, build still succeeds)."""
    files = {"translate_en": doc_rows("translate_en", 1), "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path / "a", "t27a", files)
    assert "families/book_pairs.jsonl" in res["section"]["load"]["missing_inputs"] and res["section"]["gates"]["ok"]
    files["book_pairs"] = (doc_rows("translate_en", "n1", conditions=("edition", "noised_edition"), doc_id="n1")
                           + doc_rows("translate_he", "n2", conditions=("edition",), doc_id="n2", split="dev")
                           + doc_rows("analyse_document", "n3", conditions=("edition",), doc_id="n3"))
    res2 = build(tmp_path / "b", "t27a", files)
    bp = [r for r in res2["rows"] if r["meta"]["mix_component"] == "book_pairs"]
    assert len(bp) == 4 and {r["meta"]["pgpid"] for r in bp} == {"n1"} and {r["meta"]["mix_repeats"] for r in bp} == {2}
    assert res2["section"]["load"]["dropped"]["family_not_in_mixture"] == 1
    assert res2["section"]["load"]["dropped_split"] == {"dev": 1}
    assert res2["section"]["components"]["book_pairs"]["rows"] == 4 and res2["section"]["gates"]["ok"]


def test_t27a_ignores_noise_slice(tmp_path: Path) -> None:
    """The noise slice is never part of t27a (the no-slice baseline), even if a slice row sits in a translate file."""
    files = {"translate_en": doc_rows("translate_en", 1) + [mk("translate_en", 1, "read_hard", read_model="a50")],
             "translate_he": doc_rows("translate_he", 2),
             "noise_slice": [mk("translate_he", 2, "read_hard", read_model="v21b")]}
    res = build(tmp_path, "t27a", files)
    assert not any(r["meta"]["condition"] in M.SLICE_CONDITIONS for r in res["rows"])
    assert res["section"]["load"]["dropped"]["slice_not_in_mixture"] == 1
    assert "noise_slice.jsonl" in res["section"]["load"]["ignored_files"]
    assert sum(1 for r in res["rows"] if r["meta"]["condition"] == "noised_edition") == 4


# ------------------------------------------------------------------------------------------ splits
def test_eval_splits_and_vision_flag_never_in_mixture(tmp_path: Path) -> None:
    """dev / test_indist / heldout / heldout_vision / probe rows and vision-flagged train rows are dropped."""
    files = base_t27b_files()
    for i, split in enumerate(("dev", "test_indist", "heldout", "heldout_vision", "probe")):
        files["translate_en"] += doc_rows("translate_en", 500 + i, split=split)
        files["catalogue_json"] += doc_rows("catalogue_json", 600 + i, split=split)
    files["describe_pgp"].append(mk("describe_pgp", 700, "edition", held_out_vision_benchmark=True))
    files["domain_reading"] = [mk("domain_reading", f"work/p{i:03d}", "literature", lang="EN",
                                  split="probe" if i == 0 else "train") for i in range(5)]
    res = build(tmp_path, "t27b", files)
    rows, sec = res["rows"], res["section"]
    assert {r["meta"]["split"] for r in rows} == {"train"}
    assert not any(r["meta"]["pgpid"] in {str(x) for x in range(500, 701)} for r in rows if r["meta"]["pgpid"])
    assert sec["load"]["dropped"]["held_out_vision_flag_in_train"] == 1
    assert sec["load"]["dropped_split"]["probe"] == 7  # 3 translate + 3 catalogue rows + 1 domain_reading page
    assert sum(1 for r in rows if r["meta"]["family"] == "domain_reading") == 4
    assert sec["gates"]["checks"]["train_split_only"]["pass"]


# -------------------------------------------------------------------------------------------- swap
def test_swap_rule_replaces_noised_copy_of_same_doc_family(tmp_path: Path) -> None:
    """A slice row replaces the noised copy of its (doc, family) only; without a noised copy it is kept and counted."""
    files = base_t27b_files()
    files["noise_slice"] = [mk("translate_en", 1, "read_hard", read_model="v21b", noise_band="0.45-0.6"),
                            mk("translate_he", 999, "read_hard", read_model="a50")]
    files["translate_he"] += doc_rows("translate_he", 999, conditions=("edition",))
    res = build(tmp_path, "t27b", files)
    rows, sec = res["rows"], res["section"]
    pairs = Counter((r["meta"]["pgpid"], r["meta"]["family"], r["meta"]["condition"]) for r in rows)
    assert pairs[("1", "translate_en", "noised_edition")] == 0
    assert pairs[("1", "translate_en", "read_hard")] == 2  # swapped in, ×2 like its family
    assert pairs[("2", "translate_en", "noised_edition")] == 2  # other docs keep their noised copy
    assert pairs[("101", "translate_he", "noised_edition")] == 2
    assert pairs[("999", "translate_he", "read_hard")] == 2
    swap = sec["selection"]["swap"]
    assert swap["noised_removed"] == 1 and swap["pairs_swapped"] == 1 and swap["slice_without_noised"] == 1
    assert swap["slice_rows"] == 2
    assert sec["gates"]["checks"]["swap_no_noised_beside_slice"]["pass"]
    assert sec["noise_slice"]["rows"] == 4


def test_apply_swap_direct() -> None:
    """``apply_swap`` removes exactly the noised rows of (doc, family) pairs that have slice rows."""
    its = [item(r) for r in doc_rows("describe_pgp", 1) + doc_rows("catalogue_json", 1) + doc_rows("describe_pgp", 2)]
    its.append(item(mk("describe_pgp", 1, "read_partial", read_model="a50")))
    kept, st = M.apply_swap(its)
    conds = Counter((it.meta["pgpid"], it.family, it.condition) for it in kept)
    assert conds[("1", "describe_pgp", "noised_edition")] == 0
    assert conds[("1", "catalogue_json", "noised_edition")] == 1
    assert conds[("2", "describe_pgp", "noised_edition")] == 1
    assert st["by_family"]["describe_pgp"] == {"noised_removed": 1, "slice_rows": 1, "pairs_swapped": 1}


# ------------------------------------------------------------------------------- one-condition rule
def test_one_row_per_document_across_catalogue_and_describe(tmp_path: Path) -> None:
    """catalogue ∪ describe: one row per document; a document with a slice row gets that slice row (policy single)."""
    files = base_t27b_files()
    files["noise_slice"] = [mk("describe_pgp", 12, "read_partial", read_model="a50"),
                            mk("catalogue_json", 13, "read_hard", read_model="v21b")]
    res = build(tmp_path, "t27b", files)
    rows, sec = res["rows"], res["section"]
    cd = [r for r in rows if r["meta"]["family"] in M.ONE_PER_DOC_FAMILIES]
    per_doc = Counter(r["meta"]["pgpid"] for r in cd)
    assert set(per_doc) == {str(d) for d in range(10, 20)} and set(per_doc.values()) == {1}
    chosen = {r["meta"]["pgpid"]: r["meta"]["condition"] for r in cd}
    assert chosen["12"] == "read_partial" and chosen["13"] == "read_hard"
    st = sec["selection"]["rules"]["one_per_doc"]
    assert st["docs_in_pool"] == 10 and st["rows_chosen"] == 10 and st["classes"]["slice"] == 2
    assert all(r["meta"]["mix_rule"] == "one_per_doc" and r["meta"]["mix_repeats"] == 1 for r in cd)
    assert sec["gates"]["checks"]["one_row_per_doc_catalogue_describe"]["pass"]


def test_noised_class_policy_slice_competes_in_noised_class() -> None:
    """With ``noised_class`` the slice row is only chosen when the noised class is drawn."""
    cfg = M.default_config("t27b", slice_policy="noised_class")
    slice_hits = noised_class = 0
    for d in range(600):
        its = [item(r) for r in doc_rows("describe_pgp", d, conditions=("edition", "read"))]
        its.append(item(mk("describe_pgp", d, "read_hard", read_model="v21b")))
        it, cls = M.choose_one(f"pgpid:{d}", its, cfg.seed, "catalogue_describe", cfg.class_weights, cfg.slice_policy)
        slice_hits += it.condition == "read_hard"
        noised_class += cls == "slice"
    assert slice_hits == noised_class
    assert 0.13 < slice_hits / 600 < 0.27  # noised weight 0.2


def test_class_weights_realised_and_families_balanced() -> None:
    """Over many documents the realised classes follow 0.4 / 0.4 / 0.2 and both families are used equally."""
    cfg = M.default_config("t27b")
    its = [item(r) for d in range(3000) for fam in M.ONE_PER_DOC_FAMILIES for r in doc_rows(fam, d)]
    chosen, st = M.select_one_per_doc(its, cfg, "catalogue_describe")
    n = len(chosen)
    assert n == 3000 and st["docs_without_selectable_row"] == 0
    shares = {c: st["classes"][c] / n for c in ("edition", "read", "noised")}
    assert abs(shares["edition"] - 0.4) < 0.035 and abs(shares["read"] - 0.4) < 0.035
    assert abs(shares["noised"] - 0.2) < 0.03
    assert abs(st["families"]["catalogue_json"] / n - 0.5) < 0.04


def test_class_weights_renormalised_over_present_classes() -> None:
    """Read-only documents always get their read; edition + noised documents split ≈ 2 : 1."""
    cfg = M.default_config("t27b")
    ro = [item(mk("catalogue_json", d, "read", read_model="a50")) for d in range(50)]
    chosen, st = M.select_one_per_doc(ro, cfg, "catalogue_describe")
    assert st["classes"] == {"read": 50}
    en = [item(r) for d in range(2000) for r in doc_rows("describe_pgp", d, conditions=("edition", "noised_edition"))]
    _, st2 = M.select_one_per_doc(en, cfg, "catalogue_describe")
    assert abs(st2["classes"]["edition"] / 2000 - 2 / 3) < 0.04
    lit = [item(mk("describe_pgp", 1, "literature"))]
    assert M.choose_one("pgpid:1", lit, 1, "x", cfg.class_weights) == (None, None)
    zero = M.default_config("t27b", class_weights={"edition": 0.0, "read": 0.0, "noised": 0.0})
    assert M.choose_one("pgpid:1", [item(mk("describe_pgp", 1, "edition"))], 1, "x", zero.class_weights) == (None, None)


def test_choice_independent_of_other_documents_and_order() -> None:
    """A document's choice depends only on (seed, epoch, purpose, doc), not on file order or other documents."""
    cfg = M.default_config("t27b")
    its = [item(r) for d in range(40) for fam in M.ONE_PER_DOC_FAMILIES for r in doc_rows(fam, d)]
    a, _ = M.select_one_per_doc(its, cfg, "catalogue_describe")
    shuffled = its[:]
    random.Random(1).shuffle(shuffled)
    b, _ = M.select_one_per_doc(shuffled[: len(shuffled) // 2] + shuffled[len(shuffled) // 2:], cfg, "catalogue_describe")
    assert {x.uid for x in a} == {x.uid for x in b}
    sub = [it for it in its if it.meta["pgpid"] in {"3", "7"}]
    c, _ = M.select_one_per_doc(sub, cfg, "catalogue_describe")
    assert {x.uid for x in c} == {x.uid for x in a if x.meta["pgpid"] in {"3", "7"}}
    other_epoch = M.default_config("t27b", epoch=2)
    d, _ = M.select_one_per_doc(its, other_epoch, "catalogue_describe")
    assert {x.uid for x in d} != {x.uid for x in a}


# --------------------------------------------------------------------------------------- relations
def test_relations_subsample_forty_percent_one_row_per_doc() -> None:
    """Relations: round(0.4 × rows) documents, one row each; capped at the number of documents."""
    cfg = M.default_config("t27b")
    its = [item(r) for d in range(100) for r in doc_rows("extract_relations", d)]
    chosen, st = M.select_subsample(its, cfg, "extract_relations", 0.4)
    assert len(chosen) == 100  # round(0.4 × 300) = 120 > 100 documents → one row for every document
    assert len({c.doc for c in chosen}) == len(chosen)
    assert st["target_rows"] == 120 and st["capped_at_docs"]
    single = [item(mk("extract_relations", d, "read", read_model="a50")) for d in range(50)]
    chosen2, st2 = M.select_subsample(single, cfg, "extract_relations", 0.4)
    assert len(chosen2) == 20 and not st2["capped_at_docs"]
    many = [item(r) for d in range(100) for r in doc_rows("extract_relations", d, conditions=("edition",))]
    chosen3, _ = M.select_subsample(many, cfg, "extract_relations", 0.4)
    assert len(chosen3) == 40


def test_relations_in_t27b_and_slice_single(tmp_path: Path) -> None:
    """In t27b relations keep 40 % of rows as one row per document; a sampled slice doc gets its slice row."""
    files = base_t27b_files()
    files["extract_relations"] = [r for d in range(10, 60) for r in doc_rows("extract_relations", d,
                                                                              conditions=("edition", "noised_edition"))]
    files["noise_slice"] = [mk("extract_relations", d, "read_hard", read_model="a50") for d in range(10, 60)]
    res = build(tmp_path, "t27b", files)
    rel = [r for r in res["rows"] if r["meta"]["family"] == "extract_relations"]
    assert len(rel) == 40  # 100 rows after the swap → 40 documents
    assert len({r["meta"]["pgpid"] for r in rel}) == 40
    assert {r["meta"]["condition"] for r in rel} == {"read_hard"}
    assert res["section"]["gates"]["checks"]["one_row_per_doc_extract_relations"]["pass"]


# ----------------------------------------------------------------------------- other components
def test_book_pairs_optional_and_routed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """L5 pairs: absent → logged and listed; present → translate rows ×2 as ``book_pairs``, catalogue rows join the per-doc pool."""
    with caplog.at_level(logging.WARNING, logger=M.LOG.name):
        res = build(tmp_path / "a", "t27b", base_t27b_files())
    assert "families/book_pairs.jsonl" in res["section"]["load"]["missing_inputs"]
    assert any("book_pairs" in rec.getMessage() for rec in caplog.records)
    files = base_t27b_files()
    files["book_pairs"] = (doc_rows("translate_en", "n1", conditions=("edition",), doc_id="n1")
                           + doc_rows("catalogue_json", "n1", conditions=("edition",)))
    res2 = build(tmp_path / "b", "t27b", files)
    bp = [r for r in res2["rows"] if r["meta"]["mix_component"] == "book_pairs"]
    assert len(bp) == 2 and {r["meta"]["family"] for r in bp} == {"translate_en"}
    cat = [r for r in res2["rows"] if r["meta"]["pgpid"] == "n1" and r["meta"]["family"] == "catalogue_json"]
    assert len(cat) == 1 and cat[0]["meta"]["mix_component"] == "catalogue_describe"
    assert res2["section"]["components"]["book_pairs"]["rows"] == 2


def test_analyse_twice_rag_and_domain_reading_once(tmp_path: Path) -> None:
    """analyse_document ×2; rag_context and domain_reading train rows once each."""
    files = base_t27b_files()
    files["analyse_document"] = doc_rows("analyse_document", 40, conditions=("edition", "noised_edition"))
    files["rag_context"] = [mk("rag_context", 10, "edition", task="catalogue_json", twin_row_id="x"),
                            mk("rag_context", 11, "read", read_model="a50", task="describe_pgp")]
    files["domain_reading"] = [mk("domain_reading", f"w/p{i}", "literature", lang="EN") for i in range(3)]
    res = build(tmp_path, "t27b", files)
    comp = Counter(r["meta"]["mix_component"] for r in res["rows"])
    assert comp["analyse_document"] == 4 and comp["rag_context"] == 2 and comp["domain_reading"] == 3
    assert comp["translate_en"] == 18 and comp["translate_he"] == 18
    assert comp["catalogue_describe"] == 10 and comp["extract_relations"] == 10  # round(0.4 × 30) = 12, capped at 10 docs


def test_token_cap_subsamples_domain_reading_seeded(tmp_path: Path) -> None:
    """F5: the t27b L1 budget (default 2.70 M) keeps a seeded subsample within the cap, the same rows on a rebuild;
    t27a has no cap; a cap of 0 removes it."""
    assert M.default_config("t27b").token_caps == {"domain_reading": 2_700_000, "canon_diff": 2_500_000}
    assert M.default_config("t27a").token_caps == {}
    assert M.default_config("t27b", token_caps={"domain_reading": 0}).token_caps == {"canon_diff": 2_500_000}
    files = base_t27b_files()
    files["domain_reading"] = [mk("domain_reading", f"w/p{i}", "literature", lang="EN", tokens=100 + 10 * (i % 3))
                               for i in range(20)]
    res = build(tmp_path, "t27b", files, token_caps={"domain_reading": 1000})
    dr = [r for r in res["rows"] if r["meta"]["family"] == "domain_reading"]
    st = res["section"]["selection"]["token_caps"]["domain_reading"]
    assert 0 < sum(r["meta"]["tokens"] for r in dr) == st["tokens_kept"] <= 1000 and st["rows_offered"] == 20
    assert st["rows_kept"] == len(dr) and 8 <= len(dr) < 20
    assert res["section"]["gates"]["checks"]["token_cap_domain_reading"]["pass"]
    again = build(tmp_path / "again", "t27b", files, token_caps={"domain_reading": 1000})
    assert sorted(r["meta"]["row_id"] for r in again["rows"] if r["meta"]["family"] == "domain_reading") == \
        sorted(r["meta"]["row_id"] for r in dr)
    full = build(tmp_path / "full", "t27b", files, token_caps={"domain_reading": 0})
    assert sum(1 for r in full["rows"] if r["meta"]["family"] == "domain_reading") == 20


def test_literature_floor_gate() -> None:
    """F13: the t27b floor passes when the literature components hold ≥ 17.5 % of the tokens and L1 sits within ±1 %
    of its budget; either miss fails it; per-component yields are reported.

    :returns: None.
    """
    cfg = M.default_config("t27b", token_caps={"domain_reading": 1000})
    assert cfg.literature_floor == M.T27B_LITERATURE_FLOOR == {"min_share": 0.175, "l1_tolerance": 0.01}

    def inst(family: str, tokens: int, src: Optional[str] = None) -> M.Instance:
        """One instance with a token count.

        :param family: Family.
        :param tokens: Tokens.
        :param src: Source file stem.
        :returns: Instance.
        """
        r = mk(family, 1, "edition")
        r["meta"]["tokens"] = tokens
        return M.Instance(item=item(r, src), rule="test", copy=0, repeats=1)

    base = [inst("domain_reading", 995), inst("rag_context", 100), inst("analyse_document", 50),
            inst("translate_en", 80, src="book_pairs"), inst("catalogue_json", 4000)]
    chk = M.literature_floor_check(base, cfg)
    assert chk["pass"] and chk["share"] == round(1225 / 5225, 5) and chk["tokens_by_component"]["book_pairs"] == 80
    assert not M.literature_floor_check(base[1:] + [inst("domain_reading", 980)], cfg)["pass"]       # L1 off by 2 %
    low = M.literature_floor_check(base + [inst("catalogue_json", 2000)], cfg)
    assert not low["pass"] and low["violations"] == 1
    assert M.literature_floor_check(base, M.default_config("t27b", token_caps={"domain_reading": 0}))["pass"]


def canon_rows(n_per_title: int = 12) -> List[Dict[str, Any]]:
    """Synthetic canon_diff train rows: three works, several titles each, equal row lengths.

    :param n_per_title: Rows per title.
    :returns: Rows.
    """
    titles = {"Talmud Bavli": ["Berakhot", "Shabbat", "Eruvin"], "Mishneh Torah": [f"MT {k}" for k in range(6)],
              "Mishnah": ["Pirkei Avot"]}
    out = []
    for work, ts in titles.items():
        for t in ts:
            for i in range(n_per_title if work != "Mishnah" else 2):
                out.append(mk("canon_diff", f"{t}-{i}", "edition", lang="HE", tokens=100, work=work,
                              provenance={"sefaria_title": t, "ktiv_sys_num": f"99{i}"}))
    return out


def test_weighted_rank_key_prefers_heavier_rows() -> None:
    """Weighted sampling without replacement: weight-3 rows come first more often; a weight ≤ 0 is refused."""
    us = [M.unit_hash(1, "w", i) for i in range(4000)]
    keys = sorted([(M.weighted_rank_key(u, 3.0 if i % 2 else 1.0), i % 2) for i, u in enumerate(us)])
    first = [heavy for _, heavy in keys[:1000]]
    assert sum(first) / len(first) > 0.65
    with pytest.raises(ValueError):
        M.weighted_rank_key(0.5, 0.0)
    assert M.stratum_of({"work": "W", "provenance": {"sefaria_title": "T"}}) == ("W", "T")


def test_canon_diff_work_weighted_cap_keeps_every_title(tmp_path: Path) -> None:
    """Canon fold-in: canon_diff within its budget, Bavli weighted 3× over the Mishneh Torah, every (work, title)
    kept, per-work rows / tokens in the manifest; canon_identify is not mixed; PD pairs ×2 count as literature."""
    cfg = M.default_config("t27b")
    assert cfg.cap_weights["canon_diff"] == {"Talmud Bavli": 3.0, "Mishnah": 2.0, "Mishneh Torah": 1.0}
    assert "canon_identify" not in cfg.files and "canon_diff" in cfg.files and "pd_editions" in cfg.files
    assert "canon_diff" in M.ADDITIVE_COMPONENTS and "pd_editions" in M.LITERATURE_COMPONENTS
    files = base_t27b_files()
    files["canon_diff"] = canon_rows()
    files["canon_identify"] = [mk("canon_identify", "Berakhot-0", "edition", lang="HE", work="Talmud Bavli")]
    files["pd_editions"] = [mk("translate_en", "pd1", "edition", lang="HE", source="book:pd")]
    res = build(tmp_path, "t27b", files, token_caps={"canon_diff": 4000})
    rows = res["rows"]
    canon = [r for r in rows if r["meta"]["family"] == "canon_diff"]
    st = res["section"]["selection"]["token_caps"]["canon_diff"]
    assert sum(r["meta"]["tokens"] for r in canon) == st["tokens_kept"] <= 4000 and len(canon) == 40
    assert st["strata_offered"] == 10 and st["strata_kept"] == 10 and st["strata_missing"] == []
    bw = st["by_work"]
    assert sum(v["rows_kept"] for v in bw.values()) == 40 and bw["Mishnah"]["rows_kept"] >= 1
    assert bw["Talmud Bavli"]["rows_kept"] / bw["Talmud Bavli"]["rows_offered"] > \
        bw["Mishneh Torah"]["rows_kept"] / bw["Mishneh Torah"]["rows_offered"]
    assert not any(r["meta"]["family"] == "canon_identify" for r in rows)
    assert "canon_identify.jsonl" in res["section"]["load"]["ignored_files"]
    pd = [r for r in rows if r["meta"]["mix_component"] == "pd_editions"]
    assert len(pd) == 2 and {r["meta"]["family"] for r in pd} == {"translate_en"}
    sh = res["section"]["literature_shares"]
    assert sh["additive_tokens"] == st["tokens_kept"] and sh["core_tokens"] + sh["additive_tokens"] == sh["total_tokens"]
    again = build(tmp_path / "again", "t27b", files, token_caps={"canon_diff": 4000})
    assert sorted(r["meta"]["row_id"] for r in again["rows"] if r["meta"]["family"] == "canon_diff") == \
        sorted(r["meta"]["row_id"] for r in canon)
    assert res["section"]["gates"]["checks"]["token_cap_canon_diff"]["pass"]


def test_literature_floor_on_core_epoch() -> None:
    """Design owner 2026-10-26: the floor is gated on the epoch minus the canon block; the whole-epoch share and the
    literature-plus-canon share are reported beside it."""
    cfg = M.default_config("t27b", token_caps={"domain_reading": 1000})

    def inst(family: str, tokens: int, src: Optional[str] = None) -> M.Instance:
        """One instance with a token count.

        :param family: Family.
        :param tokens: Tokens.
        :param src: Source file stem.
        :returns: Instance.
        """
        r = mk(family, 1, "edition")
        r["meta"]["tokens"] = tokens
        return M.Instance(item=item(r, src), rule="test", copy=0, repeats=1)

    core = [inst("domain_reading", 1000), inst("translate_en", 50, src="pd_editions"), inst("catalogue_json", 4950)]
    chk = M.literature_floor_check(core + [inst("canon_diff", 2000)], cfg)
    assert chk["pass"] and chk["share_core"] == chk["share"] == round(1050 / 6000, 5)
    assert chk["share_whole_epoch"] == round(1050 / 8000, 5) < cfg.literature_floor["min_share"]
    assert chk["share_literature_plus_additive_whole_epoch"] == round(3050 / 8000, 5)
    assert chk["tokens_by_component"]["pd_editions"] == 50 and chk["additive_tokens"] == 2000
    whole = M.literature_floor_check(core + [inst("canon_diff", 2000)], M.replace(cfg, additive_components=()))
    assert not whole["pass"]  # without the core rule the canon block would dilute the floor


def test_nfc_and_scrub_gates(tmp_path: Path) -> None:
    """F3 / F4: a non-NFC message or an project-name string anywhere in a row (meta included) fails the mixture gates."""
    files = base_t27b_files()
    files["translate_en"][0]["messages"][2]["content"] = "Fust\u0323a\u0304t\u0323"
    files["translate_en"][1]["meta"]["provenance"] = {"note": "image at the " + codecs.decode("Sevrqoret", "rot_13") + " site"}
    res = build(tmp_path, "t27b", files)
    checks = res["section"]["gates"]["checks"]
    assert not checks["nfc_messages"]["pass"] and checks["nfc_messages"]["violations"] == 1
    assert not checks["no_source_names_anywhere"]["pass"] and checks["no_source_names_anywhere"]["violations"] == 1
    assert not res["section"]["gates"]["ok"]
    clean = build(tmp_path / "clean", "t27b", base_t27b_files())
    assert clean["section"]["gates"]["checks"]["nfc_messages"]["pass"] and clean["section"]["gates"]["checks"]["no_source_names_anywhere"]["pass"]


def test_excluded_rows_never_enter_either_mixture(tmp_path: Path) -> None:
    """D5: rows listed in ``excluded_rows.json`` are dropped from t27a and t27b and counted per family."""
    files = base_t27b_files()
    gone = {r["meta"]["row_id"]: "ngram_target:tgt:x#w8" for r in files["translate_en"][:3]}
    gone.update({r["meta"]["row_id"]: "ngram_target:tgt:y#w8" for r in files["catalogue_json"] if r["meta"]["pgpid"] == "10"})
    ex = tmp_path / "excluded_rows.json"
    ex.write_text(json.dumps(gone))
    for name in ("t27a", "t27b"):
        res = build(tmp_path / name, name, files, exclude_rows_file=str(ex))
        ids = {r["meta"]["row_id"] for r in res["rows"]}
        assert not ids & set(gone)
        load = res["section"]["load"]
        assert load["excluded_rows"]["listed"] == len(gone) and load["excluded_rows"]["file"] == "excluded_rows.json"
        assert load["excluded_rows"]["dropped_by_family_condition"]["translate_en|edition"] == 1
    assert load["dropped"]["excluded_rows"] == 6
    assert M.load_exclusions(None) == {} and M.load_exclusions(tmp_path / "none.json") == {}
    (tmp_path / "l.json").write_text(json.dumps(["a", "b"]))
    assert M.load_exclusions(tmp_path / "l.json") == {"a": "listed", "b": "listed"}


def test_relations_target_cap_walks_the_ranking() -> None:
    """D11: identical relations targets appear at most ``max_copies`` times; the budget is filled further down."""
    cfg = M.default_config("t27b")
    assert cfg.max_copies_per_target == 5 and M.default_config("t27a").max_copies_per_target is None
    same = [item(mk("extract_relations", d, "edition", target='{"relations": []}')) for d in range(30)]
    uniq = [item(mk("extract_relations", 100 + d, "edition")) for d in range(70)]
    chosen, st = M.select_subsample(same + uniq, cfg, "extract_relations", 0.4, max_copies=5)
    per_t = Counter(M.target_hash(c.row) for c in chosen)
    assert len(chosen) == 40 and max(per_t.values()) == 5 and st["max_copies_seen"] == 5
    assert st["docs_skipped_target_cap"] >= 1 and st["docs_walked"] == 40 + st["docs_skipped_target_cap"]
    free, _ = M.select_subsample(same + uniq, cfg, "extract_relations", 0.4)
    assert max(Counter(M.target_hash(c.row) for c in free).values()) > 5
    assert M.default_config("t27b", max_copies_per_target=0).max_copies_per_target is None


def test_relations_cap_gate_in_t27b(tmp_path: Path) -> None:
    """The t27b manifest gates the per-target cap of relations."""
    files = base_t27b_files()
    files["extract_relations"] = [mk("extract_relations", d, "edition", target='{"relations": []}') for d in range(10, 40)]
    files["extract_relations"] += [mk("extract_relations", d, "edition") for d in range(40, 70)]
    res = build(tmp_path, "t27b", files)
    g = res["section"]["gates"]["checks"]["max_copies_per_target_extract_relations"]
    assert g["pass"] and g["max_seen"] == 5
    rel = [r for r in res["rows"] if r["meta"]["family"] == "extract_relations"]
    assert len(rel) == 24  # round(0.4 × 60)


def test_analyse_v2_replaces_v01_analysis_file(tmp_path: Path) -> None:
    """``analyse_document_v2.jsonl`` supersedes ``analyse_document.jsonl`` when present (D11)."""
    files = base_t27b_files()
    files["analyse_document"] = doc_rows("analyse_document", 40, conditions=("edition",))
    res = build(tmp_path / "a", "t27b", files)
    assert {r["meta"]["pgpid"] for r in res["rows"] if r["meta"]["family"] == "analyse_document"} == {"40"}
    files["analyse_document_v2"] = doc_rows("analyse_document", 41, conditions=("edition",))
    res2 = build(tmp_path / "b", "t27b", files)
    ana = [r for r in res2["rows"] if r["meta"]["family"] == "analyse_document"]
    assert {r["meta"]["pgpid"] for r in ana} == {"41"} and len(ana) == 2
    assert res2["section"]["load"]["replaced_inputs"] == {"families/analyse_document.jsonl":
                                                          "families/analyse_document_v2.jsonl"}


def test_rag_twin_coverage_reported(tmp_path: Path) -> None:
    """The manifest reports how many rag rows have their twin row in the epoch."""
    files = base_t27b_files()
    twin_in = next(r for r in files["translate_en"] if r["meta"]["condition"] == "edition")["meta"]["row_id"]
    files["rag_context"] = [mk("rag_context", 1, "literature", task="translate_en", twin_row_id=twin_in),
                            mk("rag_context", 2, "literature", task="translate_en", twin_row_id="missing")]
    res = build(tmp_path, "t27b", files)
    tw = res["section"]["rag_twins"]
    assert tw["rag_rows"] == 2 and tw["twin_in_epoch"] == 1 and tw["twin_missing"] == 1 and tw["share_in_epoch"] == 0.5
    assert tw["by_task"]["translate_en"] == {"twin_in_epoch": 1, "twin_missing": 1}


def test_rows_without_rule_dropped_and_counted(tmp_path: Path) -> None:
    """A component with repeat 0 is dropped and reported."""
    files = base_t27b_files()
    files["analyse_document"] = doc_rows("analyse_document", 40)
    res = build(tmp_path, "t27b", files, repeats={"analyse_document": 0})
    assert not any(r["meta"]["family"] == "analyse_document" for r in res["rows"])
    assert res["section"]["selection"]["dropped_no_rule"] == {"analyse_document": 3}


# ------------------------------------------------------------------------------------- load rules
def test_exposed_near_duplicate_reads_dropped(tmp_path: Path) -> None:
    """B2: reader-exposed ``read`` rows with CER < 0.05 are dropped; unexposed or noisier reads are kept."""
    files = {"translate_en": [mk("translate_en", 1, "edition"),
                              mk("translate_en", 1, "read", read_model="a50", reader_exposed=True, input_cer=0.03),
                              mk("translate_en", 1, "read", read_model="v21b", reader_exposed=True, input_cer=0.2),
                              mk("translate_en", 2, "read", read_model="a50", reader_exposed=False, input_cer=0.01)],
             "translate_he": doc_rows("translate_he", 3)}
    res = build(tmp_path / "a", "t27a", files)
    kept = {(r["meta"]["pgpid"], r["meta"]["read_model"]) for r in res["rows"] if r["meta"]["condition"] == "read"}
    assert kept == {("1", "v21b"), ("2", "a50"), ("3", "a50")}
    assert res["section"]["load"]["dropped"]["exposed_read_near_duplicate"] == 1
    res2 = build(tmp_path / "b", "t27a", files, exposed_near_dup_cer=None)
    assert sum(1 for r in res2["rows"] if r["meta"]["condition"] == "read") == 8


def test_duplicates_dropped_and_row_id_collisions_kept(tmp_path: Path) -> None:
    """An identical row in two files counts once; rows sharing an id with different content are both kept."""
    a = mk("translate_en", 1, "edition", side="r", target="recto text")
    b = mk("translate_en", 1, "edition", side="v", target="verso text")
    b["messages"][1]["content"] = "prompt for the verso"
    files = {"translate_en": [a, b, a], "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path, "t27a", files)
    load = res["section"]["load"]
    assert load["dropped"]["duplicate_row"] == 1 and load["row_id_collisions"] == 1
    assert sum(1 for r in res["rows"] if r["meta"]["pgpid"] == "1") == 4


def test_unrenderable_rows_dropped(tmp_path: Path) -> None:
    """Rows with an empty target or a wrong turn layout never reach the mixture."""
    bad = mk("translate_en", 5, "edition")
    bad["messages"][2]["content"] = ""
    worse = mk("translate_en", 6, "edition")
    worse["messages"] = worse["messages"][:2]
    files = {"translate_en": doc_rows("translate_en", 1) + [bad, worse], "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path, "t27a", files)
    assert res["section"]["load"]["dropped"]["unrenderable"] == 2
    assert not any(r["meta"]["pgpid"] in ("5", "6") for r in res["rows"])


def test_missing_required_family_raises(tmp_path: Path) -> None:
    """t27b refuses to build without its required PGP family files."""
    files = base_t27b_files()
    del files["extract_relations"]
    with pytest.raises(FileNotFoundError):
        build(tmp_path, "t27b", files)


# -------------------------------------------------------------------------------- manifest, order
def test_manifest_counts_tokens_steps_and_fingerprint(tmp_path: Path) -> None:
    """Manifest totals, per-key counts, shares, projected steps and the fingerprint match the written epoch."""
    files = base_t27b_files()
    files["noise_slice"] = [mk("translate_en", 1, "read_hard", read_model="a50", tokens=300)]
    res = build(tmp_path, "t27b", files)
    rows, sec, out = res["rows"], res["section"], res["out"]
    total = sum(r["meta"]["tokens"] for r in rows)
    assert sec["totals"]["rows"] == len(rows) and sec["totals"]["tokens"] == total
    assert sum(sec["counts"]["rows"].values()) == len(rows) and sum(sec["counts"]["tokens"].values()) == total
    assert all(k.count("|") == 3 and k.endswith("|train") for k in sec["counts"]["rows"])
    assert sec["tokens"]["translate_en"]["read_hard"]["JA"] == 600
    assert abs(sum(c["share"] for c in sec["components"].values()) - 1.0) < 0.01
    assert sec["noise_slice"] == {"rows": 2, "tokens": 600, "share": round(600 / total, 4)}
    assert sec["projected_steps"]["unpacked_steps"] == -(-len(rows) // 16)
    assert sec["projected_steps"]["packed_steps"] == -(-total // 16600)
    assert sec["fingerprint"] == R.fingerprint(rows)
    assert sec["output_sha256"] == M.file_sha256(out / "mixtures" / "t27b_epoch1.jsonl")
    assert sec["repeats_per_row"] == {"1": 20, "2": 18}
    assert set(sec["repeats_per_target"]["by_family"]) >= {"translate_en", "catalogue_json"}
    assert sec["build_args"]["seed"] == 3407 and sec["build_args"]["families_dir"] == "families"
    manifest = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())
    assert set(manifest) == {"t27b"} and manifest["t27b"]["fingerprint"] == sec["fingerprint"]
    assert any("noise-slice token share" in w for w in sec["gates"]["warnings"])


def test_manifest_keeps_other_mixture_sections(tmp_path: Path) -> None:
    """Building t27b after t27a keeps the t27a section of the shared manifest and render check."""
    out = tmp_path / "out"
    fdir = write_families(out, base_t27b_files())
    tpl, tok = make_template(tmp_path), make_tokenizer(tmp_path)
    for name in ("t27a", "t27b"):
        M.build_mixture(M.default_config(name), fdir, out, tpl, tok)
    manifest = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())
    render = json.loads((out / "mixtures" / "render_check.json").read_text())
    assert set(manifest) == {"t27a", "t27b"} and set(render) == {"t27a", "t27b"}


def test_seeded_order_deterministic_and_input_order_free(tmp_path: Path) -> None:
    """Same seed → same epoch whatever the input line order; another seed → another order, same rows."""
    files = base_t27b_files()
    a = build(tmp_path / "a", "t27b", files, render=False)
    shuffled = {k: random.Random(7).sample(v, len(v)) for k, v in files.items()}
    b = build(tmp_path / "b", "t27b", shuffled, render=False)
    assert a["section"]["fingerprint"] == b["section"]["fingerprint"]
    c = build(tmp_path / "c", "t27b", files, render=False, seed=1)
    assert c["section"]["fingerprint"] != a["section"]["fingerprint"]
    fixed = [r for r in a["rows"] if r["meta"]["mix_rule"].startswith("repeat")]
    fixed_c = [r for r in c["rows"] if r["meta"]["mix_rule"].startswith("repeat")]
    assert Counter(r["meta"]["row_id"] for r in fixed) == Counter(r["meta"]["row_id"] for r in fixed_c)


def test_order_spreads_families_through_the_epoch() -> None:
    """Single-copy rows are spread over the whole epoch, not appended after the repeated rows."""
    cfg = M.default_config("t27b")
    insts = [M.Instance(item(r), k, 2, "repeat×2") for d in range(200) for r in doc_rows("translate_en", d)
             for k in range(2)]
    insts += [M.Instance(item(r), 0, 1, "one_per_doc") for d in range(400)
              for r in doc_rows("describe_pgp", d, conditions=("edition",))]
    ordered = M.order_instances(insts, cfg)
    pos = [i for i, inst in enumerate(ordered) if inst.item.family == "describe_pgp"]
    n = len(ordered)
    assert 0.4 < sum(1 for p in pos if p < n / 2) / len(pos) < 0.6


# ------------------------------------------------------------------------------- tokens, render, gates
def test_tokens_counted_when_meta_lacks_them(tmp_path: Path) -> None:
    """Rows without ``meta.tokens`` are rendered and counted; ``--recount`` reports mismatches."""
    files = {"translate_en": [mk("translate_en", 1, "edition", tokens=None)], "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path / "a", "t27a", files)
    row = next(r for r in res["rows"] if r["meta"]["pgpid"] == "1")
    assert row["meta"]["tokens"] == R.tokens_of_row(row, tokenizer_path=make_tokenizer(tmp_path),
                                                    template_path=make_template(tmp_path))
    assert res["section"]["token_count"]["counted"] == 1
    res2 = build(tmp_path / "b", "t27a", files, recount_tokens=True)
    assert res2["section"]["token_count"]["counted"] == 4 and res2["section"]["token_count"]["mismatch"] == 3


def test_render_check_three_rows_per_family(tmp_path: Path) -> None:
    """G3 with a conforming template: 3 rows per family (the longest among them), all pass."""
    files = base_t27b_files()
    files["translate_en"].append(mk("translate_en", 77, "edition", tokens=999))
    res = build(tmp_path, "t27b", files)
    render = json.loads((res["out"] / "mixtures" / "render_check.json").read_text())["t27b"]
    fams = Counter(r["family"] for r in render["rows"])
    distinct = defaultdict(set)
    for r in res["rows"]:
        distinct[r["meta"]["family"]].add(r["meta"]["row_id"])
    assert fams == {f: min(3, len(ids)) for f, ids in distinct.items()}
    assert set(fams) == {"translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations"}
    assert render["ok"] and render["n_failed"] == 0
    assert all(r["n_empty_think"] == 1 and not r["effort_sentence"] and r["mask_ok"] for r in render["rows"])
    assert any(r["family"] == "translate_en" and r["row_id"] == files["translate_en"][-1]["meta"]["row_id"]
               for r in render["rows"])
    assert res["section"]["gates"]["checks"]["render_check"]["pass"]


def test_render_check_fails_on_effort_sentence_and_long_rows(tmp_path: Path) -> None:
    """An effort-injecting template and rows above max_seq fail the gates; drop_over_max_seq removes long rows."""
    files = {"translate_en": doc_rows("translate_en", 1), "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path / "a", "t27a", files, template=EFFORT_TEMPLATE)
    assert not res["section"]["gates"]["ok"] and not res["section"]["gates"]["checks"]["render_check"]["pass"]
    render = json.loads((res["out"] / "mixtures" / "render_check.json").read_text())["t27a"]
    assert all(r["effort_sentence"] and r["error"] for r in render["rows"])
    files["translate_en"].append(mk("translate_en", 3, "edition", tokens=7000))
    res2 = build(tmp_path / "b", "t27a", files)
    assert not res2["section"]["gates"]["checks"]["max_seq"]["pass"]
    res3 = build(tmp_path / "c", "t27a", files, drop_over_max_seq=True)
    assert res3["section"]["gates"]["checks"]["max_seq"]["pass"]
    assert res3["section"]["selection"]["dropped_over_max_seq"] == [files["translate_en"][-1]["meta"]["row_id"]]


def test_arabic_noised_rows_fail_gate(tmp_path: Path) -> None:
    """B8 guard: an Arabic noised_edition row reaching the mixture fails the gate."""
    files = {"translate_en": doc_rows("translate_en", 1, lang="AR"), "translate_he": doc_rows("translate_he", 2)}
    res = build(tmp_path, "t27a", files, render=False)
    chk = res["section"]["gates"]["checks"]["no_arabic_noised_rows"]
    assert not chk["pass"] and chk["violations"] == 2
    assert "render check skipped" in res["section"]["gates"]["warnings"]


# ------------------------------------------------------------------------------------------- CLI
def test_cli_builds_both_mixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture) -> None:
    """``main`` resolves everything through ``paths.json`` and writes both epochs."""
    out = tmp_path / "out"
    write_families(out, base_t27b_files())
    cfg = {"repo": str(tmp_path), **{k: f"data/{k}" for k in CONFIG_KEYS}}
    cfg.update({"out_dir": "out", "chat_template": str(make_template(tmp_path)),
                "tokenizer_json": str(make_tokenizer(tmp_path))})
    (tmp_path / "paths.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv(P.ENV_VAR, str(tmp_path / "paths.json"))
    argv = ["--mixture", "t27a", "t27b", "--relations-frac", "0.2", "--class-weights", "edition=0.5,read=0.5",
            "--repeat", "translate_he=1"]
    assert M.main(argv) == 1                                  # F13: no literature in the synthetic files -> floor fails
    capsys.readouterr()
    rc = M.main(argv + ["--literature-floor", "none"])
    assert rc == 0
    assert (out / "mixtures" / "t27a_epoch1.jsonl").exists() and (out / "mixtures" / "t27b_epoch1.jsonl").exists()
    manifest = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())
    args = manifest["t27b"]["build_args"]
    assert args["subsample"] == {"extract_relations": 0.2} and args["class_weights"]["edition"] == 0.5
    assert args["repeats"]["translate_he"] == 1 and args["repeats"]["translate_en"] == 2
    assert manifest["t27b"]["components"]["extract_relations"]["rows"] == 6
    assert "== t27b" in capsys.readouterr().out
    with pytest.raises(argparse_error()):
        M._parse_kv(["edition"], float)


def test_cli_exclude_rows_auto_and_cap_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--exclude-rows auto`` picks up ``<out_dir>/excluded_rows.json``; ``none`` turns it off; the cap flag is passed."""
    out = tmp_path / "out"
    files = base_t27b_files()
    write_families(out, files)
    rid = files["translate_en"][0]["meta"]["row_id"]
    (out / "excluded_rows.json").write_text(json.dumps({rid: "ngram_target:x#w8"}))
    cfg = {"repo": str(tmp_path), **{k: f"data/{k}" for k in CONFIG_KEYS}}
    cfg.update({"out_dir": "out", "chat_template": str(make_template(tmp_path)),
                "tokenizer_json": str(make_tokenizer(tmp_path))})
    (tmp_path / "paths.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv(P.ENV_VAR, str(tmp_path / "paths.json"))
    assert M.main(["--mixture", "t27b", "--max-copies-per-target", "3", "--literature-floor", "none"]) == 0
    sec = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())["t27b"]
    assert sec["load"]["dropped"]["excluded_rows"] == 1 and sec["build_args"]["exclude_rows_file"] == "excluded_rows.json"
    assert sec["build_args"]["max_copies_per_target"] == 3
    assert M.main(["--mixture", "t27b", "--exclude-rows", "none", "--literature-floor", "none"]) == 0
    sec = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())["t27b"]
    assert "excluded_rows" not in sec["load"]["dropped"] and sec["build_args"]["max_copies_per_target"] == 5
    with pytest.raises(FileNotFoundError):
        M.main(["--mixture", "t27b", "--exclude-rows", str(tmp_path / "missing.json")])
    write_families(out, {"analyse_document_v2": doc_rows("analyse_document", 41, conditions=("edition",))})
    assert M.main(["--mixture", "t27b", "--without", "analyse_document_v2", "book_pairs", "--literature-floor",
                   "none"]) == 0
    sec = json.loads((out / "mixtures" / "mixture_manifest.json").read_text())["t27b"]
    assert "analyse_document_v2" not in sec["build_args"]["files"] and sec["load"]["replaced_inputs"] == {}
    with pytest.raises(ValueError):
        M.main(["--mixture", "t27b", "--without", "translate_en"])


def argparse_error() -> type:
    """The exception type ``_parse_kv`` raises on malformed items.

    :returns: :class:`argparse.ArgumentTypeError`.
    """
    import argparse

    return argparse.ArgumentTypeError


# ------------------------------------------------------------------------------------- real template
def _real_paths() -> Optional[P.Paths]:
    """The local build configuration when the real template and tokenizer exist.

    :returns: Paths or None.
    """
    try:
        p = P.load_paths()
    except (FileNotFoundError, KeyError):
        return None
    return p if p.chat_template.exists() and p.tokenizer_json.exists() else None


@pytest.mark.skipif(_real_paths() is None, reason="local Qwen3.8 chat template / tokenizer not available")
def test_render_check_with_real_qwen_template(tmp_path: Path) -> None:
    """G3 on synthetic rows rendered through the real Qwen3.8 template and tokenizer."""
    p = _real_paths()
    out = tmp_path / "out"
    fdir = write_families(out, {"translate_en": doc_rows("translate_en", 1), "translate_he": doc_rows("translate_he", 2)})
    sec = M.build_mixture(M.default_config("t27a", recount_tokens=True), fdir, out, p.chat_template, p.tokenizer_json)
    assert sec["render_check"]["ok"] and sec["gates"]["ok"], sec["gates"]
    assert sec["token_count"]["counted"] == 6


# ---------------------------------------------------------------------------------------- hygiene
def test_every_function_has_hints_and_sphinx_docstring() -> None:
    """Every function in the module has a docstring, return annotation and annotated parameters."""
    tree = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            params = [a for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
            if (not doc or node.returns is None or any(a.annotation is None for a in params)
                    or (":returns:" not in doc) or any(f":param {a.arg}:" not in doc for a in params)):
                bad.append(node.name)
    assert not bad, bad
