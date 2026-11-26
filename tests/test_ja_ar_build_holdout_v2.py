"""Tests of ``ja_ar_build.holdout_v2`` (novelty tiers and evaluation hold-outs v2).

Synthetic records only: every text is a seeded random letter string (no edition text), every image a generated pattern.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import holdout_v2 as h
from src.finetuning.qwen_hebrew.novel_build import holdout as nh
from src.finetuning.qwen_hebrew.novel_build.images import image_keys

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def letters(n: int, seed: int, alphabet: str = HEBREW) -> str:
    """A seeded random letter string with a space every 4-6 letters.

    :param n: Letters.
    :type n: int
    :param seed: RNG seed.
    :type seed: int
    :param alphabet: Letters to draw from.
    :type alphabet: str
    :return: The text.
    :rtype: str
    """
    rng = random.Random(seed)
    out: List[str] = []
    while sum(len(w) for w in out) < n:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(4, 6))))
    return " ".join(out)[: n + len(out)]


def reg_row(dataset: str, row_id: str, *, split: str = "train", text: str = "", notes: str = "",
            **keys: Any) -> Dict[str, Any]:
    """A registry row with registry_lib-style text shingles.

    :param dataset: Registry dataset.
    :type dataset: str
    :param row_id: Row id.
    :type row_id: str
    :param split: Split name.
    :type split: str
    :param text: Target text (hashed into ``text_shingles``).
    :type text: str
    :param notes: Notes string.
    :type notes: str
    :param keys: Other registry fields.
    :type keys: Any
    :return: The row.
    :rtype: Dict[str, Any]
    """
    row = {"dataset": dataset, "split": split, "row_id": row_id, "notes": notes,
           "text_shingles": sorted({int(x) for x in h.registry_positions(text)})}
    row.update(keys)
    return row


def registry(rows: List[Dict[str, Any]], released: Optional[set] = None) -> h.SeenRegistry:
    """A finished registry index over synthetic rows.

    :param rows: Registry rows.
    :type rows: List[Dict[str, Any]]
    :param released: Released val PGP ids.
    :type released: Optional[set]
    :return: The index.
    :rtype: h.SeenRegistry
    """
    reg = h.SeenRegistry(released_val_pgpids=released)
    for r in rows:
        reg.add_row(r)
    reg.finish()
    return reg


def subject(text: str, **kw: Any) -> h.Subject:
    """A subject with defaults.

    :param text: Target text.
    :type text: str
    :param kw: Subject fields.
    :type kw: Any
    :return: The subject.
    :rtype: h.Subject
    """
    base = {"row_id": "cand:1", "side_key": "recto", "source": "pgp", "document_group": "g1"}
    base.update(kw)
    return h.Subject(text=text, **base)


def tier(sub: h.Subject, reg: h.SeenRegistry, group: Optional[h.GroupEvidence] = None) -> str:
    """Tier of a subject against a registry.

    :param sub: Subject.
    :type sub: h.Subject
    :param reg: Registry index.
    :type reg: h.SeenRegistry
    :param group: Group evidence.
    :type group: Optional[h.GroupEvidence]
    :return: The tier.
    :rtype: str
    """
    return h.side_tier(sub, h.novelty_evidence(sub, reg, h.TextRule()), group)


def write_image(path: Path, seed: int, size: int = 256) -> Path:
    """Write a seeded random-blocks greyscale PNG.

    :param path: Output file.
    :type path: Path
    :param seed: RNG seed.
    :type seed: int
    :param size: Side in pixels.
    :type size: int
    :return: The path.
    :rtype: Path
    """
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
    Image.fromarray(np.kron(blocks, np.ones((size // 16, size // 16), dtype=np.uint8)), "L").save(path)
    return path


def flip_bits(phash: str, n: int) -> str:
    """A 64-bit hash with its ``n`` lowest bits flipped.

    :param phash: 16 hex characters.
    :type phash: str
    :param n: Bits to flip.
    :type n: int
    :return: 16 hex characters.
    :rtype: str
    """
    return f"{int(phash, 16) ^ ((1 << n) - 1):016x}"


# ----------------------------------------------------------------------------------------------------------------
# Letters and the long-run text rule
# ----------------------------------------------------------------------------------------------------------------

def test_registry_letters_keep_arabic_and_drop_marks() -> None:
    """The registry letter class keeps Arabic letters and drops harakat and Hebrew points."""
    assert h.registry_letters("بِسْمِ اللّه") == "بسمالله"
    assert h.registry_letters("שָׁלוֹם 12 abc") == "שלום"
    assert len(h.registry_positions("אבגדהו")) == 2


def test_long_run_of_100_letters_holds() -> None:
    """A 100-letter shared run holds the side out (``text_run``)."""
    shared = letters(110, 1).replace(" ", "")[:100]
    side = letters(150, 2) + " " + shared + " " + letters(150, 3)
    page = np.unique(h.skeleton_positions(letters(500, 4) + shared + letters(500, 5)))
    rule = h.TextRule()
    ov = h.overlap(h.skeleton_positions(side), page, rule)
    assert ov.run_letters >= 100
    assert rule.reason(ov) == "text_run"


def test_99_letter_run_in_a_long_side_does_not_hold() -> None:
    """A 99-letter run that is a small part of a long side holds nothing."""
    shared = letters(110, 1).replace(" ", "")[:99]
    side = letters(800, 2) + " " + shared + " " + letters(800, 3)
    page = np.unique(h.skeleton_positions(letters(500, 4) + shared))
    rule = h.TextRule()
    ov = h.overlap(h.skeleton_positions(side), page, rule)
    assert ov.run_letters == 99
    assert rule.reason(ov) is None


def test_many_short_formulaic_shingles_do_not_hold() -> None:
    """Many short shared formulae (20 letters each) never hold a side out under the default rule.

    The literal reading (30 % of the side's distinct 5-grams anywhere in the document) would hold it: that is the
    formulaic bias the rule removes.
    """
    formulae = [letters(24, 100 + i).replace(" ", "")[:20] for i in range(10)]
    side = " ".join(f + " " + letters(12, 200 + i) for i, f in enumerate(formulae))
    doc = " ".join(letters(200, 300 + i) + " " + f for i, f in enumerate(formulae))
    page = np.unique(h.skeleton_positions(doc))
    positions = h.skeleton_positions(side)
    default = h.TextRule()
    ov = h.overlap(positions, page, default)
    assert ov.run_letters < 25
    assert ov.shingle_share == 0.0
    assert default.reason(ov) is None
    literal = h.TextRule(share_basis="grams")
    assert ov.share >= 0.30 and literal.reason(h.overlap(positions, page, literal)) == "text_share"


def test_short_side_contained_in_a_document_holds_by_share() -> None:
    """A side under 100 letters found whole in a document holds by the share clause."""
    side = letters(80, 7)
    page = np.unique(h.skeleton_positions(letters(300, 8) + side + letters(300, 9)))
    rule = h.TextRule()
    ov = h.overlap(h.skeleton_positions(side), page, rule)
    assert ov.run_letters < 100 and ov.shingle_share > 0.9
    assert rule.reason(ov) == "text_share"


def test_variant_rich_edition_holds_by_share() -> None:
    """Another edition of the same text, a variant every 40 letters (no 100-letter run), holds by the share."""
    base = letters(600, 11).replace(" ", "")
    variant = "".join(ch if (i % 40) != 39 else ("א" if ch != "א" else "ב") for i, ch in enumerate(base))
    page = np.unique(h.skeleton_positions(base))
    rule = h.TextRule()
    ov = h.overlap(h.skeleton_positions(variant), page, rule)
    assert ov.run_letters < 100
    assert ov.shingle_share >= 0.30
    assert rule.reason(ov) == "text_share"


def test_share_clause_needs_min_grams() -> None:
    """A tiny side gets no share verdict."""
    side = letters(25, 12)
    page = np.unique(h.skeleton_positions(side))
    ov = h.overlap(h.skeleton_positions(side), page, h.TextRule())
    assert ov.grams < h.MIN_SIDE_GRAMS and h.TextRule().reason(ov) is None


def test_text_rule_validates() -> None:
    """Unknown share bases and impossible thresholds raise."""
    with pytest.raises(ValueError):
        h.TextRule(share_basis="covered")
    with pytest.raises(ValueError):
        h.TextRule(share_threshold=0.0)


# ----------------------------------------------------------------------------------------------------------------
# Registry classes
# ----------------------------------------------------------------------------------------------------------------

def test_row_classes() -> None:
    """Definite, pool, partial-consumed, eval-only, pool-val, released-val and never-trained rows."""
    assert h.row_class({"dataset": "genizah_novel_v1", "split": "train", "notes": ["trained_status=trained"]}) == h.TRAINED
    assert h.row_class({"dataset": "genizah_novel_v1", "split": "val", "notes": ["trained_status=eval_only"]}) \
        == h.EVAL_ONLY
    assert h.row_class({"dataset": "genizah_v22_full", "split": "train",
                        "notes": "trained_status=partial;consumed=1"}) == h.TRAINED
    assert h.row_class({"dataset": "genizah_v22_full", "split": "train",
                        "notes": "trained_status=partial;consumed=0"}) == h.POOL
    assert h.row_class({"dataset": "genizah_ktiv_v4", "split": "train"}) == h.POOL
    assert h.row_class({"dataset": "genizah_ktiv_v4", "split": "val"}) == h.POOL_VAL
    assert h.row_class({"dataset": "genizah_ktiv_v4", "split": "val"}, pool_val_holdout=True) == h.EVAL_ONLY
    assert h.row_class({"dataset": "pgp_editions_v1", "split": "val", "pgpid": "7"}, {"7"}) == h.RELEASED_VAL
    assert h.row_class({"dataset": "pgp_editions_v1", "split": "val", "pgpid": "8"}, {"7"}) == h.POOL_VAL
    assert h.row_class({"dataset": "genizah_v23q_vqa", "split": "train"}) is None
    assert h.row_class({"dataset": "genizah_v23a_arabic", "split": "train_without_outside",
                        "notes": "trained_status=not_trained"}) is None


# ----------------------------------------------------------------------------------------------------------------
# Novelty tiers
# ----------------------------------------------------------------------------------------------------------------

TRAINED_NOTES = "trained_status=trained"


def test_novel_strict_without_evidence() -> None:
    """No key and no text in the registry: novel_strict."""
    reg = registry([reg_row("genizah_novel_v1", "r1", text=letters(400, 20), pgpid="99", notes=TRAINED_NOTES)])
    assert tier(subject(letters(400, 21), pgpids=["1"]), reg) == h.STRICT


def test_previously_trained_by_image_sha256_of_own_photo() -> None:
    """The side's own photograph was trained: previously_trained; the same photograph unplaced is only a flag."""
    reg = registry([reg_row("genizah_novel_v1", "r1", image_sha256="a" * 64, notes=TRAINED_NOTES)])
    photo = h.Photo(sha256="a" * 64)
    own = subject(letters(300, 22), photos=[photo], own_photos=[photo.key])
    assert tier(own, reg) == h.PREVIOUSLY_TRAINED
    unplaced = subject(letters(300, 22), photos=[photo], own_photos=[])
    ev = h.novelty_evidence(unplaced, reg, h.TextRule())
    assert h.side_tier(unplaced, ev) == h.STRICT
    assert h.pair_tiers(unplaced, ev) == {photo.key: h.PREVIOUSLY_TRAINED}
    assert any(f.startswith("photo_trained_unplaced") for f in h.novelty_record(unplaced, ev, h.GroupEvidence())["flags"])


def test_previously_trained_by_ktiv_fl_and_url() -> None:
    """A KTIV page id or the same URL is the same photograph."""
    reg = registry([reg_row("genizah_ktiv_v1", "ktiv_123_FL201234567_line1", ktiv_id="123", notes=TRAINED_NOTES),
                    reg_row("genizah_v25a_arabic", "r2", image_url="https://x/img.jpg", notes=TRAINED_NOTES)])
    p1 = h.Photo(sha256="b" * 64, fl_ids=["FL201234567"])
    assert tier(subject(letters(200, 23), photos=[p1], own_photos=[p1.key]), reg) == h.PREVIOUSLY_TRAINED
    p2 = h.Photo(sha256="c" * 64, url="https://x/img.jpg")
    assert tier(subject(letters(200, 24), photos=[p2], own_photos=[p2.key]), reg) == h.PREVIOUSLY_TRAINED


def test_previously_trained_by_long_text_witness() -> None:
    """The side's text is a trained page's text: previously_trained, with the run recorded."""
    text = letters(500, 25)
    reg = registry([reg_row("genizah_v22_pilot", "r1", text=text, image_sha256="d" * 64, notes=TRAINED_NOTES)])
    sub = subject(text)
    ev = h.novelty_evidence(sub, reg, h.TextRule())
    assert h.side_tier(sub, ev) == h.PREVIOUSLY_TRAINED
    rec = h.novelty_record(sub, ev, h.GroupEvidence())
    assert rec["tier_basis"] == "page:text"
    assert rec["witnesses"][0]["dataset"] == "genizah_v22_pilot" and rec["witnesses"][0]["run_letters"] >= 100
    assert rec["witnesses"][0]["row_ids"] == ["r1"]


def test_novel_page_level_when_only_the_document_was_trained() -> None:
    """Another side of the document was trained (PGP id): novel_page_level, the witness names the row."""
    reg = registry([reg_row("genizah_novel_v1", "r1", text=letters(400, 26), pgpid="42", notes=TRAINED_NOTES)])
    sub = subject(letters(400, 27), pgpids=["42"])
    ev = h.novelty_evidence(sub, reg, h.TextRule())
    assert h.side_tier(sub, ev) == h.PAGE_LEVEL
    rec = h.novelty_record(sub, ev, h.GroupEvidence())
    assert rec["witnesses"][0]["key"] == "pgpid" and rec["witnesses"][0]["row_ids"] == ["r1"]


@pytest.mark.parametrize("field,value,key", [("canonical_ids", "Cambridge_CUL_T_S_1_2", "canonical_id"),
                                             ("shelf_keys_clean", "cambridgects1|2", "shelf_key_clean"),
                                             ("document_group", "canonical_id:X", "document_group")])
def test_document_keys(field: str, value: str, key: str) -> None:
    """Each definite document key makes a trained document.

    :param field: Subject field carrying the key.
    :type field: str
    :param value: Key value.
    :type value: str
    :param key: Registry key kind.
    :type key: str
    """
    row_keys = {"canonical_id": value} if key == "canonical_id" else (
        {"shelf_keys_clean": [value]} if key == "shelf_key_clean" else {})
    notes = TRAINED_NOTES + (f";document_group={value}" if key == "document_group" else "")
    reg = registry([reg_row("genizah_novel_v1", "r1", notes=notes, **row_keys)])
    kw = {field: value} if field == "document_group" else {field: [value]}
    assert tier(subject(letters(300, 28), **kw), reg) == h.PAGE_LEVEL


def test_ktiv_manuscript_is_a_document_key() -> None:
    """A KTIV manuscript trained through another page: novel_page_level."""
    reg = registry([reg_row("genizah_ktiv_v1", "ktiv_555_FL200000001_li0", ktiv_id="555", notes=TRAINED_NOTES)])
    assert tier(subject(letters(300, 29), ktiv_id="555"), reg) == h.PAGE_LEVEL


def test_uncertain_history_from_pool_evidence() -> None:
    """Pool page text, pool document keys, the legacy shelf key and a trained join partner are uncertain."""
    text = letters(400, 30)
    reg = registry([reg_row("genizah_ktiv_v4", "p1", text=text),
                    reg_row("pgp_editions_v1", "p2", pgpid="77"),
                    reg_row("genizah_v25a_arabic", "t1", shelf_key="ts12|34", notes=TRAINED_NOTES),
                    reg_row("genizah_v25a_arabic", "t2", canonical_id="Partner_X", notes=TRAINED_NOTES)])
    assert tier(subject(text), reg) == h.UNCERTAIN
    assert tier(subject(letters(400, 31), pgpids=["77"]), reg) == h.UNCERTAIN
    assert tier(subject(letters(400, 32), shelf_keys_legacy=["ts12|34"]), reg) == h.UNCERTAIN
    assert tier(subject(letters(400, 33), join_partners=["Partner_X"]), reg) == h.UNCERTAIN


def test_page_pool_evidence_outranks_a_trained_document() -> None:
    """Document trained but this page sits in a pool: uncertain_history, not page-level novel."""
    text = letters(400, 34)
    reg = registry([reg_row("genizah_novel_v1", "r1", pgpid="5", notes=TRAINED_NOTES),
                    reg_row("genizah_ktiv_v4", "p1", text=text)])
    assert tier(subject(text, pgpids=["5"]), reg) == h.UNCERTAIN


def test_eval_only_and_pool_val_rows_are_not_novelty_evidence() -> None:
    """A val row (definite dataset) is a hold-out, a pool dataset's val row is neither; both leave the tier novel."""
    text = letters(400, 35)
    reg = registry([reg_row("genizah_novel_v1", "v1", split="val", text=text, pgpid="9",
                            notes="trained_status=eval_only"),
                    reg_row("genizah_ktiv_v4", "v2", split="val", text=letters(400, 36), pgpid="10")])
    sub = subject(text, pgpids=["9"])
    ev = h.novelty_evidence(sub, reg, h.TextRule())
    assert h.side_tier(sub, ev) == h.STRICT
    val = h.registry_val_witnesses(ev, text_level=True) + h.registry_val_witnesses(ev, text_level=False)
    assert {w["family"] for w in val} == {"registry_val"} and {w["set"] for w in val} == {"genizah_novel_v1"}
    other = subject(letters(400, 36), pgpids=["10"])
    ev2 = h.novelty_evidence(other, reg, h.TextRule())
    assert h.side_tier(other, ev2) == h.STRICT
    assert not h.registry_val_witnesses(ev2, True) and not h.registry_val_witnesses(ev2, False)
    assert "pool_val_evidence" in h.novelty_record(other, ev2, h.GroupEvidence())["flags"]


def test_released_val_rows_are_recorded_only() -> None:
    """The released held_out_edition_pages_v0 val pages: no hold-out, recorded as released."""
    text = letters(300, 37)
    reg = registry([reg_row("pgp_editions_v1", "v1", split="val", text=text, pgpid="7")], released={"7"})
    sub = subject(text, pgpids=["7"])
    ev = h.novelty_evidence(sub, reg, h.TextRule())
    assert h.side_tier(sub, ev) == h.STRICT
    assert not h.registry_val_witnesses(ev, True) and not h.registry_val_witnesses(ev, False)
    assert h.released_val_witnesses(ev) and h.released_val_witnesses(ev)[0]["set"] == "held_out_edition_pages_v0"


def test_never_trained_rows_are_dropped() -> None:
    """Rows of a never-trained mixture are no evidence at all."""
    text = letters(400, 38)
    reg = registry([reg_row("genizah_v23q_vqa", "q1", text=text, pgpid="3")])
    assert tier(subject(text, pgpids=["3"]), reg) == h.STRICT
    assert reg.dropped["genizah_v23q_vqa"] == 1


def test_group_member_trained_makes_page_level() -> None:
    """A sibling side trained (previously_trained) passes 'document trained' to the other side."""
    text_a, text_b = letters(400, 39), letters(400, 40)
    reg = registry([reg_row("genizah_novel_v1", "r1", text=text_a, image_sha256="e" * 64, notes=TRAINED_NOTES)])
    a = subject(text_a, side_key="recto")
    b = subject(text_b, side_key="verso")
    own = {s.subject_id: (h.side_tier(s, ev), ev) for s in (a, b)
           for ev in [h.novelty_evidence(s, reg, h.TextRule())]}
    groups = h.group_evidence([a, b], own, {})
    assert own[a.subject_id][0] == h.PREVIOUSLY_TRAINED and own[b.subject_id][0] == h.STRICT
    assert h.side_tier(b, own[b.subject_id][1], groups[b.subject_id]) == h.PAGE_LEVEL
    ledger = {"g1": [{"row_id": "other", "flags": {"verdict": "uncertain_history", "witnesses": ["x~pool"]}}]}
    c = subject(letters(300, 41), row_id="cand:2", side_key="")
    own_c = {c.subject_id: (h.STRICT, h.novelty_evidence(c, reg, h.TextRule()))}
    assert h.side_tier(c, own_c[c.subject_id][1], h.group_evidence([c], own_c, ledger)[c.subject_id]) == h.UNCERTAIN


# ----------------------------------------------------------------------------------------------------------------
# pHash confirmation
# ----------------------------------------------------------------------------------------------------------------

def test_phash_within_6_bits_needs_a_confirmed_photograph(tmp_path: Path) -> None:
    """A 4-bit pHash hit counts only when the two files are the same photograph (256-bit pHash).

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    original = write_image(tmp_path / "a.png", 1)
    buf = io.BytesIO()
    Image.open(original).convert("L").resize((200, 200)).save(buf, format="JPEG", quality=85)
    same = tmp_path / "a_small.jpg"
    same.write_bytes(buf.getvalue())
    other = write_image(tmp_path / "b.png", 2)
    stored = image_keys(original.read_bytes())
    reg = h.SeenRegistry(confirmer=h.PhashConfirmer())
    reg.add_row(reg_row("genizah_novel_v1", "r1", image_sha256=stored["sha256"], phash=stored["phash"],
                        image_path=str(original), notes=TRAINED_NOTES))
    reg.finish()
    near = flip_bits(stored["phash"], 4)
    good = h.Photo(sha256="f" * 64, phash=near, path=str(same))
    bad = h.Photo(sha256="0" * 64, phash=near, path=str(other))
    assert tier(subject(letters(200, 42), photos=[good], own_photos=[good.key]), reg) == h.PREVIOUSLY_TRAINED
    sub = subject(letters(200, 42), photos=[bad], own_photos=[bad.key])
    ev = h.novelty_evidence(sub, reg, h.TextRule())
    assert h.side_tier(sub, ev) == h.STRICT
    assert ev.photos[bad.key][0]["class"] == h.PHASH_REJECTED
    missing = h.Photo(sha256="1" * 64, phash=near, path=None)
    ev3 = h.novelty_evidence(subject("", photos=[missing], own_photos=[missing.key]), reg, h.TextRule())
    assert ev3.photos[missing.key][0]["class"] == h.PHASH_UNVERIFIED
    exact = h.Photo(sha256="2" * 64, phash=stored["phash"], path=None)
    assert tier(subject("", photos=[exact], own_photos=[exact.key]), reg) == h.PREVIOUSLY_TRAINED


def test_phash_beyond_6_bits_is_ignored() -> None:
    """A 7-bit pHash distance is no candidate at all."""
    reg = registry([reg_row("genizah_novel_v1", "r1", phash="0" * 16, notes=TRAINED_NOTES)])
    photo = h.Photo(sha256="3" * 64, phash=flip_bits("0" * 16, 7))
    ev = h.novelty_evidence(subject("", photos=[photo], own_photos=[photo.key]), reg, h.TextRule())
    assert ev.photos[photo.key] == []


# ----------------------------------------------------------------------------------------------------------------
# Evaluation hold-outs
# ----------------------------------------------------------------------------------------------------------------

def eval_set(name: str = "pgp131", family: str = "pgp131", released: bool = False, **kw: Any) -> h.EvalSet:
    """An evaluation set with a BenchmarkSet built from keyword lists.

    :param name: Set name.
    :type name: str
    :param family: Witness family.
    :type family: str
    :param released: Released reserve.
    :type released: bool
    :param kw: ``ids``, ``pgpids``, ``sys_nums``, ``marks``, ``fl``, ``names``, ``images``, ``gt``.
    :type kw: Any
    :return: The set.
    :rtype: h.EvalSet
    """
    bench = nh.BenchmarkSet(name, ids={i: i for i in kw.get("ids", [])}, pgpids=set(kw.get("pgpids", [])),
                            sys_nums={s: f"sys:{s}" for s in kw.get("sys_nums", [])})
    for mark in kw.get("marks", []):
        bench.add_mark(mark, mark, None)
    return h.EvalSet(name, family, bench, released=released, fl=dict(kw.get("fl", {})), names=dict(kw.get("names", {})),
                     images=list(kw.get("images", [])), gt=list(kw.get("gt", [])))


def held(sub: h.Subject, evals: h.EvalIndex) -> List[Dict[str, Any]]:
    """Identity and image witnesses of a subject's candidate.

    :param sub: Subject.
    :type sub: h.Subject
    :param evals: Evaluation index.
    :type evals: h.EvalIndex
    :return: Witnesses.
    :rtype: List[Dict[str, Any]]
    """
    ws = evals.identity_witnesses(h.candidate_ident(sub, None), sub.fl_ids)
    for p in sub.photos:
        ws += evals.image_witnesses(p)
    return ws


@pytest.mark.parametrize("kw,key", [({"canonical_ids": ["Cambridge_CUL_T_S_10J1_1"]}, "canonical_id"),
                                    ({"join_partners": ["Cambridge_CUL_T_S_10J1_1"]}, "canonical_id"),
                                    ({"pgpids": ["123"]}, "pgpid"),
                                    ({"ktiv_id": "990001996710205171"}, "ktiv_sys_num"),
                                    ({"fl_ids": ["FL209999999"]}, "ktiv_fl")])
def test_identity_keys_hold_out(kw: Dict[str, Any], key: str) -> None:
    """Canonical id, join partner, PGP id, KTIV manuscript and KTIV page id each hold out.

    :param kw: Subject fields.
    :type kw: Dict[str, Any]
    :param key: Expected witness key.
    :type key: str
    """
    evals = h.EvalIndex([eval_set(ids=["Cambridge_CUL_T_S_10J1_1"], pgpids=["123"], sys_nums=["990001996710205171"],
                                  fl={"FL209999999": "KTIV_1_FL209999999"})], h.TextRule())
    ws = held(subject(letters(100, 50), **kw), evals)
    assert ws and ws[0]["key"] == key and ws[0]["family"] == "pgp131" and not ws[0]["released"]


def test_shelf_mark_loose_key_holds_out() -> None:
    """A candidate whose shelf mark has the benchmark's loose key is held out."""
    evals = h.EvalIndex([eval_set(marks=["T-S 10J1.1"])], h.TextRule())
    from src.finetuning.qwen_hebrew.novel_build.schema import Candidate
    cand = Candidate("pgp:1", "pgp", "g", [], [], None, "T-S 10J1.1", "hebrew", None, "novel", [], "document", 0)
    ws = held(subject("", candidate=cand), evals)
    assert any(w["key"] in ("loose_key", "canonical_id") for w in ws)


def test_image_identity_holds_out(tmp_path: Path) -> None:
    """The same image file (sha256) or a confirmed pHash re-encode holds out; an unconfirmed pHash hit does not.

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    a = write_image(tmp_path / "eval.png", 3)
    keys = image_keys(a.read_bytes())
    buf = io.BytesIO()
    Image.open(a).convert("L").resize((180, 180)).save(buf, format="JPEG", quality=85)
    (tmp_path / "copy.jpg").write_bytes(buf.getvalue())
    b = write_image(tmp_path / "other.png", 4)
    evals = h.EvalIndex([eval_set(images=[(keys["sha256"], keys["phash"], "DOC1", str(a))])], h.TextRule())
    assert held(subject("", photos=[h.Photo(sha256=keys["sha256"])]), evals)[0]["key"] == "image_sha256"
    near = flip_bits(keys["phash"], 2)
    assert held(subject("", photos=[h.Photo(sha256="9" * 64, phash=near, path=str(tmp_path / "copy.jpg"))]),
                evals)[0]["key"] == "phash"
    assert held(subject("", photos=[h.Photo(sha256="8" * 64, phash=near, path=str(b))]), evals) == []
    assert evals.unconfirmed and evals.unconfirmed[0]["verdict"] == "rejected"


def test_text_holdout_against_ground_truth() -> None:
    """A 100-letter run with a benchmark's ground truth holds the side; formulaic overlap does not."""
    gt = letters(800, 60)
    evals = h.EvalIndex([eval_set(gt=[("DOC1", gt)])], h.TextRule())
    hit, nearest = evals.text_witnesses(letters(100, 61) + " " + gt[100:300] + " " + letters(100, 62))
    assert hit and hit[0]["key"] == "text_run" and hit[0]["item"] == "DOC1"
    miss, _ = evals.text_witnesses(" ".join(gt[i:i + 18] + " " + letters(30, 70 + i) for i in range(0, 600, 60)))
    assert miss == []


def test_released_reserve_is_not_a_holdout() -> None:
    """A released reserve's match is recorded under ``released`` and does not hold the side out."""
    evals = h.EvalIndex([eval_set("two_edition_documents_v0", "released_reserve", released=True, pgpids=["5"])],
                        h.TextRule())
    ws = held(subject("", pgpids=["5"]), evals)
    rec = h.holdout_record(ws, [])
    assert rec["eval_holdout"] is False and rec["released_sets"] == ["two_edition_documents_v0"]


def test_named_exclusion_and_registry_val_hold_out() -> None:
    """A named exclusion holds out by name / PGP id; a registry val witness holds out with family registry_val."""
    evals = h.EvalIndex([eval_set("named:x", "named_exclusion", pgpids=["35277"], names={"x": "x"})], h.TextRule())
    ws = held(subject("", pgpids=["35277"]), evals)
    assert ws[0]["family"] == "named_exclusion"
    rec = h.holdout_record([], [{"family": "registry_val", "set": "genizah_novel_v1", "released": False}])
    assert rec["eval_holdout"] and rec["eval_families"] == ["registry_val"] and rec["eval_text_holdout"]


def test_eval_registry_files_are_honoured(tmp_path: Path) -> None:
    """A registered eval-set file holds out by its identity keys and its ground-truth file.

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    """
    gt_text = letters(400, 80)
    gt_file = tmp_path / "gt.jsonl"
    gt_file.write_text(json.dumps({"item_id": "JA1", "text": gt_text}) + "\n", encoding="utf-8")
    reg_dir = tmp_path / "registered"
    reg_dir.mkdir()
    (reg_dir / "ja_dev_v1.json").write_text(json.dumps({
        "eval_set": "ja_dev_v1", "role": "dev", "built": "2026-11-12",
        "items": [{"item_id": "JA1", "pgpids": ["4242"], "ktiv_fl": ["FL200000042"], "image_sha256": ["c" * 64],
                   "row_ids": ["fjp:Some_Doc:0"], "shelfmarks": ["T-S 8J1.2"]}],
        "gt_jsonl": str(gt_file)}), encoding="utf-8")
    sets = h.load_new_eval_sets(reg_dir, None)
    assert len(sets) == 1 and sets[0].family == "new_eval"
    evals = h.EvalIndex(sets, h.TextRule())
    assert held(subject("", pgpids=["4242"]), evals)[0]["set"] == "ja_dev_v1"
    assert held(subject("", fl_ids=["FL200000042"]), evals)[0]["key"] == "ktiv_fl"
    assert held(subject("", photos=[h.Photo(sha256="c" * 64)]), evals)[0]["key"] == "image_sha256"
    assert held(subject("", row_id="fjp:Some_Doc:0"), evals)[0]["key"] == "name"
    hit, _ = evals.text_witnesses(gt_text)
    assert hit and hit[0]["set"] == "ja_dev_v1"
    assert h.load_new_eval_sets(tmp_path / "missing", None) == []
    (reg_dir / "bad.json").write_text(json.dumps({"eval_set": "bad", "items": [{"pgpids": ["1"]}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        h.load_new_eval_sets(reg_dir, None)


def test_judge_run_propagates_identity_within_group() -> None:
    """An identity hold-out of one member holds out every side of the group; a text hold-out stays on its side."""
    gt = letters(600, 90)
    evals = h.EvalIndex([eval_set(pgpids=["11"], gt=[("DOC9", gt)])], h.TextRule())
    reg = registry([])
    a = subject(letters(300, 91), row_id="cand:a", document_group="G", pgpids=["11"])
    b = subject(letters(300, 92), row_id="cand:b", document_group="G")
    c = subject(gt[:300], row_id="cand:c", document_group="H")
    d = subject(letters(300, 93), row_id="cand:d", document_group="H")
    recs = {r["row_id"]: r for r in h.judge_run([a, b, c, d], reg, evals, None, h.TextRule(), log_every=0).records}
    assert recs["cand:a"]["eval_holdout"] and recs["cand:b"]["eval_holdout"]
    assert recs["cand:c"]["eval_holdout"] and recs["cand:c"]["eval_text_holdout"]
    assert not recs["cand:d"]["eval_holdout"]
    assert recs["cand:a"]["tier"] == h.STRICT
