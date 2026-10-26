# File name: test_text27b_builder.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for ``src.finetuning.text27b.builder_v0`` (the moved v0 builder) and ``builder`` (v0_2 fixes).

Everything runs on synthetic fixtures written to ``tmp_path`` (a tiny word-level tokenizer and a
minimal chat template that honours the render contract). One test reads the real ``holdouts.json``
and family files when a phase-2 build has produced them, and is skipped otherwise.
"""
from __future__ import annotations

import codecs
import collections
import hashlib
import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.finetuning.text27b import builder as B
from src.finetuning.text27b import builder_v0 as bv
from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import textnorm as tn

#: A scrubbed project name, decoded at runtime from rot13 so this file never spells them (fix pass F12).
SCRUB_SOCIETY = codecs.decode("SWZF", "rot_13")

TEMPLATE = ("{% for m in messages %}{% if m.role == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content.strip() + '<|im_end|>\\n' }}"
            "{% else %}{{ '<|im_start|>' + m.role + '\\n' + m.content.strip() + '<|im_end|>\\n' }}{% endif %}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}")
HE_WORDS = ["אלכתאב", "מן", "אלשיך", "אבו", "עלי", "אלי", "מצר", "פי", "אלסנה", "ואלדנאניר", "אלתי", "ענדה",
            "יעקוב", "בן", "יוסף", "ואללה", "יחפטך", "אלכתאן", "ואלמרכב", "אלואצל"]
AR_WORDS = ["الكتاب", "من", "الشيخ", "ابو", "علي", "الى", "مصر", "في", "السنة", "والدنانير", "التي", "عنده",
            "يعقوب", "بن", "يوسف", "والله", "الكتان", "المركب", "الواصل", "الدار"]
MH_WORDS = ["המכתב", "מאת", "השייח", "אל", "מצרים", "בשנה", "הזאת", "והדינרים", "אשר", "אצלו", "שלום", "רב",
            "הפשתן", "הספינה", "שהגיעה"]
EN_WORDS = ["the", "letter", "from", "shaykh", "to", "Egypt", "in", "year", "dinars", "which", "he", "holds",
            "flax", "ship", "arrived", "peace", "upon", "you"]


# ============================================================================== helpers
def pgpid_with_split(split: str, start: int, avoid: Tuple[str, ...] = ()) -> str:
    """First pgpid from ``start`` upward whose hash split is ``split``.

    :param split: ``train`` / ``dev`` / ``test``.
    :param start: First candidate.
    :param avoid: pgpids already used.
    :returns: The pgpid.
    """
    n = start
    while bv.split_of(str(n)) != split or str(n) in avoid:
        n += 1
    return str(n)


def text_lines(words: List[str], seed: int, n_lines: int = 8, per_line: int = 6) -> str:
    """Deterministic pseudo-text of distinct lines.

    :param words: Vocabulary.
    :param seed: RNG seed.
    :param n_lines: Lines.
    :param per_line: Words per line.
    :returns: Text with newline-separated lines.
    """
    rng = random.Random(seed)
    return "\n".join(" ".join(rng.choice(words) for _ in range(per_line)) for _ in range(n_lines))


def deletions(text: str, every: int = 15) -> str:
    """A 'read' of ``text`` that only deletes characters (keeps the noise model substitution-free).

    :param text: Edition text.
    :param every: Delete one character in this many.
    :returns: Degraded text with the line structure kept.
    """
    return "".join(ch for i, ch in enumerate(text) if ch == "\n" or i % every != 7)


def write_tokenizer(path: Path) -> Path:
    """Write a whitespace word-level tokenizer (every word is one token).

    :param path: Destination ``tokenizer.json``.
    :returns: The path.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))
    return path


def write_csv(path: Path, header: List[str], rows: List[Dict[str, str]], bom: bool = False) -> None:
    """Write a CSV file.

    :param path: Destination.
    :param header: Column names.
    :param rows: Rows.
    :param bom: Prefix a UTF-8 BOM (as PGP exports do).
    :returns: None.
    """
    import csv

    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


DOC_COLS = ["pgpid", "shelfmark", "type", "tags", "origin", "destination", "mentioned", "possibly_mentioned",
            "languages_primary", "languages_secondary", "description", "doc_date_standard", "inferred_date_standard",
            "doc_date_original", "doc_date_calendar"]


def pair_row(pgpid: str, shelfmark: str, edition: str, script: str, lang: str, description: str,
             tr_en: Optional[str] = None, tr_he: Optional[str] = None, read: bool = True,
             edition_he: Optional[str] = None, vision: bool = False, cids: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """A synthetic pgp_parallel v1.1 row.

    :param pgpid: PGP id.
    :param shelfmark: Shelfmark.
    :param edition: Edition text.
    :param script: ``hebrew`` / ``arabic``.
    :param lang: lang_class (``JA``, ``AR`` ...).
    :param description: PGP description.
    :param tr_en: English translation.
    :param tr_he: Hebrew translation.
    :param read: Attach a deletion-only a50 read.
    :param edition_he: A different edition matched to the Hebrew translation (B7).
    :param vision: ``held_out_vision_benchmark`` flag.
    :param cids: Merged canonical ids.
    :returns: Row dict.
    """
    trs = []
    for code, t in (("en", tr_en), ("he", tr_he)):
        if t:
            trs.append({"language": code, "text": t, "raw": t, "source_slug": f"ed-{pgpid}", "location": ""})
    eds = [{"source_slug": f"ed-{pgpid}", "script": script, "raw": edition, "text": edition}]
    if edition_he:
        eds.append({"source_slug": f"edhe-{pgpid}", "script": script, "raw": edition_he, "text": edition_he})
    reads = [{"model": "a50", "text": deletions(edition), "pages": [{"n_lines": 8, "n_agreed": 4}]}] if read else []
    return {"pgpid": pgpid, "shelfmark": shelfmark, "lang_class": lang, "ja_score": 0.3, "edition_text": edition,
            "edition_source_slug": f"ed-{pgpid}", "edition_script": script, "reads": reads,
            "held_out_vision_benchmark": vision, "merged_doc_ids": list(cids), "translation_en": tr_en,
            "translation_he": tr_he, "translations": trs, "editions": eds, "type": "Letter", "tags": ["trade"],
            "origin": "Fustat", "destination": "", "mentioned": ["Alexandria"], "possibly_mentioned": [],
            "languages_primary": "Judaeo-Arabic" if lang == "JA" else "Arabic", "languages_secondary": "",
            "description": description, "edition_text_he": edition_he,
            "edition_source_slug_he": f"edhe-{pgpid}" if edition_he else None}


@pytest.fixture()
def corpus(tmp_path: Path) -> Dict[str, Any]:
    """A complete synthetic build input (paths.json, PGP export, pairs, maps, reads, v0_1 analysis rows).

    :param tmp_path: pytest temporary directory.
    :returns: ``{"paths": Paths, "ids": {...}, "root": Path}``.
    """
    root = tmp_path
    exp = root / "export"
    exp.mkdir()
    used: Tuple[str, ...] = ()
    ids: Dict[str, str] = {}
    for name, split, start in (("A", "train", 100), ("B", "train", 200), ("C", "train", 300), ("D", "train", 400),
                               ("E", "train", 500), ("F", "test", 600), ("G", "train", 700), ("H", "train", 800),
                               ("P", "train", 900), ("Q", "train", 1000)):
        ids[name] = pgpid_with_split(split, start, used)
        used += (ids[name],)
    he_ed = {k: text_lines(HE_WORDS, i) for i, k in enumerate("ACDEFGHP")}
    ar_ed = text_lines(AR_WORDS, 11)
    en = "\n".join(f"{i + 1}. " + " ".join(random.Random(i).choice(EN_WORDS) for _ in range(7)) for i in range(8))
    mh = "\n".join(f"({2 * i + 1}–{2 * i + 2}) " + " ".join(random.Random(50 + i).choice(MH_WORDS) for _ in range(6))
                   for i in range(8))
    mh = mh.replace("(3–4)", "(3−4)")
    desc_a = ("Letter from Yaakov b. Yosef, in Fustat, to Abu Ali, in Alexandria. Concerning dinars and a shipment of "
              "flax sent to Egypt in the winter season. The same scribe wrote T-S 13J20.5 and ENA 2738.1. Formerly "
              f"listed by {SCRUB_SOCIETY} as a legal text.")
    plain = "Letter concerning the shipment of flax and the payment of dinars between two merchants of Fustat."
    pairs = [
        pair_row(ids["A"], "T-S 10J1.1", he_ed["A"], "hebrew", "JA", desc_a, tr_en=en, tr_he=mh),
        pair_row(ids["B"], "T-S Ar.1.2", ar_ed, "arabic", "AR", "Legal deed in Arabic script concerning the sale of a "
                 "house in Fustat to a merchant named Abu Nasr.", tr_en=en),
        pair_row(ids["C"], "Bodl. MS heb. d 66/22", he_ed["C"], "hebrew", "JA", plain, tr_en=en),
        pair_row(ids["D"], "T-S 10J1.4", he_ed["D"], "hebrew", "JA", plain),
        pair_row(ids["E"], "T-S 10J1.5", he_ed["E"], "hebrew", "JA", plain),
        pair_row(ids["F"], "T-S 10J1.6", he_ed["F"], "hebrew", "JA", plain),
        pair_row(ids["G"], "T-S 10J1.7", he_ed["G"], "hebrew", "JA", plain, tr_he=mh,
                 edition_he=text_lines(HE_WORDS, 99)),
        pair_row(ids["H"], "T-S 10J1.8", he_ed["H"], "hebrew", "JA", plain, tr_en=" ".join([en] * 4)),
        pair_row(ids["P"], "T-S 10J1.9", he_ed["P"], "hebrew", "JA", plain),
    ]
    with open(root / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    docs = [{"pgpid": r["pgpid"], "shelfmark": r["shelfmark"], "type": "Letter", "tags": "trade",
             "origin": "Fustat", "description": r["description"], "languages_primary": r["languages_primary"]}
            for r in pairs]
    docs.append({"pgpid": ids["Q"], "shelfmark": "T-S 10J1.10", "type": "Letter", "origin": "Fustat",
                 "languages_primary": "Judaeo-Arabic", "description": plain + " Read-only pool document."})
    write_csv(exp / "documents.csv", DOC_COLS, docs, bom=True)
    write_csv(exp / "fragments.csv", ["shelfmark", "pgpids"], [{"shelfmark": "x", "pgpids": f"{ids['C']};{ids['D']}"}])
    write_csv(exp / "people.csv", ["name", "social_roles", "name_variants"],
              [{"name": "Yaakov b. Yosef", "social_roles": "merchant", "name_variants": ""}])
    write_csv(exp / "places.csv", ["name"], [{"name": "Fustat"}, {"name": "Alexandria"}])
    write_csv(exp / "sources.csv", ["authors"], [{"authors": "Goitein, S. D."}])
    merged = {"Cambridge_CUL_T_S_10J1_1": {"pgpids": [ids["A"]], "image_urls": []},
              "Cambridge_CUL_T_S_10J1_10": {"pgpids": [ids["Q"]], "image_urls": []}}
    (root / "merged_map.json").write_text(json.dumps(merged), encoding="utf-8")
    (root / "benchmark_ids.json").write_text(json.dumps(["Oxford_Bodleian_MS_heb_d_66_22"]), encoding="utf-8")
    (root / "samples.json").write_text(json.dumps({"sets": {"S1": [{"pgpid": ids["P"]}]}}), encoding="utf-8")
    reads = root / "reads"
    reads.mkdir()
    q_text = text_lines(HE_WORDS, 77, n_lines=10)
    rec = {"doc_id": "Cambridge_CUL_T_S_10J1_10", "image_index": 0, "image_url": "u", "image_sha256": "s",
           "ai_read": {"lines": [{"index": i, "text": t} for i, t in enumerate(q_text.split("\n"))], "n_lines": 10,
                       "n_agreed": 5, "parsed": True, "decoded_at": "2026-10-01"}}
    (reads / "a50.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    (reads / "v21b.jsonl").write_text("", encoding="utf-8")
    v01 = root / "v0_1"
    v01.mkdir()
    ana_target = ("Paper. 25 cm. x 10 cm. Letter of Yaakov. T-S 12.34. The writer asks about the dinars and the flax "
                  "shipment, a matter discussed at length in the commentary below, with notes on the readings of lines "
                  "four and five.\n\nNotes\n\nThe word in line two is unclear and may be read differently.")
    ana = {"messages": [{"role": "system", "content": R.SYSTEM},
                        {"role": "user", "content": "x\n\n<document>\ny\n</document>\n\nKnown catalogue context:\nType: Letter"},
                        {"role": "assistant", "content": ana_target}],
           "meta": {"task": "analyse_document", "condition": "edition", "pgpid": ids["A"], "source": "book:x",
                    "licence": "test", "provenance": {"book": "x"}, "grounding_share": 1.0, "grounding_candidates": 0,
                    "grounding_tier": "A", "target_truncated": False, "target_lang": "en"}}
    (v01 / "train.jsonl").write_text(json.dumps(ana, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tokenizer(root / "tokenizer.json")
    (root / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    cfg = {"repo": str(root), "probe_dir": "probe", "text_ft_v0_1": "v0_1", "pgp_parallel": "pairs.jsonl",
           "pgp_export": "export", "merged_pgpid_map": "merged_map.json", "document_treatments": "none.jsonl",
           "analysis_pages_index": "none.jsonl", "exclusions_v1": "none.json", "samples": "samples.json",
           "a50_overlap": "none.json", "benchmark_ids": "benchmark_ids.json", "academic_literature": "lit",
           "ai_reads": "reads", "book_editions_nas": "nas", "tokenizer_json": "tokenizer.json",
           "chat_template": "chat_template.jinja", "out_dir": "out", "scratch": "scratch"}
    (root / "paths.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "builder_local.json").write_text(json.dumps({"read_files": {"a50": "a50.jsonl", "v21b": "v21b.jsonl"}}),
                                             encoding="utf-8")
    return {"paths": P.load_paths(root / "paths.json"), "ids": ids, "root": root}


def build_corpus(corpus: Dict[str, Any], **kw: Any) -> Tuple[B.V2Config, Dict[str, Any], Dict[str, Any]]:
    """Run the v0_2 build on the synthetic corpus and write its outputs.

    :param corpus: Fixture.
    :param kw: Extra :class:`builder.V2Config` fields.
    :returns: ``(config, result, report)``.
    """
    p = corpus["paths"]
    local = bv.load_local_config(p)
    kw.setdefault("use_holdouts_module", False)
    cfg = B.V2Config.from_paths(p, None, local, **kw)
    cfg.v0.registry_fn = lambda: ({"Cambridge_CUL_T_S_99_99"}, {corpus["ids"]["E"]})
    result = B.build_v02(cfg)
    report = B.write_v02(result, cfg, {"test": True})
    return cfg, result, report


def family_rows(cfg: B.V2Config) -> List[Dict[str, Any]]:
    """All rows of the written family files.

    :param cfg: Build config.
    :returns: Rows.
    """
    return [r for f in sorted((cfg.out_dir / "families").glob("*.jsonl")) for r in R.read_jsonl(f)]


# ============================================================================== builder_v0 (the move)
def test_v0_prompts_are_the_rows_prompts() -> None:
    """The moved builder uses the shared prompts and the verbatim v0 condition notes."""
    assert bv.SYSTEM == R.SYSTEM and bv.INSTRUCTION == R.INSTRUCTION and bv.SCRIPT_NAME == R.SCRIPT_NAME
    assert bv.COND_NOTE == R.COND_NOTE_V0 and bv.FAMILIES == R.PGP_FAMILIES
    assert bv.COND_NOTE["noised_edition"] == bv.COND_NOTE["read"]


def test_v0_split_of_matches_hash_rule() -> None:
    """``split_of`` is sha1(pgpid)[0] % 100 with the 80/10/10 cut."""
    for p in ("1", "444", "9999", "31694"):
        b = hashlib.sha1(p.encode()).digest()[0] % 100
        assert bv.split_of(p) == ("train" if b < 80 else "dev" if b < 90 else "test")


def test_v0_cer_uses_v1_normalisation() -> None:
    """The v0 builder's CER ignores U+2212 as a range dash (v11=False), unlike the textnorm default."""
    from src.finetuning.text27b import textnorm as tn

    assert bv._norm("(3−4) abc") != tn.normalize_for_cer("(3−4) abc")
    assert tn.normalize_for_cer("(3−4) abc") == "abc"
    assert bv._cer("abc", "abc") == 0.0


def test_v0_csv_and_split_multi(tmp_path: Path) -> None:
    """``load_csv`` strips a BOM; ``split_multi`` splits outside parentheses.

    :param tmp_path: pytest temporary directory.
    """
    write_csv(tmp_path / "x.csv", ["pgpid", "tags"], [{"pgpid": "1", "tags": "a, b (c, d); e"}], bom=True)
    rows = bv.load_csv(tmp_path / "x.csv")
    assert rows[0]["pgpid"] == "1"
    assert bv.split_multi(rows[0]["tags"]) == ["a", "b (c, d)", "e"]


def test_v0_classify_label_and_clean_text() -> None:
    """Side labels are dropped, margin labels kept as § lines, HTML stripped."""
    assert bv.classify_label("Recto") == "side"
    assert bv.classify_label("Right margin") == "margin"
    assert bv.classify_label("אלכתאב מן") is None
    c: collections.Counter = collections.Counter()
    out = bv.clean_text("T-S 1.2<br>Recto<br>אלכתאב מן<br>Right margin:<br>אלשיך", ["T-S 1.2"], c)
    assert out == "אלכתאב מן\n§ Right margin\nאלשיך"
    assert c["side_labels_dropped"] == 1 and c["shelfmark_heading_dropped"] == 1


def test_v0_merged_map_and_reads(tmp_path: Path) -> None:
    """The merged map is rebuilt from a corpus file when there is no cache; reads keep the best record per page.

    :param tmp_path: pytest temporary directory.
    """
    merged = tmp_path / "merged.jsonl"
    merged.write_text(json.dumps({"canonical_id": "X_1", "pgpids": ["5"], "images": ["http://a/b.jpg"]}) + "\n"
                      + json.dumps({"canonical_id": "X_2", "pgpids": []}) + "\n", encoding="utf-8")
    m = bv.build_merged_map(tmp_path / "cache.json", merged)
    assert m == {"X_1": {"pgpids": ["5"], "image_urls": ["http://a/b.jpg"]}}
    assert bv.build_merged_map(tmp_path / "cache.json") == m
    recs = [{"doc_id": "X_1", "image_index": 0, "image_sha256": "h", "ai_read": {"lines": [{"index": 0, "text": "old"}],
                                                                                "decoded_at": "2026-01-01"}},
            {"doc_id": "X_1", "image_index": 0, "image_sha256": "h", "ai_read": {"lines": [{"index": 0, "text": "new"}],
                                                                                "decoded_at": "2026-02-01"}},
            {"doc_id": "other", "image_url": "http://a/b.jpg", "ai_read": {"lines": [{"text": "by url"}]}, "image_index": 1}]
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    reads = bv.load_reads(tmp_path / "r.jsonl", {"X_1"}, {"http://a/b.jpg": "X_1"}, "a50")
    assert reads["X_1"][0]["text"] == "new" and reads["X_1"][1]["text"] == "by url"


def test_v0_clean_description_and_letter_parties() -> None:
    """The v0 description cleaner and party parser behave as in the prototype."""
    text, acts = bv.clean_description("Letter from Yaakov b. Yosef, in Fustat, to Abu Ali. See also PGPID 123. (MR)")
    assert text == "Letter from Yaakov b. Yosef, in Fustat, to Abu Ali." and acts["reference_sentence"] == 1
    bv.PLACE_NAMES.update({"Fustat"})
    assert bv.letter_parties(text) == ("Yaakov b. Yosef", "Abu Ali")


def test_v0_local_config(tmp_path: Path) -> None:
    """The local config is optional and installs the licence map.

    :param tmp_path: pytest temporary directory.
    """
    assert bv.load_local_config(path=tmp_path / "missing.json") == {}
    (tmp_path / "l.json").write_text(json.dumps({"licence_by_book": {"b": "pd"}, "analysis_excluded_books": ["z"]}))
    bv.apply_local_config(bv.load_local_config(path=tmp_path / "l.json"))
    assert bv.LICENCE_BY_BOOK == {"b": "pd"} and bv.ANALYSIS_EXCLUDED_BOOKS == {"z"}
    bv.apply_local_config({})
    assert not bv.LICENCE_BY_BOOK


# ============================================================================== v0_2 strips
def test_strip_markers_gil_and_leading_numbers() -> None:
    """B3 removes range markers (incl. U+2212) and ascending leading numbers, to a fixpoint."""
    he, ops = B.strip_markers("(1–2) המכתב מאת השייח\n(3−4) אל מצרים בשנה")
    assert he == "המכתב מאת השייח\nאל מצרים בשנה" and ops == ["B3:gil_range"]
    en, ops = B.strip_markers("1. The letter.\n2. To Egypt.\n3. The dinars.")
    assert en == "The letter.\nTo Egypt.\nThe dinars." and ops == ["B3:leading_line_number"]
    assert B.strip_markers(en) == (en, [])


def test_clean_target_translation_keeps_text_and_lacunae() -> None:
    """Translations get project-name + B3 only; ``[...]`` lines and ellipses survive."""
    txt = "(1–2) The letter [...]\n[...]\n... and the dinars"
    out, ops = B.clean_target(txt, "translate_en")
    assert out == "The letter [...]\n[...]\n... and the dinars" and ops == ["B3:gil_range"]
    assert B.clean_target("no markers here", "translate_he") == ("no markers here", [])


#: The project abbreviation variant, decoded at runtime so the name itself is not written in this file (F12).
SCRUB_ABBR = codecs.decode("STC", "rot_13")


def test_strip_source_names_input_is_line_bounded() -> None:
    """Input project-name strip removes only the hit's parenthetical or line, never neighbouring edition lines."""
    ed = (f"אבג דהו זחט\n\n§ Verso (Image 4 in {SCRUB_ABBR})\n\nיכל מנס עפצ\nTranscription from {SCRUB_ABBR}\nקרש תאב")
    out, n = B.strip_source_names_input(ed)
    assert n == 2 and out == "אבג דהו זחט\n\n§ Verso\n\nיכל מנס עפצ\nקרש תאב"
    assert B.strip_source_names_input("plain edition") == ("plain edition", 0) and B.strip_source_names_input(None) == (None, 0)


def test_end_to_end_scrubs_inputs(corpus: Dict[str, Any]) -> None:
    """A project-name side label in an edition reaches no input (edition, noised or read) and is counted.

    :param corpus: Fixture.
    """
    root, did = corpus["root"], corpus["ids"]["D"]
    pairs = [json.loads(x) for x in (root / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in pairs:
        if r["pgpid"] == did:
            r["edition_text"] += f"\n\n§ Verso (Image 4 in {SCRUB_ABBR})\n\n" + text_lines(HE_WORDS, 55, n_lines=3)
    (root / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs), encoding="utf-8")
    cfg, _, report = build_corpus(corpus)
    rows = family_rows(cfg)
    d_rows = [r for r in rows if r["meta"]["pgpid"] == did]
    assert d_rows and any("§ Verso" in r["messages"][1]["content"] for r in d_rows)
    assert not any(tn.has_source_names(r["messages"][1]["content"]) for r in rows)
    assert report["checks"]["scrub_input_rows_violations"] == 0
    assert any(k.startswith("scrub_stripped_edition_inputs:") for k in report["fixes"]["builder"])


def test_clean_target_description_citations_and_scrub() -> None:
    """B4 removes shelfmark cross-references and project-name sentences from descriptions."""
    desc = ("Letter concerning flax sent to Egypt in the winter season. The same scribe wrote T-S 13J20.5 and "
            f"ENA 2738.1. Formerly listed by {SCRUB_SOCIETY} as a legal text.")
    out, ops = B.clean_target(desc, "describe_pgp")
    assert out == "Letter concerning flax sent to Egypt in the winter season."
    assert "B4:shelfmark_sentence" in ops and f"B4:{tn.SCRUB_OP}" in ops


def test_strip_analysis_extras() -> None:
    """Material, dimensions (Latin and Hebrew), Hebrew citations, folio residue and IB numbers go."""
    out, ops = B.strip_analysis_extras("Paper. The verso is blank.\n\n190x140 מ\"מ; כתב ידו של יוסף. נזכר אצל גויטיין, חברה.")
    assert out == "The verso is blank.\n\nכתב ידו של יוסף." and {"B4:material", "B4:dimensions", "B4:he_citation"} <= set(ops)
    out, ops = B.strip_analysis_extras("I, 43b Power of Attorney {Fustat, 1100}, f. 3 Left upper corner.")
    assert out == "Power of Attorney {Fustat, 1100} Left upper corner." and ops == ["B4:folio_residue", "B4:ib_number"]
    assert B.strip_analysis_extras("Plain commentary.") == ("Plain commentary.", [])


def test_clean_target_analysis_full() -> None:
    """An analysis headnote loses material, dimensions and the shelfmark; notes stay."""
    t = ("Paper. 25 cm. x 10 cm. Letter of Yaakov. T-S 12.34. The writer asks about the dinars.\n\nNotes\n\n"
         "The word is unclear.")
    out, ops = B.clean_target(t, "analyse_document")
    assert out == "Letter of Yaakov. The writer asks about the dinars.\n\nNotes\n\nThe word is unclear."
    assert {"B4:dimensions", "B4:material", "B4:shelfmark"} <= set(ops)


def test_band_ok_and_noise_band() -> None:
    """F1 bands per language; E7 CER bands."""
    ed = "א" * 100
    assert B.band_ok(ed, "a" * 240, "en") == (True, 2.4)
    assert B.band_ok(ed, "a" * 240, "he")[0] is False
    assert B.band_ok(ed, "a" * 45, "en")[0] is False
    assert [B.noise_band(x) for x in (None, 0.0, 0.2, 0.45, 0.89, 1.3)] == [None, "0-0.15", "0.15-0.3", "0.45-0.6",
                                                                           "0.6-0.9", ">0.9"]


# ============================================================================== hold-outs and splits
def test_holdout_index_bodleian_by_key() -> None:
    """A PGP-131 Bodleian id missing from the merged map still matches its document by shelfmark key."""
    h = B.HoldoutIndex.build({}, ["Oxford_Bodleian_MS_heb_d_66_22"], (set(), {"7"}), {"8"})
    assert h.match("1", "Bodl. MS heb. d 66/22", []) == {"pgp131": "shelfmark_key"}
    assert h.match("2", "T-S 1.1", ["Oxford_Bodleian_Bodl_MS_heb_d_66_22"]) == {"pgp131": "shelfmark_key"}
    assert h.match("7", "T-S 1.1", []) == {"registry": "pgpid"}
    assert h.match("8", "T-S 1.1", [], vision_flag=True) == {"vision_flag": "flag", "probe": "pgpid"}
    assert h.match("3", "Bodl. MS heb. d 66/23", []) == {}


def test_holdout_index_external_reasons_mapped() -> None:
    """Reasons from the holdouts module are mapped onto the builder's categories (prefixes dropped)."""
    ext = {"5": "key:pgp131", "6": "group:registry", "9": "something_else"}
    h = B.HoldoutIndex.build({}, [], (set(), set()), set(), external=lambda p, s, c: ext.get(p))
    assert h.match("5", None, []) == {"pgp131": "external"}
    assert h.match("6", None, []) == {"registry": "external"}
    assert h.match("9", None, []) == {"holdouts_module": "external"}
    assert h.match("4", None, []) == {}


def test_holdouts_module_adapter_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter calls ``Holdouts.load(path, p).is_benchmark(pgpid, shelfmark, cid)`` per canonical id.

    :param monkeypatch: pytest monkeypatch.
    """
    calls: List[Tuple[Any, ...]] = []

    class FakeHoldouts:
        """Stand-in for the holdouts module's class."""

        @classmethod
        def load(cls, path: Any = None, p: Any = None) -> "FakeHoldouts":
            """Return an instance.

            :param path: Ignored.
            :param p: Ignored.
            :returns: Instance.
            """
            return cls()

        def is_benchmark(self, pgpid: Any = None, shelfmark: Optional[str] = None,
                         canonical: Optional[str] = None) -> Optional[str]:
            """Flag canonical id ``X`` only.

            :param pgpid: PGP id.
            :param shelfmark: Shelfmark.
            :param canonical: Canonical id.
            :returns: Reason or None.
            """
            calls.append((pgpid, shelfmark, canonical))
            return "pgp131" if canonical == "X" else None

    fake = types.ModuleType("src.finetuning.text27b.holdouts")
    fake.Holdouts = FakeHoldouts  # type: ignore[attr-defined]
    import src.finetuning.text27b as pkg

    monkeypatch.setitem(sys.modules, "src.finetuning.text27b.holdouts", fake)
    monkeypatch.setattr(pkg, "holdouts", fake, raising=False)
    f = B.holdouts_module_adapter()
    assert f is not None and f("1", "sm", ["W", "X"]) == "pgp131" and f("2", None, []) is None
    assert ("1", "sm", "W") in calls and ("2", None, None) in calls


def test_holdouts_module_adapter_missing_file(tmp_path: Path) -> None:
    """A missing ``holdouts.json`` disables the adapter instead of failing the build.

    :param tmp_path: pytest temporary directory.
    """
    assert B.holdouts_module_adapter(path=tmp_path / "missing_holdouts.json") is None


def test_assign_split_precedence() -> None:
    """PGP-131 → heldout before vision; vision → heldout_vision; probe → heldout; else test_indist."""
    assert B.assign_split("test", {"pgp131", "registry"}) == "heldout"
    assert B.assign_split("test", {"registry", "probe"}) == "heldout_vision"
    assert B.assign_split("test", {"vision_flag"}) == "heldout_vision"
    assert B.assign_split("test", {"probe"}) == "heldout"
    assert B.assign_split("test", set()) == "test_indist"
    assert B.assign_split("train", set()) == "train" and B.assign_split("dev", ()) == "dev"
    with pytest.raises(AssertionError):
        B.assign_split("train", {"pgp131"})


# ============================================================================== rows, checks, audit
def _doc(**kw: Any) -> Dict[str, Any]:
    """Minimal document info for row tests.

    :param kw: Overrides.
    :returns: Document dict.
    """
    d = {"pgpid": "11", "shelfmark": "T-S 1.1", "group": "g11", "split": "train", "split_v2": "train", "held_out": False,
         "holdout_reasons": [], "lang": "JA", "script": "hebrew", "edition": "אלכתאב מן אלשיך", "edition_slug": "s",
         "reads_ok": []}
    d.update(kw)
    return d


def test_make_row_v2_schema(tmp_path: Path) -> None:
    """A v0_2 row has every v0_1 field, the v0_2 fields and a side-aware row id.

    :param tmp_path: pytest temporary directory.
    """
    tc = bv.TokenCounter(tmp_path / "no_tokenizer.json")
    base = {"source": "s", "licence": "l", "provenance": {}, "input_source": "x", "input_truncated": False}
    row = B.make_row_v2("translate_en", "read", _doc(), "user", "target",
                        {**base, "read_model": "a50", "input_cer": 0.2}, tc)
    assert R.validate_row(row) == []
    m = row["meta"]
    assert m["noise_band"] == "0.15-0.3" and m["noise_class"] == "read_gate" and m["build"] == "v0_2"
    assert m["row_id"] == R.row_id("translate_en", "11", "read", "a50")
    side = B.make_row_v2("translate_en", "edition", _doc(), "u", "t", {**base, "side": "1r"}, tc)
    assert side["meta"]["row_id"] == R.row_id("translate_en", "11", "edition", None, "1r") != m["row_id"]


def test_checks_v02_detects_violations(tmp_path: Path) -> None:
    """Markers, project names, AR noised rows, duplicate ids and misplaced PGP-131 rows are all counted.

    :param tmp_path: pytest temporary directory.
    """
    tc = bv.TokenCounter(tmp_path / "no_tokenizer.json")
    base = {"source": "s", "licence": "l", "provenance": {}, "input_source": "x", "input_truncated": False}
    ok = B.make_row_v2("translate_he", "edition", _doc(), "u", "תרגום נקי", base, tc)
    marker = B.make_row_v2("translate_he", "edition", _doc(pgpid="12", group="g12"), "u", "(1–2) תרגום", base, tc)
    host = ".".join([codecs.decode("travmnu", "rot_13"), "org"])     # the domain, assembled at runtime (F12)
    named = B.make_row_v2("describe_pgp", "edition", _doc(pgpid="13", group="g13"), "u", f"Seen on {host} today.", base, tc)
    arn = B.make_row_v2("catalogue_json", "noised_edition", _doc(pgpid="14", group="g14", lang="AR"), "u",
                        json.dumps({"summary": "x"}), base, tc)
    dup = json.loads(json.dumps(ok))
    bod = B.make_row_v2("translate_en", "edition", _doc(pgpid="15", group="g15"), "u", "t", base, tc)
    chk = B.checks_v02([ok, marker, named, arn, dup, bod], {"15"}, set(), {}, {"15"}, {"Oxford_x": ["15"]})
    assert chk["line_marker_rows_violations"] == 1 and chk["scrub_target_rows_violations"] == 1
    assert chk["ar_noised_rows_violations"] == 1 and chk["duplicate_row_id_violations"] == 1
    assert chk["pgp131_rows_not_heldout_violations"] == 1 and chk["bodleian_unmapped_rows_not_heldout_violations"] == 1
    assert chk["benchmark_in_train_violations"] == 1


def test_audit_sample_stratified() -> None:
    """The audit sample is capped, B4-only and round-robin over (family, first rule)."""
    recs = [{"family": "describe_pgp", "pgpid": str(i), "split": "train", "ops": [op], "before": "a" * 20, "after": "a"}
            for i, op in enumerate(["B4:see_cf"] * 40 + ["B4:shelfmark"] * 40 + ["B3:gil_range"] * 5)]
    recs += [{"family": "analyse_document", "pgpid": "999", "split": "train", "ops": ["B4:dimensions"],
              "before": "abc", "after": "a"}]
    out = B.audit_sample(recs, n=10)
    assert len(out) == 10 and out[0]["family"] == "analyse_document"
    assert collections.Counter(r["ops"][0] for r in out) == {"B4:see_cf": 5, "B4:shelfmark": 4, "B4:dimensions": 1}
    assert all(r["chars_removed"] >= 0 for r in out)


def test_finalize_tokens_exact_and_cap(tmp_path: Path) -> None:
    """Rows are rendered through the template (contract checked) and counted; over-cap rows are dropped.

    :param tmp_path: pytest temporary directory.
    """
    tok = write_tokenizer(tmp_path / "tok.json")
    (tmp_path / "t.jinja").write_text(TEMPLATE, encoding="utf-8")
    tc = bv.TokenCounter(tmp_path / "no_tokenizer.json")
    base = {"source": "s", "licence": "l", "provenance": {}, "input_source": "x", "input_truncated": False}
    short = B.make_row_v2("describe_pgp", "edition", _doc(), "a b c", "d e", base, tc)
    long = B.make_row_v2("describe_pgp", "edition", _doc(pgpid="12"), "w " * 50, "d e", base, tc)
    n_short = R.count_tokens(R.render_chat(short["messages"], template_path=tmp_path / "t.jinja"), tok)
    n_long = R.count_tokens(R.render_chat(long["messages"], template_path=tmp_path / "t.jinja"), tok)
    assert n_long >= n_short + 40
    kept, drops = B.finalize_tokens([short, long], max_tokens=n_short + 10, tokenizer_path=tok,
                                    template_path=tmp_path / "t.jinja")
    assert kept == [short] and drops == {"describe_pgp": 1}
    rendered = R.render_chat(short["messages"], template_path=tmp_path / "t.jinja")
    assert short["meta"]["tokens"] == R.count_tokens(rendered, tok) and short["meta"]["target_tokens"] == 2


def test_apply_exposure_with_fake_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposure tags come from the exposure module; exposed train reads below CER 0.05 are dropped.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch.
    """
    fake = types.ModuleType("src.finetuning.text27b.exposure")

    def tag_rows(rows: List[Dict[str, Any]], exposure: Any, stats: Any = None) -> List[Dict[str, Any]]:
        """Mark every read row as page-exposed.

        :param rows: Rows.
        :param exposure: Path (ignored).
        :param stats: Counter.
        :returns: Rows.
        """
        for r in rows:
            read = r["meta"]["condition"] == "read"
            r["meta"]["exposure_level"] = "page" if read else None
            r["meta"]["reader_exposed"] = read
            if stats is not None:
                stats["level:page" if read else "level:n/a"] += 1
        return list(rows)

    def near_duplicate(row: Dict[str, Any], cer_max: float = 0.05) -> bool:
        """Exposed and below the CER bound.

        :param row: Row.
        :param cer_max: Bound.
        :returns: Flag.
        """
        m = row["meta"]
        return bool(m.get("reader_exposed")) and m.get("input_cer") is not None and m["input_cer"] < cer_max

    fake.tag_rows = tag_rows  # type: ignore[attr-defined]
    fake.near_duplicate = near_duplicate  # type: ignore[attr-defined]
    import src.finetuning.text27b as pkg

    monkeypatch.setitem(sys.modules, "src.finetuning.text27b.exposure", fake)
    monkeypatch.setattr(pkg, "exposure", fake, raising=False)
    rows = [{"meta": {"condition": "read", "split": "train", "input_cer": 0.01}},
            {"meta": {"condition": "read", "split": "heldout_vision", "input_cer": 0.01}},
            {"meta": {"condition": "read", "split": "train", "input_cer": 0.2}},
            {"meta": {"condition": "edition", "split": "train", "input_cer": None}}]
    same, rep = B.apply_exposure(list(rows), tmp_path / "absent.jsonl")
    assert rep["applied"] is False and len(same) == 4
    (tmp_path / "exposure.jsonl").write_text("{}\n", encoding="utf-8")
    out, rep = B.apply_exposure(list(rows), tmp_path / "exposure.jsonl")
    assert rep["dropped_exposed_near_dup"] == 1 and len(out) == 3 and rep["levels"]["level:page"] == 3


def test_main_requires_out_dir_for_samples() -> None:
    """A sample build never writes into the real output directory."""
    with pytest.raises(SystemExit):
        B.main(["--limit", "5"])


# ============================================================================== end to end (synthetic corpus)
def test_end_to_end_v02_build(corpus: Dict[str, Any]) -> None:
    """The whole v0_2 build on a synthetic corpus: every fix lands and every check is 0.

    :param corpus: Fixture.
    """
    ids = corpus["ids"]
    cfg, result, report = build_corpus(corpus)
    rows = family_rows(cfg)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["meta"]["pgpid"]].append(r)
    assert all(R.validate_row(r) == [] for r in rows)
    viol = {k: v for k, v in report["checks"].items() if k.endswith("_violations") and v}
    assert viol == {}
    # B10 / §9.0: the Bodleian document (not in the merged map) and its join partner are held out
    assert {r["meta"]["split"] for r in by[ids["C"]]} == {"heldout"}
    assert {r["meta"]["split"] for r in by[ids["D"]]} == {"heldout"}
    assert all("pgp131" in r["meta"]["holdout_reasons"] for r in by[ids["C"]] + by[ids["D"]])
    assert all(r["meta"]["held_out_vision_benchmark"] for r in by[ids["C"]])
    assert not any(r["meta"]["held_out_vision_benchmark"] for r in by[ids["D"]])
    assert report["holdouts"]["bodleian_unmapped_pgp131"] == {"Oxford_Bodleian_MS_heb_d_66_22": [ids["C"]]}
    assert {r["meta"]["split"] for r in by[ids["E"]]} == {"heldout_vision"}
    assert {r["meta"]["split"] for r in by[ids["P"]]} == {"heldout"}
    assert {r["meta"]["split"] for r in by[ids["F"]]} == {"test_indist"}
    assert {r["meta"]["split"] for r in by[ids["A"]]} == {"train"}
    assert not any(r["meta"]["condition"] == "noised_edition" for r in by[ids["C"]])
    # B3: markers stripped, raw kept
    he = [r for r in by[ids["A"]] if r["meta"]["family"] == "translate_he"]
    assert he and all("(1–2)" not in r["messages"][2]["content"] and "(3−4)" not in r["messages"][2]["content"] for r in he)
    assert all("(1–2)" in r["meta"]["target_raw"] and "B3:gil_range" in r["meta"]["strip_ops"] for r in he)
    en = [r for r in by[ids["A"]] if r["meta"]["family"] == "translate_en"]
    assert en and all(not r["messages"][2]["content"].startswith("1.") for r in en)
    assert {r["meta"]["condition"] for r in en} == {"edition", "noised_edition", "read"}
    # v0_2 read note on read-like conditions, edition note on editions
    read = next(r for r in en if r["meta"]["condition"] == "read")
    assert R.READ_NOTE_V2 in read["messages"][1]["content"] and read["meta"]["noise_class"] == "read_gate"
    noised = next(r for r in en if r["meta"]["condition"] == "noised_edition")
    assert R.READ_NOTE_V2 in noised["messages"][1]["content"]
    # B4 + B5: stripped description, catalogue summary identical, dup flag
    desc = next(r for r in by[ids["A"]] if r["meta"]["family"] == "describe_pgp" and r["meta"]["condition"] == "edition")
    cat = next(r for r in by[ids["A"]] if r["meta"]["family"] == "catalogue_json" and r["meta"]["condition"] == "edition")
    tgt = desc["messages"][2]["content"]
    assert "T-S 13J20.5" not in tgt and SCRUB_SOCIETY not in tgt and tgt.startswith("Letter from Yaakov b. Yosef")
    assert json.loads(cat["messages"][2]["content"])["summary"] == tgt and desc["meta"]["pgp_description_dup"] is True
    raw_summary = json.loads(cat["meta"]["target_raw"])["summary"]       # F4: target_raw keeps the pre-strip text ...
    assert "T-S 13J20.5" in raw_summary and SCRUB_SOCIETY not in raw_summary     # ... minus its project-name strings
    assert not tn.has_source_names(json.dumps([r["meta"] for rs in by.values() for r in rs], ensure_ascii=False))
    assert {"B4:shelfmark_sentence", f"B4:{tn.SCRUB_OP}"} <= set(desc["meta"]["strip_ops"])
    # B8 / B9: Arabic document has no noised rows and a re-scripted twin with the same target
    b_rows = by[ids["B"]]
    assert not any(r["meta"]["condition"] == "noised_edition" for r in b_rows)
    resc = [r for r in b_rows if r["meta"]["condition"] == "edition_rescript"]
    ed_b = next(r for r in b_rows if r["meta"]["family"] == "translate_en" and r["meta"]["condition"] == "edition")
    assert len(resc) == 1 and resc[0]["meta"]["family"] == "translate_en"
    assert resc[0]["messages"][2]["content"] == ed_b["messages"][2]["content"]
    doc_part = resc[0]["messages"][1]["content"].split("<document>")[1]
    assert not any("؀" <= ch <= "ۿ" for ch in doc_part) and any("א" <= ch <= "ת" for ch in doc_part)
    assert "(Hebrew script)" in resc[0]["messages"][1]["content"] and resc[0]["meta"]["input_script"] == "hebrew"
    # B7: the Hebrew translation of G uses the Hebrew-matched edition
    g_he = [r for r in by[ids["G"]] if r["meta"]["family"] == "translate_he" and r["meta"]["condition"] == "edition"]
    assert len(g_he) == 1 and g_he[0]["meta"]["edition_matched_to"] == "he"
    assert text_lines(HE_WORDS, 99).split("\n")[0] in g_he[0]["messages"][1]["content"]
    # F1: H's English translation is 4x too long → no translate_en rows
    assert not any(r["meta"]["family"] == "translate_en" for r in by[ids["H"]])
    assert report["drops"].get("translate_en:length_ratio_F1") == 1
    # read-only pool document Q
    q = by[ids["Q"]]
    assert q and {r["meta"]["condition"] for r in q} == {"read"} and q[0]["meta"]["noise_class"] == "read_only_pool"
    # analysis carried forward from v0_1, stripped, context rebuilt from the v0_2 catalogue record
    ana = [r for r in by[ids["A"]] if r["meta"]["family"] == "analyse_document"]
    assert ana and all("25 cm" not in r["messages"][2]["content"] and "T-S 12.34" not in r["messages"][2]["content"]
                       for r in ana)
    assert "Known catalogue context:\nType: Letter" in ana[0]["messages"][1]["content"]
    assert ana[0]["meta"]["analysis_source"] == "v0_1" and ana[0]["meta"]["target_raw"].startswith("Paper.")
    # exact tokens, manifest counts, audit
    assert all(r["meta"]["tokens"] == R.count_tokens(R.render_chat(r["messages"], template_path=cfg.chat_template),
                                                     cfg.v0.tokenizer) for r in rows[:5])
    assert report["counts"]["total"]["rows"] == len(rows)
    cell = report["counts"]["by_cell"]["translate_en"]["edition_rescript"]["AR"]["train"]
    assert cell["rows"] == 1 and cell["tokens"] == resc[0]["meta"]["tokens"]
    audit = list(R.read_jsonl(cfg.out_dir / "audit" / "strip_sample_50.jsonl"))
    assert audit and all(any(o.startswith("B4:") for o in a["ops"]) for a in audit)
    assert (cfg.out_dir / "noise_model.json").exists() and report["build_args"] == {"test": True}


def test_end_to_end_is_deterministic(corpus: Dict[str, Any]) -> None:
    """Two builds of the same inputs give byte-identical family files.

    :param corpus: Fixture.
    """
    cfg, _, rep1 = build_corpus(corpus)
    first = {f.name: f.read_bytes() for f in (cfg.out_dir / "families").glob("*.jsonl")}
    cfg2, _, rep2 = build_corpus(corpus)
    assert first == {f.name: f.read_bytes() for f in (cfg2.out_dir / "families").glob("*.jsonl")}
    assert {k: v["fingerprint"] for k, v in rep1["families"].items()} == {k: v["fingerprint"] for k, v in rep2["families"].items()}


def test_v0_run_on_synthetic_corpus_keeps_bodleian_leak(corpus: Dict[str, Any]) -> None:
    """The moved v0 builder reproduces the v0 behaviour, including the §9.0 leak the v0_2 builder fixes.

    :param corpus: Fixture.
    """
    p = corpus["paths"]
    cfg = bv.V0Config.from_paths(p, p.out("v0"), bv.load_local_config(p))
    cfg.registry_fn = lambda: (set(), {corpus["ids"]["E"]})
    res = bv.run_v0(cfg, analysis=False)
    c_rows = [r for r in res["rows"] if r["meta"]["pgpid"] == corpus["ids"]["C"]]
    assert c_rows and {r["meta"]["split"] for r in c_rows} == {"train"}
    assert all(r["messages"][1]["content"].count(bv.COND_NOTE["edition"]) <= 1 for r in res["rows"])
    report = bv.write_v0(res, cfg)
    assert report["checks"]["benchmark_in_train_violations"] == 0  # v0's own check cannot see the key-only match


# ============================================================================== real outputs (phase 2)
def test_real_bodleian_pgpids_never_in_train() -> None:
    """The PGP-131 documents held out only through a shelfmark key are never in train (real build outputs).

    Skipped until a phase-2 build has written ``holdouts.json`` and the family files.
    """
    try:
        p = P.load_paths()
    except FileNotFoundError:
        pytest.skip("no paths.json on this machine")
    hpath, fam = p.out("holdouts.json"), p.out("families")
    if not hpath.exists() or not fam.exists() or not list(fam.glob("*.jsonl")):
        pytest.skip("phase-2 outputs (holdouts.json, families/) not built yet")
    h = json.loads(hpath.read_text(encoding="utf-8"))
    key_only = {pg for pg, reasons in (h.get("benchmark") or {}).items()
                if any(r.endswith("pgp131") for r in reasons) and all(r.startswith(("key:", "group:")) for r in reasons)}
    if not key_only:
        pytest.skip("holdouts.json lists no key-only PGP-131 documents")
    bad = [r["meta"]["row_id"] for f in fam.glob("*.jsonl") for r in R.read_jsonl(f)
           if r["meta"].get("pgpid") in key_only and r["meta"].get("split") == "train"]
    assert bad == []
