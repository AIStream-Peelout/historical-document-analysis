# File name: test_text27b_literature_l1.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``literature_l1``: the ported quality rules, page-furniture stripping, the
literature manifest (metadata matching per part, licence classes, hand-edit merge), chunking,
probe / cap selection, duplicate pages, exclusions, the project-name strip, the end-to-end build and the
CLI (manifest, dry run, build).

Everything runs on a synthetic literature tree in ``tmp_path`` with a word-level tokenizer and a
small chat template; one test also uses the real local Qwen3.8 tokenizer / template when present.
No real work names are used.
"""
import codecs
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.finetuning.text27b import literature_l1 as L
from src.finetuning.text27b import pages as G
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import textnorm as T

REQUIRED = ["probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
            "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
            "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch"]

#: Chat template with the Qwen3.8 thinking behaviour that matters (same shape as the core tests').
TEMPLATE = (
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

#: A name the project-name strip removes, decoded at runtime from rot13 so it never appears literally (F12).
SCRUB_NAME = codecs.decode("Sevrqoret", "rot_13")

VOCAB = ("the of and to in the of and merchant partner cargo flax pepper ships western ports judge sale house witnesses scribe paper hand "
         "letter brother money silk dinar court elder community widow marriage contract debt payment caravan "
         "harbour season market price wheat oil cloth travel news health family greeting blessing synagogue "
         "teacher student book copy commentary edition fragment collection library scholar history trade route "
         "city village river desert goods camel donkey storage warehouse agent commission account balance").split()


# ------------------------------------------------------------------------------------ fixtures
def prose(seed: int, n_lines: int = 12, words: int = 9) -> str:
    """Deterministic pseudo-prose: ``n_lines`` lines of ``words`` words ending in a full stop.

    :param seed: Random seed (different seeds give different texts).
    :param n_lines: Lines.
    :param words: Words per line.
    :returns: The text.
    """
    rnd = random.Random(seed)
    out = []
    for _ in range(n_lines):
        out.append(" ".join(rnd.choice(VOCAB) for _ in range(words)) + ".")
    return "\n".join(out)


def make_tokenizer(tmp_path: Path) -> Path:
    """Save a word-level tokenizer (every whitespace word = one token).

    :param tmp_path: Pytest temporary directory.
    :returns: Path of ``tokenizer.json``.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    out = tmp_path / "tokenizer.json"
    tok.save(str(out))
    return out


def make_template(tmp_path: Path) -> Path:
    """Write the synthetic chat template.

    :param tmp_path: Pytest temporary directory.
    :returns: Path of the template.
    """
    f = tmp_path / "chat_template.jinja"
    f.write_text(TEMPLATE, encoding="utf-8")
    return f


def write_page(path: Path, text: str, page_number: int, pdf: str, printed: Any = None,
               classification: str = "general_academic") -> None:
    """Write one structured page JSON.

    :param path: Destination file.
    :param text: ``full_main_text``.
    :param page_number: ``metadata.page_number``.
    :param pdf: ``metadata.pdf_name``.
    :param printed: ``extracted_page_number``.
    :param classification: Page classification.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"full_main_text": text, "classification": classification, "extracted_page_number": printed,
            "metadata": {"page_number": page_number, "pdf_name": pdf, "model_used": "m", "provider": "p"},
            "shelf_marks_mentioned": {}, "footnotes": {}, "transcriptions": {}, "summary": "MODEL SUMMARY"}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_ocr(path: Path, texts: Dict[int, str]) -> None:
    """Write an ``*_ocr_results.json`` file.

    :param path: Destination file.
    :param texts: ``{page_number: text}``.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = [{"page_number": n, "ocr_result": {"full_text": t}, "ocr_source": "text_layer"} for n, t in texts.items()]
    path.write_text(json.dumps({"pages": pages, "processing_info": {"mode": "hybrid"}}), encoding="utf-8")


def write_part(base: Path, stem: str, bodies: Dict[int, str], head: Optional[str] = "A Study of Merchants",
               offset: int = 10, ocr_extra: Optional[Dict[int, str]] = None,
               struct_override: Optional[Dict[int, str]] = None, classification: Optional[Dict[int, str]] = None) -> None:
    """Write one part: structured pages (body only) + OCR pages (page number, running head, body).

    :param base: Directory holding the part (the structuring directory and the OCR file go here).
    :param stem: Part stem.
    :param bodies: ``{page_number: body text}``.
    :param head: Running head printed on top of every OCR page (None = none).
    :param offset: Printed page = page number + offset.
    :param ocr_extra: Extra OCR-only text appended per page.
    :param struct_override: ``full_main_text`` replacing the body per page.
    :param classification: Page classification per page.
    :returns: None.
    """
    ocr = {}
    for n, body in bodies.items():
        top = f"{n + offset}\n" + (f"{head}\n" if head else "")
        ocr[n] = top + body + ("\n" + ocr_extra[n] if ocr_extra and n in ocr_extra else "")
        text = (struct_override or {}).get(n, body)
        write_page(base / f"{stem}_structured" / f"page_{n:03d}_structured.json", text, n, stem,
                   printed=[n + offset], classification=(classification or {}).get(n, "general_academic"))
    write_ocr(base / f"{stem}_ocr_results.json", ocr)


def write_meta(path: Path, data: Dict[str, Any]) -> None:
    """Write a ``*_metadata.json`` file.

    :param path: Destination.
    :param data: Record.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def lib(tmp_path: Path) -> Path:
    """A synthetic literature tree.

    * ``work_book``: one metadata file at the work root, two nested parts (vol_1/part_a, vol_2/part_b),
      12 + 4 pages, a running head, one page carrying a project-name sentence, one page whose structured
      text differs from the OCR (low containment), one plate page, one short page.
    * ``work_coll``: a collection -- ``art_one`` (root metadata, same stem), ``art_two`` (metadata in
      its own sub-directory, other stem, CC BY-NC), ``art_three`` (no metadata); art_three page 1
      duplicates art_one page 1.
    * ``work_silver``: no structured pages (derived data), one metadata file.

    :param tmp_path: Pytest temporary directory.
    :returns: Literature root.
    """
    root = tmp_path / "lit"
    wb = root / "work_book"
    write_meta(wb / "book_metadata.json", {"title": "A Book of Letters", "authors": ["Ann Author", "Ben Writer"],
                                          "publication": {"type": "Print Book", "year": 1999}})
    bodies = {n: prose(n) for n in range(1, 13)}
    bodies[5] = prose(5) + f"\nThe images are on the {SCRUB_NAME} project site. The merchant left the city."
    bodies[11] = "Plate 3."
    write_part(wb / "vol_1" / "part_a", "part_a", bodies,
               struct_override={7: prose(700)}, classification={9: "diagram/map"})
    write_part(wb / "vol_2" / "part_b", "part_b", {n: prose(100 + n) for n in range(1, 5)}, head=None, offset=200)
    wc = root / "work_coll"
    write_meta(wc / "art_one_metadata.json", {"title": "First Article", "authors": "Cy Scholar",
                                              "publication": {"type": "Journal Article", "year": "2001"},
                                              "license": "CC BY 4.0"})
    write_part(wc, "art_one", {n: prose(200 + n) for n in range(1, 5)}, head="Journal of Things")
    write_meta(wc / "art_two" / "other_name_metadata.json",
               {"title": "Second Article", "author": {"name": "Di Expert"}, "publication": {"date": "2010-05-01"},
                "access_restrictions": "CC BY-NC-ND 4.0 (Open Access)"})
    write_part(wc / "art_two", "art_two", {n: prose(300 + n) for n in range(1, 4)}, head=None)
    write_part(wc / "art_three", "art_three", {1: prose(201), 2: prose(402)}, head=None)
    (wc / "broken_metadata.json").write_text("", encoding="utf-8")
    write_meta(root / "work_silver" / "silver_metadata.json", {"title": "Derived", "authors": []})
    (root / "work_silver" / "Person").mkdir(parents=True)
    return root


def write_config(tmp_path: Path, root: Path) -> Path:
    """Write a ``paths.json`` for the CLI tests (synthetic tokenizer / template, out_dir under tmp).

    :param tmp_path: Pytest temporary directory (repository root).
    :param root: Literature root.
    :returns: Config path.
    """
    cfg: Dict[str, Any] = {"repo": str(tmp_path)}
    cfg.update({k: f"data/{k}" for k in REQUIRED})
    cfg.update({"academic_literature": str(root), "tokenizer_json": str(make_tokenizer(tmp_path)),
                "chat_template": str(make_template(tmp_path)), "out_dir": "logs/t27/v0_2"})
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return f


def page_rec(text: Optional[str], fmt: str = "", page_type: str = "general_academic", work: str = "w",
             part: str = "p", page_no: int = 1, containment: Optional[float] = 1.0) -> G.PageRec:
    """A hand-built :class:`pages.PageRec`.

    :param text: OCR text (None = no OCR).
    :param fmt: ``full_main_text``.
    :param page_type: Classification.
    :param work: Work.
    :param part: Part.
    :param page_no: Page number.
    :param containment: OCR containment.
    :returns: The record.
    """
    return G.PageRec(work=work, part=part, page_file=f"page_{page_no:03d}_structured.json", page_no=page_no,
                     path=f"/x/{work}/{part}_structured/page_{page_no:03d}_structured.json",
                     structuring_dir=f"{part}_structured", structurer_model="m", structurer_provider="p",
                     printed_page=None, full_main_text=fmt or (text or ""), page_type=page_type, shelf_marks=[],
                     ocr_text=text, ocr_source="text_layer" if text else None, ocr_containment=containment)


def word_count(texts: Sequence[str]) -> List[int]:
    """Whitespace word counts (the synthetic tokenizer's behaviour).

    :param texts: Texts.
    :returns: Counts.
    """
    return [len(t.split()) for t in texts]


# ------------------------------------------------------------------------------------ quality
def test_quality_rules_each_fire_and_clean_prose_passes() -> None:
    """Each ported rule fires on a text built to trip it; ordinary prose passes."""
    assert L.quality_reasons(prose(1)) == []
    assert "short_page" in L.quality_reasons("a short page of text")
    assert "low_letters" in L.quality_reasons("12 34 56 78 90 11 22 33 44 55 66 77 88 99 00 . " * 20)
    frag = "\n".join(["one two"] * 40 + ["the merchant wrote to his partner about the cargo"] * 5)
    assert "fragmented_lines" in L.quality_reasons(frag)
    rev_words = ["םולש", "ןתנ", "ךלמ"] + ["שלום", "ברכה"] * 15
    rev = "\n".join(" ".join(rev_words[i:i + 6]) for i in range(0, len(rev_words), 6)) * 4
    assert "reversed_hebrew" in L.quality_reasons(rev)
    pres = " ".join([chr(0xFE8D) + chr(0xFEDF) + chr(0xFEE4)] * 30 + ["كتاب"] * 5)
    assert "arabic_presentation_forms" in L.quality_reasons((pres + "\n") * 3)
    bib = "\n".join(f"Author {i}, A Title of a Book, Leiden {1900 + i}." for i in range(30))
    assert "bibliography_like" in L.quality_reasons(bib)
    idx = "\n".join(f"merchant names and places, {i}, {i + 3}" for i in range(40))
    assert "index_like" in L.quality_reasons(idx)
    assert L.quality_reasons(prose(2), page_type=L.PLATE_CLASS) == ["plate"]
    assert L.quality_reasons("a short page", rules={"short_page_chars": 5}) == []


def test_page_features_counts() -> None:
    """Feature counts follow the prototype's definitions."""
    f = L.page_features("one two\nthree four five six\n\nשלום םולש עולם 1950\nends with 12")
    assert f["lines"] == 4 and f["short_lines"] == 1
    assert f["heb_words"] == 3 and f["heb_final_start"] == 1
    assert f["year_lines"] == 1 and f["endnum_lines"] == 2


def test_quality_ok_uses_ocr_text_and_furniture() -> None:
    """``quality_ok`` judges the OCR text (furniture-stripped); missing OCR fails; ``use_ocr=False`` uses the structured text."""
    good = page_rec(prose(3))
    assert L.quality_ok(good) and L.page_quality(good) == []
    assert not L.quality_ok(page_rec(None, fmt=prose(3)))
    assert L.page_quality(page_rec(None, fmt=prose(3))) == ["no_ocr"]
    assert L.quality_ok(page_rec(None, fmt=prose(3)), use_ocr=False)
    furniture_only = page_rec("12\nDownloaded from somewhere.com on 1 Jan 2020")
    assert L.page_quality(furniture_only) == ["no_ocr"]
    plate = page_rec(prose(4), page_type=L.PLATE_CLASS)
    assert not L.quality_ok(plate)


# ------------------------------------------------------------------------------------ furniture
def test_strip_furniture_peels_numbers_heads_and_stamps() -> None:
    """Page numbers, running heads (with or without their number) and download stamps go; body lines stay."""
    heads = frozenset({L.norm_head("A Study of Merchants")})
    text = ("  — 22 —  \n22 A Study of Merchants\nThe merchant wrote to his partner.\nA Study of Merchants\n"
            "He asked for news.\nDownloaded from example.com 12/10/2020\nvia free access\n")
    out, ops = L.strip_furniture(text, heads)
    assert out == "The merchant wrote to his partner.\nA Study of Merchants\nHe asked for news."
    assert sorted(ops) == ["boilerplate", "boilerplate", "page_number", "running_head"]
    assert L.strip_furniture("body line one two three\n7\n", ())[0] == "body line one two three"
    many = "1\n2\n3\n4\nbody of the page here"
    assert L.strip_furniture(many, (), edge=3)[0] == "4\nbody of the page here"
    sm = "T-S 12.34\nbody of the page here and more"
    assert L.strip_furniture(sm, frozenset({L.norm_head("T-S 12.34")}))[0] == sm


def test_running_heads_detection_rules() -> None:
    """A head on ≥3 pages counts; short heads need a 25 % share; shelfmark lines and body lines never count."""
    pages = []
    for i in range(12):
        top = "A Study of Merchants" if i % 2 else "Chapter Two Trade"
        label = "Verso\n" if i < 3 else ""
        sm = "T-S 8J 1.2\n" if i < 4 else ""
        pages.append(f"{i + 1}\n{top}\n{label}{sm}{prose(i, 3)}")
    heads = L.running_heads(pages)
    assert L.norm_head("A Study of Merchants") in heads and L.norm_head("Chapter Two Trade") in heads
    assert "verso" not in heads and not any(h.startswith("t s") for h in heads)
    assert L.running_heads(pages[:2]) == frozenset()
    short = [f"{i}\nSmith\n{prose(i, 3)}" for i in range(8)]
    assert "smith" in L.running_heads(short)
    rare_short = [f"{i}\n" + ("Smith\n" if i < 3 else "") + prose(i, 3) for i in range(20)]
    assert "smith" not in L.running_heads(rare_short)
    by_part = L.running_heads_by_part([page_rec(t, part="p1", page_no=i) for i, t in enumerate(pages)])
    assert L.norm_head("A Study of Merchants") in by_part[("w", "p1")]


def test_page_number_and_boilerplate_lines() -> None:
    """Bare (Arabic / Roman / bracketed) numbers and download stamps are recognised; prose is not."""
    for s in ("12", " [xiv] ", "— 7 —", "(301)"):
        assert L.is_page_number_line(s)
    for s in ("12 merchants", "", "Chapter 3"):
        assert not L.is_page_number_line(s)
    assert L.is_boilerplate_line("This content downloaded from 1.2.3.4 on Fri")
    assert L.is_boilerplate_line("via free access")
    assert not L.is_boilerplate_line("The letter was downloaded by the scribe")


# ------------------------------------------------------------------------------------ manifest
def test_licence_class_mapping() -> None:
    """Licence statements map onto the five classes."""
    assert L.licence_class("CC BY 4.0") == "cc-by"
    assert L.licence_class("CC BY-SA 4.0") == "cc-by"
    assert L.licence_class("CC BY-NC-ND 3.0") == "cc-by-nc"
    assert L.licence_class("Attribution-NonCommercial 4.0 International") == "cc-by-nc"
    assert L.licence_class("public domain (early journal content)") == "pd"
    assert L.licence_class("free on the society's site (rights: the society)") == "copyright"
    assert L.licence_class("© 2014 A Publisher") == "copyright"
    assert L.licence_class("free on a repository") == "unknown"
    assert L.licence_class(None) == "unknown"


def test_parse_metadata_variants() -> None:
    """Authors (list / dict / editors), year (int, date, range), licence (fields or restriction notes) and type."""
    a = L.parse_metadata({"title": " T ", "authors": ["A", "B", "C", "D"], "publication": {"type": "PhD Dissertation",
                                                                                           "year": 2017}})
    assert a == {"type": "thesis", "author": "A et al.", "title": "T", "year": 2017, "licence": None,
                 "licence_class": "unknown"}
    b = L.parse_metadata({"title": "T", "author": {"name": "X"}, "publication": {"type": "Journal Article",
                                                                               "year": "1990-1993"},
                          "access_restrictions": "CC BY-NC-ND 4.0 (Open Access)"})
    assert b["author"] == "X" and b["year"] == 1990 and b["licence_class"] == "cc-by-nc" and b["type"] == "article"
    c = L.parse_metadata({"title": "T", "authors": [], "editors": ["E1", "E2"], "source": {"licence": "CC BY 4.0"},
                          "publication": {"type": "Book (edited volume)", "date": "2021-01-01"}})
    assert c["author"] == "E1 and E2 (ed.)" and c["licence_class"] == "cc-by" and c["year"] == 2021
    d = L.parse_metadata({"title": "T", "authors": ["A"], "access_restrictions": "Full"})
    assert d["licence"] is None and d["type"] is None


def test_build_literature_manifest_matches_parts(lib: Path) -> None:
    """Work-level metadata covers nested parts; collections get per-part records; page-less dirs are flagged.

    :param lib: Synthetic literature root.
    """
    man = L.build_literature_manifest(lib)
    w = man["works"]
    assert set(w) == {"work_book", "work_coll", "work_silver"}
    book = w["work_book"]
    assert book["type"] == "book" and book["author"] == "Ann Author and Ben Writer" and book["year"] == 1999
    assert book["n_parts"] == 2 and book["n_pages"] == 16 and "parts" not in book
    coll = w["work_coll"]
    assert coll["type"] == "collection" and coll["licence_class"] == "unknown"
    assert coll["parts"]["art_one"]["title"] == "First Article" and coll["parts"]["art_one"]["licence_class"] == "cc-by"
    assert coll["parts"]["art_two"]["author"] == "Di Expert"
    assert coll["parts"]["art_two"]["year"] == 2010
    assert coll["parts"]["art_two"]["licence_class"] == "cc-by-nc"
    assert coll["parts"]["art_three"]["title"] is None
    assert w["work_silver"]["type"] == "no_pages" and w["work_silver"]["n_pages"] == 0
    info = L.source_info(man, "work_coll", "art_two")
    assert info["title"] == "Second Article"
    assert L.source_info(man, "work_book", "vol_1/part_a")["title"] == "A Book of Letters"
    assert L.source_info(man, "nope", "x") == L._empty_info()


def test_merge_manifests_keeps_hand_edits(lib: Path) -> None:
    """Informative old values win over rebuilt ones; unknowns are refreshed; old-only works are kept.

    :param lib: Synthetic literature root.
    """
    new = L.build_literature_manifest(lib)
    old = json.loads(json.dumps(new))
    old["works"]["work_book"]["licence_class"] = "copyright"
    old["works"]["work_book"]["year"] = None
    old["works"]["work_coll"]["parts"]["art_three"]["title"] = "Hand Title"
    old["works"]["gone"] = {"work_dir": "gone", "licence_class": "pd"}
    merged = L.merge_manifests(new, old)
    assert merged["works"]["work_book"]["licence_class"] == "copyright"
    assert merged["works"]["work_book"]["year"] == 1999
    assert merged["works"]["work_coll"]["parts"]["art_three"]["title"] == "Hand Title"
    assert "gone" in merged["works"]
    assert L.merge_manifests(new, None) == new


def test_match_metadata_prefers_exact_stem() -> None:
    """Exact stem beats a prefix; a prefix beats a file in the part's own directory; unrelated files never match."""
    def m(d: str, s: str) -> L.MetaFile:
        """A metadata record with no fields.

        :param d: Directory relative to the work.
        :param s: Stem.
        :returns: The record.
        """
        return L.MetaFile(d, s, Path(f"/{d}/{s}_metadata.json"), {})

    metas = [m("", "india"), m("", "india_vol_1"), m("sub", "other"), m("", "unrelated")]
    assert L.match_metadata("", "india_vol_1", metas).stem == "india_vol_1"
    assert L.match_metadata("", "india_vol_2", metas).stem == "india"
    assert L.match_metadata("sub", "sub_part", metas).stem == "other"
    assert L.match_metadata("", "zzz", metas) is None


# ------------------------------------------------------------------------------------ row pieces
def test_stub_pages_and_language() -> None:
    """Source stub shape, page labels (filled across a consecutive pair) and language classes."""
    info = {"author": "A", "title": "T", "year": 1999}
    assert L.source_stub(info, "p. 4") == "A, *T*, 1999, p. 4"
    assert L.source_stub({"title": "T"}, None) == "*T*"
    assert L.source_stub({}, "p. 4") == ""
    assert L.format_pages([10, None], [3, 4]) == "pp. 10–11"
    assert L.format_pages([None, 11], [3, 4]) == "pp. 10–11"
    assert L.format_pages([None, None], [3, 4]) is None
    assert L.format_pages([10, None], [3, 7]) == "p. 10"
    assert L.lang_class_of(prose(1)) == "EN"
    assert L.lang_class_of("le marchand et la ville sont dans le port") == "FR"
    assert L.lang_class_of("שלום עליכם ברוך הבא " * 5 + "a b") == "HE"
    assert L.lang_class_of("كتاب " * 10) == "AR"
    assert L.lang_class_of("xyzzy qwrt") == "OTHER"


def test_split_opening_shares_and_boundaries() -> None:
    """The opening is about a quarter, ends at a sentence boundary when one is near, and no text is lost."""
    text = prose(9, 20)
    opening, rest = L.split_opening(text)
    assert 0.15 <= len(opening) / len(text) <= 0.35
    assert opening.endswith(".") and rest
    assert (opening + " " + rest).split() == text.split()
    no_space = "x" * 100
    o2, r2 = L.split_opening(no_space)
    assert o2 + r2 == no_space and len(o2) == 25


def test_make_chunks_pairs_consecutive_pages_within_budget() -> None:
    """Consecutive pages of one part pair up within the budget; gaps, part changes and over-budget pairs do not."""
    def cand(part: str, n: int, words: int) -> L.PageCand:
        """A page candidate of ``words`` words.

        :param part: Part.
        :param n: Page number.
        :param words: Words.
        :returns: The candidate.
        """
        text = " ".join(["word"] * words)
        return L.PageCand(f"w/{part}/page_{n:03d}", "w", part, n, text, words, None, "text_layer", 1.0)

    cs = [cand("a", 1, 40), cand("a", 2, 40), cand("a", 3, 40), cand("a", 5, 40), cand("b", 6, 40),
          cand("b", 7, 90)]
    chunks = L.make_chunks(cs, 100, word_count)
    assert [[p.page_no for p in c.pages] for c in chunks] == [[1, 2], [3], [5], [6], [7]]
    assert chunks[0].tokens == 82 and chunks[0].key == "w/a/page_001+w/a/page_002"
    long = L.PageCand("w/a/page_009", "w", "a", 9, "\n".join(["one two three four five"] * 10), 50, None, None, 1.0)
    pieces = L.make_chunks([long], 20, word_count)
    assert len(pieces) == 4 and all(c.tokens <= 20 for c in pieces)
    assert [c.piece for c in pieces] == [0, 1, 2, 3] and pieces[1].key.endswith("#1")


def test_find_duplicates_and_assign_probe_and_cap() -> None:
    """Near-duplicates map to the first copy; probe picks whole pages per work; the cap keeps rows under budget."""
    texts = {"a": prose(1, 20), "b": prose(2, 20), "c": prose(1, 20) + "\nan extra closing line here", "d": "short"}
    assert L.find_duplicates(texts) == {"c": "a"}
    assert L.find_duplicates({"a": prose(1, 20), "b": prose(2, 20)}) == {}

    cands = [L.PageCand(f"w/p/page_{i:03d}", "w", "p", i, "t", 1, None, None, 1.0) for i in range(40)]
    small = [L.PageCand(f"v/p/page_{i:03d}", "v", "p", i, "t", 1, None, None, 1.0) for i in range(5)]
    chunks = [L.Chunk([c], "t", 1) for c in cands + small]
    probe = L.assign_probe(chunks, 0.05, seed=1)
    assert len([k for k in probe if k.startswith("w/")]) == 2 and not any(k.startswith("v/") for k in probe)
    assert probe == L.assign_probe(chunks, 0.05, seed=1) and probe != L.assign_probe(chunks, 0.05, seed=2)
    assert len(L.assign_probe([L.Chunk([c], "t", 1) for c in cands[:10]], 0.01)) == 1
    pieces = [L.Chunk([cands[0]], "t", 1, piece=0), L.Chunk([cands[0]], "t", 1, piece=1)]
    assert L.assign_probe(pieces, 1.0, min_pages=1) == {pieces[0].key, pieces[1].key}

    rows = [{"meta": {"row_id": f"{i:016x}", "work": "w" if i < 8 else "v", "tokens": 10,
                      "split": "probe" if i == 0 else "train"}} for i in range(10)]
    kept, dropped = L.apply_cap(rows, 35)
    assert len(dropped) == 4 and all(r["meta"]["work"] == "w" for r in dropped)
    assert sum(r["meta"]["tokens"] for r in kept if r["meta"]["work"] == "w" and r["meta"]["split"] == "train") <= 35
    assert any(r["meta"]["split"] == "probe" for r in kept)
    assert L.apply_cap(rows, None) == (rows, [])


def test_load_excluded_pages_formats(tmp_path: Path) -> None:
    """Plain mapping, wrapped mapping and key list are accepted; None means no exclusions.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"w/p/page_001_structured.json": ["treatment"], "k2": "mention"}), encoding="utf-8")
    assert L.load_excluded_pages(f) == {"w/p/page_001_structured.json": ["treatment"], "k2": ["mention"]}
    f.write_text(json.dumps({"pages": {"k": ["r"]}, "version": 1}), encoding="utf-8")
    assert L.load_excluded_pages(f) == {"k": ["r"]}
    f.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert L.load_excluded_pages(f) == {"a": ["listed"], "b": ["listed"]}
    assert L.load_excluded_pages(None) == {}
    with pytest.raises(FileNotFoundError):
        L.load_excluded_pages(tmp_path / "missing.json")


# ------------------------------------------------------------------------------------ build
def run_build(lib: Path, tmp_path: Path, **kw: Any) -> Any:
    """Build rows from the synthetic tree with the synthetic tokenizer and template.

    :param lib: Literature root.
    :param tmp_path: Pytest temporary directory.
    :param kw: Extra :func:`literature_l1.build` arguments.
    :returns: ``(rows, report, pages)``.
    """
    pages = G.iter_pages(lib)
    man = L.build_literature_manifest(lib, pages)
    args: Dict[str, Any] = {"token_cap": None, "probe_frac": 0.0, "max_tokens": 400}
    args.update(kw)
    rows, rep = L.build(pages, man, tokenizer_path=make_tokenizer(tmp_path), template_path=make_template(tmp_path),
                        **args)
    return rows, rep, pages


def test_build_end_to_end(lib: Path, tmp_path: Path) -> None:
    """Rows validate, carry OCR text only, drop the gated pages with the right reasons and strip project names / furniture.

    :param lib: Synthetic literature root.
    :param tmp_path: Pytest temporary directory.
    """
    rows, rep, pages = run_build(lib, tmp_path)
    assert rows and all(R.validate_row(r) == [] for r in rows)
    assert len({r["meta"]["row_id"] for r in rows}) == len(rows)
    for r in rows:
        m = r["meta"]
        assert m["family"] == "domain_reading" and m["condition"] == "literature" and m["split"] == "train"
        assert m["tokens"] > m["target_tokens"] > 0 and m["pgpid"] is None and m["build"] == R.BUILD
        user, target = r["messages"][1]["content"], r["messages"][2]["content"]
        assert user.startswith(L.L1_INSTRUCTION)
        for content in (user, target):
            assert not T.has_source_names(content) and "MODEL SUMMARY" not in content
            assert "A Study of Merchants" not in content and "Journal of Things" not in content
    book = [r for r in rows if r["meta"]["work"] == "work_book" and r["meta"]["part"] == "vol_1/part_a"]
    assert book[0]["messages"][1]["content"].split("\n\n")[1] == "Source: Ann Author and Ben Writer, *A Book of Letters*, 1999, pp. 11–12"
    assert book[0]["meta"]["page_range"] == "1-2" and book[0]["meta"]["licence_class"] == "unknown"
    assert any(T.SCRUB_OP in r["meta"]["strip_ops"] for r in rows) and rep["source_scrub_hits_removed"] == 1
    assert rep["rows_with_source_names_after"] == 0
    assert all("page_number" in r["meta"]["strip_ops"] for r in rows)
    drops = rep["works"]["work_book"]["dropped"]
    assert drops["low_containment"]["pages"] == 1
    assert drops["quality:plate"]["pages"] == 1 and drops["quality:short_page"]["pages"] == 1
    assert rep["works"]["work_coll"]["dropped"]["duplicate"]["pages"] == 1
    keys = {k for r in rows for k in r["meta"]["provenance"]["page_keys"]}
    assert all(r["meta"]["page_keys"] == r["meta"]["provenance"]["page_keys"] for r in rows)  # where rescan looks (D8)
    assert "work_book/vol_1/part_a/page_007_structured.json" not in keys
    assert "work_coll/art_three/page_001_structured.json" not in keys
    t = rep["totals"]
    w = rep["works"]["work_book"]
    assert w["pages_total"] == 16 and w["train"]["pages"] + sum(v["pages"] for v in w["dropped"].values()) == 16
    assert t["train_rows"] == len(rows) and t["train_row_tokens"] == sum(r["meta"]["tokens"] for r in rows)
    assert rep["counts"]["rows"]["domain_reading"]["literature"]["EN"]["train"] == len(rows)
    coll = [r for r in rows if r["meta"]["work"] == "work_coll" and r["meta"]["part"] == "art_two"]
    assert coll and all(r["meta"]["licence_class"] == "cc-by-nc" for r in coll)
    assert "*Second Article*, 2010" in coll[0]["messages"][1]["content"]


def test_build_exclusions_probe_and_cap(lib: Path, tmp_path: Path) -> None:
    """Excluded pages (and their duplicates) leave; probe pages are disjoint from train; the cap limits a work.

    :param lib: Synthetic literature root.
    :param tmp_path: Pytest temporary directory.
    """
    excl = {"work_book/vol_1/part_a/page_002_structured.json": ["mention"],
            "work_coll/art_one/page_001_structured.json": ["treatment"]}
    rows, rep, _ = run_build(lib, tmp_path, excluded=excl)
    keys = {k for r in rows for k in r["meta"]["provenance"]["page_keys"]}
    assert not keys & set(excl)
    assert "work_coll/art_three/page_001_structured.json" not in keys
    assert rep["works"]["work_book"]["dropped"]["excluded"]["pages"] == 1
    assert rep["works"]["work_coll"]["dropped"]["excluded_via_duplicate"]["pages"] == 1

    rows, rep, _ = run_build(lib, tmp_path, probe_frac=0.2)
    probe = [r for r in rows if r["meta"]["split"] == "probe"]
    train = [r for r in rows if r["meta"]["split"] == "train"]
    assert probe and train
    pk = {k for r in probe for k in r["meta"]["provenance"]["page_keys"]}
    tk = {k for r in train for k in r["meta"]["provenance"]["page_keys"]}
    assert not pk & tk and rows.index(probe[0]) > rows.index(train[-1])

    full, _, _ = run_build(lib, tmp_path)
    book_tokens = sum(r["meta"]["tokens"] for r in full if r["meta"]["work"] == "work_book")
    capped, rep, _ = run_build(lib, tmp_path, token_cap=book_tokens // 2)
    assert sum(r["meta"]["tokens"] for r in capped if r["meta"]["work"] == "work_book") <= book_tokens // 2
    assert rep["works"]["work_book"]["dropped"]["work_cap"]["pages"] > 0 and rep["capped_rows"] > 0


def test_build_reproducible(lib: Path, tmp_path: Path) -> None:
    """Two builds with the same seed give identical rows (fingerprint).

    :param lib: Synthetic literature root.
    :param tmp_path: Pytest temporary directory.
    """
    a, _, _ = run_build(lib, tmp_path, probe_frac=0.2, token_cap=600)
    b, _, _ = run_build(lib, tmp_path, probe_frac=0.2, token_cap=600)
    assert R.fingerprint(a) == R.fingerprint(b)


# ------------------------------------------------------------------------------------ CLI
def test_cli_manifest_dry_run_and_build(lib: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture) -> None:
    """``manifest`` writes the literature manifest (keeping hand edits); ``build --dry-run`` writes only the
    report; ``build`` writes the family file and its manifest.

    :param lib: Synthetic literature root.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch.
    :param capsys: Pytest capture.
    """
    monkeypatch.setenv("TEXT27B_PATHS", str(write_config(tmp_path, lib)))
    assert L.main(["manifest"]) == 0
    man_path = tmp_path / "logs" / "t27" / "literature_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert set(man["works"]) == {"work_book", "work_coll", "work_silver"}
    man["works"]["work_book"]["licence_class"] = "copyright"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    assert L.main(["manifest"]) == 0
    assert json.loads(man_path.read_text(encoding="utf-8"))["works"]["work_book"]["licence_class"] == "copyright"

    assert L.main(["build", "--dry-run", "--token-cap", "0"]) == 0
    out = tmp_path / "logs" / "t27" / "v0_2"
    dry = json.loads((out / "literature_l1_dry_report.json").read_text(encoding="utf-8"))
    assert dry["dry_run"] and dry["inputs"]["excluded_pages"] is None
    assert not (out / "families" / "domain_reading.jsonl").exists()
    assert dry["works"]["work_book"]["licence_class"] == "copyright"

    excl = tmp_path / "excluded_pages.json"
    excl.write_text(json.dumps({"work_book/vol_1/part_a/page_001_structured.json": ["m"]}), encoding="utf-8")
    assert L.main(["build", "--excluded-pages", str(excl), "--works", "work_book"]) == 0
    fam = out / "families" / "domain_reading.jsonl"
    rows = R.load_jsonl(fam)
    rep = json.loads((out / "families" / "domain_reading_manifest.json").read_text(encoding="utf-8"))
    assert rows and rep["fingerprint"] == R.fingerprint(rows) and rep["inputs"]["excluded_page_keys"] == 1
    assert rep["inputs"]["excluded_pages"] == "excluded_pages.json" and rep["build_args"]["works"] == ["work_book"]
    assert all(r["meta"]["work"] == "work_book" for r in rows)
    assert "wrote" in capsys.readouterr().out


def test_real_template_and_tokenizer(lib: Path) -> None:
    """With the local Qwen3.8 template / tokenizer (skipped when absent) every row meets the render contract.

    :param lib: Synthetic literature root.
    """
    from src.finetuning.text27b import paths as P

    try:
        p = P.load_paths()
    except FileNotFoundError:
        pytest.skip("no paths.json")
    if not (p.tokenizer_json.exists() and p.chat_template.exists()):
        pytest.skip("local Qwen3.8 tokenizer / template not present")
    pages = G.iter_pages(lib, works=["work_book"])
    rows, rep = L.build(pages, L.build_literature_manifest(lib, pages), token_cap=None, probe_frac=0.0)
    assert rows
    for r in rows:
        rendered = R.render_chat(r["messages"])
        assert rendered.count(R.EMPTY_THINK) == 1 and R.EFFORT_MARKER not in rendered
        assert r["meta"]["tokens"] == R.count_tokens(rendered)


# ------------------------------------------------------------------------------------ prep-pass hygiene (2026-10-26)
BIB_PAGE = "\n".join([
    "Abbot, Ann. 1999. 'A Study of Trade'. Journal of Things 12: 33-45.",
    "Baker, Ben. 2001. Merchants of the Sea. Leiden: Brill.",
    "———. 2004. 'Ports and Partners'. In Essays on Trade, edited by C. Cole, 10-20.",
    "Cole, Carl, and Dan Dee. 2010. Letters and Ledgers. Oxford: Oxford University Press.",
    "Dee, Dan. 1987. The Silk Road Again. London: Routledge.",
    "Eve, Emma. 2015. 'Widows in Court'. Speculum 90: 1-30.",
    "Fox, Fay. 1990. Caravans. New York: Harper."])
#: Wrapped bibliography entries (fewer than 45 % of the lines carry a year, so the ported year rule lets it through).
BIB_WRAPPED = "\n".join([
    "Abbot, Ann. 1999. 'A Study of Trade in the Ports of the", "Western Sea and the Merchants Who Sailed There'.",
    "Journal of Things 12: 33-45.",
    "Baker, Ben. 2001. Merchants of the Sea and Their", "Partners in the East: Letters, Ledgers and",
    "Lawsuits. Leiden: Brill.",
    "Cole, Carl, and Dan Dee. 2010. Letters and Ledgers of", "the Geniza Merchants and Their Families in",
    "the Long Twelfth Century. Oxford: Oxford University Press.",
    "Dee, Dan. 1987. The Silk Road Again: A Survey of", "Trade Routes Between the Mediterranean and",
    "the Indian Ocean. London: Routledge."])
PROSE_WITH_NOTES = ("The merchant wrote to his partner about the cargo of flax and pepper that had reached the western\n"
                    "ports during the sailing season, and he asked for news of the market price.12 The partner\n"
                    "answered that the ships were late and that the judge had sealed the house of the debtor.13\n"
                    "This shows how the partnership worked in practice and why the letters were kept together.\n"
                    "12 See Goitein, Med. Soc., I, p. 33.\n"
                    "13 Cf. ibid., p. 45; the same court is discussed in Baker (2001), 12.")


def test_reference_density_and_page_kinds() -> None:
    """Bibliography / endnotes / index / contents / front matter are recognised; argumentative prose is body text."""
    assert L.reference_density(BIB_PAGE) > L.REFERENCE_DENSITY_MAX
    assert L.reference_density(prose(1)) == 0.0 and L.reference_density("") == 0.0
    assert L.page_kind(BIB_PAGE, 300) == "bibliography" and L.page_kind(BIB_WRAPPED, 300) == "bibliography"
    assert not L.quality_reasons(BIB_WRAPPED)  # the earlier rules alone kept such pages
    assert L.page_kind("Bibliography\n" + prose(2, 8), 300) == "bibliography"
    notes = "\n".join(f"{n}. See Goitein, Med. Soc., I, p. {n + 3}; ibid., p. {n + 9}." for n in range(40, 52))
    assert L.page_kind(notes, 200) == "endnotes"
    index = "Index\n" + "\n".join(f"topic {c}, {n}, {n + 4}-{n + 6}, {n + 30}" for n, c in zip(range(10, 30), "abcdefghijklmnopqrst"))
    assert L.page_kind(index, 400) == "index"
    toc = "Contents\nIntroduction ........ 1\nThe Letters ........ 15\nThe Courts ........ 40"
    assert L.page_kind(toc, 5) == "toc"
    front = "First published 2001\nAll rights reserved\nISBN 978-0-00-000000-0\nPrinted and bound in somewhere"
    assert L.page_kind(front, 4) == "front_matter" and L.page_kind(front, 200) is None  # front matter only early
    assert L.page_kind(prose(3), 50) is None
    # a line-numbered translation is not an endnote page
    trans = "\n".join(f"{n}. and he sent the money to his brother with the caravan" for n in range(1, 15))
    assert L.page_kind(trans, 60) is None


def test_footnote_cut_structured_and_fallback() -> None:
    """The footnote region goes: located from the page's footnote strings, else by the note-number fallback, which
    needs a matching note reference in the body (line-numbered edition text is never cut)."""
    body, how = L.cut_footnotes(PROSE_WITH_NOTES)
    assert how == "fallback" and body.endswith("kept together.") and "Goitein" not in body
    rec = page_rec(PROSE_WITH_NOTES)
    rec.data = {"footnotes": {"12": "See Goitein, Med. Soc., I, p. 33."}}
    body2, how2 = L.cut_footnotes(PROSE_WITH_NOTES, rec)
    assert how2 == "structured" and body2 == body
    edition = "\n".join(f"{n} and he wrote to his brother about the flax and the pepper and the price" for n in range(1, 25))
    assert L.cut_footnotes(edition) == (edition, None)
    assert L.footnote_block_start("no notes here at all") is None


def test_join_broken_lines() -> None:
    """Hyphenated and mid-sentence line breaks join; sentence ends, numbered lines and Judaeo-Arabic lines stay."""
    text = ("The merchants of the western ports wrote long histor-\nical letters to their partners in the east about\n"
            "the cargo and the season.\nA new paragraph starts here with a capital letter.\n"
            "The Judaeo-\nArabic letters were written in Hebrew script by the writer himself.")
    out, n = L.join_broken_lines(text)
    assert "historical letters" in out and "east about the cargo" in out and "Judaeo-Arabic" in out
    assert "season.\nA new paragraph" in out and n["hyphen"] == 2 and n["latin"] >= 1
    numbered = "1 and he sent the money to his brother with\n2 the caravan of the merchants of the west"
    assert L.join_broken_lines(numbered)[0] == numbered
    he = ("והסוחרים שבנמלי המערב כתבו מכתבים ארוכים אל שותפיהם שבמזרח והם מספרים לנו את\n"
          "סיפור המסחר של התקופה בפירוט רב")
    out_he, n_he = L.join_broken_lines(he)
    assert n_he["rtl"] == 1 and "\n" not in out_he
    ja = ("ואעלמך אן אלכתאב וצל אליה מן אכיה פי אלשהר אלמאצי ואנא אשכר אללה עלי דלך כתירא ואלסלאם עליך\n"
          "ואלסלאם עליך ורחמה אללה")
    assert L.join_broken_lines(ja)[0] == ja  # quoted Judaeo-Arabic keeps its lines


def test_text_kind_labels() -> None:
    """argument / narrative / edition_apparatus / other."""
    arg = ("However, the evidence suggests that the reading is probably wrong, because the scribe seems to have "
           "confused two letters; therefore we must conclude that the date is uncertain, as argued above. ") * 3
    nar = ("The merchant was born in Fustat and became a judge; he married the daughter of a trader, went to Aden, "
           "returned in 1140 and died there. His son was a scribe and wrote many letters. ") * 3
    ed = "\n".join(f"{n}. ואעלמך אן אלכתאב וצל אליה מן אכיה פי אלשהר אלמאצי" for n in range(1, 12))
    assert L.text_kind(arg) == "argument" and L.text_kind(nar) == "narrative"
    assert L.text_kind(ed) == "edition_apparatus" and L.text_kind("") == "other"
    assert L.text_kind("cargo flax pepper ships ports judge sale house " * 6) == "other"
    assert set(L.TEXT_KINDS) == {"argument", "edition_apparatus", "narrative", "other"}


def test_build_hygiene_drops_apparatus_cuts_notes_and_labels(tmp_path: Path) -> None:
    """build(): a bibliography page is dropped as page_kind, a reference-dense page as reference_density, the notes of
    a body page are cut (counted per work) and every row carries a text_kind; hygiene=False keeps the earlier path."""
    pages = [page_rec(prose(9, 6) + "\n" + PROSE_WITH_NOTES, work="w", page_no=1),
             page_rec(BIB_WRAPPED, work="w", page_no=2),
             page_rec(prose(4), work="w", page_no=3)]
    tok, tpl = make_tokenizer(tmp_path), make_template(tmp_path)
    rows, rep = L.build(pages, {}, token_cap=None, probe_frac=0.0, max_tokens=400, tokenizer_path=tok, template_path=tpl)
    w = rep["works"]["w"]
    assert w["dropped"]["page_kind:bibliography"]["pages"] == 1
    assert w["cuts"]["footnotes"]["pages"] == 1 and w["cuts"]["footnotes"]["tokens"] > 0
    assert rep["totals"]["footnote_cut_pages"] == 1 and rep["hygiene"] is True
    text = " ".join(m["content"] for r in rows for m in r["messages"])
    assert "ibid." not in text and "Goitein" not in text
    assert all(r["meta"]["text_kind"] in L.TEXT_KINDS for r in rows)
    assert any("footnote_cut:fallback" in r["meta"]["strip_ops"] for r in rows)
    assert sum(c["rows"] for c in rep["text_kinds"]["train"].values()) == len(rows)
    raw, rep_raw = L.build(pages, {}, token_cap=None, probe_frac=0.0, max_tokens=400, tokenizer_path=tok,
                           template_path=tpl, hygiene=False)
    assert "page_kind:bibliography" not in rep_raw["works"]["w"]["dropped"] and rep_raw["hygiene"] is False
    dense = [page_rec("\n".join(f"the merchant {i} see p. {i}; ibid., pp. 3-4." for i in range(10)) + "\n" + prose(5, 6),
                      work="v", page_no=40)]
    _, rep_d = L.build(dense, {}, token_cap=None, probe_frac=0.0, max_tokens=400, tokenizer_path=tok, template_path=tpl)
    assert rep_d["works"]["v"]["dropped"]["reference_density"]["pages"] == 1
