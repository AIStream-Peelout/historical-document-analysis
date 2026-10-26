# File name: test_text27b_contamination.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``contamination``: n-gram hashing and the leak rule, the stop-list (with and
without background documents), persistence, the RAG block rule, the project-name scan, the eval
references, the literature page exclusions and the mixture re-scan (functions and CLI).

Everything runs on synthetic texts and the synthetic world of ``test_text27b_holdouts.py``.
"""
import codecs
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.finetuning.text27b import canon_diff as CD
from src.finetuning.text27b import contamination as C
from src.finetuning.text27b import holdouts as H
from src.finetuning.text27b import pages as PG
from src.finetuning.text27b import paths as P

from test_text27b_holdouts import REGISTRY, make_world, v01_row

#: Scrubbed project names, decoded at runtime from rot13 so this file never spells them (fix pass F12).
SCRUB_SURNAME = codecs.decode("Sevrqoret", "rot_13")
SCRUB_SOCIETY = codecs.decode("SWZF", "rot_13")

#: 400 distinct words (a long English reference).
PROSE = " ".join(f"alpha{i}" for i in range(400))
#: A 12-word formula used by several documents.
FORMULA = "may god prolong your days and make your honour last for ever amen"
_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"


def hebrew_text(seed: int, n_letters: int = 240) -> List[str]:
    """Deterministic pseudo-Hebrew words (4–5 letters each).

    :param seed: Random seed.
    :param n_letters: Approximate number of letters.
    :returns: Words.
    """
    rnd = random.Random(seed)
    words, total = [], 0
    while total < n_letters:
        w = "".join(rnd.choice(_LETTERS) for _ in range(rnd.choice((4, 5))))
        words.append(w)
        total += len(w)
    return words


#: A Hebrew edition with brackets and line breaks, and the same letters re-spaced as a book prints it.
HE_WORDS = hebrew_text(7)
HE_EDITION = "\n".join(" ".join(f"[{w}]" if i % 7 == 3 else w for i, w in enumerate(HE_WORDS[j:j + 6]))
                       for j in range(0, len(HE_WORDS), 6))
HE_BOOK = " ".join(a + b for a, b in zip(HE_WORDS[::2], HE_WORDS[1::2]))


def mk_page(work: str, part: str, n: int, text: str, ocr: Optional[str] = None, keys: Optional[List[str]] = None) -> PG.PageRec:
    """A synthetic literature page.

    :param work: Work directory.
    :param part: Part.
    :param n: Page number.
    :param text: ``full_main_text``.
    :param ocr: OCR text.
    :param keys: Raw ``shelf_marks_mentioned`` keys.
    :returns: The page.
    """
    return PG.PageRec(work=work, part=part, page_file=f"page_{n:03d}_structured.json", page_no=n, path="",
                      structuring_dir="x_structured", structurer_model=None, structurer_provider=None,
                      printed_page=None, full_main_text=text, page_type=None, shelf_marks=list(keys or []),
                      ocr_text=ocr)


# ------------------------------------------------------------------------------------ tokens and hashes
def test_word_tokens_and_hebrew_runs() -> None:
    """Tokens are lower-cased without points; Hebrew runs keep letters only, fold finals, break on Latin.

    :returns: None.
    """
    assert C.word_tokens("The Fusṭāṭ letter, 1040!") == ["the", "fusṭāṭ", "letter", "1040"]
    assert C.word_tokens("שָׁלוֹם עליכם") == ["שלום", "עליכם"]
    assert C.hebrew_runs("[של]ום ע-ל\nיכם") == ["שלומעליכמ"]
    assert C.hebrew_runs("אבג abc דהו") == ["אבג", "דהו"]
    assert C.hebrew_runs("אבג abc דהו", min_len=4) == []
    assert C.word_tokens(None) == [] and C.hebrew_runs("") == []


def test_gram_hashes_stable_salted_and_invariant() -> None:
    """Hashes are deterministic, kinds never share a hash, and c20 ignores brackets / spacing.

    :returns: None.
    """
    w = C.gram_hashes(PROSE, "w8")
    assert len(w) == 393 and w.dtype == np.uint64
    assert np.array_equal(w, C.gram_hashes(PROSE, "w8"))
    c1, c2 = C.gram_hashes(HE_EDITION, "c20"), C.gram_hashes(HE_BOOK, "c20")
    assert len(c1) > 100 and np.array_equal(c1, c2)
    assert not np.intersect1d(C.gram_hashes(HE_EDITION, "w8"), c1).size
    assert len(C.gram_hashes("too short text", "w8")) == 0
    with pytest.raises(ValueError):
        C.gram_hashes("x", "w3")
    g = C.text_grams(HE_EDITION)
    assert set(g) == {"w8", "c20"}


# ------------------------------------------------------------------------------------ the index
@pytest.fixture()
def index() -> C.NgramIndex:
    """An index over a long English reference, a short one and a Hebrew edition.

    :returns: The index.
    """
    short = " ".join(f"beta{i}" for i in range(20))
    return C.NgramIndex.build([("tgt:translate_en:1", "1", PROSE), ("tgt:describe_pgp:2", "2", short),
                               ("inp:3", "3", HE_EDITION)])


def test_w8_count_rule(index: C.NgramIndex) -> None:
    """Five shared word 8-grams (12 words) leak a long reference; four do not.

    :param index: Fixture index.
    :returns: None.
    """
    words = PROSE.split()
    page12 = "some page text " + " ".join(words[100:112]) + " more text here"
    page11 = "some page text " + " ".join(words[100:111]) + " more text here"
    assert index.page_hits(page12)[0] == 5
    assert [r[:3] for r in index.leak_refs(page12)] == [("tgt:translate_en:1", "w8", 5)]
    n, share = index.page_hits(page11)
    assert n == 4 and share < 0.02
    assert not index.is_leak(page11)


def test_share_rule_on_short_reference(index: C.NgramIndex) -> None:
    """One shared 8-gram of a 13-gram reference is ≥ 2 % of it.

    :param index: Fixture index.
    :returns: None.
    """
    page = "intro " + " ".join(f"beta{i}" for i in range(5, 13)) + " outro"
    hits = index.ref_hits(page)
    assert hits["tgt:describe_pgp:2#w8"][0] == 1 and hits["tgt:describe_pgp:2#w8"][1] == pytest.approx(1 / 13)
    assert index.is_leak(page)


def test_c20_catches_respaced_edition(index: C.NgramIndex) -> None:
    """A book page re-spacing the edition shares no word 8-gram but leaks through c20.

    :param index: Fixture index.
    :returns: None.
    """
    leaks = index.leak_refs("Chapter 3\n" + HE_BOOK + "\nNotes")
    assert leaks and leaks[0][0] == "inp:3" and leaks[0][1] == "c20" and leaks[0][2] >= 25
    assert "inp:3#w8" not in index.ref_hits(HE_BOOK)


def test_c20_short_shared_run_is_not_a_leak() -> None:
    """A ~30-letter shared run (a formula) in a long edition stays under both c20 thresholds.

    :returns: None.
    """
    long_ed = " ".join(hebrew_text(11, 3000))
    idx = C.NgramIndex.build([("inp:9", "9", long_ed)])
    run = "".join(C.hebrew_runs(long_ed))[500:532]
    n, share = idx.page_hits("text " + run + " text")
    assert 0 < n < 25 and share < 0.02
    assert not idx.is_leak(run)
    strict = C.NgramIndex.build([("inp:9", "9", long_ed)], min_shared={"c20": 5})
    assert strict.is_leak(run)


def test_clean_page(index: C.NgramIndex) -> None:
    """An unrelated text has no hits.

    :param index: Fixture index.
    :returns: None.
    """
    assert index.page_hits("nothing to see here at all in this little page of prose") == (0, 0.0)
    assert index.leak_refs("") == [] and index.ref_hits(None) == {}


def test_stop_list_and_background() -> None:
    """A formula in five documents is stop-listed; background documents count toward the five.

    :returns: None.
    """
    refs = [(f"tgt:x:{i}", str(i), f"unique{i} words{i} here{i} " + FORMULA) for i in range(5)]
    idx = C.NgramIndex.build(refs)
    assert idx.page_hits("a page quoting " + FORMULA + " in passing") == (0, 0.0)
    refs2 = refs[:2]
    idx2 = C.NgramIndex.build(refs2)
    assert idx2.is_leak("a page quoting " + FORMULA + " in passing")
    bg = [(f"b{i}", "other text " + FORMULA) for i in range(3)]
    idx3 = C.NgramIndex.build(refs2, background=bg)
    assert idx3.n_background_docs == 3 and len(idx3.stop) > 0
    assert not idx3.is_leak("a page quoting " + FORMULA + " in passing")
    same_doc = C.NgramIndex.build(refs2, background=[("0", FORMULA), ("1", FORMULA), ("b", FORMULA)])
    assert same_doc.n_background_docs == 1 and same_doc.is_leak(FORMULA)


def test_identical_texts_indexed_once() -> None:
    """Duplicate reference texts are indexed once (the describe target = catalogue summary case).

    :returns: None.
    """
    idx = C.NgramIndex.build([("a", "1", PROSE), ("b", "1", PROSE + "  "), ("c", "1", "")])
    assert idx.stats()["references"] == 1 and idx.stats()["reference_kinds"] == {"w8": 1}


def test_save_load_roundtrip(index: C.NgramIndex, tmp_path: Path) -> None:
    """A saved index answers exactly like the original.

    :param index: Fixture index.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    f = index.save(tmp_path / "i.npz")
    j = C.NgramIndex.load(f)
    page = " ".join(PROSE.split()[10:40]) + " " + HE_BOOK
    assert j.leak_refs(page) == index.leak_refs(page)
    assert j.stats() == index.stats()


def test_empty_index() -> None:
    """An index without references never leaks.

    :returns: None.
    """
    idx = C.NgramIndex.build([("a", "1", "short")])
    assert idx.page_hits(PROSE) == (0, 0.0) and not idx.is_leak(PROSE)


# ------------------------------------------------------------------------------------ helpers
def test_target_share_and_block_ok() -> None:
    """The RAG block rule measures the target's 8-grams found anywhere in the block.

    :returns: None.
    """
    words = PROSE.split()
    target = " ".join(words[:108])
    block = ["page one " + " ".join(words[:13]), "page two " + " ".join(words[50:56])]
    assert C.target_share(target, block) == pytest.approx(6 / 101)
    assert C.target_share(target, block[0]) == pytest.approx(6 / 101)
    assert C.block_ok(target, "page one " + " ".join(words[:12]))
    assert not C.block_ok(target, block)
    assert C.block_ok(target, "nothing shared")
    assert C.target_share("short", block) == 0.0


def test_source_scan() -> None:
    """Scrubbed project names and hosts are found; clean text gives nothing.

    :returns: None.
    """
    host = ".".join([SCRUB_SOCIETY.lower(), codecs.decode("travmnu", "rot_13"), "org"])  # assembled at runtime (F12)
    assert C.source_scan(f"See the {SCRUB_SURNAME} Genizah Project and https://{host}/x.") == [SCRUB_SURNAME, f"https://{host}/x."]
    assert C.source_scan("A letter from Fustat.") == [] and C.source_scan(None) == []


def test_prose_of_and_document_input() -> None:
    """JSON targets reduce to their long strings; ``<document>`` text is extracted.

    :returns: None.
    """
    js = json.dumps({"document_type": "Letter", "date": None, "people": [{"name": "Natan"}],
                     "summary": "A letter from a merchant in Alexandria about a shipment of flax."})
    assert C.prose_of(js) == "A letter from a merchant in Alexandria about a shipment of flax."
    assert C.prose_of("{not json") == "{not json" and C.prose_of("Plain text.") == "Plain text."
    assert C.prose_of('{"a": "short"}') == "" and C.prose_of(None) == ""
    assert C.document_input("Do.\n\n<document>\nאבג\nדהו\n</document>\n\nMore") == "אבג\nדהו"
    assert C.document_input("<document>\nopen ended") == "open ended"
    assert C.document_input("no tags") == ""


def test_reference_text_strips_citations_from_catalogue_prose() -> None:
    """Description-like references lose citation clauses; translations keep every sentence.

    :returns: None.
    """
    d = "A letter about flax. See Smith, Trade Letters, 1999, p. 33. The writer asks for news."
    assert C.reference_text(d, "describe_pgp") == "A letter about flax. The writer asks for news."
    assert C.reference_text(d, "description") == "A letter about flax. The writer asks for news."
    assert C.reference_text(d, "translate_en") == d
    js = json.dumps({"summary": d, "document_type": "Letter"})
    assert C.reference_text(js, "catalogue_json") == "A letter about flax. The writer asks for news."


# ------------------------------------------------------------------------------------ world
@pytest.fixture()
def cworld(tmp_path: Path) -> P.Paths:
    """The holdouts world plus texts: a dev English target, a test Hebrew edition, a train target.

    :param tmp_path: Pytest temporary directory.
    :returns: Its paths.
    """
    return make_world(tmp_path, v01_extra={
        "dev": [v01_row("9020", "dev", "T-S 77J3.4", task="translate_en", target=PROSE)],
        "test": [v01_row("9021", "test", "T-S 76J3.4", task="translate_he", target="תרגום קצר", document=HE_EDITION)],
        "train": [v01_row("9010", "train", "T-S 88J2.1", task="translate_en", target="train only words " * 10)]},
        pairs_extra=[{"pgpid": "9010", "shelfmark": "T-S 88J2.1", "translation_en": "train pair " * 10}])


@pytest.fixture()
def cheld(cworld: P.Paths) -> H.Holdouts:
    """H over the contamination world.

    :param cworld: Paths.
    :returns: The hold-out set.
    """
    return H.build_holdouts(cworld, registry=REGISTRY)


def test_reference_texts(cworld: P.Paths, cheld: H.Holdouts) -> None:
    """References are the held documents' targets, edition inputs, pair texts and descriptions only.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :returns: None.
    """
    refs = {r: (d, t) for r, d, t in C.reference_texts(cworld, cheld)}
    assert refs["tgt:translate_en:9020"] == ("9020", PROSE)
    assert refs["inp:9021:2"][1] == HE_EDITION and refs["inp:9021"][1] == "שלום עליכם"
    assert "tgt:describe_pgp:9002" in refs and "desc:9001" in refs
    assert not any(d == "9010" for d, _ in refs.values())
    bg = list(C.background_texts(cworld, cheld))
    assert {d for d, _ in bg} == {"9010"}


def test_build_reference_index_cache(cworld: P.Paths, cheld: H.Holdouts, monkeypatch: pytest.MonkeyPatch) -> None:
    """The second build loads the cached index; a different setting rebuilds.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param monkeypatch: Pytest monkeypatch.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    assert idx.is_leak(" ".join(PROSE.split()[:30])) and idx.n_background_docs == 1
    assert list(cworld.out("cache").glob("ngram_index_*.npz"))

    def boom(*a: Any, **k: Any) -> None:
        """Fail if called.

        :param a: Ignored.
        :param k: Ignored.
        :returns: None.
        """
        raise AssertionError("rebuilt")

    monkeypatch.setattr(C.NgramIndex, "build", boom)
    again = C.build_reference_index(cworld, cheld)
    assert again.stats() == idx.stats()
    with pytest.raises(AssertionError):
        C.build_reference_index(cworld, cheld, background=False)


# ------------------------------------------------------------------------------------ page exclusions
def test_page_resolver() -> None:
    """Triples join exactly or by the last part component; neighbours stay inside the part.

    :returns: None.
    """
    r = C.PageResolver(["w/vol1/sec/page_001_structured.json", "w/vol1/sec/page_002_structured.json",
                        "w/other/page_002_structured.json"])
    assert r.resolve("w", "vol1/sec", "page_002_structured.json") == "w/vol1/sec/page_002_structured.json"
    assert r.resolve("w", "sec", "page_001_structured.json") == "w/vol1/sec/page_001_structured.json"
    assert r.resolve("w", "nope", "page_001_structured.json") is None
    assert r.resolve_key("w/sec/page_002_structured.json") == "w/vol1/sec/page_002_structured.json"
    assert r.resolve_key("bad") is None
    assert r.neighbours("w/vol1/sec/page_001_structured.json") == ["w/vol1/sec/page_002_structured.json"]


def _lit_world(p: P.Paths) -> List[PG.PageRec]:
    """Write a treatments file, the old page index, a RAG index and an extra list; return the pages.

    :param p: Paths.
    :returns: Synthetic pages (bookA/p1 pages 1–7, bookB/p pages 1–3).
    """
    pages = [mk_page("bookA", "p1", i, f"Plain page {i} about the history of trade in the city.") for i in range(1, 8)]
    pages[1] = mk_page("bookA", "p1", 2, "As T-S 77J3.4 shows, the merchant wrote twice.")
    pages[6] = mk_page("bookA", "p1", 7, "Discussed in Bodl. MS heb. d 99/7 and elsewhere.", keys=["Bodl. MS heb. d 99/7"])
    pages += [mk_page("bookB", "p", 1, "Translation: " + " ".join(PROSE.split()[:40])),
              mk_page("bookB", "p", 2, "An appendix page."), mk_page("bookB", "p", 3, "Clean page.", ocr=HE_BOOK)]
    with open(p.document_treatments, "w") as fh:
        fh.write(json.dumps({"book": "bookA", "part": "p1", "pgpids": ["9001"], "merged_doc_ids": [],
                             "pages": ["page_005_structured.json"]}) + "\n")
        fh.write(json.dumps({"book": "bookA", "part": "p1", "pgpids": ["9010"], "pages": ["page_004_structured.json"]}) + "\n")
    with open(p.analysis_pages_index, "w") as fh:
        fh.write(json.dumps({"book": "bookA", "part": "p1", "page_file": "page_004_structured.json", "pgpids": ["9010"],
                             "merged_doc_ids": [], "shelfmark_raw": "T-S 88J2.1", "located_in_text": True}) + "\n")
    p.out_dir.mkdir(parents=True, exist_ok=True)
    with open(p.out(C.RAG_INDEX_FILE), "w") as fh:
        fh.write(json.dumps({"page_key": "bookA/p1/page_001_structured.json", "pgpids": [],
                             "merged_doc_ids": ["Cambridge_CUL_T_S_66J5_5"], "tier": "mention", "evidence": "located"}) + "\n")
    (p.out_dir.parent / C.EXTRA_EXCLUDED_FILE).write_text(json.dumps({"bookB/p/page_002_*": "appendix"}))
    return pages


def test_build_excluded_pages(cworld: P.Paths, cheld: H.Holdouts) -> None:
    """Every exclusion source contributes its reason; clean pages stay out of the map.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :returns: None.
    """
    pages = _lit_world(cworld)
    idx = C.build_reference_index(cworld, cheld)
    ex, man = C.build_excluded_pages(cworld, cheld, pages, idx)
    a = "bookA/p1/page_{:03d}_structured.json".format
    assert ex[a(2)] == ["mention:text:key:dev", "window:mention:rag:extra:sealed_x"]
    assert "window:mention:text:key:dev" in ex[a(1)] and "window:mention:text:key:dev" in ex[a(3)]
    assert "mention:rag:extra:sealed_x" in ex[a(1)]
    assert ex[a(5)] == ["treatment:pgp131"]
    assert ex[a(7)] == ["mention:page_keys:key:pgp131", "mention:text:key:pgp131"]
    assert ex[a(6)] == ["window:mention:page_keys:key:pgp131", "window:mention:text:key:pgp131"]
    assert a(4) not in ex
    assert ex["bookB/p/page_001_structured.json"] == ["ngram:tgt:translate_en:9020#w8"]
    assert ex["bookB/p/page_002_structured.json"] == ["extra:appendix"]
    assert ex["bookB/p/page_003_structured.json"] == ["ngram:inp:9021:2#c20"]
    assert man["inputs"] == {"document_treatments": True, "analysis_pages_index": True, "rag_index": True,
                             "extra_excluded_pages": True, "train_dup": False}
    c = man["counts"]
    assert c["excluded_pages"] == len(ex) and c["treatments"]["spans_held"] == 1
    assert c["pages_by_reason_class"]["ngram"] == 2 and c["by_work"]["bookB"]["pages"] == 3
    assert c["pages_by_tier"]["benchmark"] >= 3


def test_excluded_pages_without_optional_inputs(cworld: P.Paths, cheld: H.Holdouts) -> None:
    """A missing RAG index or extra list is logged and skipped; the n-gram gate can be switched off.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :returns: None.
    """
    pages = _lit_world(cworld)
    cworld.out(C.RAG_INDEX_FILE).unlink()
    (cworld.out_dir.parent / C.EXTRA_EXCLUDED_FILE).unlink()
    ex, man = C.build_excluded_pages(cworld, cheld, pages, use_ngram=False)
    assert man["inputs"]["rag_index"] is False and man["inputs"]["extra_excluded_pages"] is False
    assert "bookB/p/page_001_structured.json" not in ex and "bookB/p/page_002_structured.json" not in ex
    assert not any(r.startswith("mention:rag") for rs in ex.values() for r in rs)
    assert "ngram_index" not in man["counts"]


def test_load_extra_excluded(tmp_path: Path) -> None:
    """The extra list accepts a list of keys or a mapping of keys / globs to reasons.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    r = C.PageResolver(["w/p/page_001_structured.json", "w/p/page_002_structured.json"])
    (tmp_path / "a.json").write_text(json.dumps(["w/p/page_001_structured.json", "w/p/page_009_structured.json"]))
    out, unmatched = C.load_extra_excluded(tmp_path / "a.json", r)
    assert dict(out) == {"w/p/page_001_structured.json": {"extra:list"}} and unmatched == 1
    (tmp_path / "b.json").write_text(json.dumps({"w/p/*": ["held", "boi"]}))
    out, _ = C.load_extra_excluded(tmp_path / "b.json", r)
    assert out["w/p/page_002_structured.json"] == {"extra:held", "extra:boi"}
    assert C.load_extra_excluded(None, r) == ({}, 0)


# ------------------------------------------------------------------------------------ D5: dedupe + rows
#: 200 distinct words: a long train translation target.
TRAIN_TR = " ".join(f"beta{i}" for i in range(200))


def _fam_row(family: str, split: str, pgpid: str, target: str, condition: str = "edition",
             row_id: Optional[str] = None) -> Dict[str, Any]:
    """A family-file row (v0_2 meta subset).

    :param family: Family.
    :param split: Split.
    :param pgpid: PGP id.
    :param target: Assistant text.
    :param condition: Condition.
    :param row_id: Row id (default derived from the other fields).
    :returns: The row.
    """
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Do it."},
                         {"role": "assistant", "content": target}],
            "meta": {"family": family, "task": family, "split": split, "pgpid": pgpid, "condition": condition,
                     "row_id": row_id or f"{family[:4]}{pgpid}{condition[:3]}{split[:2]}"}}


def _write_fams(d: Path, files: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Write family files ``<name>.jsonl`` under ``d``.

    :param d: Directory.
    :param files: File stem → rows.
    :returns: ``d``.
    """
    d.mkdir(parents=True, exist_ok=True)
    for stem, rows in files.items():
        (d / f"{stem}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return d


def test_train_target_dedupe(tmp_path: Path) -> None:
    """D5: a page holding ≥ 20 % of a TRAIN translate / analyse target's 8-grams is a duplicate; dev
    targets and other families do not count; the L2 / L5 extra files count when present.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    words = TRAIN_TR.split()
    ana = " ".join(f"gamma{i}" for i in range(100))
    fams = _write_fams(tmp_path / "families", {
        "translate_en": [_fam_row("translate_en", "train", "1", TRAIN_TR),
                         _fam_row("translate_en", "train", "1", TRAIN_TR, condition="read"),
                         _fam_row("translate_en", "dev", "2", " ".join(f"delta{i}" for i in range(100)))],
        "describe_pgp": [_fam_row("describe_pgp", "train", "3", " ".join(f"eps{i}" for i in range(100)))],
        "analyse_document_v2": [_fam_row("analyse_document", "train", "4", ana)]})
    refs = list(C.train_target_texts(fams))
    assert sorted(r for r, _, _ in refs) == ["analyse_document:4", "translate_en:1", "translate_en:1"]
    idx = C.build_train_target_index(refs)
    assert idx.stats()["references"] == 2  # identical targets indexed once
    pages = [mk_page("bk", "p", 1, "Intro. " + " ".join(words[:60]) + " end."),       # ≈ 27 % of the target
             mk_page("bk", "p", 2, " ".join(words[100:130])),                         # ≈ 12 %: kept
             mk_page("bk", "p", 3, " ".join(f"delta{i}" for i in range(100))),      # dev target: kept
             mk_page("bk", "p", 4, " ".join(f"eps{i}" for i in range(100))),        # describe target: kept
             mk_page("bk", "p", 5, "x " + " ".join(f"gamma{i}" for i in range(30)))]  # analyse (L2 file): 23 %
    dup = C.train_dup_pages(pages, idx)
    assert sorted(dup) == ["bk/p/page_001_structured.json", "bk/p/page_005_structured.json"]
    assert dup["bk/p/page_001_structured.json"] == {"train_target_dup:translate_en:1"}
    assert dup["bk/p/page_005_structured.json"] == {"train_target_dup:analyse_document:4"}


def test_excluded_pages_with_train_dedupe(cworld: P.Paths, cheld: H.Holdouts) -> None:
    """The dedupe pass adds its reason to ``build_excluded_pages`` and its counts to the manifest.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :returns: None.
    """
    pages = _lit_world(cworld)
    pages.append(mk_page("bookB", "p", 4, " ".join(TRAIN_TR.split()[:80])))
    idx = C.build_train_target_index([("translate_en:1", "1", TRAIN_TR)])
    ex, man = C.build_excluded_pages(cworld, cheld, pages, use_ngram=False, train_dup_index=idx)
    assert ex["bookB/p/page_004_structured.json"] == ["train_target_dup:translate_en:1"]
    assert man["counts"]["train_dup_pages"] == 1 and man["inputs"]["train_dup"] is True
    assert man["build_args"]["train_dup_share"] == C.TRAIN_DUP_SHARE
    assert man["counts"]["pages_by_reason_class"]["train_target_dup"] == 1


def test_build_excluded_rows(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """D5 rows: TRAIN rows of PGP families / slice / RAG whose target leaks an eval reference; dev rows
    and literature rows are not scanned; the file round-trips through ``load_excluded_rows``.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    leak = " ".join(PROSE.split()[100:140])
    fams = _write_fams(tmp_path / "families", {
        "translate_en": [_fam_row("translate_en", "train", "9010", leak), _fam_row("translate_en", "train", "9011", "Clean.")],
        "noise_slice": [_fam_row("translate_en", "train", "9010", leak, condition="read_hard")],
        "rag_context": [_fam_row("rag_context", "train", "9010", leak, condition="literature")],
        "describe_pgp": [_fam_row("describe_pgp", "dev", "9020", leak)],
        "domain_reading": [_fam_row("domain_reading", "train", "x", leak, condition="literature")]})
    ex, man = C.build_excluded_rows(fams, idx)
    assert set(ex) == {"tran9010editr", "tran9010reatr", "rag_9010littr"}
    assert all(v == "ngram_target:tgt:translate_en:9020#w8" for v in ex.values())
    assert man["excluded_rows"] == 3 and man["excluded_by_family_condition"]["translate_en|read_hard"] == 1
    assert man["train_rows_scanned"]["translate_en|edition"] == 2 and "describe_pgp|edition" not in man["train_rows_scanned"]
    (tmp_path / "x.json").write_text(json.dumps(ex))
    assert C.load_excluded_rows(tmp_path / "x.json") == ex
    assert C.load_excluded_rows(None) == {} and C.load_excluded_rows(tmp_path / "missing.json") == {}


def test_l1_only_reasons_bind_domain_reading_only(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """A page excluded only by the train-target dedupe fails an L1 row but not a RAG row in the re-scan.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    assert C.excluded_for_family(["train_target_dup:translate_en:1"], "domain_reading")
    assert not C.excluded_for_family(["train_target_dup:translate_en:1"], "rag_context")
    assert C.excluded_for_family(["train_target_dup:translate_en:1", "window:mention:rag:test"], "rag_context")
    k = "bookA/p1/page_004_structured.json"
    rows = [_row("rag_context", "9010", None, "S.", user="[SOURCE 1] x", lit_sources=[{"page_key": k}]),
            _row("domain_reading", None, None, "Prose.", user="Source: y", page_keys=[k])]
    f = tmp_path / "m.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = C.rescan(f, cworld, cheld, use_ngram=False, excluded={k: ["train_target_dup:translate_en:1"]})
    assert rep["checks"]["excluded_page_hits"]["value"] == 1
    assert rep["examples"]["excluded_page_hits"][0]["family"] == "domain_reading"


def test_pair_exclusions_catch_boundary_mentions_and_union_leaks(cworld: P.Paths, cheld: H.Holdouts) -> None:
    """Two consecutive pages: a held shelfmark split over the page break, and an eval passage half on each
    page, are caught only on the joined (furniture-stripped) chunk text; already excluded pages are skipped.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :returns: None.
    """
    w = PROSE.split()
    pages = [mk_page("bk", "p", 1, "Plain opening page on trade."),
             mk_page("bk", "p", 2, "As the deed in Bodl."),
             mk_page("bk", "p", 3, "MS heb. d 99/7 shows, the merchant wrote twice."),
             mk_page("bk", "p", 4, "Plain middle page on ships."),
             mk_page("bk", "p", 5, " ".join(w[100:111])),
             mk_page("bk", "p", 6, " ".join(w[111:122])),
             mk_page("bk", "p", 7, "Closing page."),
             mk_page("bk", "p", 9, "Not consecutive with page 7.")]
    pages = [mk_page(pg.work, pg.part, pg.page_no, pg.full_main_text, ocr=pg.full_main_text) for pg in pages]
    keys = [k for _, _, k in [(a, b, a) for a, b, _ in C.page_pairs(pages)]]
    assert len(keys) == 6  # pairs (1,2) .. (6,7); page 9 has no neighbour
    texts = {(a, b): t for a, b, t in C.page_pairs(pages)}
    assert texts[("bk/p/page_002_structured.json", "bk/p/page_003_structured.json")] == (
        "As the deed in Bodl.\n\nMS heb. d 99/7 shows, the merchant wrote twice.")
    idx = C.build_reference_index(cworld, cheld)
    assert not idx.is_leak(pages[4].full_main_text) and not idx.is_leak(pages[5].full_main_text)
    res = C.PageResolver(pg.page_key for pg in pages)
    out, cnt = C.pair_exclusions(pages, cheld, idx, set(), res)
    a = "bk/p/page_{:03d}_structured.json".format
    assert "mention:pair:key:pgp131" in out[a(2)] and "mention:pair:key:pgp131" in out[a(3)]
    assert out[a(1)] == {"window:mention:pair:key:pgp131"}  # page 3 also names it, so pair (3, 4) is flagged too
    assert "mention:pair:key:pgp131" in out[a(4)]
    assert "ngram_pair:tgt:translate_en:9020#w8" in out[a(5)] and out[a(6)] == {"ngram_pair:tgt:translate_en:9020#w8"}
    assert cnt["pairs_mention"] == 2 and cnt["pairs_ngram"] == 1 and cnt["pairs_checked"] == 6
    out2, cnt2 = C.pair_exclusions(pages, cheld, None, {a(5)}, res)
    assert a(6) not in out2 and cnt2["pairs_checked"] == 4 and "pairs_ngram" not in cnt2


def test_page_mentions_see_shelfmarks_split_after_trailing_blanks(cheld: H.Holdouts) -> None:
    """``T-S ⏎77J3.4`` (a blank before the line break) is a mention, as the L1 row text will show it.

    :param cheld: Hold-out set.
    :returns: None.
    """
    pg = mk_page("bk", "p", 1, "As T-S \n77J3.4 shows, the merchant wrote twice.")
    assert C._page_mentions([pg], cheld) == {"bk/p/page_001_structured.json": {"mention:text:key:dev"}}


def test_excluded_rows_benchmark_text_mentions(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """A train row whose input names a benchmark document is excluded (the re-scan's mention rule); an
    eval-tier mention is not.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    bench = _fam_row("translate_en", "train", "9010", "Clean target words here.")
    bench["messages"][1]["content"] = "Edition:\nשטר; Bodl. MS heb. d 99/7 - שטר אחר"
    evald = _fam_row("describe_pgp", "train", "9011", "Another clean target.")
    evald["messages"][1]["content"] = "See T-S 77J3.4."
    fams = _write_fams(tmp_path / "families", {"translate_en": [bench], "describe_pgp": [evald]})
    ex, man = C.build_excluded_rows(fams, idx, holdouts=cheld)
    assert ex == {bench["meta"]["row_id"]: "benchmark_text_mention:key:pgp131"}
    assert man["excluded_by_kind"] == {"benchmark_text_mention": 1}
    assert C.build_excluded_rows(fams, idx)[0] == {}


def test_excluded_rows_split_conflicts(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """A TRAIN row whose document another family file holds out (by pgpid / pgpids, by a ``book:`` doc id
    or by a book entry in provenance) is excluded with a ``split_conflict`` reason; switched off, it stays.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    tr_book = _fam_row("translate_en", "train", "", "A clean book translation.", row_id="bookA17")
    tr_book["meta"].update({"pgpid": None, "doc_id": "book:srcA:17"})
    l2_test = _fam_row("analyse_document", "test_indist", "", "Clean commentary.", row_id="l2A17")
    l2_test["meta"].update({"pgpid": None, "provenance": {"book": "srcA", "entry_no": 17}})
    rel = _fam_row("extract_relations", "train", "9100", "Clean relations.")
    held = _fam_row("analyse_document", "heldout", "9101", "Clean intro.", row_id="bookB12")
    held["meta"]["pgpids"] = ["9101", "9100"]
    ok = _fam_row("extract_relations", "train", "9102", "Clean other relations.")
    assert C.doc_identities(held["meta"]) == {"pgpid:9101", "pgpid:9100"}
    assert C.doc_identities(l2_test["meta"]) == {"book:srcA:17"} == C.doc_identities(tr_book["meta"])
    fams = _write_fams(tmp_path / "families", {"book_pairs": [tr_book, held], "analyse_document_v2": [l2_test],
                                               "extract_relations": [rel, ok]})
    ex, man = C.build_excluded_rows(fams, idx, holdouts=cheld)
    assert ex == {"bookA17": "split_conflict:test_indist:book:srcA:17",
                  rel["meta"]["row_id"]: "split_conflict:heldout:pgpid:9100"}
    assert man["excluded_by_kind"] == {"split_conflict": 2}
    assert C.build_excluded_rows(fams, idx, holdouts=cheld, split_conflicts=False)[0] == {}


def test_rescan_reads_provenance_page_keys(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """A domain-reading row whose page keys sit only in ``provenance.page_keys`` is checked, not skipped.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    rows = [_row("domain_reading", None, None, "Plain prose.", user="Source: x", work="bookA", page_range="2",
                 provenance={"page_keys": ["bookA/p1/page_002_structured.json"]}),
            _row("domain_reading", None, None, "Other prose.", user="Source: y", work="bookA", page_range="4",
                 provenance={"page_keys": ["bookA/p1/page_004_structured.json"]})]
    f = tmp_path / "m.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = C.rescan(f, cworld, cheld, use_ngram=False, excluded={"bookA/p1/page_002_structured.json": ["x"]})
    assert rep["checks"]["excluded_page_hits"]["value"] == 1
    assert "literature_rows_pages_unchecked" not in rep["reported"]


# ------------------------------------------------------------------------------------ re-scan
def _row(family: str, pgpid: Optional[str], shelfmark: Optional[str], target: str, user: str = "Translate.",
         **meta: Any) -> Dict[str, Any]:
    """A v0_2-style mixture row.

    :param family: Family.
    :param pgpid: PGP id.
    :param shelfmark: Shelfmark.
    :param target: Assistant text.
    :param user: User text.
    :param meta: Extra meta fields.
    :returns: The row.
    """
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": user},
                         {"role": "assistant", "content": target}],
            "meta": dict({"family": family, "task": family, "pgpid": pgpid, "shelfmark": shelfmark, "split": "train",
                          "row_id": f"r{pgpid}{family[:3]}{len(target)}"}, **meta)}


def test_rescan_checks(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """Each failure kind is counted by its check; a clean mixture passes.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    clean = [_row("translate_en", "9010", "T-S 88J2.1", "A clean translation of a train document."),
             _row("rag_context", "9010", "T-S 88J2.1", "Summary.", user="<document>\nx\n</document>\n[SOURCE 1] A, *B*, 2000, p. 3",
                  lit_sources=[{"page_key": "bookA/p1/page_004_structured.json"}])]
    f = tmp_path / "clean.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in clean))
    rep = C.rescan(f, cworld, cheld, idx, excluded={"bookA/p1/page_002_structured.json": ["x"]})
    assert rep["ok"] and rep["rows"] == 2 and rep["reported"] == {}
    bad = clean + [
        _row("describe_pgp", "9002", "Bodl. MS heb. d 99/7", "A deed."),
        _row("describe_pgp", "9020", "T-S 77J3.4", "A dev doc."),
        _row("describe_pgp", "9040", "T-S 66J5.5", "A sealed doc."),
        _row("describe_pgp", "9010", "T-S 88J2.1", f"Cf. the {SCRUB_SURNAME} images."),
        _row("describe_pgp", "9010", "T-S 88J2.1", "Joined to T-S 99J1.2 by the editor."),
        _row("rag_context", "9010", None, "S.", user="[SOURCE 1] x", lit_sources=[{"page_key": "bookA/p1/page_002_structured.json"}]),
        _row("domain_reading", None, None, " ".join(PROSE.split()[:60]), user="Source: x", work="bookB", page_range="1"),
        _row("translate_en", "9010", "T-S 88J2.1", " ".join(PROSE.split()[200:230])),
    ]
    g = tmp_path / "bad.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in bad))
    rep = C.rescan(g, cworld, cheld, idx, excluded={"bookA/p1/page_002_structured.json": ["x"]})
    v = {k: c["value"] for k, c in rep["checks"].items()}
    assert v == {"benchmark_meta_hits": 1, "eval_split_meta_hits": 1, "excluded_page_hits": 1,
                 "benchmark_text_mentions": 1, "literature_text_mentions": 0, "unmapped_entry_rows": 0, "scrub_rows": 1,
                 "non_nfc_rows": 0, "literature_ngram_leaks": 1, "canon_holdout_hits": 0}
    assert not rep["ok"]
    assert rep["reported"]["extra_benchmark_meta_hits"] == 1
    assert rep["reported"]["literature_rows_pages_unchecked"] == 1
    assert rep["ngram_leak_rows_by_family"]["translate_en"] == {"target": 1}
    assert rep["examples"]["benchmark_meta_hits"][0]["reason"] == "key:pgp131"
    assert "content" not in json.dumps(rep["examples"])


def test_canon_holdout_route(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """Canon rows are checked by KTIV system number, by H over pgpid / catalogue pgpids / canonical id / shelfmark key
    (both tiers) and by GT shingle overlap, in the re-scan and in the row exclusions.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    gt = "אבגדהוזחטיכלמנסעפצקרשת" * 3
    ktiv = ({"990000000000000001": "ktiv_benchmark:religious_synth"}, CD.shingle_set([gt]))
    frag = "<document>\nשלום שלום\n</document>"

    def canon(fam: str, sys_num: str, **meta: Any) -> Dict[str, Any]:
        """A canon-family row.

        :param fam: Family.
        :param sys_num: KTIV system number.
        :param meta: Extra meta fields.
        :returns: The row.
        """
        prov = {"ktiv_sys_num": sys_num, "catalogue_pgpids": meta.pop("catalogue_pgpids", [])}
        return _row(fam, meta.pop("pgpid", None), meta.pop("shelfmark", None), "{}", user=meta.pop("user", frag),
                    sys_num=sys_num, provenance=prov, row_id=f"c{sys_num[-2:]}{fam[6:9]}", **meta)

    clean = [canon("canon_diff", "990000000000000002"), canon("canon_identify", "990000000000000002")]
    bad = clean + [
        canon("canon_diff", "990000000000000001"),                                    # KTIV benchmark sys_num
        canon("canon_diff", "990000000000000003", catalogue_pgpids=["9020"]),         # eval-tier pgpid (2nd id)
        canon("canon_diff", "990000000000000004", canonical_id="Oxford_Bodleian_Bodl_MS_heb_d_99_7"),  # PGP-131 id
        canon("canon_identify", "990000000000000005", shelfmark="Bodl. MS heb. d 99/7"),              # shelfmark key
        canon("canon_identify", "990000000000000006", user=f"<document>\n{gt}\n</document>"),       # GT overlap
    ]
    assert C.canon_holdout_reason(bad[2]["meta"], frag, cheld, ktiv) == "sys_num:ktiv_benchmark:religious_synth"
    assert C.canon_holdout_reason(bad[3]["meta"], frag, cheld, ktiv).startswith("id:")
    assert C.canon_holdout_reason(bad[6]["meta"], bad[6]["messages"][1]["content"], cheld, ktiv) == "gt_overlap"
    assert C.canon_holdout_reason(clean[0]["meta"], frag, cheld, ktiv) is None
    f, g = tmp_path / "clean.jsonl", tmp_path / "bad.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf-8")
    g.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bad), encoding="utf-8")
    rep = C.rescan(f, cworld, cheld, use_ngram=False, ktiv=ktiv)
    assert rep["ok"] and rep["reported"]["canon_rows_checked"] == 2
    rep = C.rescan(g, cworld, cheld, use_ngram=False, ktiv=ktiv)
    assert rep["checks"]["canon_holdout_hits"]["value"] == 5 and not rep["ok"]
    fam_dir = tmp_path / "fam"
    fam_dir.mkdir()
    (fam_dir / "canon_diff.jsonl").write_text(g.read_text(encoding="utf-8"), encoding="utf-8")
    idx = C.build_reference_index(cworld, cheld)
    out, man = C.build_excluded_rows(fam_dir, idx, holdouts=cheld, ktiv=ktiv, split_conflicts=False)
    assert sorted(v.split(":")[0] for v in out.values()) == [C.CANON_HOLDOUT_REASON] * 5
    assert man["excluded_by_kind"][C.CANON_HOLDOUT_REASON] == 5
    assert "canon_diff" in C.ROW_SCAN_FAMILIES and "canon_identify" in C.ROW_SCAN_FAMILIES
    assert "pd_editions.jsonl" in C.TRAIN_DUP_EXTRA_FILES and "canon_diff" not in C.TRAIN_DUP_FAMILIES


def test_rescan_without_excluded_or_ngram(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """Without ``excluded_pages.json`` and n-grams the page and leak checks report as skipped.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    f = tmp_path / "m.jsonl"
    f.write_text(json.dumps(_row("translate_en", "9010", "T-S 88J2.1", "Fine.")) + "\n")
    rep = C.rescan(f, cworld, cheld, use_ngram=False)
    assert rep["ok"] and rep["checks"]["excluded_page_hits"]["skipped"] and rep["checks"]["literature_ngram_leaks"]["skipped"]
    assert rep["ngram_index"] is None


# ------------------------------------------------------------------------------------ CLI
def _write_lit_tree(root: Path) -> None:
    """A two-page structured literature tree (``iter_pages`` layout, no OCR files).

    :param root: Literature root.
    :returns: None.
    """
    d = root / "bookC" / "part1_structured"
    d.mkdir(parents=True)
    (d / "page_001_structured.json").write_text(json.dumps({"full_main_text": "Translation: " + " ".join(PROSE.split()[:30]),
                                                            "metadata": {"page_number": 1}}))
    (d / "page_002_structured.json").write_text(json.dumps({"full_main_text": "A page on T-S 76J3.4 and its date.",
                                                            "metadata": {"page_number": 2}}))
    (d / "page_003_structured.json").write_text(json.dumps({"full_main_text": "Nothing here.", "metadata": {"page_number": 3}}))
    (d / "page_004_structured.json").write_text(json.dumps({"full_main_text": "Nor here.", "metadata": {"page_number": 4}}))


def test_cli(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``ngram-index``, ``excluded-pages`` and ``rescan`` write their files; ``rescan`` exits 1 on a failure.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest output capture.
    :returns: None.
    """
    cheld.save(p=cworld)
    _write_lit_tree(cworld.academic_literature)
    cfg = str(cworld.source_file)
    assert C.main(["--paths", cfg, "ngram-index"]) == 0
    assert json.loads(cworld.out("ngram_index_manifest.json").read_text())["stats"]["references"] > 0
    assert C.main(["--paths", cfg, "excluded-pages"]) == 0
    ex = json.loads(cworld.out(C.EXCLUDED_PAGES_FILE).read_text())
    assert ex["bookC/part1/page_001_structured.json"] == ["ngram:tgt:translate_en:9020#w8", "window:mention:text:key:test"]
    assert ex["bookC/part1/page_002_structured.json"] == ["mention:text:key:test"]
    assert ex["bookC/part1/page_003_structured.json"] == ["window:mention:text:key:test"]
    assert "bookC/part1/page_004_structured.json" not in ex
    man = json.loads(cworld.out(C.EXCLUDED_MANIFEST_FILE).read_text())
    assert man["inputs"]["rag_index"] is False and man["build_args"]["background"] is True
    fam_dir = _write_fams(cworld.out("families"), {"translate_en": [
        _fam_row("translate_en", "train", "9010", " ".join(PROSE.split()[50:90]))]})
    assert C.main(["--paths", cfg, "excluded-rows"]) == 0
    assert json.loads(cworld.out(C.EXCLUDED_ROWS_FILE).read_text()) == {"tran9010editr": "ngram_target:tgt:translate_en:9020#w8"}
    assert json.loads(cworld.out(C.EXCLUDED_ROWS_MANIFEST_FILE).read_text())["files_scanned"] == {"translate_en.jsonl": 1}
    assert fam_dir.is_dir()
    m = tmp_path / "mix.jsonl"
    m.write_text(json.dumps(_row("translate_en", "9002", None, "x")) + "\n")
    capsys.readouterr()
    assert C.main(["--paths", cfg, "rescan", str(m)]) == 1
    rep = json.loads(cworld.out(C.RESCAN_FILE).read_text())
    assert rep["checks"]["benchmark_meta_hits"]["value"] == 1 and rep["build_args"]["mixture"] == "mix.jsonl"


# ------------------------------------------------------------------------------------ fix pass F1-F4 (2026-10-22)
def test_mention_key_candidates_spaced_spellings() -> None:
    """Spaced, abbreviated and folio spellings give the canonical form's key; a number is never cut in two.

    :returns: None.
    """
    assert C.mention_tokens("Cambridge T-S 98 J 7 6, ed.") == ["t", "s", "98", "j", "7", "6", "ed"]
    assert C.compact_tokens(["ts", "ns", "999", "19"]) == "tsns999_19"
    assert C.mention_key_candidates("TS 98 J 7 6")[0] == "ts98j7_6"
    assert "ts98j7" not in C.mention_key_candidates("TS 98 J 7 6")
    assert C.mention_key_candidates("Moss. IX,249.")[0] == "ix249"
    assert "mshebd99_22" in C.mention_key_candidates("Bodl. MS Heb. d 99, fol. 22")
    assert C.mention_key_candidates("ENA 9822 A.56 contains")[-1] == "ena9822"
    assert "ena9822a56" in C.mention_key_candidates("ENA 9822 A.56 contains")
    assert C.mention_key_candidates("TS") == []
    text = "see T-S ns. 999.19: a paper; and TS 96.67.\n15 Tosefta"
    spans = C.mention_spans(text)
    assert len(spans) == 2 and text[spans[1][0]:spans[1][1]] == "TS 96.67.\n15 Tosefta"   # F10: one line break
    cands = C.mention_key_candidates(text[spans[1][0]:spans[1][1]])
    assert cands.index("ts96_67_15") < cands.index("ts96_67")       # the next line's number may be cut off
    assert C.mention_spans("T-S 96.67\nsecond line\nthird 5")[0] == (0, 21)   # never past a second line break


def test_mention_index_resolves_spellings_the_old_regex_missed(cheld: H.Holdouts) -> None:
    """``TS 99 J 1 2`` / ``T-S 99 J 1.2`` name the PGP-131 document T-S 99J1.2; ``T-S ns. 999.5`` the registry one;
    ``T-S 77 J 3.4`` the dev document (eval tier only); a longer number names nothing.

    :param cheld: Hold-out set of the synthetic world.
    :returns: None.
    """
    from src.finetuning.text27b import textnorm as T
    mi = C.mention_index(cheld)
    assert C.mention_index(cheld) is mi
    for text in ("as TS 99 J 1 2, ed. X, shows", "in T-S 99 J 1.2 the merchant"):
        old = sorted({text[a:b] for a, b in T.find_shelfmarks(text)})
        assert not (old and cheld.match_any(shelfmarks=old)), text            # the pre-fix route missed it
        assert mi.reason([text], include_eval=False) == "key:pgp131"
        assert C.text_mention_reason([text], cheld, include_eval=False) == "key:pgp131"
    assert C.text_mention_reason(["T-S ns. 999.5: a single paper"], cheld, include_eval=False) == "key:registry"
    assert C.text_mention_reason(["T-S NS 999, f. 5"], cheld, include_eval=False) == "key:registry"
    assert C.text_mention_reason(["see T-S 77 J 3.4"], cheld, include_eval=False) is None
    assert C.text_mention_reason(["see T-S 77 J 3.4"], cheld) == "key:dev"
    assert C.text_mention_reason(["TS 99 J 1 25 and TS 99 J 1"], cheld) is None
    assert C.text_mention_reason(["Bodl. MS heb. d 99/7"], cheld, include_eval=False) == "key:pgp131"   # old route kept
    found = mi.find("TS 99 J 1 2 and T-S 77J3.4")
    assert [(k, any(r.endswith("pgp131") for r in rs)) for _, k, rs in found] == [("ts99j1_2", True), ("ts77j3_4", False)]


def test_spaced_mentions_reach_pages_rows_and_rescan(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """F1: a page, a train row (input or target) and a mixture row naming a held document in a spaced spelling are
    caught by the page exclusions, the row exclusions and the re-scan alike.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    pg = mk_page("bk", "p", 1, "The letter TS 99 J 1 2 was sent from Fustat.")
    assert C._page_mentions([pg], cheld) == {"bk/p/page_001_structured.json": {"mention:text:key:pgp131"}}
    idx = C.build_reference_index(cworld, cheld)
    tgt = _fam_row("describe_pgp", "train", "9010", "Reused by the same scribe as T-S 99 J 1.2.")
    inp = _fam_row("catalogue_json", "train", "9011", "Clean summary target.")
    inp["messages"][1]["content"] = "Edition:\nשטר\n(Editor's note: compare TS 99 J 1 2.)"
    fams = _write_fams(tmp_path / "families", {"describe_pgp": [tgt], "catalogue_json": [inp]})
    ex, _ = C.build_excluded_rows(fams, idx, holdouts=cheld)
    assert ex == {tgt["meta"]["row_id"]: "benchmark_text_mention:key:pgp131",
                  inp["meta"]["row_id"]: "benchmark_text_mention:key:pgp131"}
    mix = tmp_path / "m.jsonl"
    rows = [_row("domain_reading", None, None, "A page naming TS 99 J 1 2 in passing.", user="Source: x",
                 page_keys=["bk/p/page_009_structured.json"]),
            _row("domain_reading", None, None, "A page naming T-S 77 J 3.4.", user="Source: y",
                 page_keys=["bk/p/page_010_structured.json"])]
    mix.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = C.rescan(mix, cworld, cheld, use_ngram=False, excluded={})
    assert rep["checks"]["benchmark_text_mentions"]["value"] == 1 and rep["reported"]["eval_text_mentions"] == 1
    assert rep["distinct_rows"]["benchmark_text_mentions"] == 1
    assert rep["examples"]["benchmark_text_mentions"][0]["mentions"][0][1] == "ts99j1_2"


def test_page_rule_covers_analyse_and_book_rows(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """F2: analyse / book-pair train rows on an excluded page are excluded (L1-only reasons do not bind them) and
    the re-scan checks their page keys; ``provenance.work_pages`` counts when ``meta.page_keys`` is absent.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    a = "bk/p/page_00{}_structured.json".format
    ana = _fam_row("analyse_document", "train", "9010", "Commentary on the letter.", row_id="ana1")
    ana["meta"]["page_keys"] = [a(1), a(2)]
    dup = _fam_row("analyse_document", "train", "9010", "Other commentary.", row_id="ana2")
    dup["meta"]["page_keys"] = [a(3)]
    book = _fam_row("translate_he", "train", "", "Hebrew translation.", row_id="bk1")
    book["meta"].update({"pgpid": None, "provenance": {"work_pages": {"translation": [a(4)]}}})
    excluded = {a(2): ["extra:int:l5:dev:book:x:1:window"], a(3): ["train_target_dup:analyse_document:9010"],
                a(4): ["mention:text:key:test"]}
    fams = _write_fams(tmp_path / "families", {"analyse_document_v2": [ana, dup], "book_pairs": [book]})
    ex, man = C.build_excluded_rows(fams, idx, holdouts=cheld, excluded_pages=excluded)
    assert ex == {"ana1": f"excluded_page:{a(2)}", "bk1": f"excluded_page:{a(4)}"}
    assert man["excluded_by_kind"] == {"excluded_page": 2}
    assert C.build_excluded_rows(fams, idx, holdouts=cheld)[0] == {}
    assert C.row_page_keys(book["meta"], None) == ([a(4)], True)
    book["meta"]["page_keys"] = []
    assert C.row_page_keys(book["meta"], None) == ([], True)
    mix = tmp_path / "m.jsonl"
    mix.write_text("".join(json.dumps(r) + "\n" for r in (ana, dup)))
    rep = C.rescan(mix, cworld, cheld, use_ngram=False, excluded=excluded)
    assert rep["checks"]["excluded_page_hits"]["value"] == 1
    assert rep["rows_with_page_keys_by_family"] == {"analyse_document": 2}


def test_rescan_nfc_and_scrub_in_meta(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """F3 / F4: a decomposed message fails the NFC check; an project-name string in meta (``target_raw``) fails the project-name check.

    :param cworld: Paths.
    :param cheld: Hold-out set.
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    decomposed = "Fusṭāṭ"
    rows = [_row("translate_en", "9010", "T-S 88J2.1", f"From {decomposed}."),
            _row("describe_pgp", "9010", "T-S 88J2.1", "A deed.",
                 target_raw="A deed. Image at the " + SCRUB_SURNAME + " site."),
            _row("describe_pgp", "9010", "T-S 88J2.1", "A clean deed.")]
    f = tmp_path / "m.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    rep = C.rescan(f, cworld, cheld, use_ngram=False, excluded={})
    v = {k: c["value"] for k, c in rep["checks"].items()}
    assert v["non_nfc_rows"] == 1 and v["scrub_rows"] == 1 and not rep["ok"]
    assert rep["examples"]["scrub_rows"][0]["where"] == "meta"
    assert C.row_source_scan(rows[1]) and not C.row_source_scan(rows[2])


# ------------------------------------------------------------------------------------ fix pass 2: F10 / F11 (2026-10-22)
def _spell(*parts: Any) -> str:
    """A shelfmark spelling assembled at runtime from its parts (synthetic numbers; no real benchmark list here).

    :param parts: Pieces.
    :returns: The joined string.
    """
    return "".join(str(x) for x in parts)


#: ``(canonical shelfmark, prose naming it in the spelling form the second verification found, tier)``: an
#: ``Arabic Box`` series word, an ``NLI`` prefix, a line break inside the shelfmark, a ``BL OR`` prefix, a ``TS G``
#: series, a ``(Cat. N)`` insert and a line break before ``fol.`` -- with synthetic numbers.
F10_CASES = [
    (_spell("T-S Ar.", 77, ".", 255), _spell("“a quote,” TS Arabic Box ", 77, ".", 255, ", l. 15; translation"), "bench"),
    (_spell("NLI ", 977, ".", 3, "/", 5), _spell("the sender's signing of NLI ", 977, ".", 3, "/", 5, " in 1133 CE."), "bench"),
    (_spell("T-S Misc.", 99, ".", 24), _spell("in the commentary to T-S Misc.\n", 99, ".", 24, "r:7 [no. 23]."), "bench"),
    (_spell("BL OR ", 9566, "B.", 1), _spell("the next note (to which BL OR ", 9566, "B.", 1, ", discovered by"), "eval"),
    (_spell("T-S G", 9, ".", 18), _spell("כ\"י TS G", 9, ".", 18, " (נדפס אצל"), "eval"),
    (_spell("Bodl. MS heb. d ", 98, "/", 139),
     _spell("שנת 1151/2 Bodl. MS. Heb. d. ", 98, " (Cat. ", 9989, "), fol. ", 139, " חומר"), "eval"),
    (_spell("ENA NS ", 7, ".", 38), _spell("is also mentioned in ENA NS ", 7, ",\nfol. ", 38, ", line 4, a poorly"), "eval"),
]


def _f10_index() -> C.MentionIndex:
    """A mention index over the F10 cases' canonical keys (normaliser keys), a MIAC inventory number and two lettered
    leaves of one number, only one of them held.

    :returns: The index.
    """
    from src.finetuning.text27b import shelfmarks as SM
    bench: Dict[str, List[str]] = {}
    evals: Dict[str, List[str]] = {}
    for canon, _, tier in F10_CASES:
        (bench if tier == "bench" else evals)[SM.keys(canon)[0]] = ["pgp131" if tier == "bench" else "dev"]
    evals[SM.keys(_spell("MIAC ", 99973, ".", 46))[0]] = ["test"]
    bench[SM.keys(_spell("T-S NS ", 999, ".", "1b"))[0]] = ["registry"]
    return C.MentionIndex(bench, evals)


def test_f10_spellings_resolve_to_the_canonical_key() -> None:
    """F10: each spelling form of the second verification resolves to the key the normaliser gives its canonical form,
    in the right tier; a neighbouring number names nothing.

    :returns: None.
    """
    from src.finetuning.text27b import shelfmarks as SM
    mi = _f10_index()
    for canon, prose, tier in F10_CASES:
        found = mi.find(prose)
        assert [k for _, k, _ in found] == [C.mention_norm(SM.keys(canon)[0])], (canon, found)
        assert (mi.reason([prose], include_eval=False) is not None) == (tier == "bench"), canon
        assert mi.reason([prose]) is not None, canon
    assert mi.find(_spell("Bodl. MS. Heb. d. ", 98, " (Cat. ", 9989, "), fol. ", 140)) == []
    assert mi.find(_spell("TS Arabic Box ", 77, ".", 256)) == [] and mi.find("or 5 dinars, OR 77") == []
    assert C.mention_text(_spell("d. ", 98, " (Cat. ", 9989, "), fol. ", 139)) == _spell("d. ", 98, ", fol. ", 139)


def test_f10_bare_inventory_number_and_ambiguous_leaf() -> None:
    """F10: a bare MIAC inventory number on an index page resolves; a leaf number without its letter (``1r``, bare) is
    held when any lettered leaf is held, an explicit other letter is not.

    :returns: None.
    """
    mi = _f10_index()
    assert mi.reason([_spell("Museum of Islamic Art inventory number: ", 99973, ".", 46, " Excavation photograph")]) == "key:test"
    assert mi.reason([_spell("Index: ", 99973, ".", 47, "; ", 12345, ".", 46)]) is None
    for text in (_spell("a list of pious endowments, TS NS ", 999, " 1r.", 15, ", as noted"),
                 _spell("source: TS NS ", 999, " ", 1, ", TS ", 12, " ", 482), _spell("T-S NS ", 999, ".", 1, " recto")):
        assert mi.reason([text], include_eval=False) == "key:registry", text
    assert [k for _, k, _ in mi.find(_spell("TS NS ", 999, " 1r.", 15))] == ["tsns999_1b"]
    assert mi.find(_spell("T-S NS ", 999, ".", "1a", " is a different leaf")) == []
    assert C.mention_candidates(_spell("TS NS ", 999, " 1r.", 15))[-1] == ("tsns999_1", True)
    assert ("tsns999_1", False) in C.mention_candidates(_spell("T-S NS ", 999, ".", "1a"))


def test_f10_literature_mentions_and_f11_unmapped_rows(cworld: P.Paths, cheld: H.Holdouts, tmp_path: Path) -> None:
    """F10: a train RAG row whose literature block names an eval-tier document is excluded and the re-scan counts the
    literature mention (L1 and RAG); F11: a parsed-entry row without page keys is excluded and counted.

    :param cworld: Paths.
    :param cheld: Hold-out set (T-S 77J3.4 is a dev document).
    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    idx = C.build_reference_index(cworld, cheld)
    dev_spelling = _spell("TS ", 77, " J ", 3, ",\nfol. ", 4)
    rag = _fam_row("rag_context", "train", "9010", "Summary of a train document.", row_id="rag1")
    rag["messages"][1]["content"] = f"<document>\nx\n</document>\n[SOURCE 1] A, *B*, 2000, p. 3\nCompare {dev_spelling}."
    rag_ok = _fam_row("rag_context", "train", "9010", "Another summary.", row_id="rag2")
    rag_ok["messages"][1]["content"] = "<document>\nT-S 77J3.4 is named before the sources\n</document>\n[SOURCE 1] A"
    nas = _fam_row("translate_en", "train", "", "An English translation.", row_id="nas1")
    nas["meta"].update({"pgpid": None, "source": "book:bk", "page_keys": [], "page_keys_source": C.NAS_PAGE_SOURCE,
                        "page_keys_unmapped": "no_work", "provenance": {"source": "bk", "doc": "3", "entry_no": 3}})
    legacy = _fam_row("translate_en", "train", "", "Another translation.", row_id="nas2")
    legacy["meta"].update({"pgpid": None, "source": "book:bk", "provenance": {"source": "bk", "doc": "4", "entry_no": 4}})
    mapped = _fam_row("translate_en", "train", "", "A third translation.", row_id="nas3")
    mapped["meta"].update({"pgpid": None, "source": "book:bk", "page_keys": ["bk/p/page_009_structured.json"],
                           "page_keys_source": C.NAS_PAGE_SOURCE, "provenance": {"entry_no": 5}})
    fams = _write_fams(tmp_path / "families", {"rag_context": [rag, rag_ok], "book_pairs": [nas, legacy, mapped]})
    ex, man = C.build_excluded_rows(fams, idx, holdouts=cheld, excluded_pages={})
    assert ex == {"rag1": "literature_text_mention:key:dev", "nas1": "unmapped_entry_pages:no_work",
                  "nas2": "unmapped_entry_pages:no_page_keys"}
    assert man["excluded_by_kind"] == {"literature_text_mention": 1, "unmapped_entry_pages": 2}
    assert C.unmapped_entry_row(mapped["meta"]) is None and C.unmapped_entry_row(rag["meta"]) is None
    lit = _row("domain_reading", None, None, f"A page naming {dev_spelling} in passing.", user="Source: x",
               page_keys=["bk/p/page_009_structured.json"])
    mix = tmp_path / "m.jsonl"
    mix.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (rag, rag_ok, lit, nas, mapped)))
    rep = C.rescan(mix, cworld, cheld, use_ngram=False, excluded={})
    v = {k: c["value"] for k, c in rep["checks"].items()}
    assert v["literature_text_mentions"] == 2 and v["unmapped_entry_rows"] == 1 and v["benchmark_text_mentions"] == 0
    assert not rep["ok"] and rep["examples"]["unmapped_entry_rows"][0]["why"] == "no_work"
