"""Tests for ``novel_build.targets``: per-source target sides, letters, pointers and reasons (synthetic inputs only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.finetuning.qwen_hebrew import build_pgp_editions as bpe
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build import targets as T
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, read_jsonl, write_jsonl

# Synthetic lines: invented letter groups, no real transcription.
R1, R2, R3 = "אבגד הוזח טיכל מנסע", "פצקר שתאב גדהו זחטי", "כלמנ סעפצ קרשת אבגד"
V1, V2 = "הוזח טיכל מנסע פצקר", "שתאב גדהו זחטי כלמנ"
M1 = "סעפצ קרשת אבגד הוזח"
A1, A2, A3 = "ابتث جحخد ذرزس", "شصضط [ظعغف] قكلم", "نهوي ابتث جحخد"


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as JSON lines.

    :param path: File.
    :param rows: Rows.
    :return: The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _cand(row_id: str, source: str, **kw: Any) -> Candidate:
    """A minimal candidate.

    :param row_id: Row id.
    :param source: Source.
    :param kw: Field overrides.
    :return: The candidate.
    """
    fields: Dict[str, Any] = dict(row_id=row_id, source=source, document_group=f"g:{row_id}", canonical_ids=[],
                                  pgpids=[], ktiv_id=None, shelfmark="", script="hebrew", languages_stated=None,
                                  verdict="novel", witnesses=[], unit="document", n_letters=0)
    fields.update(kw)
    return Candidate(**fields)


# ----------------------------------------------------------------------------- letters and script


def test_count_letters_uses_the_spec_ranges() -> None:
    """Hebrew U+05D0-05EA; Arabic U+0621-064A plus U+0671-06D3; marks, digits and Latin are not letters."""
    assert T.count_letters("אב גד") == (4, 0)
    assert T.count_letters("ابت ٱ") == (0, 4)
    assert T.count_letters("abc 123 ְً") == (0, 0)


@pytest.mark.parametrize("hebrew,arabic,expected", [
    (0, 0, "none"), (30, 0, "hebrew"), (0, 30, "arabic"), (180, 20, "mixed"), (200, 20, "hebrew"),
    (19, 19, "hebrew"), (20, 25, "mixed"), (5, 25, "arabic"),
])
def test_measure_script_thresholds(hebrew: int, arabic: int, expected: str) -> None:
    """Mixed needs both scripts >= 20 letters and the minority >= 10 % of the letters; a tie is Hebrew.

    :param hebrew: Hebrew letters.
    :param arabic: Arabic letters.
    :param expected: Expected script.
    """
    assert T.measure_script("א" * hebrew + " " + "ب" * arabic) == expected


# ----------------------------------------------------------------------------- pgp


def test_pgp_labelled_edition_one_side_per_label_with_margin_appended() -> None:
    """Recto main lines first, then its margin; verso apart; markers, pointers and letters per side."""
    text = "\n".join(["Recto", R1, R2, "", "Right margin", M1, "", "Verso", V1, V2])
    built = T.pgp_hebrew_sides(text, "101")
    recto, verso = built.sides
    assert (recto.side_key, verso.side_key) == ("recto", "verso")
    assert recto.lines == [R1, R2, M1] and recto.text == "\n".join([R1, R2, M1]) == recto.raw_text
    assert recto.side_marker == "Recto" and verso.side_marker == "Verso"
    assert recto.pointer == {"source": "pgp_export_footnotes", "pgpid": "101", "parser": "parse_edition",
                             "block_indices": [0, 1], "kinds": ["main", "margin"]}
    assert verso.pointer["block_indices"] == [2] and verso.lines == [V1, V2]
    assert (recto.letters_hebrew, recto.letters_arabic) == (48, 0) and verso.letters_hebrew == 32
    assert not built.reasons and not built.flags


def test_pgp_text_before_first_label_goes_to_recto_and_unlabelled_is_one_side() -> None:
    """Blocks before the first side label belong to the recto; no label at all gives one side ``""``."""
    built = T.pgp_hebrew_sides("\n".join([R1, "", "Verso", V1]), "7")
    assert [(s.side_key, s.lines) for s in built.sides] == [("recto", [R1]), ("verso", [V1])]
    assert built.sides[0].side_marker is None
    single = T.pgp_hebrew_sides("\n".join([R1, R2]), "7")
    assert [(s.side_key, s.side_marker) for s in single.sides] == [("", None)]


def test_pgp_cleaner_runs() -> None:
    """The builder's line cleaner gives ``[...]`` for restorations."""
    built = T.pgp_hebrew_sides("\n".join([R1, "אבגד [הוזח] טיכל"]), "7")
    assert built.sides[0].lines[1] == "אבגד [...] טיכל"


def _sections(built: T.Built) -> List[Any]:
    """Key, marker, block indices and lines of each built side.

    :param built: Built sides.
    :return: One tuple per side.
    """
    return [(s.side_key, s.side_marker, s.pointer["block_indices"], s.lines) for s in built.sides]


def test_pgp_complex_edition_one_side_per_section_label() -> None:
    """Folio labels (most followed by a blank line) give one side each, keyed and marked by the label."""
    text = "\n".join(["Fol. 1r", "", R1, R2, "", "Fol. 1v", "", V1, "", "Fol. 2r", R3, "", "Fol. 2v", V2])
    built = T.pgp_hebrew_sides(text, "7")
    assert _sections(built) == [("Fol. 1r", "Fol. 1r", [0], [R1, R2]), ("Fol. 1v", "Fol. 1v", [1], [V1]),
                                ("Fol. 2r", "Fol. 2r", [2], [R3]), ("Fol. 2v", "Fol. 2v", [3], [V2])]
    assert built.flags == {key: ["complex_side_labels"] for key in ("Fol. 1r", "Fol. 1v", "Fol. 2r", "Fol. 2v")}
    assert built.sides[0].pointer == {"source": "pgp_export_footnotes", "pgpid": "7", "parser": "parse_edition",
                                      "block_indices": [0], "kinds": ["main"]}


def test_pgp_complex_same_side_name_on_two_folios_is_not_joined() -> None:
    """Two pages named recto stay apart, with page numbers or as a repeated bare label (``~2`` key)."""
    paged = T.pgp_hebrew_sides("\n".join(["Recto (page 1)", R1, "Verso (page 2)", V1, "Recto (page 3)", R2,
                                          "Verso (page 4)", V2]), "7")
    assert [(s.side_key, s.lines) for s in paged.sides] == [
        ("Recto (page 1)", [R1]), ("Verso (page 2)", [V1]), ("Recto (page 3)", [R2]), ("Verso (page 4)", [V2])]
    repeated = T.pgp_hebrew_sides("\n".join(["Recto", R1, "Verso", V1, "Recto", R2]), "7")
    assert _sections(repeated) == [("Recto", "Recto", [0], [R1]), ("Verso", "Verso", [1], [V1]),
                                   ("Recto~2", "Recto", [2], [R2])]
    assert repeated.flags["Recto~2"] == ["complex_side_labels"]


def test_pgp_complex_unlabelled_blocks_regions_and_places_on_the_page() -> None:
    """Blocks before the first section label form side ``""`` (flagged); a margin and a place on the page
    (``top of the page``) stay in the section in force."""
    text = "\n".join([R1, "", "Right margin", M1, "", "page b", "", V1, "", "top of the page", V2])
    built = T.pgp_hebrew_sides(text, "7")
    assert _sections(built) == [("", None, [0, 1], [R1, M1]), ("page b", "page b", [2, 3], [V1, V2])]
    assert built.flags == {"": ["complex_side_labels", "unlabelled_block_in_complex_edition"],
                           "page b": ["complex_side_labels"]}


def test_pgp_complex_enumerators_and_regions_that_name_another_side() -> None:
    """``I``/``II`` under a named side number its parts, without one they open sections; a region label naming
    another side opens a section, one naming no side stays."""
    numbered = T.pgp_hebrew_sides("\n".join(["(Recto)", "I", R1, "II", R2, "(Verso)", V1]), "7")
    assert _sections(numbered) == [("(Recto)", "(Recto)", [0, 1], [R1, R2]), ("(Verso)", "(Verso)", [2], [V1])]
    bare = T.pgp_hebrew_sides("\n".join(["I", R1, "II", R2]), "7")
    assert [(s.side_key, s.lines) for s in bare.sides] == [("I", [R1]), ("II", [R2])]
    regions = T.pgp_hebrew_sides("\n".join(["Fol. 1 recto", R1, "Right margin", M1, "Verso, upside down", V1]), "7")
    assert _sections(regions) == [("Fol. 1 recto", "Fol. 1 recto", [0, 1], [R1, M1]),
                                  ("Verso, upside down", "Verso, upside down", [2], [V1])]


def test_walk_edition_finds_the_parser_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk sees the parser's blocks (numbered lines, commentary, an unrendered line, a rule, the apparatus);
    a walk that disagrees leaves the candidate without sides."""
    text = "\n".join(["Recto", f"1 {R1}", f"2 {R2}", f"3 {R3}", "", "This is an editor's note about the hand.",
                      "[...]", "_____", "Verso", f"4 {V1}", "Footnotes", "אבגד הוזח"])
    lines, spans = T.edition_lines(text)
    edition = bpe.parse_edition("\n".join(lines))
    walk = T.walk_edition(lines, spans)
    assert edition.numbered and edition.n_commentary == 1 and edition.n_tail == 1
    assert T.walk_matches(walk, edition) and walk.labels == ["Recto", "Verso"]
    assert [(b.section, b.lines, b.n_dropped) for b in walk.blocks] == [(0, [R1, R2, R3], 0), (0, [], 1), (1, [V1], 0)]
    monkeypatch.setattr(T, "walk_edition", lambda lines, spans: T.EditionWalk([], []))
    assert T.pgp_hebrew_sides(text, "7").reasons == ["edition_walk_mismatch"]


def test_unreverse_supplied_spans() -> None:
    """``>x<`` becomes ``<x>``; the ink between two supplied spans and doubled brackets stay."""
    assert T.unreverse_supplied("אבג>ד<הו") == ("אבג<ד>הו", 1)
    assert T.unreverse_supplied("x >אב< y >גד<") == ("x <אב> y <גד>", 2)
    assert T.unreverse_supplied("א<ל>גאני אב<ו>") == ("א<ל>גאני אב<ו>", 0)
    assert T.unreverse_supplied(">>אבג<<") == (">>אבג<<", 0)


def test_pgp_reversed_supplied_letters_are_dropped_like_supplied_letters() -> None:
    """A reversed supplied span goes the way of ``<x>`` (letters and brackets), the side is flagged."""
    built = T.pgp_hebrew_sides("\n".join(["Recto", "אבגד הוז>ח< טיכל מנסע", "פצקר <ש>תאב גד>הו< זחטי", "Verso", V1]),
                               "7")
    assert built.sides[0].lines == ["אבגד הוז טיכל מנסע", "פצקר תאב גד זחטי"]
    assert built.flags == {"recto": ["supplied_reversed"]}


def test_pgp_arabic_reversed_supplied_span_keeps_the_line() -> None:
    """In an Arabic-script edition the reversed span is dropped and the rest of its line stays."""
    built = T.pgp_arabic_sides("\n".join(["Recto", "ابتث جح>خ<د ذرزس", A3, "Verso", A1]), "9")
    assert [(s.side_key, s.lines) for s in built.sides] == [("recto", ["ابتث جح د ذرزس", A3]), ("verso", [A1])]
    assert built.flags == {"recto": ["supplied_reversed"]}


def test_pgp_arabic_edition_uses_the_arabic_splitter() -> None:
    """Arabic-script editions are cut by ``split_sections`` with the ``[...]`` gap token."""
    built = T.pgp_arabic_sides("\n".join(["Recto", A1, A2, "Verso", A3]), "102")
    recto, verso = built.sides
    assert recto.lines == [A1, "شصضط [...] قكلم"] and verso.lines == [A3]
    assert recto.pointer == {"source": "pgp_export_footnotes", "pgpid": "102", "parser": "split_sections",
                             "block_indices": [0], "kinds": ["section"]}
    assert recto.letters_arabic == 20 and recto.side_marker == "recto"
    outside = T.pgp_arabic_sides("\n".join([A1, "Verso", A3]), "102")
    assert [s.side_key for s in outside.sides] == ["", "verso"] and outside.flags == {"": ["text_outside_side_labels"]}


@pytest.mark.parametrize("script,text,arabic", [
    ("arabic", R1, True), ("mixed", A1 + "\n" + A3 + "\nאב", True), ("mixed", R1 + "\n" + A1, False),
    ("hebrew", A1, False),
])
def test_arabic_splitter_choice(script: str, text: str, arabic: bool) -> None:
    """Ledger ``arabic``, or ``mixed`` with Arabic >= Hebrew letters, goes to the Arabic splitter.

    :param script: Ledger script.
    :param text: Edition text.
    :param arabic: Expected choice.
    """
    assert T.use_arabic_splitter(text, script) is arabic


# ----------------------------------------------------------------------------- legacy store


@pytest.fixture()
def sections_file(tmp_path: Path) -> Path:
    """A cleaned-sections file: section 0 of document A holds two records, section 1 one.

    :param tmp_path: Pytest temp dir.
    :return: The file.
    """
    return _jsonl(tmp_path / "cleaned_sections.jsonl", [
        {"canonical_id": "Testville_Lib_A_1", "doc_id": "d1", "section_index": 0, "side": "1r", "side_source": "marker",
         "side_decision": "marker", "column": None, "family": "documentary_plain", "text": R1, "flags": {"digits": 2}},
        {"canonical_id": "Testville_Lib_A_1", "doc_id": "d1", "section_index": 0, "side": "1v", "side_source": "marker",
         "side_decision": "marker", "column": None, "family": "documentary_plain", "text": V1, "flags": {}},
        {"canonical_id": "Testville_Lib_A_1", "doc_id": "d1", "section_index": 1, "side": None, "side_source": None,
         "side_decision": "undecidable", "column": None, "family": "documentary_plain", "text": M1, "flags": {}},
        {"canonical_id": "Testville_Lib_C_3", "doc_id": "d3", "section_index": 0, "side": None, "side_source": None,
         "side_decision": "single_image", "column": None, "family": "documentary_plain", "text": R3, "flags": {}},
    ])


def test_legacy_record_join_uses_the_record_index(sections_file: Path) -> None:
    """``#r<k>`` names the record with ``record_index`` k; a suffix-less id needs a one-record section."""
    sections = T.load_sections(sections_file)
    built = T.legacy_section_sides("Testville_Lib_A_1:0#r1", sections, "cleaned_sections.jsonl", V1)
    side = built.sides[0]
    assert (side.side_key, side.text, side.side_marker) == ("1v", V1, "1v")
    assert side.pointer == {"source": "legacy_store_sections", "file": "cleaned_sections.jsonl",
                            "canonical_id": "Testville_Lib_A_1", "doc_id": "d1", "section_index": 0, "record_index": 1,
                            "column": None, "family": "documentary_plain", "side_source": "marker",
                            "side_decision": "marker"}
    assert T.legacy_section_sides("Testville_Lib_A_1:0#r0", sections, "f").flags == {"1r": ["digits"]}
    unsided = T.legacy_section_sides("Testville_Lib_A_1:1", sections, "f").sides[0]
    assert (unsided.side_key, unsided.side_marker, unsided.pointer["record_index"]) == ("", None, 2)
    assert T.legacy_section_sides("Testville_Lib_A_1:0", sections, "f").reasons == ["legacy_section_ambiguous"]
    assert T.legacy_section_sides("Testville_Lib_A_1:0#r5", sections, "f").reasons == ["legacy_record_not_found"]
    assert T.legacy_section_sides("Nowhere_1:0", sections, "f").reasons == ["legacy_section_not_found"]
    assert T.legacy_section_sides("no-section-index", sections, "f").reasons == ["row_id_unparsed"]
    assert T.legacy_section_sides("Testville_Lib_C_3:0", sections, "f", "other text").flags == {
        "": ["section_text_differs_from_candidate"]}


def test_scrape_text_goes_through_the_sigla_rules() -> None:
    """Restorations become ``[...]``, digits go; the pointer names the scrape entry, block and part."""
    built = T.scrape_sides("1 אבגד הוזח [טיכל] מנסע\n2 פצקר שתאב גדהו", None, "Testville_Lib_B_2",
                           "shape=string; part=str; entry=0/1; block=0/1")
    side = built.sides[0]
    assert side.side_key == "" and side.lines == ["אבגד הוזח [...] מנסע", "פצקר שתאב גדהו"]
    assert side.raw_text.startswith("1 אבגד")
    assert side.pointer == {"source": "legacy_store_site_scrape", "canonical_id": "Testville_Lib_B_2", "entry": 0,
                            "block": 0, "part": "str", "sigla_side": 0, "sigla_sides": 1, "family": "documentary_plain"}
    assert built.flags == {"": ["digits"]}
    assert T.scrape_sides(R1 + " " + R2, "r", "X", "").sides[0].side_key == "r"
    assert T.scrape_sides("1 2 3", None, "X", "").reasons == ["no_text_after_sigla"]


def test_scrape_side_headings_split_sides() -> None:
    """Side headings found by the sigla rules give one side each, keyed by the rules' label."""
    built = T.scrape_sides(f"1r {R1} {R2} 1v {V1} {V2}", None, "X", "")
    assert [(s.side_key, s.side_marker, s.pointer["sigla_side"]) for s in built.sides] == [
        ("1r", "1r", 0), ("1v", "1v", 1)]
    assert T.unique_keys(["", "r", "", ""]) == ["", "r", "~2", "~3"]


#: A scrape text with a final letter stored before its word (``ן שאי`` for ``שאין``).
SCRAPE_REVERSED = "אבגד הוזח ן שאי טיכל מנסע"


def test_load_lexicon_and_scrape_word_order_repair(tmp_path: Path) -> None:
    """The lexicon holds the Hebrew words (2+ letters) of the records' text; with it the rules re-order a final letter
    stored before its word, without it they leave the text alone."""
    path = _jsonl(tmp_path / "words.jsonl", [{"text": "שאין א [...] בג"}, {"text": None}, {"side": "1r"}])
    lexicon = T.load_lexicon(path)
    assert lexicon == frozenset({"שאין", "בג"})
    assert T.scrape_sides(SCRAPE_REVERSED, None, "X", "", lexicon).sides[0].text == "אבגד הוזח שאין טיכל מנסע"
    assert T.scrape_sides(SCRAPE_REVERSED, None, "X", "").sides[0].text == SCRAPE_REVERSED


@pytest.fixture()
def scrape_inputs(tmp_path: Path) -> Dict[str, Path]:
    """A legacy-store scrape candidate, a sections file that knows the repaired word and an empty word file.

    :param tmp_path: Pytest temp dir.
    :return: Paths by name.
    """
    cdir = tmp_path / "candidates"
    _jsonl(cdir / "fjp_scrape.jsonl", [{"row_id": "Testville_Lib_B_2:s0", "edition_text": SCRAPE_REVERSED,
                                        "side": None, "notes": "entry=0/1; block=0/1"}])
    return {"candidates": cdir, "sections": _jsonl(tmp_path / "sections.jsonl", [{"text": f"{R1} שאין"}]),
            "empty": _jsonl(tmp_path / "empty.jsonl", [{"text": R1}]), "blocks": _jsonl(tmp_path / "b.jsonl", []),
            "run2": _jsonl(tmp_path / "run2.jsonl", []), "root": tmp_path / "datasets"}


def test_attach_targets_scrape_lexicon_comes_from_the_sections_file(scrape_inputs: Dict[str, Path]) -> None:
    """By default the scrape's lexicon is built from ``sections_path``; another file or None can replace it."""
    def scrape_text(**kw: Any) -> str:
        """Attach the scrape candidate's target and return its text.

        :param kw: Extra :func:`targets.attach_targets` arguments.
        :return: The target text.
        """
        cand = _cand("Testville_Lib_B_2:s0", "fjp_scrape")
        T.attach_targets([cand], candidates_dir=scrape_inputs["candidates"], sections_path=scrape_inputs["sections"],
                         lit_blocks_path=scrape_inputs["blocks"], run2_path=scrape_inputs["run2"],
                         datasets_root=scrape_inputs["root"], **kw)
        return cand.targets[0].text

    assert scrape_text() == "אבגד הוזח שאין טיכל מנסע"
    assert scrape_text(lexicon_path=None) == SCRAPE_REVERSED
    assert scrape_text(lexicon_path=scrape_inputs["empty"]) == SCRAPE_REVERSED


def test_targets_main_lexicon_options(tmp_path: Path, scrape_inputs: Dict[str, Path]) -> None:
    """``main`` uses the sections file as lexicon unless ``--lexicon`` names another or ``--no-lexicon`` is given."""
    work = tmp_path / "work"
    stage = paths.stage_path(work, "candidates")

    def run(*extra: str) -> str:
        """Re-attach the scrape candidate through ``main``.

        :param extra: Extra CLI arguments.
        :return: The stored target text.
        """
        write_jsonl(stage, [_cand("Testville_Lib_B_2:s0", "fjp_scrape")])
        T.main(["--work-dir", str(work), "--candidates-dir", str(scrape_inputs["candidates"]),
                "--sections", str(scrape_inputs["sections"]), "--lit-blocks", str(scrape_inputs["blocks"]),
                "--run2", str(scrape_inputs["run2"]), "--datasets-root", str(scrape_inputs["root"]), *extra])
        return next(read_jsonl(stage, Candidate)).targets[0].text

    assert run() == "אבגד הוזח שאין טיכל מנסע"
    assert run("--no-lexicon") == SCRAPE_REVERSED
    assert run("--lexicon", str(scrape_inputs["empty"])) == SCRAPE_REVERSED


# ----------------------------------------------------------------------------- ktiv


def test_ktiv_page_side_and_pointer() -> None:
    """One side ``page``; pointer: shape, system number, ``fl`` from the row id, bundle from the notes."""
    built = T.ktiv_sides(f"{R1}\n{R2}\n", "990000000000205171", "990000000000205171:FL100",
                         "shape=api; bundle=ktiv_test_bundle.json; n_lines=2", "nli_ktiv_transcription_api")
    side = built.sides[0]
    assert (side.side_key, side.lines, side.text) == ("page", [R1, R2], f"{R1}\n{R2}")
    assert side.pointer == {"source": "ktiv_api", "sys_num": "990000000000205171", "fl": "FL100",
                            "bundle": "ktiv_test_bundle.json"}
    dom = T.ktiv_sides(R1, "9", "9:img_0001", "shape=dom; bundle=b.json", "nli_ktiv_transcription_dom")
    assert dom.sides[0].pointer["source"] == "ktiv_dom" and dom.sides[0].pointer["fl"] == "img_0001"


# ----------------------------------------------------------------------------- literature


@pytest.fixture()
def lit_blocks(tmp_path: Path) -> Dict[Any, Any]:
    """Literature blocks: the candidate's key on pages 13 and 14, another key on 13, the same key on page 15.

    :param tmp_path: Pytest temp dir.
    :return: :func:`targets.load_lit_blocks` result.
    """
    def block(page: int, key: str, text: str, label: str = "") -> Dict[str, Any]:
        """One block record.

        :param page: Page number.
        :param key: Shelf key.
        :param text: Block text.
        :param label: Block label.
        :return: The record.
        """
        return {"work": "testwork", "part": "partA", "page_file": f"testwork/partA/s/page_{page:03d}_structured.json",
                "page_number": page, "shelf_key": key, "label": label, "lines": text.count("\n") + 1, "text": text}

    return T.load_lit_blocks(_jsonl(tmp_path / "lit_blocks.jsonl", [
        block(14, "TS X 1", R3), block(13, "TS X 2", V1), block(13, "TS X 1", f"{R1}\n{R2}"), block(15, "TS X 1", V2),
    ]))


def test_literature_blocks_in_page_order(lit_blocks: Dict[Any, Any]) -> None:
    """Blocks of the candidate's key and pages, page order, checked against its text; one side, its side or ``""``."""
    text = f"{R1}\n{R2}\n{R3}"
    built = T.literature_sides("literature:testwork/partA pages 13,14", ["TS X 1"], None, text, lit_blocks)
    side = built.sides[0]
    assert (side.side_key, side.text, side.lines) == ("", text, [R1, R2, R3])
    assert side.pointer == {"source": "scanned_edition", "work": "testwork", "part": "partA", "shelf_key": "TS X 1",
                            "page_files": ["testwork/partA/s/page_013_structured.json",
                                           "testwork/partA/s/page_014_structured.json"]}
    sided = T.literature_sides("literature:testwork/partA pages 14", ["TS X 1"], "verso", R3, lit_blocks).sides[0]
    assert (sided.side_key, sided.side_marker) == ("verso", "verso")


def test_literature_skips_blocks_of_another_candidate_and_reports_misses(lit_blocks: Dict[Any, Any]) -> None:
    """A same-key block the text does not continue with is skipped; no block or no rebuild gives a reason."""
    built = T.literature_sides("literature:testwork/partA pages 13,14", ["TS X 1"], None, R3, lit_blocks)
    assert built.sides[0].pointer["page_files"] == ["testwork/partA/s/page_014_structured.json"]
    assert T.literature_sides("literature:testwork/partA pages 13", ["TS X 9"], None, R1, lit_blocks).reasons == [
        "lit_blocks_not_found"]
    assert T.literature_sides("literature:testwork/partA pages 13", ["TS X 1"], None, "other", lit_blocks).reasons == [
        "lit_blocks_mismatch"]
    assert T.literature_sides("not a literature source", [], None, R1, lit_blocks).reasons == [
        "edition_source_unparsed"]


# ----------------------------------------------------------------------------- run 2


def test_run2_lines_grouped_by_side() -> None:
    """Sides in order of appearance, ``unknown`` as ``""``; pointer with the file and the line numbers."""
    record = {"doc_id": "lit:testbook:1", "transcription_lines": [
        {"n": 1, "side": "recto", "part": None, "text": R1}, {"n": 2, "side": "recto", "part": None, "text": R2},
        {"n": 1, "side": "verso", "part": None, "text": V1}, {"n": 1, "side": "unknown", "part": None, "text": M1}]}
    built = T.run2_sides(record, "run2.jsonl")
    assert [(s.side_key, s.lines, s.side_marker) for s in built.sides] == [
        ("recto", [R1, R2], "recto"), ("verso", [V1], "verso"), ("", [M1], None)]
    assert built.sides[0].pointer == {"source": "run2_edition", "file": "run2.jsonl", "doc_id": "lit:testbook:1",
                                      "line_numbers": [1, 2]}
    assert T.run2_sides({"doc_id": "x", "transcription_lines": []}, "f").reasons == ["run2_no_lines"]


def test_load_run2_reads_several_files(tmp_path: Path) -> None:
    """Several run-2 files: each id keeps its file name."""
    first = _jsonl(tmp_path / "a.jsonl", [{"doc_id": "d1", "transcription_lines": []}])
    second = _jsonl(tmp_path / "b.jsonl", [{"doc_id": "d2", "transcription_lines": []}])
    loaded = T.load_run2([first, second])
    assert loaded["d1"][1] == "a.jsonl" and loaded["d2"][1] == "b.jsonl"
    assert set(T.load_run2(first)) == {"d1"}


# ----------------------------------------------------------------------------- pass-through and attach_targets


@pytest.fixture()
def target_inputs(tmp_path: Path, sections_file: Path) -> Dict[str, Path]:
    """Candidate files, literature blocks, a run-2 file and a datasets root with an APD and an Aodeh manifest.

    :param tmp_path: Pytest temp dir.
    :param sections_file: The cleaned-sections fixture.
    :return: Paths by name.
    """
    cdir = tmp_path / "candidates"
    _jsonl(cdir / "pgp.jsonl", [{"row_id": "pgp:101", "edition_text": "\n".join(["Recto", R1, "Verso", V1]),
                                 "text_shingles": ["x"]}])
    _jsonl(cdir / "fjp.jsonl", [{"row_id": "Testville_Lib_A_1:0#r1", "edition_text": V1}])
    _jsonl(cdir / "apd_v3.jsonl", [{"row_id": "apd_v3:apd_test_0_page", "edition_text": f"{A1}\n{A3}"}])
    _jsonl(cdir / "aodeh_v2_1.jsonl", [{"row_id": "aodeh_v2_1:aodeh_test_1_page", "edition_text": A3}])
    blocks = _jsonl(tmp_path / "lit_blocks.jsonl", [])
    run2 = _jsonl(tmp_path / "run2.jsonl", [{"doc_id": "lit:testbook:1", "transcription_lines": [
        {"n": 1, "side": "unknown", "part": None, "text": R1}]}])
    root = tmp_path / "datasets"
    apd_dir = root / "arabic_external" / "apd" / "apd_arabic_pages_v3"
    fallback = apd_dir.with_name("apd_arabic_pages_v3_images") / "apd_test_0_page.jpg"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"jpeg")
    _jsonl(apd_dir / "manifest.jsonl", [{"stem": "apd_test_0_page", "side": "verso", "canvas": 1,
                                         "image_path": str(tmp_path / "moved" / "gone.jpg"),
                                         "image_url": "https://example.invalid/apd/1.jpg"}])
    aodeh_image = tmp_path / "aodeh.jpg"
    aodeh_image.write_bytes(b"jpeg")
    _jsonl(root / "book_edition_aodeh_1992_v2_1" / "manifest.jsonl", [
        {"stem": "aodeh_test_1_page", "side": "recto", "canvas_index": 0, "canvas_label": "1r",
         "image_path": str(aodeh_image), "image_sha256": "ab" * 32, "image_url": "https://example.invalid/a.jpg"}])
    return {"candidates": cdir, "sections": sections_file, "blocks": blocks, "run2": run2, "root": root}


def _attach(cands: List[Candidate], inputs: Dict[str, Path]) -> None:
    """Run :func:`targets.attach_targets` on the fixture inputs.

    :param cands: Candidates.
    :param inputs: Fixture paths.
    """
    T.attach_targets(cands, candidates_dir=inputs["candidates"], sections_path=inputs["sections"],
                     lit_blocks_path=inputs["blocks"], run2_path=inputs["run2"], datasets_root=inputs["root"])


def test_attach_targets_per_source_and_reasons(target_inputs: Dict[str, Path]) -> None:
    """Every source dispatches to its builder; missing inputs leave reasons; pass-through images replace the list."""
    cands = [_cand("pgp:101", "pgp", pgpids=["101"]), _cand("pgp:999", "pgp"),
             _cand("Testville_Lib_A_1:0#r1", "fjp"), _cand("lit:testbook:1", "lit_run2"),
             _cand("lit:testbook:9", "lit_run2"),
             _cand("apd_v3:apd_test_0_page", "apd_v3", script="arabic", notes={"stem": "apd_test_0_page"}),
             _cand("aodeh_v2_1:aodeh_test_1_page", "aodeh_v2_1", script="arabic"),
             _cand("apd_v3:apd_unknown_page", "apd_v3")]
    _attach(cands, target_inputs)
    by_id = {c.row_id: c for c in cands}
    assert [t.side_key for t in by_id["pgp:101"].targets] == ["recto", "verso"]
    assert by_id["pgp:999"].targets == [] and by_id["pgp:999"].notes["target_reasons"] == ["no_candidate_record"]
    assert by_id["Testville_Lib_A_1:0#r1"].targets[0].side_key == "1v"
    assert "target_flags" not in by_id["Testville_Lib_A_1:0#r1"].notes
    assert [t.side_key for t in by_id["lit:testbook:1"].targets] == [""]
    assert by_id["lit:testbook:9"].notes["target_reasons"] == ["run2_record_not_found"]
    apd = by_id["apd_v3:apd_test_0_page"]
    assert apd.targets[0].side_key == "verso" and apd.targets[0].pointer == {"source": "apd_arabic_pages_v3",
                                                                            "stem": "apd_test_0_page"}
    assert apd.targets[0].letters_arabic == 24
    image = apd.images[0]
    assert (image.origin, image.status, image.canvas_index, image.sha256) == ("apd_local", "present", 1, None)
    assert image.path.endswith("apd_arabic_pages_v3_images/apd_test_0_page.jpg")
    aodeh = by_id["aodeh_v2_1:aodeh_test_1_page"].images[0]
    assert (aodeh.origin, aodeh.status, aodeh.sha256, aodeh.label) == ("aodeh_local", "present", "ab" * 32, "1r")
    assert by_id["apd_v3:apd_unknown_page"].notes["target_reasons"] == ["manifest_row_not_found"]


def test_attach_targets_replaces_earlier_targets(target_inputs: Dict[str, Path],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A second run replaces targets and target notes; a stubbed ``path_exists`` marks a pass-through image missing."""
    cand = _cand("pgp:101", "pgp", pgpids=["101"])
    _attach([cand], target_inputs)
    _attach([cand], target_inputs)
    assert [t.side_key for t in cand.targets] == ["recto", "verso"]
    monkeypatch.setattr(T, "path_exists", lambda path: False)
    apd = _cand("apd_v3:apd_test_0_page", "apd_v3", notes={"stem": "apd_test_0_page"})
    _attach([apd], target_inputs)
    assert apd.images[0].status == "missing" and apd.images[0].path.endswith("gone.jpg")


def test_targets_main_reattaches_in_place(tmp_path: Path, target_inputs: Dict[str, Path]) -> None:
    """``main`` re-attaches the targets of a stage-0 file; ``--missing-only`` leaves done candidates alone."""
    work = tmp_path / "work"
    done = _cand("Testville_Lib_A_1:0#r1", "fjp")
    done.targets = [T.make_side("kept", R1, R1, {"source": "earlier"})]
    write_jsonl(paths.stage_path(work, "candidates"), [_cand("pgp:101", "pgp", pgpids=["101"]), done])
    T.main(["--work-dir", str(work), "--candidates-dir", str(target_inputs["candidates"]),
            "--sections", str(target_inputs["sections"]), "--lit-blocks", str(target_inputs["blocks"]),
            "--run2", str(target_inputs["run2"]), "--datasets-root", str(target_inputs["root"]), "--missing-only"])
    stored = list(read_jsonl(paths.stage_path(work, "candidates"), Candidate))
    assert [t.side_key for t in stored[0].targets] == ["recto", "verso"]
    assert [t.side_key for t in stored[1].targets] == ["kept"]
