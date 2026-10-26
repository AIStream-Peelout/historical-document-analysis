# File name: test_rover_reads.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""ROVER over pipeline page reads: grouping, pass-through, vote, agreement re-derivation, provenance, schema, CLI.

Toy records and raw-cache entries written to ``tmp_path``; no data files, no model, no network. The pipeline's
helpers are compared with the originals compiled from ``two_reader_lines.py`` without importing it (its import loads
``.env``).
"""
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.consensus import describe_from_reads as dfr
from src.datasets.consensus import line_rule
from src.datasets.consensus import rover_reads as rr

KEY = "MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"
PIPELINE = Path(rr.__file__).with_name("two_reader_lines.py")
PUA = chr(0xE000)          # the first private-use code point (the codebook's first stand-in)
P1 = "https://img/p1.jpg"


# --------------------------------------------------------------------------------------------- fixtures


def band(i: int) -> Tuple[float, float, float, float]:
    """VLM box of line ``i`` (one 50-unit band per line).

    :param i: Line position.
    :return: Box.
    """
    return (100.0, 100.0 + 50 * i, 900.0, 140.0 + 50 * i)


def frag_box(i: int) -> Tuple[float, float, float, float]:
    """Kraken box inside band ``i``.

    :param i: Line position.
    :return: Box.
    """
    return (150.0, 105.0 + 50 * i, 850.0, 135.0 + 50 * i)


def ai_line(i: int, text: str, status: str = "unconfirmed", agreement: Optional[float] = None,
            htr_text: Optional[str] = None) -> Dict[str, Any]:
    """One ``ai_read.lines`` entry in the pipeline's key order.

    :param i: Line index.
    :param text: VLM text.
    :param status: Status.
    :param agreement: Similarity.
    :param htr_text: Kraken text.
    :return: Line dict.
    """
    fragments = [rr.clamp_box(frag_box(i))] if htr_text else []
    return {"index": i, "text": text, "bbox": rr.clamp_box(band(i)), "agreement": agreement, "status": status,
            "htr_text": htr_text, "htr_fragments": fragments}


def record(reader: str, texts: Sequence[str], url: str = P1, sha: Optional[str] = "sha-p1",
           doc_id: str = "DOC_1", image_index: int = 0, decoded_at: str = "2026-10-08T00:00:00+00:00",
           parsed: bool = True, htr_texts: Optional[Sequence[Optional[str]]] = None) -> Dict[str, Any]:
    """A ``two_reader_lines`` record in the pipeline's key order.

    :param reader: VLM checkpoint.
    :param texts: Line texts.
    :param url: Image URL.
    :param sha: Image hash (None leaves it empty).
    :param doc_id: Recorded doc id.
    :param image_index: Recorded image position.
    :param decoded_at: Decode time.
    :param parsed: Reply parsed.
    :param htr_texts: Kraken text per line (None: no Kraken text).
    :return: Record.
    """
    htr_texts = list(htr_texts) if htr_texts is not None else [None] * len(texts)
    lines = [ai_line(i, t, htr_text=h) for i, (t, h) in enumerate(zip(texts, htr_texts))]
    ai = {"vlm_model": reader, "vlm_revision": f"rev-{reader}", "htr_model": "MiDRASH_Gen_01",
          "rule_version": line_rule.RULE_VERSION, "decoded_at": decoded_at, "parsed": parsed,
          "n_lines": len(lines), "n_agreed": 0, "lines": lines, "htr_cache_key": KEY}
    return {"doc_id": doc_id, "source_index": "genizah_merged_v9", "image_index": image_index, "image_url": url,
            "image_width": 1000, "image_height": 1400, "image_sha256": sha, "ai_read": ai}


def page_read(reader: str, texts: Sequence[str], kraken: Sequence[str] = (), key: str = "sha-p1",
              drift: float = 0.0, **kwargs: Any) -> rr.PageRead:
    """A :class:`PageRead` built in memory: line ``i`` in band ``i``, Kraken fragment ``i`` inside band ``i``.

    :param reader: VLM checkpoint.
    :param texts: VLM line texts.
    :param kraken: Kraken fragment texts (Kraken's order).
    :param key: Page key.
    :param drift: Vertical shift of the VLM boxes (a drifting reader).
    :param kwargs: Passed to :func:`record`.
    :return: The read.
    """
    rec = record(reader, texts, **kwargs)
    boxes = [(b[0], b[1] + drift, b[2], b[3] + drift) for b in map(band, range(len(texts)))]
    return rr.PageRead(page_key=key, record=rec, record_id=f"{reader}.jsonl:1",
                       vlm_lines=[rr.SourceLine(t, b) for t, b in zip(texts, boxes)], box_source="raw_cache",
                       kraken_lines=[rr.SourceLine(t, frag_box(i)) for i, t in enumerate(kraken)],
                       kraken_source="raw_cache" if kraken else "none")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]], partial: Optional[str] = None) -> Path:
    """Write rows as JSONL, optionally followed by an unterminated line (a record still being appended).

    :param path: Destination.
    :param rows: Rows.
    :param partial: Text written after the last newline.
    :return: The path.
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + (partial or ""),
                    encoding="utf-8")
    return path


def write_raw(raw_dir: Path, rec: Dict[str, Any], vlm_lines: Sequence[Dict[str, Any]],
              frags: Sequence[Dict[str, Any]], **overrides: Any) -> Path:
    """A raw-cache entry as ``two_reader_lines.stage2`` writes it under ``--htr-cache-key``.

    :param raw_dir: Cache directory.
    :param rec: The record it belongs to.
    :param vlm_lines: ``{text, box}`` before the loop-repeat dedupe.
    :param frags: Kraken fragments.
    :param overrides: Fields to replace (e.g. a different ``image_url``).
    :return: The file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    entry = dict(doc_id=rec["doc_id"], image_index=rec["image_index"], image_url=rec["image_url"], width=1000,
                 height=1400, sha256=rec["image_sha256"], vlm_model=rec["ai_read"]["vlm_model"], vlm_raw="[]",
                 vlm_lines=list(vlm_lines), parsed=True, frags=list(frags), htr_model=KEY)
    entry.update(overrides)
    path = raw_dir / f"{rr.raw_stem(rec, rec['ai_read']['vlm_model'])}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return path


def pipeline_namespace(*names: str) -> Dict[str, Any]:
    """Compile named top-level functions and constants of ``two_reader_lines.py`` without importing the module.

    :param names: Function / constant names.
    :return: Namespace holding them.
    """
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    wanted = set(names)
    body = [node for node in tree.body
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted)
            or (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))]
    namespace: Dict[str, Any] = {"re": re, "datetime": datetime, "timezone": timezone, "Any": Any, "Dict": Dict,
                                 "List": List, "Optional": Optional, "Tuple": Tuple}
    namespace.update({k: getattr(line_rule, k) for k in ("RULE_VERSION", "assign_fragments", "evidence_box",
                                                         "letters", "line_status", "similarity")})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PIPELINE), "exec"), namespace)  # noqa: S102
    assert wanted <= set(namespace), f"not found in {PIPELINE.name}: {wanted - set(namespace)}"
    return namespace


# --------------------------------------------------------------------------------------------- grouping


def test_page_key_strategies() -> None:
    """A page is its hash, else the hash its URL has elsewhere, else its URL, else ``doc_id#image_index``."""
    with_sha = record("a", ["x"], url="u1", sha="s1")
    no_sha_known_url = record("b", ["x"], url="u1", sha=None)
    no_sha_new_url = record("b", ["x"], url="u2", sha=None)
    nothing = record("b", ["x"], url="", sha=None, doc_id="D", image_index=3)
    url_sha = rr.url_hashes([with_sha, no_sha_known_url, no_sha_new_url, nothing])
    assert url_sha == {"u1": "s1"}
    assert [rr.page_key(r, url_sha) for r in (with_sha, no_sha_known_url, no_sha_new_url, nothing)] == \
        ["s1", "s1", "u2", "D#3"]
    two_hashes = rr.url_hashes([record("a", ["x"], url="u3", sha="s3"), record("b", ["x"], url="u3", sha="s4")])
    assert "u3" not in two_hashes                  # the bytes behind the URL changed: no hash is borrowed


def test_collect_groups_reads_across_key_strategies(tmp_path: Path) -> None:
    """Reads of one page meet across files whichever key identifies it; a half-written line is skipped.

    :param tmp_path: Pytest temporary directory.
    """
    file_a = write_jsonl(tmp_path / "a.jsonl", [
        record("vlm-a", ["אבג"], url="u1", sha="s1"),
        record("vlm-a", ["דהו"], url="u2", sha=None),                         # joins s2 through its URL
        record("vlm-a", ["זחט"], url="", sha=None, doc_id="D", image_index=5),
    ], partial='{"doc_id": "half-written')
    file_b = write_jsonl(tmp_path / "b.jsonl", [
        record("vlm-b", ["זחט"], url="", sha=None, doc_id="D", image_index=5),
        record("vlm-b", ["אבג"], url="u1-renamed", sha="s1"),                 # same bytes under another name
        record("vlm-b", ["דהו"], url="u2", sha="s2"),
    ])
    stats = rr.Counter()
    pages = rr.collect_page_reads([file_a, file_b], raw_dir=None, stats=stats)
    assert list(pages) == ["s1", "s2", "D#5"]                       # first appearance
    assert all([r.reader for r in reads] == ["vlm-a", "vlm-b"] for reads in pages.values())
    assert pages["s1"][1].record_id == "b.jsonl:2" and stats["records"] == 6
    assert pages["s2"][0].vlm_lines[0].text == "דהו"


def test_collect_keeps_job_pages_and_one_read_per_reader(tmp_path: Path) -> None:
    """Only job pages are kept, in job order, and one read per reader: parsed first, then the newest.

    :param tmp_path: Pytest temporary directory.
    """
    reads = write_jsonl(tmp_path / "reads.jsonl", [
        record("vlm-a", ["old"], url="u1", sha="s1", decoded_at="2026-10-01T00:00:00+00:00"),
        record("vlm-a", ["outside"], url="u9", sha="s9"),
        record("vlm-a", ["new"], url="u1", sha="s1", decoded_at="2026-10-02T00:00:00+00:00"),
        record("vlm-a", ["unparsed"], url="u1", sha="s1", decoded_at="2026-10-03T00:00:00+00:00", parsed=False),
        record("vlm-a", ["second"], url="u2", sha="s2"),
    ])
    jobs = [{"doc_id": "DOC_2", "image_index": 1, "image_url": "u2"},
            {"doc_id": "DOC_1", "image_index": 0, "image_url": "u1"}]
    stats = rr.Counter()
    pages = rr.collect_page_reads([reads], jobs, raw_dir=None, stats=stats)
    assert list(pages) == ["s2", "s1"]                              # job order
    assert pages["s1"][0].vlm_lines[0].text == "new"                # parsed beats newer-but-unparsed
    assert pages["s1"][0].job == jobs[1]
    assert stats["outside_jobs"] == 1 and stats["duplicate_reads"] == 2


def test_raw_cache_supplies_vlm_boxes_and_all_kraken_fragments(tmp_path: Path) -> None:
    """The raw cache gives the VLM's own boxes (after the loop dedupe) and every fragment; another read's file
    is ignored and the record's evidence boxes and ``htr_text`` stand in.

    :param tmp_path: Pytest temporary directory.
    """
    raw_dir = tmp_path / "raw"
    loop = "שורה ארוכה שחוזרת"
    rec = record("vlm-a", ["אבגד", loop, "הוזח"], htr_texts=["אבגד", None, None])
    raw_boxes = [[110.0, 101.0, 890.0, 139.0], [111.0, 151.0, 889.0, 189.0], [111.0, 151.0, 889.0, 189.0],
                 [112.0, 201.0, 888.0, 239.0]]
    vlm_lines = [{"text": t, "box": b} for t, b in zip(["אבגד", loop, loop, "הוזח"], raw_boxes)]
    frags = [{"text": "אבגד", "conf": 0.9, "box": list(frag_box(0))},
             {"text": "unassigned", "conf": 0.5, "box": [0.0, 900.0, 50.0, 950.0]}]
    write_raw(raw_dir, rec, vlm_lines, frags)
    path = write_jsonl(tmp_path / "reads.jsonl", [rec])
    read = rr.collect_page_reads([path], raw_dir=raw_dir)["sha-p1"][0]
    assert read.box_source == "raw_cache" and read.kraken_source == "raw_cache"
    assert [ln.box for ln in read.vlm_lines] == [tuple(raw_boxes[0]), tuple(raw_boxes[1]), tuple(raw_boxes[3])]
    assert [ln.text for ln in read.kraken_lines] == ["אבגד", "unassigned"]   # every fragment, Kraken's order
    assert read.htr_key == KEY and read.agreed == [False, False, False]

    write_raw(raw_dir, rec, vlm_lines, frags, image_url="https://img/other.jpg")   # another read's cache file
    read = rr.collect_page_reads([path], raw_dir=raw_dir)["sha-p1"][0]
    assert read.box_source == "record_bbox" and read.kraken_source == "record_htr_text"
    assert [ln.text for ln in read.kraken_lines] == ["אבגד"]
    assert read.vlm_lines[0].box == tuple(float(v) for v in rr.clamp_box(band(0)))


# --------------------------------------------------------------------------------------------- pass-through


def test_single_vlm_read_with_kraken_is_passed_through_unchanged() -> None:
    """One VLM read + Kraken is the two-voter degenerate case: the read's lines and statuses are kept as recorded."""
    read = page_read("vlm-a", ["אבגד הוז", "חטיכ למנ"], kraken=["אבגד הוז", "חטיכ למס"])
    read.record["ai_read"]["lines"][0].update(status="agreed", agreement=1.0, htr_text="אבגד הוז")
    page = rr.vote_page([read])
    assert not page.voted and page.reason == "single_vlm_read" and page.voters == ["vlm-a", rr.KRAKEN]
    out = rr.build_record(page)
    assert out["ai_read"]["lines"] == read.record["ai_read"]["lines"]
    assert out["ai_read"]["vlm_model"] == "rover2"
    assert out["ai_read"]["vlm_revision"] == f"vlm-a=rev-vlm-a; kraken={KEY}"
    for key in ("rule_version", "decoded_at", "n_agreed", "htr_cache_key", "parsed"):
        assert out["ai_read"][key] == read.record["ai_read"][key]
    assert out["rover"]["voted"] is False and out["rover"]["sources"][0]["record"] == "vlm-a.jsonl:1"
    assert out["rover"]["lines"][1]["full"] == ["vlm-a"] and out["rover"]["n_hybrid_lines"] == 0


def test_empty_read_does_not_vote_and_the_other_passes_through() -> None:
    """A read without lines is left out (it would vote NULL everywhere); the remaining read passes through."""
    empty = page_read("vlm-b", [], kraken=["אבגד"])
    full = page_read("vlm-a", ["אבגד"], kraken=["אבגד"])
    page = rr.vote_page([empty, full])
    assert not page.voted and page.voting == [full] and page.model == "rover2"
    assert page.skipped == [{"reader": "vlm-b", "record": "vlm-b.jsonl:1", "reason": "no_lines"}]


# --------------------------------------------------------------------------------------------- vote


def test_three_voters_keep_the_majority_character_and_the_line_structure() -> None:
    """Two VLMs + Kraken: the majority character wins, newlines keep the lines, agreement is re-derived."""
    a = page_read("vlm-a", ["אבגד הוז", "חטיכ למנ"], kraken=["אבגד הוז", "חטיכ למס"])
    b = page_read("vlm-b", ["אבגה הוז", "חטיכ למנ"], kraken=["אבגד הוז", "חטיכ למס"])
    page = rr.vote_page([a, b])
    assert page.voted and page.model == "rover3" and page.voters == ["vlm-a", "vlm-b", rr.KRAKEN]
    assert [ln["text"] for ln in page.lines] == ["אבגד הוז", "חטיכ למנ"]
    assert [ln["status"] for ln in page.lines] == ["agreed", "agreed"]      # records said unconfirmed: re-derived
    assert [ln["agreement"] for ln in page.lines] == [1.0, round(1 - 1 / 7, 4)]
    assert page.provenance[0]["attested_by"] == {"vlm-a": 7, "vlm-b": 6, rr.KRAKEN: 7}
    assert page.provenance[0]["full"] == ["vlm-a", rr.KRAKEN] and page.provenance[1]["full"] == ["vlm-a", "vlm-b"]
    assert page.provenance[0]["box_from"] == "vlm-a#0" and page.geometry_reader == "vlm-a"
    assert rr.agreed_on_voted_pages([page]) == {"vlm-a": (2, 2), "vlm-b": (2, 2), "output": (2, 2)}
    record_ = rr.build_record(page, decoded_at="2026-10-08T12:00:00+00:00")
    assert record_["ai_read"]["vlm_revision"] == f"vlm-a=rev-vlm-a; vlm-b=rev-vlm-b; kraken={KEY}"
    assert record_["ai_read"]["n_agreed"] == 2 and record_["ai_read"]["htr_cache_key"] == KEY
    assert record_["rover"]["sources"][1]["reader"] == "vlm-b"


def test_geometry_comes_from_the_reader_whose_boxes_pair_with_kraken() -> None:
    """A reader whose boxes miss every fragment does not supply the voted lines' boxes."""
    texts, kraken = ["אבגד הוז", "חטיכ למנ"], ["אבגד הוז", "חטיכ למנ"]
    drifting = page_read("vlm-a", texts, kraken=kraken, drift=25.0)   # its bands miss every fragment
    steady = page_read("vlm-b", texts, kraken=kraken)
    page = rr.vote_page([drifting, steady])
    assert page.geometry_scores == {"vlm-a": 0, "vlm-b": 2} and page.geometry_reader == "vlm-b"
    assert [p["box_from"] for p in page.provenance] == ["vlm-b#0", "vlm-b#1"]
    assert [ln["status"] for ln in page.lines] == ["agreed", "agreed"]
    assert rr.build_record(page)["rover"]["geometry_scores"] == {"vlm-a": 0, "vlm-b": 2}


@pytest.mark.parametrize("unit", ["char", "word"])
def test_voted_text_is_rover_consensus_text(unit: str) -> None:
    """The slot-level vote here writes exactly ``rover_consensus.rover``'s text, backbone and network size.

    :param unit: ``char`` or ``word``.
    """
    texts = {"vlm-a": ["שלום עליכם אחי", "היקר מה שלומך", "ושלום ביתך"],
             "vlm-b": ["שלום עליכם", "אחי היקר מה שלומך", "ושלום ביתך"],
             "vlm-c": ["שלם עליכם אחי", "היקר מה שלומכ"]}
    kraken = ["שלום עלכם אחי", "היקר מה שלמך", "ושלום ביתך כלו"]
    reads = [page_read(name, lines, kraken=kraken) for name, lines in texts.items()]
    page = rr.vote_page(reads, unit=unit)
    expected = rr.rc.rover(["\n".join(lines) for lines in texts.values()] + ["\n".join(kraken)], unit=unit)
    assert "\n".join(ln["text"] for ln in page.lines) == expected.text
    assert page.backbone == page.voters[expected.backbone] and page.n_slots == expected.n_slots


def test_two_vlm_reads_without_kraken_return_the_backbone() -> None:
    """Two voters tie on every disagreement, so the vote copies the backbone; Kraken still decides agreement."""
    a = page_read("vlm-a", ["אבגד"], kraken=["אבגה"])
    b = page_read("vlm-b", ["אבגו"], kraken=["אבגה"])
    page = rr.vote_page([a, b], kraken_voter=False)
    assert page.voted and page.voters == ["vlm-a", "vlm-b"] and page.model == "rover2"
    assert [ln["text"] for ln in page.lines] == ["אבגד"] and page.backbone == "vlm-a"
    assert page.lines[0]["htr_text"] == "אבגה"


def test_gap_marks_vote_as_one_token_and_are_written_back() -> None:
    """``[...]`` votes as one token and comes back as written; a mark holding a letter stays plain text."""
    kraken = ["אבג זח דהו"]
    both = rr.vote_page([page_read("vlm-a", ["אבג [...] דהו"], kraken=kraken),
                         page_read("vlm-b", ["אבג [...] דהו"], kraken=kraken)])
    assert [ln["text"] for ln in both.lines] == ["אבג [...] דהו"]
    one = rr.vote_page([page_read("vlm-a", ["אבג [?] דהו"], kraken=["אבג דהו"]),
                        page_read("vlm-b", ["אבג דהו"], kraken=["אבג דהו"])])
    assert [ln["text"] for ln in one.lines] == ["אבג דהו"]
    book = rr.mark_codebook(["א [...] ב [?] [א]", PUA])
    assert set(book) == {"[...]", "[?]"} and PUA not in book.values()     # never a character the texts hold
    assert rr.decode_marks(rr.encode_marks("א [...] ב", book), book) == "א [...] ב"


# --------------------------------------------------------------------------------------------- agreement


def test_agreement_threshold_is_the_pipeline_rule() -> None:
    """``agreed`` at letters-only similarity >= 0.8 with Kraken text; no fragment means no agreement."""
    lines = [{"text": "אבגדה", "box": list(band(0))}, {"text": "הוזחט", "box": list(band(1))},
             {"text": "אבגד", "box": list(band(2))}, {"text": "טיכלמ", "box": list(band(3))}]
    frags = [{"text": "אבגדו", "box": list(frag_box(0))}, {"text": "הוזיכ", "box": list(frag_box(1))},
             {"text": "אבגה", "box": list(frag_box(2))}]
    out, keep = rr.rederive_lines(lines, frags)
    assert keep == [0, 1, 2, 3]
    assert [(ln["agreement"], ln["status"]) for ln in out] == [
        (0.8, "agreed"), (0.6, "unconfirmed"), (0.75, "unconfirmed"), (None, "unconfirmed")]
    assert out[3]["htr_text"] is None and out[3]["htr_fragments"] == []
    assert out[0]["bbox"] == rr.clamp_box(band(0)) and out[0]["htr_fragments"] == [rr.clamp_box(frag_box(0))]


def test_rederive_lines_and_helpers_match_the_pipeline_source() -> None:
    """``rederive_lines`` and the mirrored helpers give what ``two_reader_lines``'s own functions give."""
    ns = pipeline_namespace("HTR_MODEL_NAME", "LEGACY_HTR_MODEL", "_clamp_box", "drop_loop_duplicates", "sidecar",
                            "raw_stem", "cached_frags")
    loop = "שורה ארוכה שחוזרת שוב"
    vlm_lines = [{"text": "אבגד הוז", "box": [100.0, 100.0, 900.0, 140.0]},
                 {"text": loop, "box": [400.0, 150.0, 600.0, 190.0]},          # narrow box: whole cluster taken
                 {"text": loop, "box": [100.0, 200.0, 900.0, 240.0]},          # loop repeat: dropped
                 {"text": "חטיכ למנ", "box": [100.0, 250.0, 900.0, 290.0]},
                 {"text": "-1.5 overflow", "box": [-20.0, 300.0, 1100.0, 340.0]}]
    frags = [{"text": "הוז", "box": [150.0, 105.0, 400.0, 135.0]},
             {"text": "אבגד", "box": [450.0, 104.0, 860.0, 136.0]},
             {"text": "שורה ארוכה", "box": [150.0, 152.0, 500.0, 188.0]},
             {"text": "שחוזרת שוב", "box": [520.0, 152.0, 880.0, 188.0]},
             {"text": "חטיכ למס", "box": [120.0, 255.0, 880.0, 285.0]},
             {"text": "far away", "box": [0.0, 900.0, 50.0, 950.0]}]
    expected = ns["sidecar"](vlm_lines, frags, True, "m", "r")
    got, keep = rr.rederive_lines(vlm_lines, frags)
    assert got == expected["lines"] and keep == [0, 1, 3, 4]
    assert rr.drop_loop_duplicates(vlm_lines) == ns["drop_loop_duplicates"](vlm_lines)
    for box in ([-5, 10.4, 1200, 3], [10, 20, 30, 40]):
        assert rr.clamp_box(box) == ns["_clamp_box"](box)
    job = {"doc_id": "Cambridge T-S 12.34/a b", "image_index": 2}
    assert rr.raw_stem(job, "qwen3-vl-8b-heb-v21b-step1200") == ns["raw_stem"](job, "qwen3-vl-8b-heb-v21b-step1200")
    entries = [{"frags": [1]}, {"frags": [2], "htr_model": KEY}, {"frags": [3], "frags_by_htr": {"other": [4]}}]
    for entry in entries:
        for key in ("MiDRASH_Gen_01", KEY, "other"):
            assert rr.cached_frags(entry, key) == ns["cached_frags"](entry, key)
    assert rr.LEGACY_HTR_MODEL == ns["LEGACY_HTR_MODEL"]


# --------------------------------------------------------------------------------------------- provenance


def test_hybrid_line_and_attested_by_counts() -> None:
    """A line no reader wrote in full is hybrid and its new word is a non-input word; a shared line is not."""
    a = page_read("vlm-a", ["אבגד", "שלום עולם"], kraken=["אזגה", "שלום עולם"])
    b = page_read("vlm-b", ["אבחה", "שלום עולם"], kraken=["אזגה", "שלום עולם"])
    page = rr.vote_page([a, b])
    assert [ln["text"] for ln in page.lines] == ["אבגה", "שלום עולם"]
    first, second = page.provenance
    assert first["attested_by"] == {"vlm-a": 3, "vlm-b": 3, rr.KRAKEN: 3} and first["n_tokens"] == 4
    assert first["full"] == [] and first["hybrid"] and first["non_input_words"] == ["אבגה"]
    assert first["kraken_only"] == 0 and first["source_lines"] == {"vlm-a": [0], "vlm-b": [0], rr.KRAKEN: [0]}
    assert second["full"] == ["vlm-a", "vlm-b", rr.KRAKEN] and not second["hybrid"]
    block = rr.build_record(page)["rover"]
    assert block["n_hybrid_lines"] == 1 and block["n_non_input_words"] == 1 and block["n_words"] == 3


def test_kraken_only_characters_are_counted_when_kraken_is_the_backbone() -> None:
    """A three-way split goes to the backbone; when that is Kraken the character is counted as Kraken-only."""
    kraken = ["אבגלהטז"]
    page = rr.vote_page([page_read("vlm-a", ["אבגדהוז"], kraken=kraken),
                         page_read("vlm-b", ["אבגיהטכ"], kraken=kraken)])
    assert page.backbone == rr.KRAKEN and [ln["text"] for ln in page.lines] == ["אבגלהטז"]
    prov = page.provenance[0]
    assert prov["kraken_only"] == 1
    assert prov["full"] == [rr.KRAKEN] and prov["attested_by"] == {"vlm-a": 5, "vlm-b": 5, rr.KRAKEN: 7}
    assert prov["box_from"] == "vlm-a#0"             # geometry still comes from a VLM line


# --------------------------------------------------------------------------------------------- schema and CLI


def test_voted_records_load_unchanged_in_describe_from_reads(tmp_path: Path) -> None:
    """Voted and pass-through records load in ``describe_from_reads`` as written, with or without an image map.

    :param tmp_path: Pytest temporary directory.
    """
    p1 = [page_read("vlm-a", ["ליוסף אחי", "שלחתי לך"], kraken=["ליוסף אחי", "שלחתי לד"]),
          page_read("vlm-b", ["ליוסף אחו", "שלחתי לך"], kraken=["ליוסף אחי", "שלחתי לד"])]
    p2 = [page_read("vlm-a", ["ובפסטאט"], kraken=["ובפסטאט"], key="sha-p2", url="https://img/p2.jpg",
                    sha="sha-p2", image_index=1)]
    p2[0].record["ai_read"]["lines"][0].update(status="agreed", agreement=1.0, htr_text="ובפסטאט")
    rows = [rr.build_record(rr.vote_page(p1)), rr.build_record(rr.vote_page(p2))]
    path = write_jsonl(tmp_path / "rover.jsonl", rows)
    for line in path.read_text(encoding="utf-8").splitlines():
        assert dfr.record_head(line) is not None
    docs = dfr.load_document_reads([path])
    assert list(docs) == ["DOC_1"]
    pages = docs["DOC_1"].pages
    assert [p.vlm_model for p in pages] == ["rover3", "rover2"] and [p.image_index for p in pages] == [0, 1]
    assert [(ln.text, ln.agreed) for ln in pages[0].lines] == [("ליוסף אחי", True), ("שלחתי לך", True)]
    assert pages[1].lines[0].agreed and all(p.htr_model == KEY for p in pages)
    jobs = write_jsonl(tmp_path / "jobs.jsonl", [{"doc_id": "V9_DOC", "image_index": 0, "image_url": P1}])
    mapped = dfr.load_document_reads([path], image_map=dfr.load_image_map(jobs))
    assert list(mapped) == ["V9_DOC"] and mapped["V9_DOC"].n_lines == 2


def test_cli_dry_run_reports_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``--dry-run`` prints the voter combinations and writes nothing; without it the records are written.

    :param tmp_path: Pytest temporary directory.
    :param capsys: Pytest output capture.
    """
    raw_dir = tmp_path / "raw"
    rec_a = record("vlm-a", ["אבגד הוז"])
    rec_b = record("vlm-b", ["אבגה הוז"])
    frags = [{"text": "אבגד הוז", "conf": 0.9, "box": list(frag_box(0))}]
    for rec in (rec_a, rec_b):
        write_raw(raw_dir, rec, [{"text": rec["ai_read"]["lines"][0]["text"], "box": list(band(0))}], frags)
    file_a = write_jsonl(tmp_path / "a.jsonl", [rec_a])
    file_b = write_jsonl(tmp_path / "b.jsonl", [rec_b])
    jobs = write_jsonl(tmp_path / "jobs.jsonl", [{"doc_id": "DOC_1", "image_index": 0, "image_url": P1}])
    out = tmp_path / "rover.jsonl"
    common = ["--reads", str(file_a), str(file_b), "--jobs", str(jobs), "--out", str(out), "--raw-dir", str(raw_dir)]
    rr.main(common + ["--dry-run"])
    printed = capsys.readouterr().out
    assert "rover3  vlm-a + vlm-b + kraken  [voted]" in printed and "dry run: nothing written" in printed
    assert "pages with Kraken fragments 1/1" in printed and not out.exists()
    rr.main(common)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["ai_read"]["lines"][0]["text"] == "אבגד הוז"
    assert rows[0]["ai_read"]["lines"][0]["status"] == "agreed"
    assert rows[0]["rover"]["kraken"] == {"record": "a.jsonl:1", "source": "raw_cache", "htr_key": KEY,
                                          "n_fragments": 1}


def test_cli_requires_out_unless_dry_run(tmp_path: Path) -> None:
    """``--out`` is required for a real run, and a missing reads file is an error, not an empty input.

    :param tmp_path: Pytest temporary directory.
    """
    reads = write_jsonl(tmp_path / "a.jsonl", [record("vlm-a", ["אבג"])])
    with pytest.raises(SystemExit):
        rr.main(["--reads", str(reads)])
    with pytest.raises(SystemExit):
        rr.main(["--reads", str(tmp_path / "missing.jsonl"), "--dry-run"])
