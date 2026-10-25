# File name: test_describe_from_reads.py
# Date: 10/8/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Generated catalogue descriptions from consensus reads: loading, prompts, validation, grounding, text, client, CLI.

No model and no network: every LM Studio exchange goes through a fake transport or a fake client.
"""
import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets.consensus import describe_from_reads as dfr
from src.embeddings.semantic_text import is_semantic_eligible, semantic_text

V8 = "genizah_merged_v8"


# --------------------------------------------------------------------------------------------- fixtures


def read_line(text: str, status: str = "agreed", agreement: Optional[float] = 0.9,
              htr_text: Optional[str] = "x") -> Dict[str, Any]:
    """One ``ai_read.lines`` entry.

    :param text: VLM text.
    :param status: ``agreed`` / ``unconfirmed`` (``None`` leaves the key out).
    :param agreement: Similarity.
    :param htr_text: Kraken text.
    :return: Line dict.
    """
    line = {"index": 0, "text": text, "bbox": [0, 0, 10, 10], "agreement": agreement, "htr_text": htr_text,
            "htr_fragments": []}
    if status is not None:
        line["status"] = status
    return line


def record(doc_id: str, image_index: int, url: str, lines: List[Dict[str, Any]], vlm: str = "vlm-a",
           parsed: bool = True, sha: Optional[str] = None, decoded_at: str = "2026-10-01T00:00:00+00:00",
           htr_key: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """A ``two_reader_lines`` record in the pipeline's key order.

    :param doc_id: Recorded doc id.
    :param image_index: Recorded image position.
    :param url: Image URL.
    :param lines: Line dicts.
    :param vlm: VLM checkpoint.
    :param parsed: Reply parsed.
    :param sha: Image hash (default derived from the URL).
    :param decoded_at: Decode time.
    :param htr_key: ``htr_cache_key``.
    :param extra: Extra top-level keys (e.g. ``read_health``).
    :return: Record.
    """
    ai = {"vlm_model": vlm, "vlm_revision": "rev", "htr_model": "MiDRASH_Gen_01", "rule_version": "lines-v3",
          "decoded_at": decoded_at, "parsed": parsed, "n_lines": len(lines),
          "n_agreed": sum(l.get("status") == "agreed" for l in lines), "lines": lines}
    if htr_key:
        ai["htr_cache_key"] = htr_key
    return {"doc_id": doc_id, "source_index": V8, "image_index": image_index, "image_url": url, "image_width": 100,
            "image_height": 100, "image_sha256": sha or f"sha-{url}", "ai_read": ai, **extra}


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


LETTER_LINES = [read_line("ליוסף אחי יקירי שלום"), read_line("שלחתי לך פלפל ושמן בספינה", "unconfirmed", 0.4),
                read_line("[...] בפסטאט בחדש ניסן", "agreed")]


def letter_doc(tmp_path: Path) -> dfr.DocumentReads:
    """A one-page letter read.

    :param tmp_path: Temporary directory.
    :return: Document reads.
    """
    path = write_jsonl(tmp_path / "reads.jsonl", [record("Doc_A", 0, "u/a0", LETTER_LINES)])
    return dfr.load_document_reads([path])["Doc_A"]


def valid_output(**overrides: Any) -> Dict[str, Any]:
    """A reply that passes the schema.

    :param overrides: Keys to replace.
    :return: Reply dict.
    """
    out = {"genre": "documentary", "document_type": "Letter", "language": ["Judaeo-Arabic"], "script": "Hebrew",
           "summary": "Letter to Yosef about a shipment of pepper and oil sent by ship. It mentions Fustat and the "
                      "month of Nisan.",
           "identified_work": None,
           "entities": {"people": ["יוסף"], "places": ["פסטאט"], "dates": ["בחדש ניסן"], "commodities": ["פלפל"],
                        "institutions": []},
           "themes": ["trade", "shipping"], "confidence": 0.8, "notes": ""}
    out.update(overrides)
    return out


MERGED = {"canonical_id": "Doc_A", "shelfmark_display": "T-S 13J8.14", "institution": "Cambridge University Library",
          "collection": "Cambridge CUL", "subcollection": None, "description": "Letters", "date": None,
          "sources": {"pgp": {"documents": [{"type": "Letter", "languages_primary": "Judaeo-Arabic",
                                             "languages_secondary": "", "description": "Letter about pepper.",
                                             "url": "https://example.org/doc/1", "doc_date_standard": ""}]},
                      "ktiv": {"basic_catalog": {"title": "קטע גניזה", "language": "Hebrew",
                                                 "date": "Unknown/לא ידוע"},
                               "full_catalog": {"notes": ["See https://example.org/notes"]}},
                      "unlisted_source": {"description": "NEVER SHOWN"}}}


class FakeClient:
    """Stands in for LMStudioChatClient: replies from a queue, records every request."""

    def __init__(self, replies: Sequence[Optional[str]], model: str = "fake/model", reasoning: str = "off") -> None:
        """:param replies: Reply texts in order (None = a failed request).
        :param model: Reported model key.
        :param reasoning: Reported reasoning mode.
        """
        self.replies = list(replies)
        self.model = model
        self.reasoning = reasoning
        self.requests: List[List[Dict[str, str]]] = []

    def chat(self, messages: List[Dict[str, str]], schema: Any = None) -> dfr.ChatResult:
        """:param messages: Chat messages.
        :param schema: Ignored.
        :return: The next queued reply.
        """
        self.requests.append(messages)
        reply = self.replies.pop(0)
        return dfr.ChatResult(content=reply, finish_reason="stop", schema_used=True,
                              error=None if reply is not None else "HTTP 500: boom")


# --------------------------------------------------------------------------------------------- loading reads


def test_load_picks_the_most_agreed_read_per_page_and_orders_pages(tmp_path):
    weak = record("Doc_A", 1, "u/a1", [read_line("אחד", "unconfirmed"), read_line("שנים", "unconfirmed")], vlm="old")
    strong = record("Doc_A", 1, "u/a1", [read_line("אחד"), read_line("שנים", "unconfirmed")], vlm="new")
    pages = [record("Doc_A", i, f"u/a{i}", [read_line(f"שורה {i}")]) for i in (0, 2, 3, 4)]
    f1 = write_jsonl(tmp_path / "one.jsonl", [weak, pages[1], pages[0]])
    f2 = write_jsonl(tmp_path / "two.jsonl", [strong, pages[2], pages[3]])
    doc = dfr.load_document_reads([f1, f2], max_pages=3)["Doc_A"]
    assert [p.image_index for p in doc.pages] == [0, 1, 2] and doc.n_pages_read == 5
    page1 = doc.pages[1]
    assert page1.vlm_model == "new" and page1.alternatives == 1 and page1.source_file == "two.jsonl"
    assert doc.readers == {"vlm": ["new", "vlm-a"], "htr": ["MiDRASH_Gen_01"]}
    assert doc.source_files == ["one.jsonl", "two.jsonl"]
    assert (doc.n_lines, doc.n_agreed, doc.agreed_lines_pct) == (4, 3, 75.0)


def test_load_skips_a_partial_last_line_and_filters_doc_ids(tmp_path):
    path = write_jsonl(tmp_path / "reads.jsonl", [record("Doc_A", 0, "u/a0", LETTER_LINES),
                                                  record("Doc_B", 0, "u/b0", LETTER_LINES)],
                       partial='{"doc_id": "Doc_C", "source_index": "v", "image_index": 0, "image_url": "u/c0", ')
    assert set(dfr.load_document_reads([path])) == {"Doc_A", "Doc_B"}
    assert set(dfr.load_document_reads([path], doc_ids={"Doc_B", "Doc_X"})) == {"Doc_B"}


def test_image_map_rekeys_reads_by_exact_url(tmp_path):
    # the read was recorded under an older id and position; the v9 pool lists that image under two documents
    old = record("Old_Id", 0, "u/shared", [read_line("טקסט")])
    stray = record("New_Id", 1, "u/not-in-map", [read_line("אחר")])
    path = write_jsonl(tmp_path / "reads.jsonl", [old, stray])
    image_map = {"u/shared": [("New_Id", 2), ("Sibling_Id", 0)]}
    docs = dfr.load_document_reads([path], doc_ids={"New_Id", "Sibling_Id"}, image_map=image_map)
    assert set(docs) == {"New_Id", "Sibling_Id"}
    page = docs["New_Id"].pages[0]
    assert (page.image_index, page.record_doc_id, page.record_image_index) == (2, "Old_Id", 0)
    assert docs["New_Id"].n_pages_read == 1          # the URL outside the map is ignored
    assert docs["Sibling_Id"].pages[0].image_index == 0


def test_same_bytes_under_two_urls_count_once(tmp_path):
    a = record("Doc_A", 0, "u/p1.jpg", [read_line("אחד")], sha="same")
    b = record("Doc_A", 3, "u/p1 2.jpg", [read_line("אחד"), read_line("שנים")], sha="same")
    doc = dfr.load_document_reads([write_jsonl(tmp_path / "r.jsonl", [a, b])])["Doc_A"]
    assert len(doc.pages) == 1 and doc.pages[0].image_index == 0 and doc.pages[0].n_agreed == 2


def test_prefer_reader_outranks_agreement(tmp_path):
    strong = record("Doc_A", 0, "u/a0", [read_line("אחד"), read_line("שנים")], vlm="strong")
    chosen = record("Doc_A", 0, "u/a0", [read_line("אחד", "unconfirmed")], vlm="chosen")
    path = write_jsonl(tmp_path / "r.jsonl", [strong, chosen])
    assert dfr.load_document_reads([path])["Doc_A"].pages[0].vlm_model == "strong"
    assert dfr.load_document_reads([path], prefer=["chosen"])["Doc_A"].pages[0].vlm_model == "chosen"


def test_page_flags_health_and_derived_agreement(tmp_path):
    rows = [record("Doc_A", 0, "u/a0", [], parsed=False),
            record("Doc_A", 1, "u/a1", [read_line("שלום", status=None, agreement=0.85),
                                        read_line("עולם", status=None, agreement=0.5)],
                   read_health={"label": "loop", "flags": ["loop"]}, htr_key="k7")]
    doc = dfr.load_document_reads([write_jsonl(tmp_path / "r.jsonl", rows)])["Doc_A"]
    assert doc.pages[0].flags == ["unparsed", "no_lines"]
    assert doc.pages[1].flags == ["health:loop"] and doc.pages[1].health["label"] == "loop"
    assert [line.agreed for line in doc.pages[1].lines] == [True, False]    # line_rule: >= 0.8 with Kraken text
    assert doc.pages[1].htr_model == "k7"


def test_pool_helpers(tmp_path):
    pool = write_jsonl(tmp_path / "pool.jsonl", [
        {"doc_id": "A", "image_index": 0, "image_url": "u/1", "reread_class": "already_read"},
        {"doc_id": "A", "image_index": 1, "image_url": "u/2", "reread_class": "already_read"},
        {"doc_id": "B", "image_index": 0, "image_url": "u/3", "reread_class": "already_read"},
        {"doc_id": "B", "image_index": 1, "image_url": "u/4", "reread_class": "never_read"},
        {"doc_id": "C", "image_index": 0, "image_url": "u/1", "reread_class": "already_read"}])
    assert dfr.pool_doc_ids(pool, "already_read") == {"A", "C"}
    assert dfr.pool_doc_ids(pool) == {"A", "B", "C"}
    assert dfr.load_image_map(pool)["u/1"] == [("A", 0), ("C", 0)]
    ids_txt = tmp_path / "ids.txt"
    ids_txt.write_text("A\n\nB\n")
    ids_jsonl = write_jsonl(tmp_path / "ids.jsonl", [{"doc_id": "C"}])
    assert dfr.read_doc_ids(ids_txt) == {"A", "B"} and dfr.read_doc_ids(ids_jsonl) == {"C"}
    assert dfr.resolve_doc_ids(str(ids_txt), str(pool), "already_read") == {"A"}


# --------------------------------------------------------------------------------------------- catalogue and prompt


def test_catalogue_metadata_whitelists_fields_and_drops_links():
    meta = dfr.catalogue_metadata(MERGED)
    assert meta["Shelf mark"] == "T-S 13J8.14" and meta["Institution"] == "Cambridge University Library"
    assert meta["Existing catalogue label"] == "Letters" and meta["PGP document type"] == "Letter"
    assert meta["PGP languages"] == "Judaeo-Arabic" and meta["KTIV title"] == "קטע גניזה"
    assert "KTIV date" not in meta and "KTIV notes" not in meta          # placeholder and link dropped
    assert not any("http" in v or "NEVER SHOWN" in v for v in meta.values())
    hit = {"_id": "Doc_A", "_source": {"doc_id": "Doc_A", "shelf_mark": "T-S 13J8.14", "main_language": "Hebrew",
                                       "other_languages": ["Aramaic"], "date_display": "1120",
                                       "document_type": "letter", "description": "Letters"}}
    meta = dfr.catalogue_metadata(hit)
    assert meta["Language"] == "Hebrew; Aramaic" and meta["Document type"] == "letter"
    assert dfr.catalogue_date(hit) == "1120" and dfr.catalogue_date(MERGED) is None
    assert dfr.catalogue_metadata(None) == {}


def test_load_catalogue_reads_merged_and_index_exports(tmp_path):
    path = write_jsonl(tmp_path / "cat.jsonl", [MERGED, {**MERGED, "canonical_id": "Doc_Z"},
                                                {"_index": "i", "_id": "Doc_H", "_source": {"doc_id": "Doc_H"}}])
    got = dfr.load_catalogue(path, {"Doc_A", "Doc_H"})
    assert set(got) == {"Doc_A", "Doc_H"}


def test_build_prompt_prefixes_lines_and_carries_the_catalogue(tmp_path):
    doc = letter_doc(tmp_path)
    messages = dfr.build_prompt(doc, MERGED)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == dfr.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "- Shelf mark: T-S 13J8.14" in user and "- PGP document type: Letter" in user
    assert "Script of the transcription (counted from its characters): Hebrew" in user
    body = user[user.index("=== Page 1 of 1 (image 0) ==="):].splitlines()
    assert body[1:4] == ["[agreed] ליוסף אחי יקירי שלום", "[unverified] שלחתי לך פלפל ושמן בספינה",
                         "[agreed] [...] בפסטאט בחדש ניסן"]
    assert "Transcription: 1 page(s), 3 lines, 2 of them [agreed]." in user
    assert dfr.OUTPUT_INSTRUCTIONS in user and "http" not in user and "NEVER SHOWN" not in user
    assert "- (none known)" in dfr.build_prompt(doc, None)[1]["content"]


def test_build_prompt_cuts_long_pages_to_the_budget(tmp_path):
    long_page = [read_line("א" * 40) for _ in range(10)]
    short_page = [read_line("ב" * 10)]
    rows = [record("Doc_A", 0, "u/0", long_page), record("Doc_A", 1, "u/1", short_page, parsed=False)]
    doc = dfr.load_document_reads([write_jsonl(tmp_path / "r.jsonl", rows)])["Doc_A"]
    selected = dfr.select_prompt_lines(doc, max_chars=130)
    assert [len(lines) for _, lines in selected] == [3, 1]          # the short page whole, the long one cut
    user = dfr.build_prompt(doc, None, max_chars=130)[1]["content"]
    assert "Only 4 of the 11 lines fit the character budget" in user
    assert "=== Page 2 of 2 (image 1) (read flagged: unparsed) ===" in user
    assert dfr.page_budgets([10, 400], 130) == [10, 120]


# --------------------------------------------------------------------------------------------- parsing and schema


def test_validate_output_accepts_the_schema_and_names_errors():
    assert dfr.validate_output(valid_output()) == []
    assert dfr.validate_output(valid_output(identified_work={"title": "Mishnah Berakhot", "section": None,
                                                             "confidence": 0.4})) == []
    bad = valid_output(genre="poem", confidence=1.5, extra=1)
    del bad["notes"]
    errors = dfr.validate_output(bad)
    assert any(e.startswith("genre:") for e in errors) and any(e.startswith("confidence:") for e in errors)
    assert any("notes" in e for e in errors) and any("extra" in e for e in errors)
    bad_entities = valid_output(entities={"people": "יוסף", "places": [], "dates": [], "commodities": [],
                                          "institutions": []})
    assert any(e.startswith("entities/people") for e in dfr.validate_output(bad_entities))


def test_parse_json_object_variants():
    obj = valid_output()
    assert dfr.parse_json_object("```json\n" + json.dumps(obj) + "\n```")[0] == obj
    assert dfr.parse_json_object("Here it is: " + json.dumps(obj) + " Done.")[0] == obj
    assert dfr.parse_json_object("<think>draft {\"a\": 1}</think>" + json.dumps(obj))[0] == obj
    assert dfr.parse_json_object("[1, 2]")[1] is not None
    assert dfr.parse_json_object("no json")[1] == "no JSON object in the reply"
    assert dfr.parse_json_object("prose " + json.dumps(obj), whole=True)[0] is None
    assert dfr.parse_and_validate('{"genre": "x"}')[1]


# --------------------------------------------------------------------------------------------- grounding


TRANSCRIPTION = ("ליוסף אחי יקירי שלום\n"
                 "שלחתי לך פלפל ושמן בספינה\n"
                 "[...] בפסטאט בחדש ניסן\n"
                 "ר\N{HEBREW PUNCTUATION GERESH} אברהם בן יצחק הכהן")
AGREED = "ליוסף אחי יקירי שלום\n[...] בפסטאט בחדש ניסן"


def test_grounding_keeps_verbatim_spans_drops_the_rest_and_caps_confidence():
    entities = {"people": ["יו\N{HEBREW POINT HOLAM}ס\N{HEBREW POINT TSERE}ף",   # niqqud stripped
                           "ר' אברהם בן יצחק",                                        # ASCII apostrophe vs geresh
                           "שמואל הנגיד",                                             # not in the text
                           "יוסף"],                                                   # duplicate after normalising
                "places": ["פסטאטן", "פסטאט"], "dates": ["בחדש ניסן"], "commodities": ["פלפל", "ש"],
                "institutions": []}
    report = dfr.grounding_check(valid_output(entities=entities), TRANSCRIPTION, agreed_text=AGREED)
    kept = report.grounded_json["entities"]
    assert kept["people"] == ["יו\N{HEBREW POINT HOLAM}ס\N{HEBREW POINT TSERE}ף", "ר' אברהם בן יצחק"]
    assert kept["places"] == ["פסטאט"] and kept["commodities"] == ["פלפל"] and kept["dates"] == ["בחדש ניסן"]
    assert {(d["span"], d["reason"]) for d in report.dropped} == {
        ("שמואל הנגיד", "absent"), ("פסטאטן", "absent"), ("ש", "too_short")}
    assert report.support["פלפל"] == "unverified" and report.support["פסטאט"] == "agreed"
    assert report.confidence_capped and report.confidence == dfr.CONFIDENCE_CAP == report.grounded_json["confidence"]
    assert report.confidence_model == 0.8 and report.violations == 3
    assert report.summary()["entities_dropped"] == 3 and report.summary()["entities_kept"] == 5


def test_grounding_final_forms_are_folded_by_default():
    text = "שלחתי לך פלפל ושמנ בספינה"         # the reader wrote a medial nun at the word end
    entities = {"people": [], "places": [], "dates": [], "commodities": ["שמן"], "institutions": []}
    assert dfr.grounding_check(valid_output(entities=entities), text).grounded_json["entities"]["commodities"] == ["שמן"]
    strict = dfr.grounding_check(valid_output(entities=entities), text, fold_finals=False)
    assert strict.dropped == [{"kind": "commodities", "span": "שמן", "reason": "absent"}]


def test_grounding_flags_original_script_quotes_absent_from_the_text():
    ok = valid_output(summary='Letter opening "ליוסף אחי יקירי שלום". It mentions pepper sent by ship.')
    report = dfr.grounding_check(ok, TRANSCRIPTION)
    assert report.summary_spans == ["ליוסף אחי יקירי שלום"] and report.summary_spans_absent == []
    assert not report.confidence_capped and report.summary_sentences == 2
    bad = valid_output(summary='Letter citing "בשם רחמנא" and the רמב"ם. It mentions pepper.')
    report = dfr.grounding_check(bad, TRANSCRIPTION)
    assert report.summary_spans_absent == ["בשם רחמנא", 'רמב"ם'] and report.confidence_capped


def test_grounding_corrects_script_and_language_and_marks_the_work_unverified():
    reply = valid_output(script="Arabic", language=["Arabic", "Hebrew"],
                         identified_work={"title": "Mishnah Berakhot", "section": "2:1", "confidence": 0.6},
                         confidence=0.9)
    report = dfr.grounding_check(reply, TRANSCRIPTION)
    grounded = report.grounded_json
    assert grounded["script"] == "Hebrew" and report.script == {"model": "Arabic", "computed": "Hebrew",
                                                                 "used": "Hebrew"}
    assert grounded["language"] == ["Judaeo-Arabic", "Hebrew"] and report.language_corrections
    assert grounded["identified_work"]["verified"] is False and report.identified_work == "unverified"
    assert grounded["confidence"] == 0.9 and not report.confidence_capped
    verified = dfr.grounding_check(reply, TRANSCRIPTION, verify_work=lambda work, text: True)
    assert verified.identified_work == "verified" and verified.grounded_json["identified_work"]["verification"] == "symbolic"


# --------------------------------------------------------------------------------------------- embedding text


def test_description_text_has_the_record_shape_and_embeds_like_a_real_record():
    grounded = dfr.grounding_check(valid_output(), TRANSCRIPTION).grounded_json
    text = dfr.description_text(grounded, catalogue_date="1120")
    lines = text.splitlines()
    assert lines[0].startswith("Description: Letter to Yosef") and lines[0].endswith(dfr.MACHINE_TAG)
    assert lines[1:] == ["Language: Judaeo-Arabic", "Document Type: Letter", "Date: standard_date: 1120"]
    assert is_semantic_eligible(semantic_text(text))
    assert dfr.embed_decision(grounded, "1120") == (True, "eligible")
    no_date = dfr.description_text(grounded)
    assert no_date.splitlines()[-1] == "Date: in_text: בחדש ניסן"


def test_empty_summary_is_not_eligible_and_the_tag_never_makes_a_text_eligible():
    empty = dfr.description_text(valid_output(summary=""))
    assert not empty.startswith("Description:") and not is_semantic_eligible(semantic_text(empty))
    assert dfr.embed_decision(valid_output(summary="")) == (False, "short_summary")
    tiny = valid_output(summary="Damaged.", entities={k: [] for k in dfr.ENTITY_KINDS}, document_type="Unidentified")
    assert is_semantic_eligible(semantic_text(dfr.description_text(tiny)))      # the record rule alone lets it pass
    assert dfr.embed_decision(tiny) == (False, "short_summary")
    label = valid_output(summary="Teaching Aids,Pen Trials,Writing Exercises,Scribblings,Jotting",
                         entities={k: [] for k in dfr.ENTITY_KINDS})
    assert is_semantic_eligible(semantic_text(dfr.description_text(label)))     # only because of the tag
    assert dfr.embed_decision(label) == (False, "semantic:boilerplate_only")


def test_embed_decision_blocks_low_confidence_and_unclear_content():
    assert dfr.embed_decision(valid_output(confidence=0.1)) == (False, "low_confidence")
    unclear = valid_output(genre="unclear", entities={k: [] for k in dfr.ENTITY_KINDS},
                           summary="Small fragment with a few legible words in Hebrew script; content unclear.")
    assert dfr.embed_decision(unclear) == (False, "unclear_content")
    work = valid_output(identified_work={"title": "Mishnah Berakhot", "section": "2:1", "confidence": 0.5,
                                         "verified": False})
    assert "Possible identification (unverified): Mishnah Berakhot, 2:1." in dfr.description_text(work)


# --------------------------------------------------------------------------------------------- LM Studio client


class FakeTransport:
    """Scripted ``(status, body)`` replies; records every request."""

    def __init__(self, replies: Sequence[Any]) -> None:
        """:param replies: ``(status, body)`` pairs in order."""
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url: str, payload: Dict[str, Any], timeout: float) -> Any:
        """:param url: Endpoint.
        :param payload: Request body.
        :param timeout: Timeout.
        :return: The next reply.
        """
        self.calls.append({"url": url, "payload": json.loads(json.dumps(payload)), "timeout": timeout})
        return self.replies.pop(0)


def ok_body(content: str, reasoning: str = "") -> Dict[str, Any]:
    """A 200 chat-completions body.

    :param content: Message content.
    :param reasoning: ``reasoning_content``.
    :return: Body.
    """
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10}}


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_client_payload_text_only_schema_and_reasoning_off():
    transport = FakeTransport([(200, ok_body('{"a": 1}'))])
    client = dfr.LMStudioChatClient(base_url="http://localhost:1234/v1/", transport=transport, sleep=lambda s: None)
    result = client.chat(MESSAGES, dfr.OUTPUT_SCHEMA)
    call = transport.calls[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions" and result.content == '{"a": 1}'
    payload = call["payload"]
    assert payload["model"] == dfr.DEFAULT_MODEL and payload["messages"] == MESSAGES
    assert payload["temperature"] == dfr.TEMPERATURE and payload["max_tokens"] == dfr.MAX_TOKENS["off"]
    assert payload["reasoning_effort"] == "low" and payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == dfr.OUTPUT_SCHEMA
    assert "image_url" not in json.dumps(payload)
    on = dfr.LMStudioChatClient(reasoning="on", transport=FakeTransport([]))
    assert on.payload(MESSAGES, None)["reasoning_effort"] == "xhigh" and on.max_tokens == dfr.MAX_TOKENS["on"]
    with pytest.raises(ValueError):
        dfr.LMStudioChatClient(reasoning="maybe")
    with pytest.raises(ValueError):
        dfr.text_chat_payload("m", MESSAGES, 0.1, 10, {"model": "other"})


def test_client_yields_to_prod_chat_and_registers_its_own_call_per_request(monkeypatch):
    """Each request waits on the yield gate with the client's URL and runs inside ``own_call(model)`` (a retry and
    the schema fallback are requests too), so a description batch neither ignores an active prod chat nor passes
    for one in other gated jobs."""
    events = []
    monkeypatch.setattr(dfr.lms_yield_gate, "wait_for_chat_idle",
                        lambda site, base_url=None, **kw: events.append(("gate", site, base_url)) or 0.0)

    @contextlib.contextmanager
    def own_call(model, **kw):
        events.append(("own", model, "start"))
        yield
        events.append(("own", model, "end"))

    monkeypatch.setattr(dfr.lms_yield_gate, "own_call", own_call)
    transport = FakeTransport([(500, "busy"), (200, ok_body('{"a": 1}'))])
    client = dfr.LMStudioChatClient(base_url="http://localhost:1234/v1", transport=transport, max_retries=1,
                                    sleep=lambda s: None)
    assert client.chat(MESSAGES, schema={"type": "object"}).content == '{"a": 1}'
    gate = ("gate", "describe_from_reads", "http://localhost:1234/v1")
    assert events == [gate, ("own", dfr.DEFAULT_MODEL, "start"), ("own", dfr.DEFAULT_MODEL, "end")] * 2
    assert len(transport.calls) == 2


def test_client_falls_back_without_the_schema_when_the_server_refuses_it():
    transport = FakeTransport([(400, {"error": "'response_format.type' must be 'json_schema' or 'text'"}),
                               (200, ok_body('{"a": 1}')), (200, ok_body('{"b": 2}'))])
    client = dfr.LMStudioChatClient(transport=transport, max_retries=0, sleep=lambda s: None)
    first = client.chat(MESSAGES, dfr.OUTPUT_SCHEMA)
    assert first.content == '{"a": 1}' and not first.schema_used and client.schema_supported is False
    client.chat(MESSAGES, dfr.OUTPUT_SCHEMA)
    assert ["response_format" in c["payload"] for c in transport.calls] == [True, False, False]


def test_client_recovers_json_from_the_reasoning_channel_only_when_it_is_the_whole_channel():
    transport = FakeTransport([(200, ok_body("", reasoning='{"a": 1}')),
                               (200, ok_body("", reasoning='Draft: {"a": 1} ... wait, no.'))])
    client = dfr.LMStudioChatClient(transport=transport, sleep=lambda s: None)
    assert client.chat(MESSAGES, dfr.OUTPUT_SCHEMA).content == '{"a": 1}'
    failed = client.chat(MESSAGES, dfr.OUTPUT_SCHEMA)
    assert failed.content is None and "empty content" in failed.error


def test_client_retries_server_errors_and_never_retries_a_plain_400():
    sleeps: List[float] = []
    transport = FakeTransport([(500, "busy"), (0, "ConnectionError"), (500, "busy")])
    client = dfr.LMStudioChatClient(transport=transport, max_retries=2, retry_delay=1.0, sleep=sleeps.append)
    result = client.chat(MESSAGES)
    assert result.content is None and result.error.startswith("HTTP 500") and sleeps == [1.0, 2.0]
    transport = FakeTransport([(400, {"error": "context length exceeded"})])
    assert dfr.LMStudioChatClient(transport=transport, sleep=sleeps.append).chat(MESSAGES).content is None
    assert len(transport.calls) == 1


def test_client_doubles_the_budget_when_a_reply_is_cut_by_it():
    cut = {"choices": [{"message": {"content": "", "reasoning_content": "thinking ..."}, "finish_reason": "length"}]}
    truncated = {"choices": [{"message": {"content": '{"genre": "docu'}, "finish_reason": "length"}]}
    transport = FakeTransport([(200, cut), (200, truncated), (200, ok_body('{"a": 1}'))])
    client = dfr.LMStudioChatClient(transport=transport, sleep=lambda s: None)
    assert client.chat(MESSAGES, dfr.OUTPUT_SCHEMA).content == '{"a": 1}'
    assert [c["payload"]["max_tokens"] for c in transport.calls] == [2048, 4096, 8192]
    capped = FakeTransport([(200, cut)] * 4)
    result = dfr.LMStudioChatClient(reasoning="on", transport=capped, sleep=lambda s: None).chat(MESSAGES)
    assert result.content is None and [c["payload"]["max_tokens"] for c in capped.calls] == [8192, 16384]


# --------------------------------------------------------------------------------------------- one document


def test_describe_document_repairs_once_then_writes_a_provenance_row(tmp_path):
    doc = letter_doc(tmp_path)
    client = FakeClient(["not json at all", json.dumps(valid_output())])
    row, failure = dfr.describe_document(doc, MERGED, client)
    assert failure is None and row["attempts"] == 2
    repair_turn = client.requests[1]
    assert repair_turn[-2] == {"role": "assistant", "content": "not json at all"}
    assert "did not pass validation" in repair_turn[-1]["content"]
    assert row["doc_id"] == "Doc_A" and row["n_pages"] == 1 and row["n_lines"] == 3
    assert row["agreed_lines_pct"] == 66.7 and row["readers"]["vlm"] == ["vlm-a"]
    assert row["source_files"] == ["reads.jsonl"] and row["pages"][0]["image_url"] == "u/a0"
    assert row["model"] == "fake/model" and row["prompt_version"] == dfr.PROMPT_VERSION
    assert row["generated_at"].endswith("+00:00") and row["machine_derived"] is True
    assert row["description_source"] == dfr.DESCRIPTION_SOURCE and row["embed_eligible"] is True
    assert row["description_text"].startswith("Description: Letter to Yosef")
    assert row["grounding"]["dropped"] == [] and row["catalogue"]["Shelf mark"] == "T-S 13J8.14"
    assert row["json"]["entities"]["commodities"] == ["פלפל"]


def test_describe_document_gives_up_after_one_repair_or_a_failed_request(tmp_path):
    doc = letter_doc(tmp_path)
    row, failure = dfr.describe_document(doc, None, FakeClient(['{"genre": "x"}', '{"genre": "y"}']))
    assert row is None and failure["stage"] == "invalid_output" and failure["error"]
    row, failure = dfr.describe_document(doc, None, FakeClient([None]))
    assert row is None and failure["stage"] == "lm_studio" and "HTTP 500" in failure["error"]


# --------------------------------------------------------------------------------------------- CLI


def cli_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """Parsed CLI arguments over a two-document reads file and a catalogue.

    :param tmp_path: Temporary directory.
    :param extra: Further arguments.
    :return: Namespace.
    """
    reads = write_jsonl(tmp_path / "reads.jsonl", [record("Doc_A", 0, "u/a0", LETTER_LINES),
                                                   record("Doc_B", 0, "u/b0", LETTER_LINES),
                                                   record("Doc_E", 0, "u/e0", [])])
    catalogue = write_jsonl(tmp_path / "merged.jsonl", [MERGED])
    return dfr.build_parser().parse_args(["--reads", str(reads), "--catalogue", str(catalogue),
                                          "--out", str(tmp_path / "out" / "gen.jsonl"), *extra])


def refuse_client(args: argparse.Namespace) -> Any:
    """Client factory a dry run must never call.

    :param args: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("a dry run created a model client")


def test_dry_run_writes_prompts_only(tmp_path):
    stats = dfr.run(cli_args(tmp_path, "--dry-run"), client_factory=refuse_client)
    prompts = tmp_path / "out" / "gen.prompts.jsonl"
    rows = [json.loads(l) for l in prompts.read_text(encoding="utf-8").splitlines()]
    assert [r["doc_id"] for r in rows] == ["Doc_A", "Doc_B"] and not (tmp_path / "out" / "gen.jsonl").exists()
    assert stats["prompts"] == 2 and stats["skipped_no_lines"] == 1 and stats["with_catalogue"] == 1
    assert stats["median_lines"] == 3 and stats["median_agreed_pct"] == 66.7
    assert rows[0]["messages"][1]["content"].count("[agreed]") >= 2 and rows[0]["prompt_version"] == dfr.PROMPT_VERSION


def test_run_appends_rows_logs_failures_and_resumes(tmp_path):
    client = FakeClient([json.dumps(valid_output()), None])
    counts = dfr.run(cli_args(tmp_path), client_factory=lambda args: client)
    out = tmp_path / "out" / "gen.jsonl"
    assert counts == {"described": 1, "embed_eligible": 1, "failed": 1, "no_lines": 1}
    assert [json.loads(l)["doc_id"] for l in out.read_text(encoding="utf-8").splitlines()] == ["Doc_A"]
    failures = [json.loads(l) for l in (tmp_path / "out" / "gen.failures.jsonl").read_text().splitlines()]
    assert {(f["doc_id"], f["stage"]) for f in failures} == {("Doc_B", "lm_studio"), ("Doc_E", "no_lines")}
    retry = FakeClient([json.dumps(valid_output())])
    dfr.run(cli_args(tmp_path), client_factory=lambda args: retry)
    assert len(retry.requests) == 1                                  # only the failed document is asked again
    assert [json.loads(l)["doc_id"] for l in out.read_text(encoding="utf-8").splitlines()] == ["Doc_A", "Doc_B"]
    stats = dfr.run(cli_args(tmp_path, "--dry-run"), client_factory=refuse_client)
    assert stats["prompts"] == 0                                     # a dry run skips described documents too


def test_pool_read_class_needs_a_pool(tmp_path, capsys):
    with pytest.raises(SystemExit):
        dfr.main(["--reads", "x.jsonl", "--out", str(tmp_path / "o.jsonl"), "--pool-read-class", "already_read"])
