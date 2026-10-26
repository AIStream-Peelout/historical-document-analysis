"""KTIV "Varying form of title" -> ``alt_titles`` and KTIV scholarly identifications -> ``scholarly_frames`` end to end."""

from src.datasets.consensus.describe_from_reads import catalogue_metadata
from src.datasets.document_models.genizah_document import (
    GenizahDocument,
    _ktiv_alt_titles,
    _ktiv_scholarly_frames,
)
from src.datasets.indexing.backfill_alt_titles import alt_title_update_actions
from src.datasets.indexing.elastic_index_genizah import ALT_TITLES_MAPPING, SCHOLARLY_FRAMES_MAPPING

CID = "Cambridge_CUL_T_S_NS_219_40"
VARYING = ["Title in English: Talmud Bavli: Megillah 2 a – b", 'תלמוד בבלי: מגילה ב ע"א – ע"ב']


def _merged(varying=VARYING):
    return {
        "canonical_id": CID,
        "shelfmark_display": "Cambridge University Library, Cambridge, England Ms. T-S NS 219.40",
        "institution": "Cambridge University Library",
        "sources_present": ["ktiv"],
        "description": 'ספרות הלכתית ופרשנות תלמודית;ספרות חז"ל;תלמוד בבלי [טקסט',
        "date": None,
        "images": {"ktiv": {"image_urls": []}, "preferred_source": None},
        "sources": {
            "pgp": None, "fjp": [], "bodleian": None,
            "ktiv": {"sys_num": "990051590520205171",
                     "basic_catalog": {"title": "x", "varying_form_of_title": varying}},
        },
    }


def test_ktiv_alt_titles_strips_english_label_and_dedupes():
    ktiv = {"basic_catalog": {"varying_form_of_title": VARYING + ["  title in english:  Talmud Bavli: Megillah 2 a – b "]}}
    assert _ktiv_alt_titles(ktiv) == ["Talmud Bavli: Megillah 2 a – b", 'תלמוד בבלי: מגילה ב ע"א – ע"ב']


def test_ktiv_alt_titles_handles_missing_and_scalar():
    assert _ktiv_alt_titles(None) == []
    assert _ktiv_alt_titles({"basic_catalog": {}}) == []
    assert _ktiv_alt_titles({"basic_catalog": {"varying_form_of_title": "Only one"}}) == ["Only one"]


def test_from_merged_format_surfaces_alt_titles_everywhere():
    doc = GenizahDocument.from_merged_format(_merged())
    assert doc.alt_titles == ["Talmud Bavli: Megillah 2 a – b", 'תלמוד בבלי: מגילה ב ע"א – ע"ב']
    assert "Megillah" in doc.create_full_text_content()
    assert "Alternate titles: Talmud Bavli: Megillah 2 a – b;" in doc.create_text_representation()
    es_doc = doc.to_elasticsearch_document()
    assert es_doc["alt_titles"] == doc.alt_titles
    assert "Megillah" in es_doc["full_text_content"]
    assert "Megillah" not in es_doc["description"]  # kept as its own field


def test_alt_titles_change_embedding_cache_key():
    with_titles = GenizahDocument.from_merged_format(_merged())
    without = GenizahDocument.from_merged_format(_merged(varying=[]))
    assert without.alt_titles == []
    assert with_titles.get_embedding_cache_key() != without.get_embedding_cache_key()


def test_update_actions_only_for_docs_with_alt_titles():
    docs = [GenizahDocument.from_merged_format(_merged()),
            GenizahDocument.from_merged_format(_merged(varying=[]))]
    actions = list(alt_title_update_actions(docs, "genizah_merged_v6"))
    assert len(actions) == 1
    (action,) = actions
    assert action["_op_type"] == "update" and action["_index"] == "genizah_merged_v6"
    assert action["_id"] == CID
    assert action["doc"]["alt_titles"] == docs[0].alt_titles
    assert "Megillah" in action["doc"]["full_text_content"]
    assert set(action["doc"]) == {"alt_titles", "full_text_content"}


def test_alt_titles_mapping_is_searchable_text():
    assert ALT_TITLES_MAPPING["alt_titles"]["type"] == "text"
    assert ALT_TITLES_MAPPING["alt_titles"]["analyzer"] == "multilingual"


def test_scholarly_frames_mapping_has_the_alt_titles_shape():
    assert SCHOLARLY_FRAMES_MAPPING["scholarly_frames"] == ALT_TITLES_MAPPING["alt_titles"]
    assert SCHOLARLY_FRAMES_MAPPING["scholarly_frames"]["fields"]["keyword"]["type"] == "keyword"


# --- scholarly identifications and subject keywords (2026-10-09) -------------------------------------------------

from src.datasets.document_models.genizah_document import _merged_genres, _merged_subjects  # noqa: E402
from src.datasets.indexing.elastic_index_genizah import GENRES_MAPPING, SUBJECTS_MAPPING  # noqa: E402


def _entry(frame=None, title=None, domain=None):
    block = {k: v for k, v in (("frame", frame), ("title", title), ("domain", domain)) if v is not None}
    return {"subsections": {"writing_characteristics": block}}


def _ktiv(entries, subjects=None, varying=None):
    return {"basic_catalog": {"title": "קטע גניזה", "varying_form_of_title": varying or []},
            "full_catalog": {"subjects": subjects or []},
            "scholarly_entries": entries}


def test_scholarly_frames_and_specific_titles_become_scholarly_frames_not_alt_titles():
    ktiv = _ktiv([_entry(frame="[Bible]: Numbers Bemidbar 2:12 – 3:29", title="Bible [Text]",
                         domain="Bible: Texts # Bible: Texts and Translations"),
                  _entry(title="not define"),
                  _entry(frame="(Hoshanot): ''למען אב ידעך''@", title="Sefer ha-Arukh")],
                 varying=["Title in English: Bible: Numbers Bemidbar 2:12 – 3:29"])
    assert _ktiv_alt_titles(ktiv) == ["Bible: Numbers Bemidbar 2:12 – 3:29"]   # varying titles only, label stripped
    assert _ktiv_scholarly_frames(ktiv) == [
        # the frame "[Bible]: Numbers Bemidbar 2:12 – 3:29" restates the varying title: not repeated
        '(Hoshanot): "למען אב ידעך"',                   # frame: incipit quotes normalised, "@" dropped
        "Sefer ha-Arukh",                               # a specific work title
    ]
    assert _ktiv_scholarly_frames(None) == [] and _ktiv_scholarly_frames({"basic_catalog": {}}) == []


def test_titles_and_frames_collapse_bracket_quote_space_and_case_variants():
    ktiv = _ktiv([_entry(frame="[Talmud Bavli]: Megillah 2 a – b"),
                  _entry(frame="(Hoshanot): ''למען אב ידעך''@"),
                  _entry(frame='(Hoshanot):  "למען אב ידעך"'),
                  _entry(title="SEFER HA-ARUKH"), _entry(title="Sefer ha-Arukh")],
                 varying=["Title in English: Talmud Bavli: Megillah 2 a – b", "talmud bavli: megillah 2 a – b"])
    assert _ktiv_alt_titles(ktiv) == ["Talmud Bavli: Megillah 2 a – b"]
    assert _ktiv_scholarly_frames(ktiv) == ['(Hoshanot): "למען אב ידעך"', "SEFER HA-ARUKH"]


def test_generic_work_titles_are_subjects_and_domains_are_genres_not_frames():
    ktiv = _ktiv([_entry(title="Talmud Bavli [Text]", domain="Rabbinic Literature # Talmud Bavli # Talmud Bavli: Texts and Anthologies"),
                  _entry(title="Mishneh Torah"), _entry(title="not define"),
                  _entry(frame="[טוביה בן אליעזר, לקח טוב]: אסתר, רות, שיר השירים, קהלת ואיכה", title="Leqakh Tov")])
    assert _ktiv_alt_titles(ktiv) == []
    assert _ktiv_scholarly_frames(ktiv) == ["[טוביה בן אליעזר, לקח טוב]: אסתר, רות, שיר השירים, קהלת ואיכה"]
    subjects = _merged_subjects(ktiv, [])
    assert subjects == ["Talmud Bavli", "Mishneh Torah", "Leqakh Tov"]
    # domain levels only; a level that is already a subject ("Talmud Bavli") is not repeated
    assert _merged_genres(ktiv, subjects) == ["Rabbinic Literature", "Talmud Bavli: Texts and Anthologies"]
    assert _merged_genres(ktiv) == ["Rabbinic Literature", "Talmud Bavli", "Talmud Bavli: Texts and Anthologies"]
    assert _merged_genres(None) == []


def test_identification_cleaning_handles_lists_brackets_and_empty_work_slots():
    ktiv = _ktiv([_entry(frame=["[Talmud Yerushalmi]: Yoma 41 b – c", "42 d"]),
                  _entry(frame="[Piyyut]@"),
                  _entry(frame=": ''אוחילה''[much injured]@")])
    assert _ktiv_alt_titles(ktiv) == []
    assert _ktiv_scholarly_frames(ktiv) == ["[Talmud Yerushalmi]: Yoma 41 b – c; 42 d", "Piyyut",
                                            '"אוחילה"[much injured]']


def test_subjects_drop_generic_labels_ids_workflow_and_legacy_names():
    ktiv = _ktiv([_entry(domain="Unidentified # Unspecified Domain")],
                 subjects=["Cairo Genizah fragments", "986067066900041", "Karaites", "Karaites-Liturgy-Texts"])
    pgp_docs = [{"tags": "ketubba, Marriage; DIMME; FGP stub; to edit; ib-addenda; Karaite"},
                {"tags": ["India", "trade", "cmp", "friedberg scan"]}]
    assert _merged_subjects(ktiv, pgp_docs) == ["Karaites", "Karaites-Liturgy-Texts", "ketubba", "Marriage",
                                                 "Karaite", "India", "trade"]
    assert _merged_subjects(None, []) == [] and _merged_genres(ktiv) == []


def test_frames_and_subjects_reach_text_full_text_index_doc_and_cache_key():
    merged = _merged(varying=[])
    merged["sources"]["ktiv"] = _ktiv([_entry(frame="[Talmud Bavli]: Megillah 2 a – b", title="Talmud Bavli [Text]")],
                                      subjects=["Jewish law"])
    merged["sources"]["pgp"] = {"documents": [{"tags": "Karaite"}]}
    doc = GenizahDocument.from_merged_format(merged)
    assert doc.alt_titles == [] and doc.scholarly_frames == ["[Talmud Bavli]: Megillah 2 a – b"]
    assert doc.subjects == ["Jewish law", "Talmud Bavli", "Karaite"] and doc.genres == []
    text = doc.create_text_representation()
    assert text.endswith("Alternate titles: [Talmud Bavli]: Megillah 2 a – b\nSubjects: Jewish law; Talmud Bavli; Karaite")
    # without the frames the line has nothing left to list; every other line stays
    assert doc.create_text_representation(include_scholarly_frames=False) == text.replace(
        "Alternate titles: [Talmud Bavli]: Megillah 2 a – b\n", "")
    merged["sources"]["ktiv"]["scholarly_entries"][0]["subsections"]["writing_characteristics"]["domain"] = \
        "Rabbinic Literature # Talmud Bavli"
    doc = GenizahDocument.from_merged_format(merged)
    assert doc.genres == ["Rabbinic Literature"]            # "Talmud Bavli" is already a subject
    assert "Rabbinic Literature" not in doc.create_text_representation()
    full = doc.create_full_text_content()
    assert "Karaite" in full and "Rabbinic Literature" in full   # genres reach lexical search, not the embedding text
    assert "Megillah 2 a – b" in full                             # so do the frames, whatever the embedding text holds
    es_doc = doc.to_elasticsearch_document()
    assert es_doc["subjects"] == doc.subjects and es_doc["alt_titles"] == []
    assert es_doc["scholarly_frames"] == ["[Talmud Bavli]: Megillah 2 a – b"]
    assert es_doc["genres"] == ["Rabbinic Literature"]
    without = GenizahDocument.from_merged_format(_merged(varying=[]))
    assert without.subjects == [] and doc.get_embedding_cache_key() != without.get_embedding_cache_key()


def test_alternate_titles_line_lists_varying_titles_then_frames_only_with_the_flag():
    merged = _merged()
    merged["sources"]["ktiv"]["scholarly_entries"] = [
        _entry(frame="[Talmud Bavli]: Megillah 2 a – b"),                     # restates the varying title
        _entry(frame="[Talmud Bavli]: Megillah 3 a", title="Sefer ha-Arukh")]
    doc = GenizahDocument.from_merged_format(merged)
    assert doc.alt_titles == ["Talmud Bavli: Megillah 2 a – b", 'תלמוד בבלי: מגילה ב ע"א – ע"ב']
    assert doc.scholarly_frames == ["[Talmud Bavli]: Megillah 3 a", "Sefer ha-Arukh"]
    varying_line = 'Alternate titles: Talmud Bavli: Megillah 2 a – b; תלמוד בבלי: מגילה ב ע"א – ע"ב'
    assert varying_line + "; [Talmud Bavli]: Megillah 3 a; Sefer ha-Arukh" in doc.create_text_representation().split("\n")
    assert varying_line in doc.create_text_representation(include_scholarly_frames=False).split("\n")
    full = doc.create_full_text_content()
    assert "Megillah 3 a" in full and "Sefer ha-Arukh" in full


def test_scholarly_frames_change_embedding_cache_key():
    merged = _merged(varying=[])
    merged["sources"]["ktiv"] = _ktiv([_entry(frame="[Talmud Bavli]: Sukkah 29 a – b")])
    framed = GenizahDocument.from_merged_format(merged)
    merged["sources"]["ktiv"] = _ktiv([])
    plain = GenizahDocument.from_merged_format(merged)
    assert framed.scholarly_frames == ["[Talmud Bavli]: Sukkah 29 a – b"] and plain.scholarly_frames == []
    assert framed.get_embedding_cache_key() != plain.get_embedding_cache_key()


def test_subjects_and_genres_mappings_are_searchable_text_with_keyword():
    for mapping, name in ((SUBJECTS_MAPPING, "subjects"), (GENRES_MAPPING, "genres")):
        assert mapping[name]["type"] == "text" and mapping[name]["analyzer"] == "multilingual"
        assert mapping[name]["fields"]["keyword"]["type"] == "keyword"


def test_description_prompt_sees_varying_titles_then_scholarly_identifications():
    merged = _merged()
    merged["sources"]["ktiv"]["scholarly_entries"] = [_entry(frame="[Talmud Bavli]: Megillah 3 a", title="Sefer ha-Arukh")]
    meta = catalogue_metadata(merged)
    assert meta["KTIV alternate titles"] == 'Talmud Bavli: Megillah 2 a – b; תלמוד בבלי: מגילה ב ע"א – ע"ב'
    assert meta["KTIV scholarly identifications"] == "[Talmud Bavli]: Megillah 3 a; Sefer ha-Arukh"
    labels = list(meta)
    assert labels.index("KTIV scholarly identifications") == labels.index("KTIV alternate titles") + 1
