"""KTIV "Varying form of title" -> ``alt_titles`` end to end."""

from src.datasets.document_models.genizah_document import (
    GenizahDocument,
    _ktiv_alt_titles,
)
from src.datasets.indexing.backfill_alt_titles import alt_title_update_actions
from src.datasets.indexing.elastic_index_genizah import ALT_TITLES_MAPPING

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
