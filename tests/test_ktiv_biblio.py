"""Tests for the KTIV bibliography parser (real strings from the KTIV scrape)."""
from src.datasets.indexing.neo4j.ktiv_biblio import parse_ktiv_citation


def test_latin_multi_author_with_page_and_tag():
    c = parse_ktiv_citation("Davis, Malcolm C.; Knopf, Henry; Outhwaite, Ben, Hebrew Bible manuscripts in the "
                            "Cambridge Genizah Collections. Cambridge University Pressעמוד 405 (איזכור)")
    assert c["authors"] == ["Davis, Malcolm C.", "Knopf, Henry", "Outhwaite, Ben"]
    assert c["title"] == "Hebrew Bible manuscripts in the Cambridge Genizah Collections"
    assert c["citedonpages"] == "405"
    assert c["mention_type"] == {"mentioned": "True"}
    assert c["language"] == "English" and c["source"] == "ktiv" and c["parsed"]


def test_hebrew_author_with_hebrew_year_pages_and_tag():
    c = parse_ktiv_citation("הורביץ, אלעזר, קטלוג קטעי גניזת קאהיר: בספריית ווסטמינסטר קולג', ב: קטלוג קטעי הגניזה. "
                            "מכון גניזת קאהיר - ישיבה אוניב (תשס\"ו), עמוד 29-30 (איזכור)")
    assert c["authors"] == ["הורביץ, אלעזר"]
    assert c["title"].startswith("קטלוג קטעי גניזת קאהיר")
    assert c["citedonpages"] == "29-30"
    assert c["year"] == "תשס\"ו"
    assert c["mention_type"] == {"mentioned": "True"} and c["language"] == "Hebrew"


def test_discussion_tag_with_image_flag():
    c = parse_ktiv_citation("Goitein, S. D., A Mediterranean Society vol. 2, Berkeley 1971 pp. 240-241 (דיון, יש תמונה)")
    assert c["mention_type"] == {"discussion": "True"} and c["has_image"]
    assert c["citedonpages"] == "240-241" and c["year"] == "1971"
    assert c["authors"] == ["Goitein, S. D."]


def test_unparseable_keeps_raw_as_title():
    raw = "ראה: י' אביבי, אהל שם, רשימת כתבי היד אשר באוסף מוסיוף, ירושלים, תשנ\"ב, מס' 39"
    c = parse_ktiv_citation(raw)
    assert c["raw"] == raw and c["title"] and not c["parsed"]
