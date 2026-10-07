# File name: test_two_edition_documents.py
# Date: 10/4/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the documents that two scholars edited."""
import csv

import pytest

from src.datasets.evaluations import two_edition_documents as ted

LETTER = "בשם רחמן\nשלום רב לאהובי היקר\nכתאבי אליך יום אלכמיס\nואנא בכיר ואלחמד ללה"
LETTER_OTHER_READING = "בשם רחמן\nשלום רב לאחובי היקר\nכתאבי אליך יום אלכמים\nואנא בכיר ואלחמד ללה"
DEED = "ראובן בן יעקב העד\nשמעון בר יצחק נע\nאנחנא שהדי דחתימין לתתא\nכן הוה באחד בשבת"
COLUMNS = ["document", "document_id", "source", "source_slug", "location", "doc_relation", "emendations", "notes", "url", "content"]


def write_footnotes(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for pgpid, source, relation, content in rows:
            writer.writerow({"document": f"doc {pgpid}", "document_id": pgpid, "source": source, "doc_relation": relation, "content": content})


@pytest.fixture
def footnotes(tmp_path):
    path = tmp_path / "footnotes.csv"
    write_footnotes(path, [
        ("1", "Goitein, unpublished", "Edition", LETTER),
        ("1", "Gil, printed", "Edition", LETTER_OTHER_READING),
        ("1", "Cohen, translation", "Translation", LETTER),                    # not an edition
        ("2", "Goitein, unpublished", "Edition", LETTER),                      # one source only
        ("2", "Goitein, unpublished", "Edition", LETTER[:20]),
        ("3", "Goitein, unpublished", "Edition", LETTER),
        ("3", "Gil, printed", "Edition", DEED),                                # two different texts under one document
        ("4", "Goitein, unpublished", "Edition", LETTER),
        ("4", "Gil, printed", "Edition, Translation", LETTER),                 # a copy: no second opinion
        ("5", "Goitein, unpublished", "Edition", LETTER),
        ("5", "Gil, printed", "Edition", "בשם רחמן"),                          # second edition too short
    ])
    return path


def test_visible_text_and_letters():
    assert ted.edition_visible_text("Recto\n1 בשם [רחמן]\n\nשלום רב") == "בשם שלום רב"
    assert ted.hebrew_letters("בשם، abc 12 רחמן") == "בשםרחמן"


def test_shared_ngram_share_is_relative_to_the_smaller_side():
    assert ted.shared_ngram_share("אבגדהוז", "אבגדהוז") == 1.0
    assert ted.shared_ngram_share("אבגדהוזחטי", "אבגדה") == 1.0
    assert ted.shared_ngram_share("אבגדהוז", "תשרקצפע") == 0.0
    assert ted.shared_ngram_share("אבג", "אבגדהוז") == 0.0                      # shorter than one n-gram


def test_editions_by_source_keeps_the_longest_edition_of_each_source(footnotes):
    per_source = ted.editions_by_source(footnotes)
    assert set(per_source["1"]) == {"Goitein, unpublished", "Gil, printed"}
    assert per_source["2"] == {"Goitein, unpublished": LETTER}


def test_only_two_different_readings_of_the_same_text_qualify(footnotes):
    found = ted.two_edition_documents(footnotes, min_letters=20)
    assert set(found) == {"1"} and ted.pgpids(found) == {"1"}
    assert sorted(found["1"]["sources"]) == ["Gil, printed", "Goitein, unpublished"]
    assert found["1"]["letters"] == [len(ted.hebrew_letters(LETTER))] * 2 and 0.5 <= found["1"]["shared"] < 1.0
    assert ted.two_edition_documents(footnotes, min_letters=500) == {}          # the real threshold drops short fixtures


def test_pair_texts_returns_both_readings(footnotes):
    first, second = ted.pair_texts("1", footnotes)
    assert {first, second} == {ted.edition_visible_text(LETTER), ted.edition_visible_text(LETTER_OTHER_READING)}
    with pytest.raises(KeyError):
        ted.pair_texts("2", footnotes)
