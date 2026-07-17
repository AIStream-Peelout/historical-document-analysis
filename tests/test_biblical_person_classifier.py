"""Regression tests for the known-historical-author gazetteer.

Added during the 2026-07-17 KG audit: Maimonides, Saadia Gaon, and Yehuda
ha-Levi were each split across a mislabeled Scholar node (from the
academic-literature pipeline, which treats an "author"/"editor" role as
evidence of a modern post-discovery scholar) and a separate correct Person
node (from the FJP/KTIV merge pipeline). is_known_historical_author() is the
gazetteer that stops the mislabeling at the source (see coreference_resolver
_classify_people).
"""

import pytest

from src.datasets.document_models import biblical_person_classifier as bib


@pytest.mark.parametrize("name", [
    "Maimonides",
    "Moses Maimonides",
    "Rambam",
    "Saadia Gaon",
    "Yehuda ha-Levi",
    "Judah Halevi",
    "Rashi",
    "Al-Maqrizi",
    "al-Maqrizi",
    "Abraham bar Ḥayya",
    "Abraham Bar Ḥayya",
])
def test_known_historical_authors_recognised(name):
    assert bib.is_known_historical_author(name) is True


@pytest.mark.parametrize("name", [
    "Estara Arrant",
    "Mark Cohen",
    "David",
    "",
])
def test_modern_or_unrelated_names_not_flagged(name):
    assert bib.is_known_historical_author(name) is False
