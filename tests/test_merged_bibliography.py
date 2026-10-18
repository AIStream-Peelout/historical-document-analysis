"""Bibliography assembly in GenizahDocument.from_merged_format: FJP, PGP and KTIV citations, source-tagged."""
from src.datasets.document_models.genizah_document import GenizahDocument


def _merged(**over):
    base = {
        "canonical_id": "Cambridge_CUL_T_S_12_148", "shelfmark_display": "Cambridge CUL: T-S 12.148",
        "sources_present": ["fjp", "pgp", "ktiv"], "description": "Letter", "institution": "Cambridge",
        "images": {}, "sources": {
            "fjp": [{"bibliography": [{"citation": "Goitein, Med. Soc. II", "location": "p. 240", "relations": ["Edition"]}]}],
            "pgp": {"documents": [{"pgpid": "1", "scholarship_records": "Gil, Palestine, doc. 12"}]},
            "ktiv": {"bibliography": [
                "אברמסון, שרגא, במרכזים ובתפוצות בתקופת הגאונים. מוסד הרב קוק (1965), עמוד 104 (איזכור)",
                "Goitein, S. D., A Mediterranean Society vol. 2, Berkeley 1971 pp. 240-241 (דיון, יש תמונה)",
            ]},
        },
    }
    base.update(over)
    return base


def test_bibliography_merges_three_sources_with_tags():
    doc = GenizahDocument.from_merged_format(_merged())
    by_source = {}
    for b in doc.bibliography:
        by_source.setdefault(b.source, []).append(b)
    assert set(by_source) == {"fjp", "pgp", "ktiv"}
    assert len(by_source["ktiv"]) == 2
    heb = next(b for b in by_source["ktiv"] if b.citation.startswith("אברמסון"))
    assert heb.authors == ["אברמסון, שרגא"] and heb.year == "1965" and heb.location == "104" and heb.relations == ["Mention"]
    eng = next(b for b in by_source["ktiv"] if b.citation.startswith("Goitein"))
    assert eng.relations == ["Discussion", "Image"] and eng.location == "240-241" and eng.title.startswith("A Mediterranean Society")
    assert by_source["fjp"][0].location == "p. 240" and by_source["pgp"][0].citation.startswith("Gil")


def test_ktiv_citations_deduplicated_and_serialised():
    m = _merged()
    m["sources"]["ktiv"]["bibliography"].append(m["sources"]["ktiv"]["bibliography"][0])   # duplicate
    doc = GenizahDocument.from_merged_format(m)
    assert sum(1 for b in doc.bibliography if b.source == "ktiv") == 2
    es = doc.to_elasticsearch_document() if hasattr(doc, "to_elasticsearch_document") else None
    if es is not None:
        assert es["has_bib"] is True and any(b.get("source") == "ktiv" for b in es["bibliography"])
