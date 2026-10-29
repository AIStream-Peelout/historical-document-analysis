"""LLM-free web-items ingest: page building, layout the pipeline discovers, dedup, idempotence, late PDFs."""
import json
from pathlib import Path
from typing import Any, Dict, List

import fitz
import pytest

from src.datasets.document_models.bibliography_document import BibliographyDocument
from src.datasets.indexing.bibliography import web_items_ingest as W
from src.datasets.indexing.bibliography.academic_pipeline_status import survey_pdf
from src.datasets.indexing.bibliography.index_all_bibliography import discover_indexing_tasks

PARA = ("The fragment T-S 13J24.22 is a letter from Fustat about a partnership in flax, written in Judaeo-Arabic "
        "and later reused as scrap paper by a copyist who wrote liturgical poems on its verso side. ")


def _pdf(path: Path, pages: List[str]) -> Path:
    """Write a text-layer PDF, one string per page.

    :param path: Output file.
    :param pages: Page texts.
    :returns: The path.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=9)
    doc.save(path)
    return path


def _source(root: Path, name: str = "web_src") -> Path:
    """Build a fetched source dir: HTML post, PDF article, landing-only item, fetch duplicate, index page.

    :param root: ``academic_literature`` root.
    :param name: Source directory name.
    :returns: The source directory.
    """
    src = root / name
    (src / "raw").mkdir(parents=True)
    (src / "text").mkdir()
    html_body = "\n\n".join(f"Paragraph {i}. " + PARA + " ".join(f"word{i}x{j}" for j in range(80)) + "."
                              for i in range(12))
    (src / "raw" / "post_a.html").write_text("<html><head><title>Post A – Example Site</title></head><body>x</body></html>")
    (src / "text" / "post_a.txt").write_text(f"Post A – Example Site\nhttps://example.org/post_a\n\n{html_body}")
    _pdf(src / "raw" / "art_b.pdf", [PARA * 4, "   ", PARA * 3 + " ENA 4020.1 is cited here."])
    _pdf(src / "raw" / "cover_g.pdf", ["Early Journal   Content on JSTOR,\nFree to Anyone in the World. " + PARA,
                                       PARA * 3])
    (src / "raw" / "land_c.html").write_text("<html><head><meta property='og:title' content='Landing C Title - Publisher'>"
                                             "</head></html>")
    (src / "text" / "land_c.txt").write_text("land c\nhttps://example.org/c\n\nLanding C Title")
    (src / "raw" / "land_f.html").write_text("<html><head><meta property='og:title' content='Other F - Publisher'>"
                                             "</head></html>")
    items: List[Dict[str, Any]] = [
        {"slug": "post_a", "title": "Post A – Example Site", "authors": ["A. Author"], "year": 2015, "language": "en",
         "kind": "post", "source_url": "https://example.org/post_a", "fetched": "2026-10-26", "licence_terms": "© X",
         "shelfmarks_mentioned": ["T-S 13J24.22"], "files": {"html": "raw/post_a.html", "text": "text/post_a.txt"}},
        {"slug": "art_b", "title": "Article B – Example Site", "authors": "B. Writer", "citation": "Journal 3 (1999)",
         "language": "en", "kind": "item", "source_url": "https://example.org/b.pdf", "fetched": "2026-10-26",
         "doi": None, "files": {"raw": ["raw/art_b.pdf"]}},
        {"slug": "land_c", "title": "landing c", "language": "he", "kind": "landing_page_only",
         "source_url": "https://example.org/c", "children_failed": ["https://files.example.org/x/GK9.8e.pdf"],
         "files": {"raw": ["raw/land_c.html"], "text": "text/land_c.txt"}},
        {"slug": "land_f", "title": "other f", "kind": "landing_page_only", "files": {"raw": ["raw/land_f.html"]}},
        {"slug": "cover_g", "title": "Cover G – Example Site", "kind": "item", "language": "en", "files": {"raw": ["raw/cover_g.pdf"]}},
        {"slug": "dup_d", "title": "Post A again – Example Site", "kind": "post", "duplicate_of": "post_a",
         "files": {"raw": ["raw/post_a.html"], "text": "text/post_a.txt"}},
        {"slug": "index_e", "title": "Index", "kind": "index_page", "files": {"text": "text/post_a.txt"}},
    ]
    (src / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    rows = [{"url": "https://example.org/post_a", "slug": "post_a", "status": 200, "fetched": "2026-10-26T10:00:00",
             "file": "raw/post_a.html"},
            {"url": "https://example.org/b.pdf", "slug": "art_b", "status": 200, "fetched": "2026-10-26T10:01:00",
             "file": "raw/art_b.pdf", "final_url": "https://example.org/b.pdf"}]
    (src / "fetch_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return src


def _config(root: Path, name: str = "web_src") -> Dict[str, W.SourceConfig]:
    """Write the settings file and load it.

    :param root: ``academic_literature`` root.
    :param name: Source directory name.
    :returns: Loaded settings.
    """
    (root / W.CONFIG_NAME).write_text(json.dumps({"sources": {name: {
        "publication_type": "web article", "licence_report": "© Example",
        "ocr_reliability": {"method": "html_text", "latin": "reliable", "hebrew": "reliable"}}}}))
    return W.load_sources_config(root)


def _run(root: Path, sources: Dict[str, W.SourceConfig], **kw: Any) -> Dict[str, Dict[str, Any]]:
    """Run the ingest without corpus or Elasticsearch and key the rows by slug.

    :param root: Root.
    :param sources: Settings.
    :param kw: Extra :func:`W.run` arguments.
    :returns: Slug -> manifest row.
    """
    rows = W.run(root, sources, corpus=None, **kw)
    return {r["slug"]: r for r in rows}


def test_chunk_text_is_verbatim_and_cut_on_paragraphs():
    text = "\n\n".join(("word " * 150).strip() + f" end{i}." for i in range(10))
    chunks = W.chunk_text(text, target=2000)
    assert all(c in text for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert "".join(c.replace("\n", "").replace(" ", "") for c in chunks) == text.replace("\n", "").replace(" ", "")
    assert max(len(c) for c in chunks) <= 2500


def test_fetch_header_and_site_suffix():
    assert W.strip_fetch_header("Title\nhttps://x.org/a\n\nBody\nmore") == "Body\nmore"
    assert W.strip_fetch_header("No header\nline two") == "No header\nline two"
    titles = ["מעשה – המרכז לחקר בין-תחומי של הגניזה", "קבלה – המרכז לחקר בין-תחומי של הגניזה", "Other"]
    suffix = W.common_title_suffix(titles)
    assert suffix == [" – המרכז לחקר בין-תחומי של הגניזה"]
    assert W.item_title([(None, []), (titles[0], suffix)]) == "מעשה"


def test_legacy_hebrew_remap_and_word_order():
    stored = "Â˙·Ò ÌÈÒÂ"
    assert W._unreverse_hebrew_tokens(W._remap_mac_hebrew(stored)) == "סבתו ונסים"
    assert W._unreverse_hebrew_tokens("JNUL Heb. 577") == "JNUL Heb. 577"


def test_garbled_ocr_lines_are_flagged_not_dropped():
    text = "the Chushiel of this letter with the captive Rabbi\nl^nHJ^ny^v^jg^^ kn3V'Vn\nT-S 13J24.22 is cited."
    spans = W.garbled_line_spans(text, skip=[(text.index("T-S"), text.index("T-S") + 12)])
    assert len(spans) == 1
    assert text[spans[0][0]:spans[0][1]] == "l^nHJ^ny^v^jg^^ kn3V'Vn"


def test_materialised_layout_is_what_the_pipeline_reads(tmp_path):
    root = tmp_path / "academic_literature"
    _source(root)
    rows = _run(root, _config(root), manifest_dir=tmp_path / "out")
    assert rows["post_a"]["verdict"] == "new" and rows["post_a"]["status"] == "written"
    assert rows["art_b"]["status"] == "written"
    assert rows["land_c"]["verdict"] == "skipped:landing_page_only_awaiting_pdf"
    assert rows["dup_d"]["verdict"] == "skipped:fetch_duplicate_of:post_a"
    assert rows["index_e"]["verdict"] == "skipped:kind:index_page"

    work = root / "web_src" / "post_a"
    meta = json.loads((work / "post_a_metadata.json").read_text())
    assert meta["title"] == "Post A" and meta["authors"] == ["A. Author"]
    assert meta["publication"]["year"] == "2015" and meta["publication"]["language"] == "English"
    assert meta["source"]["licence"] == "© X" and meta["source"]["fetch_record"]["items_line"] == 1
    assert meta["files"] == ["post_a.html", "post_a.txt"]
    assert (work / "post_a.html").read_bytes() == (root / "web_src" / "raw" / "post_a.html").read_bytes()
    assert (root / "web_src" / "raw" / "post_a.html").exists(), "copied, never moved"
    pages = sorted((work / "post_a" / "post_a_structured_webtext").glob("page_*_structured.json"))
    assert len(pages) >= 2
    page = json.loads(pages[0].read_text())
    assert page["summary"] == "" and page["ingest"]["route"] == W.ROUTE
    assert page["full_main_text"] in (root / "web_src" / "text" / "post_a.txt").read_text()
    assert "T-S 13J24.22" in page["shelf_marks_mentioned"]
    doc = BibliographyDocument.from_structured_json(pages[0], meta)
    assert doc.title == "Post A" and doc.page_number == 1 and doc.shelf_mark == "T-S 13J24.22"
    assert doc.to_elasticsearch_document()["bibliography"][0]["url"] == "https://example.org/post_a"

    assert W.list_works(root, ["web_src"]) == ["art_b", "cover_g", "post_a"]
    tasks = {t["book"]: t for t in discover_indexing_tasks(str(root))}
    assert set(tasks) == {"post_a", "art_b", "cover_g"}
    cover = json.loads((root / "web_src" / "cover_g" / "cover_g" / "cover_g_structured_webtext" /
                        "page_001_structured.json").read_text())
    assert cover["not_this_work"] == "jstor_early_journal_content_cover"
    assert tasks["art_b"]["metadata_file"].name == "art_b_metadata.json"

    pdf_book = root / "web_src" / "art_b" / "art_b"
    assert sorted(p.name for p in (pdf_book / "art_b_structured_webtext").iterdir()) == \
        ["page_001_structured.json", "page_003_structured.json"], "blank PDF page 2 has no page file"
    status = survey_pdf(root / "web_src" / "art_b" / "art_b.pdf", root)
    assert status.metadata_kind == "exact" and status.next_step == "pass2-4"
    assert (tmp_path / "out" / "MANIFEST.md").exists()
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["totals"]["web_src"]["new"] == 3


def test_rerun_is_idempotent_and_late_pdf_becomes_a_work(tmp_path):
    root = tmp_path / "academic_literature"
    src = _source(root)
    cfg = _config(root)
    _run(root, cfg)
    page = root / "web_src" / "post_a" / "post_a" / "post_a_structured_webtext" / "page_001_structured.json"
    before = page.stat().st_mtime_ns
    again = _run(root, cfg)
    assert again["post_a"]["status"] == "unchanged" and again["art_b"]["status"] == "unchanged"
    assert page.stat().st_mtime_ns == before
    assert _run(root, cfg, rebuild=True)["post_a"]["status"] == "written"

    _pdf(src / "raw" / "GK9.8e.pdf", [PARA * 2 + " landing item text " * 5])
    late = _run(root, cfg)
    assert late["land_c"]["verdict"] == "new" and late["land_c"]["status"] == "written"
    meta = json.loads((root / "web_src" / "land_c" / "land_c_metadata.json").read_text())
    assert meta["title"] == "Landing C Title" and meta["files"][0] == "land_c.pdf"
    assert meta["source"]["provenance"].startswith("user-supplied PDF")


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "academic_literature"
    _source(root)
    rows = _run(root, _config(root), dry_run=True)
    assert rows["post_a"]["status"] == "would_write"
    assert not (root / "web_src" / "post_a").exists()


def test_local_copy_of_a_work_is_a_duplicate(tmp_path):
    root = tmp_path / "academic_literature"
    _source(root)
    book = root / "older_articles" / "post_a_pdf"
    sd = book / "post_a_pdf_structured_gemini"
    sd.mkdir(parents=True)
    body = W.strip_fetch_header((root / "web_src" / "text" / "post_a.txt").read_text())
    for i, chunk in enumerate(W.chunk_text(body), 1):
        (sd / f"page_{i:03d}_structured.json").write_text(json.dumps({"full_main_text": chunk}))
    (book / "post_a_pdf_metadata.json").write_text(json.dumps({"title": "Post A (PDF)"}))
    rows = _run(root, _config(root))
    assert rows["post_a"]["verdict"] == "local_duplicate"
    assert rows["post_a"]["status"] == "not_materialised"
    assert any(r.startswith("local:text=older_articles/post_a_pdf@") for r in rows["post_a"]["dedup_reasons"])
    assert not (root / "web_src" / "post_a").exists()
    allowed = _run(root, _config(root), allow=["post_a"])
    assert allowed["post_a"]["status"] == "written"


def test_index_matches_give_already_indexed():
    dd = W.Deduper("index")
    dd.add("uuid-1", title="Article B", text=PARA * 10, urls=["https://example.org/b.pdf"], book_uuid="uuid-1")
    m = dd.match("Something else", "unrelated words " * 50, ["https://example.org/b.pdf/"])
    assert W.verdict_from({"book_uuid": None, "url": [], "doi": [], "title": [], "containment": [], "n_shingles": 0},
                          m)[0] == "already_indexed"
    m2 = dd.match("Article B", "unrelated words " * 50)
    verdict, reasons = W.verdict_from(dd.match(None, "", []), m2)
    assert verdict == "new" and reasons == ["index:same_title=uuid-1"]


@pytest.mark.parametrize("name", ["bad slug", "../x"])
def test_unsafe_slugs_are_skipped(tmp_path, name):
    root = tmp_path / "academic_literature"
    src = _source(root)
    items = [json.loads(line) for line in (src / "items.jsonl").read_text().splitlines()]
    items[0]["slug"] = name
    (src / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    rows = _run(root, _config(root))
    assert rows[name]["verdict"] == "skipped:unsafe_slug"


def test_config_overrides_fill_missing_catalogue_fields(tmp_path):
    root = tmp_path / "academic_literature"
    _source(root)
    cfg = _config(root)
    cfg["web_src"].item_overrides = {"art_b": {"title": "Article B, catalogued", "year": "1999",
                                               "ocr_reliability": {"latin": "column_order_unreliable"}}}
    rows = _run(root, cfg)
    assert rows["art_b"]["title"] == "Article B, catalogued"
    meta = json.loads((root / "web_src" / "art_b" / "art_b_metadata.json").read_text())
    assert meta["ingest"]["overrides"] == ["title", "year"]
    assert meta["ingest"]["ocr_reliability"]["latin"] == "column_order_unreliable"
    assert _run(root, cfg, rebuild=True)["art_b"]["title"] == "Article B, catalogued"


def test_index_copy_of_this_very_work_is_not_a_duplicate(tmp_path, monkeypatch):
    root = tmp_path / "academic_literature"
    _source(root)
    from src.datasets.document_models import corpus_ids

    own = corpus_ids.book_uuid(corpus_ids.book_key({}, "post_a"))
    body = W.strip_fetch_header((root / "web_src" / "text" / "post_a.txt").read_text())
    idx = W.Deduper("index")
    idx.add(own, title="Post A", text=body, urls=["https://example.org/post_a"], book_uuid=own)
    idx.pdf_names[own] = {"post_a"}
    other = W.Deduper("index")
    other.add(own, title="Post A", text=body, book_uuid=own)
    other.pdf_names[own] = {"someone_elses_pdf"}
    monkeypatch.setattr(W, "index_works", lambda index, env: idx)
    rows = _run(root, _config(root), es_index="any", check_local=False)
    assert rows["post_a"]["verdict"] == "new" and rows["post_a"]["status"] == "written"
    monkeypatch.setattr(W, "index_works", lambda index, env: other)
    W.run(root, _config(root), corpus=None, es_index="any", check_local=False)  # own dir exists: still itself
    import shutil
    shutil.rmtree(root / "web_src" / "post_a")
    rows = _run(root, _config(root), es_index="any", check_local=False)
    assert rows["post_a"]["verdict"] == "already_indexed"
