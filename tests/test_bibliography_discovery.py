"""Bibliography discovery: a book takes its own metadata file, never an alphabetical sibling's."""
import json
from pathlib import Path

from src.datasets.indexing.bibliography.index_all_bibliography import (
    discover_indexing_tasks,
    find_book_metadata,
    pick_book_metadata,
)


def _book(root: Path, *parts: str, pages: int = 2, two_level: bool = False) -> Path:
    """Create ``<root>/<parts>/<book>_structured_x[/model]/page_NNN_structured.json`` and return the book dir.

    :param root: Tree root.
    :param parts: Path parts; the last is the book name.
    :param pages: Pages to write.
    :param two_level: Put the pages in a model subdirectory.
    :returns: Book directory.
    """
    book = root.joinpath(*parts)
    sd = book / f"{parts[-1]}_structured_x"
    if two_level:
        sd = sd / "qwen3_vl_8b"
    sd.mkdir(parents=True)
    for i in range(1, pages + 1):
        (sd / f"page_{i:03d}_structured.json").write_text(json.dumps({"page": i, "main_text": f"text {i}"}))
    return book


def _meta(path: Path, title: str) -> Path:
    path.write_text(json.dumps({"title": title, "authors": ["A"]}))
    return path


def test_sibling_metadata_files_go_to_their_own_books(tmp_path):
    root = tmp_path / "lit"
    coll = root / "cambridge_articles"
    for name in ("HiddenHekhalotToward", "barley_av", "judeo-spanish"):
        _book(root, "cambridge_articles", name, two_level=True)
        _meta(coll / f"{name}_metadata.json", name)
    tasks = {t["book"]: t for t in discover_indexing_tasks(str(root))}
    assert {b: Path(t["metadata_file"]).name for b, t in tasks.items()} == {
        "HiddenHekhalotToward": "HiddenHekhalotToward_metadata.json",
        "barley_av": "barley_av_metadata.json",
        "judeo-spanish": "judeo-spanish_metadata.json",
    }


def test_prefix_match_unique_fallback_nearest_wins_and_ambiguity_skips(tmp_path, caplog):
    root = tmp_path / "lit"
    ry = root / "rylands_articles"
    long_name = "searching-for-the-last-genizah-fragment-in-late-ottoman-cairo"
    _book(root, "rylands_articles", long_name); _book(root, "rylands_articles", "6david")
    _meta(ry / "searching-for_metadata.json", "Searching"); _meta(ry / "6david_metadata.json", "David")
    _meta(ry / "bjrl-article-p159_metadata.json", "Bulletin")           # a book without pages yet
    # parts of one volume share the collection-level file (unique there)
    _book(root, "india_traders", "india_trader_1_50"); _book(root, "india_traders", "india_trader_258_349")
    _meta(root / "india_traders" / "india_trader_metadata.json", "India traders")
    # a per-book file beats the collection-level one
    own = _book(root, "india_traders", "india_trader_350_424")
    _meta(own / "own_metadata.json", "Own")
    # ambiguous: two unrelated siblings and no matching name
    _book(root, "misc", "mystery")
    _meta(root / "misc" / "alpha_metadata.json", "Alpha"); _meta(root / "misc" / "beta_metadata.json", "Beta")
    tasks = {t["book"]: Path(t["metadata_file"]).name for t in discover_indexing_tasks(str(root))}
    assert tasks[long_name] == "searching-for_metadata.json" and tasks["6david"] == "6david_metadata.json"
    assert tasks["india_trader_1_50"] == tasks["india_trader_258_349"] == "india_trader_metadata.json"
    assert tasks["india_trader_350_424"] == "own_metadata.json"
    assert "mystery" not in tasks and "Ambiguous metadata for book 'mystery'" in caplog.text
    assert find_book_metadata(root / "misc" / "mystery", root) is None


def test_pick_book_metadata_rules(tmp_path):
    a, b, c = (tmp_path / n for n in ("alpha_metadata.json", "alphabet_meta.json", "meta.json"))
    assert pick_book_metadata([a], "anything") == a
    assert pick_book_metadata([a, b], "alpha") == a                 # exact beats prefix
    assert pick_book_metadata([a, b], "alphabet-soup") == b          # the one related stem
    assert pick_book_metadata([a, b], "alph") is None                # two related stems: ambiguous
    assert pick_book_metadata([a, c], "zeta") is None                # bare meta.json never matches by name
