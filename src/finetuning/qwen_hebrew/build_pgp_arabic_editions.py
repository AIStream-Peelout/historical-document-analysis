# File name: build_pgp_arabic_editions.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Arabic-script documentary pages for training, from Princeton Geniza Project editions.

The fine-tuning mixtures hold Arabic *language* in Hebrew letters (Judaeo-Arabic) but next to no
Arabic *script*. This builder makes page-transcription rows from the PGP editions of documents
written in Arabic script that no benchmark holds out.

Documents: rows of ``merged/arabic_scrape_priority_queue.csv`` with a permitted image (tiers 1 to
3), a single fragment, a library IIIF manifest, and an edition that is at least 90 % Arabic
letters. Held out: every registered benchmark document (``benchmark_registry``, which includes the
Arabic-script benchmark) and every fragment of the Hebrew-script benchmarks
(``build_documentary_grounding.decontam_reason``).

Pages: a library image is used only when the record says which text is on it:

* the manifest has one image and the edition one side;
* or the image's label (``1r`` / ``1v``) names the side of an edition section (``Recto`` /
  ``Verso``) and exactly one image carries that label;
* or the edition has no side labels, PGP's ``side`` field names one side, and exactly one image
  carries that label.

Everything else is skipped with a reason. Images of the image store carry no side label, so
only library images are used. A page is also skipped when its text is too dense for the image
(more than 1,000 letters per megapixel: Cambridge serves 2,000 pixels on the long side, and a long
scroll at that size has letters a few pixels wide; such a row would teach the model to invent).
An image gets one row: when PGP has two records of the same text on it, the fuller edition is
kept; when two different documents share the side, the image is dropped (neither edition is the
whole page).

Rows (KTIV ``FEATURES`` schema, ``label_source="pgp_edition"``): one ``fragment_transcribe`` row
per page, asking with :data:`ARABIC_FRAGMENT_PROMPT` for the visible text of that side, one line
per line, with ``[...]`` where the editor restored text or marked a loss
(``arabic_script.split_sections`` with a gap token). Splits: ``train_page`` and ``val``; a
document is in one split, chosen by a hash of its PGP id, and a document whose fragment is
already in another training set is never put in ``val``.

Output::

    <out>/            saved DatasetDict + manifest.jsonl + stats.json
    <out>_images/     the page images (<pgpid>_<n>.jpg)

Export it with ``images_once`` before adding it to a mixture.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.build_pgp_arabic_editions [--out DIR] [--limit N]
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image as PILImage

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.finetuning.qwen_hebrew import build_documentary_grounding as bdg
from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

DEFAULT_OUT = bab.NAS_DATASETS / "pgp_arabic_editions_v1"
GAP = "[...]"
MIN_PAGE_LETTERS = 40
MIN_ARABIC_SHARE = 0.9
VAL_SHARE = 0.05
SAME_TEXT_F1 = 0.5              # two editions with this much letter 5-gram overlap are editions of one text
TASK = "fragment_transcribe"
SECTION = "pgp_arabic_page"
LABEL_SOURCE = "pgp_edition"
_HEBREW_SCRIPT_CLAUSE = "handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic)"
ARABIC_FRAGMENT_PROMPT = FRAGMENT_TRANSCRIBE_PROMPT.replace(_HEBREW_SCRIPT_CLAUSE, "handwritten Arabic script")
assert ARABIC_FRAGMENT_PROMPT != FRAGMENT_TRANSCRIBE_PROMPT, "the fragment prompt no longer names its script the way this builder expects"


def candidate_documents(queue_csv: Path, held_ids: Set[str], held_pgpids: Set[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """PGP documents of the Arabic queue that may become training pages.

    :param queue_csv: ``arabic_scrape_priority_queue.csv``.
    :type queue_csv: Path
    :param held_ids: Canonical ids held out by a registered benchmark.
    :type held_ids: Set[str]
    :param held_pgpids: PGP ids held out by a registered benchmark.
    :type held_pgpids: Set[str]
    :return: ``({pgpid: {canonical_id, shelfmark, library, tier, pgp_side, manifests}}, {pgpid: reason
        it was left out})`` in queue order.
    :rtype: Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]
    """
    docs: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    with open(queue_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pgpid = row["pgpid"]
            manifests = [u for u in re.split(r"[;|\s]+", (row["iiif_urls"] or row["suggested_iiif"] or "").strip()) if u.startswith("http")]
            if pgpid in held_pgpids or row["canonical_id"] in held_ids:
                skipped[pgpid] = "held out by a benchmark"
            elif row["tier"] not in ("1", "2", "3"):
                skipped[pgpid] = "no permitted image"
            elif row["single_fragment"] != "True":
                skipped[pgpid] = "join: the edition covers several fragments"
            elif not manifests:
                skipped[pgpid] = "no library manifest"
            else:
                docs[pgpid] = {"canonical_id": row["canonical_id"], "shelfmark": row["shelfmark"], "library": row["library"],
                               "tier": row["tier"], "pgp_side": row["side"], "manifests": manifests}
    return docs, skipped


def best_sections(editions: List[str]) -> List[Tuple[str, List[str]]]:
    """The fullest edition of a document, cut by side, with losses marked.

    :param editions: Edition texts of one PGP document.
    :type editions: List[str]
    :return: ``(side, lines)`` of the edition with the most visible Arabic letters (empty when
        there is no edition text).
    :rtype: List[Tuple[str, List[str]]]
    """
    best: Tuple[int, List[Tuple[str, List[str]]]] = (0, [])
    for content in editions:
        sections = ar.split_sections(content, GAP)
        letters = len(ar.arabic_letters("\n".join(line for _, lines in sections for line in lines)))
        if letters > best[0]:
            best = (letters, sections)
    return best[1]


def single_side(pgp_side: str) -> str:
    """The one side PGP's ``side`` field names, if it names exactly one.

    :param pgp_side: Field value such as ``"verso"``, ``"recto and verso"`` or ``"verso ; verso"``.
    :type pgp_side: str
    :return: ``"recto"``, ``"verso"`` or ``""``.
    :rtype: str
    """
    named = {side for side in ("recto", "verso") if side in pgp_side.lower()}
    return named.pop() if len(named) == 1 else ""


def assign_pages(sections: List[Tuple[str, List[str]]], canvas_sides: List[str], pgp_side: str = "") -> Tuple[Dict[int, List[str]], str]:
    """Decide which image holds which lines of an edition, only where the record says so.

    :param sections: ``(side, lines)`` of the edition, ``side`` in ``{"", "recto", "verso"}``.
    :type sections: List[Tuple[str, List[str]]]
    :param canvas_sides: Side named by each image's label (``""`` when the label does not say).
    :type canvas_sides: List[str]
    :param pgp_side: PGP's ``side`` field of the document.
    :type pgp_side: str
    :return: ``({image index: lines}, why nothing or not everything was assigned)``.
    :rtype: Tuple[Dict[int, List[str]], str]
    """
    by_side: Dict[str, List[str]] = {}
    for side, lines in sections:
        by_side.setdefault(side, []).extend(lines)
    labelled = [side for side in by_side if side]
    if not by_side or not canvas_sides:
        return {}, "no edition text" if not by_side else "no image"
    if "" in by_side and labelled:
        return {}, "edition has text outside its side labels"
    if not labelled:
        side = single_side(pgp_side)
        if len(canvas_sides) == 1:
            return {0: by_side[""]}, ""
        if side and canvas_sides.count(side) == 1:
            return {canvas_sides.index(side): by_side[""]}, ""
        return {}, "edition without side labels and several images"
    if len(labelled) == 1 and len(canvas_sides) == 1 and canvas_sides[0] == "":
        return {0: by_side[labelled[0]]}, ""
    pages = {canvas_sides.index(side): by_side[side] for side in labelled if canvas_sides.count(side) == 1}
    missing = [side for side in labelled if canvas_sides.count(side) != 1]
    return pages, "; ".join(f"{canvas_sides.count(side)} images labelled {side}" for side in missing)


def resolve_shared_images(pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    """Keep at most one page per image.

    :param pages: Planned pages; each has ``url`` (the image) and ``text`` (its target).
    :type pages: List[Dict[str, Any]]
    :return: ``(pages kept, in their original order; counts of pages dropped by reason)``. Several
        editions of one text on an image: the one with the most Arabic letters stays. Different
        texts on one image: none stays.
    :rtype: Tuple[List[Dict[str, Any]], Counter]
    """
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        by_url.setdefault(page["url"], []).append(page)
    dropped: Counter = Counter()
    keep = set()
    for group in by_url.values():
        letters = [ar.arabic_letters(page["text"]) for page in group]
        fullest = max(range(len(group)), key=lambda i: len(letters[i]))
        if all(ar.ngram_overlap(letters[i], letters[fullest])["f1"] >= SAME_TEXT_F1 for i in range(len(group)) if i != fullest):
            keep.add(id(group[fullest]))
            dropped["another edition of the same page is fuller"] += len(group) - 1
        else:
            dropped["several documents on one image"] += len(group)
    return [page for page in pages if id(page) in keep], dropped


def split_of(pgpid: str, val_share: float) -> str:
    """Deterministic split of a document.

    :param pgpid: PGP document id.
    :type pgpid: str
    :param val_share: Share of documents that go to validation.
    :type val_share: float
    :return: ``"val"`` or ``"train_page"``.
    :rtype: str
    """
    bucket = int(hashlib.sha1(pgpid.encode("utf-8")).hexdigest(), 16) % 10_000
    return "val" if bucket < round(val_share * 10_000) else "train_page"


def page_row(image_path: Path, text: str, stem: str, width: int, height: int) -> Dict[str, Any]:
    """One page-transcription row in the KTIV ``FEATURES`` schema.

    :param image_path: Page image on disk.
    :type image_path: Path
    :param text: Target: the visible text of the page, one line per line.
    :type text: str
    :param stem: Row id.
    :type stem: str
    :param width: Image width in pixels.
    :type width: int
    :param height: Image height in pixels.
    :type height: int
    :return: Feature dict.
    :rtype: Dict[str, Any]
    """
    return {"image": str(image_path), "question": ARABIC_FRAGMENT_PROMPT, "answer": text, "task": TASK, "section": SECTION,
            "stem": stem, "label_source": LABEL_SOURCE, "target_chars": len(text), "target_tokens": 0,
            "image_width": width, "image_height": height}


def build(out: Path, limit: Optional[int] = None, delay_s: float = 0.5, max_side: int = 4000, max_canvases: int = 8,
          val_share: float = VAL_SHARE, held_out: Optional[Tuple[Set[str], Set[str]]] = None,
          benchmark_index: Optional[bdg.BenchmarkIndex] = None, in_training: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Build the dataset directory.

    :param out: Output directory of the saved DatasetDict (images go to ``<out>_images``).
    :type out: Path
    :param limit: Only the first N candidate documents (dry run).
    :type limit: Optional[int]
    :param delay_s: Pause between requests to library servers.
    :type delay_s: float
    :param max_side: Long-side cap of the requested images.
    :type max_side: int
    :param max_canvases: A manifest with more images is a whole volume and is skipped.
    :type max_canvases: int
    :param val_share: Share of documents for validation.
    :type val_share: float
    :param held_out: ``(canonical ids, PGP ids)`` held out by benchmarks; default: the registry.
    :type held_out: Optional[Tuple[Set[str], Set[str]]]
    :param benchmark_index: Hebrew-script benchmark index; default: the real one.
    :type benchmark_index: Optional[bdg.BenchmarkIndex]
    :param in_training: ``build_arabic_benchmark.training_documents`` result; default: the real sets.
    :type in_training: Optional[Dict[str, str]]
    :return: The statistics (also written to ``stats.json``).
    :rtype: Dict[str, Any]
    """
    held_ids, held_pgpids = held_out if held_out is not None else registered_benchmark_documents()
    index = benchmark_index if benchmark_index is not None else bdg.load_benchmark_index()
    trained = in_training if in_training is not None else bab.training_documents(bab.TRAINING_MANIFESTS, bab.TRAINING_ID_LISTS)
    docs, skipped = candidate_documents(bab.QUEUE_CSV, held_ids, held_pgpids)
    if limit:
        docs = dict(list(docs.items())[:limit])
    editions = bab.load_editions(bab.FOOTNOTES_CSV, docs)
    images_dir = out.with_name(out.name + "_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[Dict[str, Any]]] = {"train_page": [], "val": []}
    manifest: List[Dict[str, Any]] = []
    planned: List[Dict[str, Any]] = []
    page_skips: Counter = Counter()
    for pgpid, doc in docs.items():
        reason = bdg.decontam_reason(doc["canonical_id"] or doc["shelfmark"], (), bdg.DocRefs(shelfmarks={doc["shelfmark"]}), index)
        if reason:
            skipped[pgpid] = f"fragment of a benchmark ({reason})"
            continue
        sections = best_sections(editions.get(pgpid, []))
        text = "\n".join(line for _, lines in sections for line in lines)
        arabic, hebrew, share = ar.script_share(text)
        if arabic < MIN_PAGE_LETTERS:
            skipped[pgpid] = f"fewer than {MIN_PAGE_LETTERS} visible Arabic letters"
            continue
        if share < MIN_ARABIC_SHARE:
            skipped[pgpid] = "mixed script"
            continue
        canvases, note = bab.planned_images(pgpid, doc, [], True, max_side, max_canvases, delay_s)
        pages, problem = assign_pages(sections, [c["side"] for c in canvases], doc["pgp_side"])
        if not pages:
            skipped[pgpid] = re.sub(r"https?://\S+: ", "", note) or problem
            continue
        if problem:
            page_skips[problem] += 1
        split = "train_page" if bab.training_set_of(doc["canonical_id"], [pgpid], trained) else split_of(pgpid, val_share)
        for canvas_index, lines in sorted(pages.items()):
            page_text = "\n".join(lines)
            letters = len(ar.arabic_letters(page_text))
            if letters < MIN_PAGE_LETTERS:
                page_skips[f"page under {MIN_PAGE_LETTERS} Arabic letters"] += 1
                continue
            planned.append({"pgpid": pgpid, "doc": doc, "split": split, "image_index": canvas_index, "text": page_text,
                            "lines": len(lines), "letters": letters, **{k: canvases[canvas_index][k] for k in ("url", "label", "side")}})
    planned, shared = resolve_shared_images(planned)
    page_skips.update(shared)
    for page in planned:
        pgpid, doc = page["pgpid"], page["doc"]
        path = images_dir / f"{pgpid}_{page['image_index']}.jpg"
        try:
            bab.save_image(page["url"], path)
        except OSError:                                                  # refused or missing at the library
            page_skips["image download failed"] += 1
            continue
        with PILImage.open(path) as image:
            width, height = image.size
        if page["letters"] / (width * height / 1e6) > bab.MAX_LETTERS_PER_MEGAPIXEL:
            page_skips["text too dense for the image"] += 1
            continue
        stem = f"pgpar_{pgpid}_{page['image_index']}_page"
        splits[page["split"]].append(page_row(path, page["text"], stem, width, height))
        manifest.append({"stem": stem, "pgpid": pgpid, "canonical_id": doc["canonical_id"], "shelfmark": doc["shelfmark"],
                         "tier": doc["tier"], "split": page["split"], "image_index": page["image_index"], "label": page["label"],
                         "side": page["side"], "image_url": page["url"], "image_path": str(path), "lines": page["lines"],
                         "arabic_letters": page["letters"]})
    stats = {"candidate_documents": len(docs), "documents": len({m["pgpid"] for m in manifest}), "pages": len(manifest),
             "rows": {name: len(rows) for name, rows in splits.items()},
             "arabic_letters": sum(m["arabic_letters"] for m in manifest),
             "documents_already_in_another_training_set": len({m["pgpid"] for m in manifest
                                                               if bab.training_set_of(m["canonical_id"], [m["pgpid"]], trained)}),
             "by_tier": dict(Counter(m["tier"] for m in manifest)), "skipped": dict(Counter(skipped.values())),
             "page_skips": dict(page_skips), "skipped_documents": skipped, "prompt": ARABIC_FRAGMENT_PROMPT}
    save_dataset(splits, out, {"manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                               "stats.json": json.dumps(stats, ensure_ascii=False, indent=1)})
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="only the first N candidate documents")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests to library servers")
    args = parser.parse_args(argv)
    stats = build(args.out, limit=args.limit, delay_s=args.delay)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("skipped_documents", "prompt")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
