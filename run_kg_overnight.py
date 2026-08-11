#!/usr/bin/env python3
"""Driver: run Passes 2-4 (entity tag -> coref -> relations) on every book that
has Pass-1 structured output, regenerating with the current pipeline.

Per-book and resumable: each finished book gets a `.<version>_complete`
sentinel; on restart, books with the sentinel (or in SKIP) are skipped.
overwrite=True so stale prior output is fully regenerated. Errors are caught per
book/per pass so one bad book never aborts the run. Logs to ~/kg_overnight.log.

Lives in the repo (not /tmp) so it survives reboots and /tmp cleanups.

Full corpus run:
    PYTHONPATH=. nohup .venv/bin/python run_kg_overnight.py &

Single book (adding new material, or testing a pipeline change) — does not
depend on sentinel state, so it works even on a fresh version tree:
    PYTHONPATH=. .venv/bin/python run_kg_overnight.py --only malk-ish-1-188

List what would run without doing any work:
    PYTHONPATH=. .venv/bin/python run_kg_overnight.py --list
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

_LOG = os.path.expanduser("~/kg_overnight.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(_LOG), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("overnight")

from src.models.llm.academic.llm_client import LLMClient
from src.models.llm.academic.entity_tagger import (
    EntityTagger, _safe_tag, _is_structured_dir, _book_dir_for_structured,
)
from src.models.llm.academic.coreference_resolver import CoreferenceResolver
from src.models.llm.academic.relationship_extractor import (
    RelationExtractor, _RELATIONS_ROOT, PIPELINE_VERSION,
)

ROOT = Path("src/datasets/raw_data/cairo_genizah/academic_literature")
MODEL = "qwen3.6-35b-a3b"
# Run tag = pipeline version + model. The version prefix is what stops a
# re-run on the same model from overwriting the previous generation's
# resolved-entity files (book_entities_resolved_<run_tag>.json) in place.
RUN_TAG = f"{PIPELINE_VERSION}_{_safe_tag(MODEL)}"
SKIP: set = set()  # resumability is handled by the .<version>_complete sentinels


def discover_books() -> list:
    seen, books = set(), []
    for p in ROOT.rglob("page_*_structured.json"):
        if _is_structured_dir(p.parent):
            bd = _book_dir_for_structured(p)
            if bd not in seen:
                seen.add(bd)
                books.append(bd)
    return sorted(books)


def sentinel(book_dir: Path) -> Path:
    return _RELATIONS_ROOT / book_dir.name / RUN_TAG / f".{PIPELINE_VERSION}_complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", metavar="BOOK", action="append", default=None,
                        help="Process only this book directory name (repeatable). "
                             "Ignores sentinels, so it works on a fresh version tree "
                             "— use this to add one new book or test a change.")
    parser.add_argument("--list", action="store_true",
                        help="List discovered books and their sentinel state, then exit.")
    args = parser.parse_args()

    books = discover_books()

    if args.only:
        wanted = set(args.only)
        books = [b for b in books if b.name in wanted]
        missing = wanted - {b.name for b in books}
        if missing:
            log.error(f"No discovered book directory named: {', '.join(sorted(missing))}")
            if not books:
                sys.exit(1)

    if args.list:
        for b in books:
            state = "done" if sentinel(b).exists() else "pending"
            print(f"  [{state:7s}] {b.name}")
        print(f"\n{len(books)} book(s); run_tag={RUN_TAG}")
        return

    client = LLMClient(backend="lm_studio", lms_model=MODEL)
    tagger = EntityTagger(client, run_tag=RUN_TAG)
    resolver = CoreferenceResolver(lms_model=MODEL, run_tag=RUN_TAG)
    extractor = RelationExtractor(client, run_tag=RUN_TAG)

    log.info(f"RUN START — {len(books)} books, run_tag={RUN_TAG}")
    done = skipped = failed = 0

    for i, bd in enumerate(books, 1):
        name = bd.name
        # --only is an explicit request for these books, so it overrides the
        # sentinel skip (otherwise re-testing a book would silently no-op).
        if not args.only and (name in SKIP or sentinel(bd).exists()):
            skipped += 1
            log.info(f"[{i}/{len(books)}] SKIP {name} (done/excluded)")
            continue

        log.info(f"[{i}/{len(books)}] START {name}")
        t0 = time.time()
        ok = True
        for label, fn in (
            ("pass2_tag",       lambda: tagger.tag_book(bd, overwrite=True)),
            ("pass3_resolve",   lambda: resolver.resolve_book(bd, overwrite=True)),
            ("pass4_relations", lambda: extractor.extract_book(bd, overwrite=True)),
        ):
            try:
                fn()
            except Exception as e:
                ok = False
                log.error(f"[{i}/{len(books)}] {name} {label} FAILED: {type(e).__name__}: {e}")
                break

        # A pass can "succeed" while silently degrading (Pass 3 skips failed
        # dedup batches and keeps going — the June 2026 run wrote success
        # sentinels for 227 skipped batches' worth of un-deduplicated books).
        # Refuse the sentinel so the book is retried on the next run.
        if ok and resolver.dedup_skipped_batches > 0:
            ok = False
            log.error(f"[{i}/{len(books)}] {name} DEGRADED: "
                      f"{resolver.dedup_skipped_batches} dedup batch(es) skipped — "
                      f"no sentinel written; book will be retried next run")

        if ok:
            sentinel(bd).parent.mkdir(parents=True, exist_ok=True)
            sentinel(bd).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            done += 1
            log.info(f"[{i}/{len(books)}] DONE {name} in {time.time()-t0:.0f}s "
                     f"(done={done} failed={failed} skipped={skipped})")
        else:
            failed += 1

    log.info(f"RUN COMPLETE — done={done} failed={failed} skipped={skipped} / {len(books)}")


if __name__ == "__main__":
    main()
