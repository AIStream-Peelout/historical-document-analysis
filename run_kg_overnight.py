#!/usr/bin/env python3
"""Driver: run Passes 2-4 (entity tag -> coref -> relations) on every book that
has Pass-1 structured output, regenerating with the current pipeline.

Per-book and resumable: each finished book gets a `.v2_complete` sentinel; on
restart, books with the sentinel (or in SKIP) are skipped. overwrite=True so
stale prior output is fully regenerated. Errors are caught per book/per pass so
one bad book never aborts the run. Logs to ~/kg_overnight.log.

Lives in the repo (not /tmp) so it survives reboots and /tmp cleanups.
Run: PYTHONPATH=. nohup .venv/bin/python run_kg_overnight.py &
"""
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
from src.models.llm.academic.relationship_extractor import RelationExtractor, _RELATIONS_V2

ROOT = Path("src/datasets/raw_data/cairo_genizah/academic_literature")
MODEL = "qwen3.6-35b-a3b"
RUN_TAG = _safe_tag(MODEL)
SKIP: set = set()  # resumability is handled by the .v2_complete sentinels


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
    return _RELATIONS_V2 / book_dir.name / RUN_TAG / ".v2_complete"


def main() -> None:
    client = LLMClient(backend="lm_studio", lms_model=MODEL)
    tagger = EntityTagger(client, run_tag=RUN_TAG)
    resolver = CoreferenceResolver(lms_model=MODEL, run_tag=RUN_TAG)
    extractor = RelationExtractor(client, run_tag=RUN_TAG)

    books = discover_books()
    log.info(f"RUN START — {len(books)} books, run_tag={RUN_TAG}")
    done = skipped = failed = 0

    for i, bd in enumerate(books, 1):
        name = bd.name
        if name in SKIP or sentinel(bd).exists():
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
