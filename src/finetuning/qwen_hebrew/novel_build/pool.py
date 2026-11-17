"""Stage 0 of the novel-rows build: the novelty ledger's candidates, normalised across sources.

Every ledger record that passes the source and verdict filters is joined with the record of the same ``row_id`` in
``<candidates_dir>/<source>.jsonl`` (the source extraction, which carries the edition text and source fields the
ledger left out; where both hold a field the ledger wins) and normalised to one :class:`~.schema.Candidate`:

* identifiers as lists: ``canonical_ids`` = the ledger's search-index ids (``cids_es``), then ``canonical_id``, then
  ``canonical_ids``; ``pgpids`` = ``pgpid`` and ``pgpids``, plus the PGP ids of a resolved, unambiguous resolution
  (a literature candidate keeps only its first one in ``pgpid``);
* the source's language statement verbatim; the ledger's script, unit, letter count and document group as given;
* ``witnesses``: the evidence strings of the ledger's image and document axes, its best text matches (only for a
  verdict other than ``novel``: a novel record's matches are below the ledger's thresholds) and its hold-out reason;
* ``rnl``: :data:`RNL_RE` over the record's ids, shelf marks, image names, resolution and notes (the institution);
* the photographs the record names, as :class:`~.schema.ImageRef`: every photograph of a legacy-store record
  (``images[]``; the label is the side its shelf-mark slip names, else its file-name token), else one reference to the
  record's local file, else to its photograph URL (a manifest URL is no photograph); stage 1 completes the lists;
* ``notes``: the record's provenance fields (:data:`NOTE_KEYS`) and the ledger's flags.

:func:`main` then attaches the targets (:func:`.targets.attach_targets`), writes the stage file and prints per-source
counts with the reasons of every candidate left without a target.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.novel_build.pool [--sources pgp,ktiv] [--verdicts novel]
        [--limit N] [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

from src.finetuning.qwen_hebrew.build_legacy_store_rows import sides_conflict
from src.finetuning.qwen_hebrew.novel_build import paths, targets
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl

#: Ledger sources this build knows.
SOURCES = ("pgp", "fjp", "fjp_scrape", "ktiv", "literature", "lit_run2", "apd_v3", "aodeh_v2_1")
#: Ledger verdicts.
VERDICTS = ("held_out", "previously_trained", "uncertain_history", "novel")
#: Sources whose record lists every photograph of its document (``images[]``).
LEGACY_STORE_SOURCES = ("fjp", "fjp_scrape")
#: Russian National Library holdings: the canonical-id prefix, the RNL / NLR sigla and the collection names (Yevr. /
#: Evr., Firkovich in either spelling, Antonin), and the city in shelf marks that name no collection siglum.
RNL_RE = re.compile(r"StPetersburg_NLR|\bRNL\b|\bNLR\b|Yevr|\bEvr\b|Firkovi(?:t)?ch|Antonin|St\.?[\s_]*Petersburg",
                    re.IGNORECASE)
_RNL_FIELDS = ("row_id", "document_group", "canonical_id", "canonical_ids", "cids_es", "shelfmark", "shelf_key",
               "shelf_keys", "shelf_keys_clean", "ktiv_id", "stem", "image_url", "image_path", "images", "resolution",
               "notes")
#: Record fields kept in ``Candidate.notes`` (when set).
NOTE_KEYS = ("edition_source", "image_url", "image_path", "image_sha256", "phash", "image_sources", "n_images_known",
             "notes", "resolution", "funnel", "side", "stem", "family", "domain", "held_out_hint", "shelf_key",
             "shelf_keys", "shelf_keys_clean", "identity_uncertain", "shared_text_docs", "shared_text_role",
             "dedup_keep", "dup_of", "dedup_new_fraction", "pgpid_sources", "exact_text_copy_of")
#: Candidate-record fields not joined into the ledger record (the targets stage reads the edition text itself).
NOT_JOINED = ("edition_text", "text_shingles")
#: Origin of a photograph URL by host (anything else is ``iiif_other``).
URL_ORIGINS = (("storage.googleapis.com", "fjp_bucket"), ("iiif.nli.org.il", "iiif_nli"),
               ("lib.cam.ac.uk", "iiif_cudl"), ("bodleian", "bodleian"))
_MIRROR_NAME_RE = re.compile(r"^\d+_image\.\w+$")
_PHOTO_PATH_RE = re.compile(r"\.(?:jpe?g|png|tiff?|jp2|webp)$|/default\.(?:jpe?g|png)$", re.IGNORECASE)


def path_exists(path: str) -> bool:
    """Whether a local file exists (tests replace this function).

    :param path: File path.
    :return: True when the path exists.
    """
    return Path(path).exists()


def as_list(value: Any) -> List[str]:
    """A string or a list of values as a list of non-empty strings.

    :param value: None, a scalar or a list.
    :return: The values as strings.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [str(v) for v in items if v is not None and str(v) != ""]


def ledger_witnesses(flags: Dict[str, Any]) -> List[str]:
    """The ledger's evidence against a candidate's novelty, as strings.

    :param flags: The ledger record's ``flags``.
    :return: Image and document evidence strings (definite and uncertain), the best text matches as
        ``text_<kind>:<witness>@<share>/<shared shingles>`` when the verdict is not ``novel``, and
        ``held_out:<reason>``; empty for a novel record.
    """
    out: List[str] = []
    for key in ("image", "image_uncertain", "document", "document_uncertain"):
        out.extend(as_list(flags.get(key)))
    if flags.get("verdict") != "novel":
        for kind in ("definite", "uncertain"):
            best = flags.get(f"text_best_{kind}")
            if isinstance(best, (list, tuple)) and len(best) == 3:
                out.append(f"text_{kind}:{best[2]}@{float(best[0]):.3f}/{best[1]}")
    if flags.get("held_out"):
        out.append(f"held_out:{flags['held_out']}")
    return list(dict.fromkeys(out))


def resolved_pgpids(record: Dict[str, Any]) -> List[str]:
    """PGP ids of a resolved, unambiguous resolution (literature and run-2 candidates).

    :param record: Ledger record.
    :return: The ids, or an empty list.
    """
    resolution = record.get("resolution")
    if isinstance(resolution, dict) and resolution.get("status") == "resolved" and not resolution.get("ambiguous"):
        return as_list(resolution.get("pgpids"))
    return []


def is_rnl(record: Dict[str, Any]) -> bool:
    """Whether a record names the Russian National Library holdings (:data:`RNL_RE`).

    :param record: Ledger record joined with its candidate record.
    :return: True on a match in any of the record's ids, marks, image names, resolution or notes.
    """
    parts = []
    for key in _RNL_FIELDS:
        value = record.get(key)
        if value not in (None, "", [], {}):
            parts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return bool(RNL_RE.search("\n".join(parts)))


def image_label(token: str, slip_side: str) -> str:
    """The side a legacy-store photograph shows, as the legacy-store row build reads it.

    The side its shelf-mark slip names replaces the file-name token (also when the two disagree); an agreeing token
    only adds the leaf to a slip side without one (slip ``v``, token ``1v`` -> ``1v``).

    :param token: File-name side token (``1r``) or ``""``.
    :param slip_side: Slip side or ``""``.
    :return: The label, ``""`` when neither is known.
    """
    if not slip_side:
        return token
    if len(slip_side) == 1 and len(token) > 1 and not sides_conflict(token, slip_side):
        return token
    return slip_side


def legacy_store_images(entries: Sequence[Dict[str, Any]]) -> List[ImageRef]:
    """Every photograph a legacy-store record lists.

    :param entries: ``images[]`` (file_name, path, url, image_sha256, phash, side_token, slip_side, slip_outcome).
    :return: ``fjp_local`` references for local files (``present`` when the file exists, else ``listed``),
        ``fjp_bucket`` references for URL-only entries; the note keeps the token and slip side when a slip exists.
    """
    refs: List[ImageRef] = []
    for index, entry in enumerate(entries):
        path = entry.get("path") or None
        origin = "fjp_local" if path else "fjp_bucket"
        token, slip = str(entry.get("side_token") or ""), str(entry.get("slip_side") or "")
        note = ""
        if slip:
            note = f"side_token={token or '-'}; slip_side={slip}"
            note += "; conflict" if sides_conflict(token, slip) else ""
        refs.append(ImageRef(image_id=f"{origin}:{index}", origin=origin, url=entry.get("url") or None, path=path,
                             sha256=entry.get("image_sha256") or None, phash=entry.get("phash") or None,
                             label=image_label(token, slip) or None, slip_outcome=entry.get("slip_outcome") or None,
                             status="present" if path and path_exists(path) else "listed", note=note))
    return refs


def is_photo_url(url: str) -> bool:
    """Whether a URL names a photograph (an image file or an IIIF image request), not a manifest.

    :param url: URL.
    :return: True for a photograph URL.
    """
    return bool(_PHOTO_PATH_RE.search(urlsplit(url).path))


def url_origin(url: str) -> str:
    """Origin of a photograph URL.

    :param url: Photograph URL.
    :return: ``pgpid_mirror`` for a ``<pgpid>_image`` file, else the :data:`URL_ORIGINS` origin of its host, else
        ``iiif_other``.
    """
    if _MIRROR_NAME_RE.match(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]):
        return "pgpid_mirror"
    host = urlsplit(url).netloc.lower()
    return next((origin for marker, origin in URL_ORIGINS if marker in host), "iiif_other")


def seed_images(record: Dict[str, Any]) -> List[ImageRef]:
    """The one photograph a non-legacy-store record names (stage 1 completes the list).

    The record's local file wins (with its sha256 and phash; origin = the record's first ``*_local`` image source);
    else its URL when the URL names a photograph (origin by :func:`url_origin`). The local file and the URL of one
    record may be different photographs, so they are never merged into one reference.

    :param record: Ledger record.
    :return: Zero or one reference.
    """
    labels = [str(x) for x in record.get("image_sources") or [] if x and x != "none"]
    path, url = record.get("image_path") or None, record.get("image_url") or None
    if path:
        origin = next((x for x in labels if x.endswith("_local")), labels[0] if labels else "local")
        return [ImageRef(image_id=f"{origin}:0", origin=origin, path=path, sha256=record.get("image_sha256") or None,
                         phash=record.get("phash") or None, status="present" if path_exists(path) else "listed")]
    if url and is_photo_url(url):
        origin = url_origin(url)
        return [ImageRef(image_id=f"{origin}:0", origin=origin, url=url, status="listed")]
    return []


def normalise(record: Dict[str, Any]) -> Candidate:
    """One ledger record (joined with its candidate record) as a :class:`Candidate`.

    :param record: The joined record.
    :return: The candidate (no targets yet).
    """
    flags = record.get("flags") or {}
    source = str(record.get("source") or "")
    canonical_ids = (as_list(record.get("cids_es")) + as_list(record.get("canonical_id"))
                     + as_list(record.get("canonical_ids")))
    pgpids = as_list(record.get("pgpid")) + as_list(record.get("pgpids")) + resolved_pgpids(record)
    images = legacy_store_images(record.get("images") or []) if source in LEGACY_STORE_SOURCES else seed_images(record)
    notes: Dict[str, Any] = {key: record[key] for key in NOTE_KEYS if record.get(key) is not None}
    notes["ledger_flags"] = flags
    return Candidate(row_id=str(record["row_id"]), source=source,
                     document_group=str(record.get("document_group") or ""),
                     canonical_ids=list(dict.fromkeys(canonical_ids)), pgpids=list(dict.fromkeys(pgpids)),
                     ktiv_id=str(record["ktiv_id"]) if record.get("ktiv_id") else None,
                     shelfmark=str(record.get("shelfmark") or ""), script=str(record.get("script") or ""),
                     languages_stated=record.get("languages"), verdict=str(flags.get("verdict") or ""),
                     witnesses=ledger_witnesses(flags), unit=str(record.get("unit") or ""),
                     n_letters=int(record.get("n_letters") or 0), images=images, rnl=is_rnl(record), notes=notes)


def load_candidates(pool: Path, candidates_dir: Path, sources: Optional[Sequence[str]] = None,
                    verdicts: Sequence[str] = ("novel",), limit: Optional[int] = None) -> List[Candidate]:
    """The ledger's candidates of the wanted sources and verdicts, joined and normalised.

    :param pool: The novelty ledger (JSON lines; verdict in ``flags.verdict``).
    :param candidates_dir: Directory of the per-source candidate files.
    :param sources: Ledger sources to keep (all when None or empty).
    :param verdicts: Verdicts to keep.
    :param limit: At most this many candidates per source, in ledger order (so a small dry run covers every source).
    :return: Candidates in ledger order; ``notes["candidate_record"]`` = ``joined`` or ``missing``.
    """
    wanted_sources = set(sources) if sources else None
    wanted_verdicts = set(verdicts)
    taken: Counter = Counter()
    picked: List[Dict[str, Any]] = []
    for record in read_jsonl(pool):
        source = record.get("source")
        if wanted_sources is not None and source not in wanted_sources:
            continue
        if (record.get("flags") or {}).get("verdict") not in wanted_verdicts:
            continue
        if limit is not None and taken[source] >= limit:
            continue
        taken[source] += 1
        picked.append(record)
    ids_by_source: Dict[str, set] = defaultdict(set)
    for record in picked:
        ids_by_source[str(record["source"])].add(str(record["row_id"]))
    joined = {source: targets.source_records(candidates_dir, source, ids, drop=NOT_JOINED)
              for source, ids in ids_by_source.items()}
    out: List[Candidate] = []
    for record in picked:
        extra = joined[str(record["source"])].get(str(record["row_id"]))
        cand = normalise({**(extra or {}), **record})
        cand.notes["candidate_record"] = "joined" if extra is not None else "missing"
        out.append(cand)
    return out


def summarize(cands: Sequence[Candidate], examples: int = 10) -> Dict[str, Any]:
    """Per-source counts and the reasons of candidates left without a target.

    :param cands: Candidates with targets attached.
    :param examples: Row ids listed per reason.
    :return: ``{"by_source": {source: counts}, "empty_targets": {source: {reason: {count, examples}}}}``.
    """
    by_source: Dict[str, Counter] = defaultdict(Counter)
    empty: Dict[str, Dict[str, List[str]]] = {}
    for cand in cands:
        counts = by_source[cand.source]
        counts["candidates"] += 1
        counts["with_targets"] += bool(cand.targets)
        counts["sides"] += len(cand.targets)
        counts["images"] += len(cand.images)
        counts["images_present"] += sum(image.status == "present" for image in cand.images)
        counts["rnl"] += cand.rnl
        if not cand.targets:
            counts["empty_targets"] += 1
            for reason in cand.notes.get("target_reasons") or ["no_target"]:
                empty.setdefault(cand.source, {}).setdefault(reason, []).append(cand.row_id)
    return {"by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
            "empty_targets": {source: {reason: {"count": len(ids), "examples": ids[:examples]}
                                       for reason, ids in sorted(reasons.items())}
                              for source, reasons in sorted(empty.items())}}


def split_names(text: str, allowed: Sequence[str], what: str, parser: argparse.ArgumentParser) -> List[str]:
    """A comma-separated CLI list, checked against the allowed names.

    :param text: Comma-separated names.
    :param allowed: Valid names.
    :param what: Name of the option (for the error message).
    :param parser: The parser (exits with an error on an unknown name).
    :return: The names.
    """
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in allowed]
    if unknown:
        parser.error(f"unknown {what}: {', '.join(unknown)} (known: {', '.join(allowed)})")
    return names


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface.

    :return: The parser.
    """
    ap = argparse.ArgumentParser(description="Stage 0: load and normalise the ledger's candidates, attach targets.")
    ap.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR,
                    help="Build work dir (env NOVEL_BUILD_WORK_DIR); the stage file goes to stages/ inside it.")
    ap.add_argument("--pool", type=Path, default=paths.DEFAULT_POOL, help="The novelty ledger (JSON lines).")
    ap.add_argument("--candidates-dir", type=Path, default=paths.DEFAULT_CANDIDATES_DIR,
                    help="Directory of the per-source candidate files.")
    ap.add_argument("--sources", default="", help="Comma-separated ledger sources (default: all).")
    ap.add_argument("--verdicts", default="novel", help="Comma-separated verdicts to keep (default: novel).")
    ap.add_argument("--limit", type=int, default=None, help="At most N candidates per source (ledger order).")
    ap.add_argument("--sections", type=Path, default=targets.DEFAULT_SECTIONS,
                    help="Legacy-store cleaned side records (env NOVEL_BUILD_SECTIONS).")
    ap.add_argument("--lit-blocks", type=Path, default=targets.DEFAULT_LIT_BLOCKS,
                    help="Literature blocks (env NOVEL_BUILD_LIT_BLOCKS).")
    ap.add_argument("--run2", type=Path, nargs="+", default=list(targets.DEFAULT_RUN2),
                    help="Run-2 edition files (env NOVEL_BUILD_RUN2).")
    ap.add_argument("--datasets-root", type=Path, default=paths.DATASETS_ROOT,
                    help="Datasets root of the pass-through sources (env NOVEL_BUILD_DATASETS_ROOT).")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Load the candidates, attach their targets, write the stage-0 file and print per-source counts.

    :param argv: Arguments (default: ``sys.argv``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    sources = split_names(args.sources, SOURCES, "source", parser)
    verdicts = split_names(args.verdicts, VERDICTS, "verdict", parser)
    cands = load_candidates(args.pool, args.candidates_dir, sources=sources or None, verdicts=verdicts,
                            limit=args.limit)
    targets.attach_targets(cands, candidates_dir=args.candidates_dir, sections_path=args.sections,
                           lit_blocks_path=args.lit_blocks, run2_path=args.run2, datasets_root=args.datasets_root)
    out = paths.stage_path(args.work_dir, "candidates")
    written = write_jsonl(out, cands)
    print(json.dumps({"stage_file": str(out), "candidates": written, **summarize(cands)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
