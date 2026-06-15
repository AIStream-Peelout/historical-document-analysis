#!/usr/bin/env python3
"""Merge PGP, FJP and KTIV shelfmark data into one consolidated JSONL.

Each output line is one physical fragment, keyed by the canonical shelfmark id
produced by :class:`ShelfmarkNormalizer`. The three sources are unioned: a
fragment present in any source gets a record. PGP is authoritative on field
conflicts (precedence ``PGP > KTIV > FJP``); every source's raw payload is
retained under ``sources.*`` so nothing is lost.

This pass is **metadata only** — image references are recorded as pointers
(FJP GCP filenames, KTIV PNX/zip ids) with a ``preferred_source`` flag, but no
archives are unzipped and no files are moved.

Run::

    python -m src.datasets.merging.merge_shelfmarks
    python -m src.datasets.merging.merge_shelfmarks --out /tmp/merged.jsonl

Outputs ``merged_shelfmarks.jsonl`` and ``merge_report.json`` under
``src/datasets/raw_data/cairo_genizah/merged/`` by default.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

RAW_DIR = os.path.join("src", "datasets", "raw_data", "cairo_genizah")
PGP_FRAGMENTS = os.path.join(RAW_DIR, "pgp_raw", "data", "fragments.csv")
PGP_DOCUMENTS = os.path.join(RAW_DIR, "pgp_raw", "data", "documents.csv")
FJP_FILE = os.path.join(
    RAW_DIR, "fjp_all", "merged_princeton_friedberger_all_documents_final.json"
)
KTIV_GLOB = os.path.join(RAW_DIR, "ktiv", "*.json")
DEFAULT_OUT_DIR = os.path.join(RAW_DIR, "merged")

MERGED_JSONL = "merged_shelfmarks.jsonl"
REPORT_FILE = "merge_report.json"
STATE_FILE = "merge_state.json"          # snapshot of id-sets for run-over-run diff
DIFF_FILE = "merge_diff.json"            # what changed since the previous run
IMAGE_GAP_FILE = "image_gap_targets.jsonl"  # worklist of records lacking images

# Collection anchors used to detect and split FJP multi-shelfmark strings.
_ANCHOR_RE = re.compile(
    r"(?=\b(?:T-S|ENA|MS\s+heb|Or\.|Moss\.?|Mosseri|Yevr\.?|EVR|Halper|CAJS|"
    r"Gaster|L-G|AIU|Kaufmann|DK)\b)",
    flags=re.IGNORECASE,
)
# An institution context prefix like "Cambridge CUL:" or "Paris AIU:".
_INST_PREFIX_RE = re.compile(r"\b([A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.()]*){0,3}):\s*")


# ─────────────────────────── shelfmark splitting ───────────────────────────

def looks_multi(shelf_mark: str) -> bool:
    """Heuristically decide whether *shelf_mark* lists more than one shelfmark.

    :param shelf_mark: Raw FJP ``shelf_mark`` string.
    :returns: True if it appears to enumerate multiple fragments (a join string).
    """
    if " - " in shelf_mark:
        return True
    return len(_ANCHOR_RE.findall(shelf_mark)) > 1 or shelf_mark.count(":") > 1


def split_shelfmarks(shelf_mark: str) -> List[str]:
    """Split a (possibly multi-fragment) FJP ``shelf_mark`` into individual marks.

    The institution context (text before a ``:``) is carried forward to segments
    that lack their own prefix, so ``"Cambridge CUL: T-S A.1 T-S B.2"`` yields two
    Cambridge shelfmarks. Range endpoints (``A - B``) are kept as separate marks,
    not expanded.

    :param shelf_mark: Raw FJP ``shelf_mark`` string.
    :returns: List of individual shelfmark strings (at least one).
    """
    # Drop parentheticals first ("(Alt: 1)", "(2nd Series: P 764)") so their
    # inner ":" is never mistaken for an institution prefix boundary.
    shelf_mark = re.sub(r"\([^)]*\)", " ", shelf_mark).strip()

    if not looks_multi(shelf_mark):
        return [shelf_mark]

    marks: List[str] = []
    current_inst = ""
    for inst, body in _split_by_institution(shelf_mark):
        if inst:
            current_inst = inst
        for part in _split_body(body):
            part = part.strip(" -,")
            if not part:
                continue
            marks.append(f"{current_inst}: {part}" if current_inst else part)
    return marks or [shelf_mark]


def _split_by_institution(shelf_mark: str) -> List[Tuple[str, str]]:
    """Split *shelf_mark* into ``(institution, body)`` chunks on ``Prefix:`` marks.

    :param shelf_mark: Raw shelfmark string.
    :returns: Ordered list of ``(institution_or_empty, body)`` tuples.
    """
    matches = list(_INST_PREFIX_RE.finditer(shelf_mark))
    if not matches:
        return [("", shelf_mark)]
    chunks: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        chunks.append(("", shelf_mark[: matches[0].start()]))
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(shelf_mark)
        chunks.append((m.group(1), shelf_mark[body_start:body_end]))
    return chunks


def _split_body(body: str) -> List[str]:
    """Split an institution body into individual shelfmarks on anchors / ranges.

    :param body: Shelfmark text for a single institution context.
    :returns: List of individual shelfmark strings.
    """
    # Treat a range ("A - B") as two separate marks by dropping the dash.
    body = body.replace(" - ", " ")
    # The anchor regex is a zero-width lookahead, so splitting on it inserts a
    # boundary before each collection prefix without consuming any characters.
    parts = [p.strip() for p in _ANCHOR_RE.split(body) if p.strip()]
    return parts or [body.strip()]


# ─────────────────────────────── PGP source ────────────────────────────────

def load_pgp() -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Load PGP fragments + documents into canonical-keyed records and an alias map.

    :returns: ``(records, alias)`` where ``records`` maps canonical id -> a PGP
        block (fragment row + attached documents), and ``alias`` maps every
        canonical-id variant (primary + historic) to its primary canonical id.
    """
    # Index documents by pgpid.
    docs_by_pgpid: Dict[str, dict] = {}
    with open(PGP_DOCUMENTS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pgpid = (row.get("pgpid") or "").strip()
            if pgpid:
                docs_by_pgpid[pgpid] = row

    records: Dict[str, dict] = {}
    alias: Dict[str, str] = {}
    with open(PGP_FRAGMENTS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            primary = ShelfmarkNormalizer.to_canonical_id(row["shelfmark"])
            if not primary:
                continue
            alias[primary] = primary
            for variant in _split_semicolons(row.get("shelfmarks_historic")):
                cid = ShelfmarkNormalizer.to_canonical_id(variant)
                if cid:
                    alias.setdefault(cid, primary)

            pgpids = [p.strip() for p in (row.get("pgpids") or "").split(",") if p.strip()]
            documents = [docs_by_pgpid[p] for p in pgpids if p in docs_by_pgpid]
            block = {"fragment": row, "documents": documents}
            # Merge if two fragment rows collapse to the same canonical id.
            if primary in records:
                records[primary].setdefault("_extra_fragments", []).append(row)
            else:
                records[primary] = block
    return records, alias


def _split_semicolons(value: Optional[str]) -> List[str]:
    """Split a ``;``-separated PGP field into trimmed non-empty parts.

    :param value: Raw field value or ``None``.
    :returns: List of trimmed parts.
    """
    return [v.strip() for v in (value or "").split(";") if v.strip()]


# ─────────────────────────────── FJP source ────────────────────────────────

def load_fjp(alias: Dict[str, str]) -> Tuple[Dict[str, List[Tuple[str, dict]]], dict]:
    """Group FJP records by resolved canonical id, exploding multi-shelfmark rows.

    :param alias: PGP alias map (variant cid -> primary cid).
    :returns: ``(by_cid, stats)`` where ``by_cid`` maps canonical id -> list of
        ``(matched_shelfmark, raw_record)`` tuples. The matched shelfmark is the
        single constituent this id came from (the whole join string for a
        multi-shelfmark record would be misleading as a per-fragment display).
    """
    with open(FJP_FILE, encoding="utf-8") as fh:
        fjp = json.load(fh)

    by_cid: Dict[str, List[Tuple[str, dict]]] = collections.defaultdict(list)
    stats = {"records": len(fjp), "exploded": 0, "segments": 0, "empty_cid": 0}
    for rec in fjp.values():
        sm = rec.get("shelf_mark") or ""
        marks = split_shelfmarks(sm)
        if len(marks) > 1:
            stats["exploded"] += 1
        for mark in marks:
            cid = ShelfmarkNormalizer.to_canonical_id(mark)
            if not cid:
                stats["empty_cid"] += 1
                continue
            stats["segments"] += 1
            key = alias.get(cid, cid)
            by_cid[key].append((mark, rec))
    return by_cid, stats


# ────────────────────────────── KTIV source ────────────────────────────────

def load_ktiv(alias: Dict[str, str]) -> Tuple[Dict[str, dict], dict]:
    """Load KTIV manuscripts, de-duplicating ``(1)`` copies (keep richest).

    :param alias: PGP alias map.
    :returns: ``(by_cid, stats)`` mapping canonical id -> the chosen KTIV record.
    """
    best: Dict[str, dict] = {}
    files = glob.glob(KTIV_GLOB)
    bad: List[str] = []
    empty_cid = 0
    for fpath in files:
        # KTIV is scraped incrementally over months; a single malformed scrape
        # must not abort a merge of the whole (eventually 200k+) corpus.
        try:
            with open(fpath, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError):
            bad.append(os.path.basename(fpath))
            continue
        cid = ShelfmarkNormalizer.to_canonical_id(doc.get("shelf_mark") or "")
        if not cid:
            empty_cid += 1
            continue
        key = alias.get(cid, cid)
        incumbent = best.get(key)
        if incumbent is None or _ktiv_richness(doc) > _ktiv_richness(incumbent):
            best[key] = doc
    return best, {
        "files": len(files),
        "distinct": len(best),
        "empty_shelfmark": empty_cid,
        "unparseable_files": bad,
    }


def _ktiv_richness(doc: dict) -> int:
    """Score a KTIV record so the most complete duplicate wins.

    :param doc: Parsed KTIV manuscript JSON.
    :returns: Integer richness score.
    """
    return (
        int(doc.get("scholarly_entry_count") or 0) * 10
        + len(json.dumps(doc, ensure_ascii=False))
    )


# ─────────────────────────────── merging ───────────────────────────────────

def _first(*values: Optional[str]) -> Optional[str]:
    """Return the first truthy value (PGP > KTIV > FJP precedence at call site).

    :param values: Candidate values in precedence order.
    :returns: First non-empty value, or ``None``.
    """
    for v in values:
        if v:
            return v
    return None


def build_merged_record(
    cid: str,
    pgp: Optional[dict],
    fjp: List[Tuple[str, dict]],
    ktiv: Optional[dict],
) -> dict:
    """Assemble one merged record from the per-source blocks for *cid*.

    Top-level scalar fields follow ``PGP > KTIV > FJP`` precedence; every raw
    source block is retained under ``sources``.

    :param cid: Canonical shelfmark id (the record key).
    :param pgp: PGP block from :func:`load_pgp`, or ``None``.
    :param fjp: List of ``(matched_shelfmark, raw_record)`` tuples for this id.
    :param ktiv: Chosen KTIV record, or ``None``.
    :returns: The merged record dict.
    """
    pgp_frag = (pgp or {}).get("fragment") or {}
    pgp_docs = (pgp or {}).get("documents") or []
    fjp_marks = [mark for mark, _ in fjp]
    fjp_recs = [rec for _, rec in fjp]
    fjp0 = fjp_recs[0] if fjp_recs else {}
    ktiv = ktiv or {}

    sources_present = [
        name for name, present in
        (("pgp", pgp), ("fjp", fjp), ("ktiv", ktiv)) if present
    ]

    display = _first(
        pgp_frag.get("shelfmark"),
        ktiv.get("shelf_mark"),
        fjp_marks[0] if fjp_marks else None,  # constituent mark, not the join string
    )
    # Feed the prefix-stripped core so institution lookup sees "T-S AS 62.645"
    # rather than "Cambridge CUL: T-S AS 62.645".
    inst_info = ShelfmarkNormalizer.get_institution_info(
        ShelfmarkNormalizer._strip_institution(display or "")
    )

    description = _first(
        next((d.get("description") for d in pgp_docs if d.get("description")), None),
        (ktiv.get("basic_catalog") or {}).get("title"),
        fjp0.get("description"),
    )

    # Image pointers (metadata only — nothing unzipped).
    fjp_images = sorted({img for rec in fjp_recs for img in (rec.get("images") or [])})
    ktiv_pnx = ktiv.get("pnx_id")
    images = {
        "fjp": fjp_images,
        "ktiv": {
            "pnx_id": ktiv_pnx,
            "iiif_manifest_url": ktiv.get("iiif_manifest_url"),
            "friedberg_image_no": (ktiv.get("shelfmarks") or {}).get(
                "friedberg_geniza_project_images_no"
            ),
        } if ktiv else None,
        "preferred_source": "ktiv" if ktiv_pnx else ("fjp" if fjp_images else None),
    }

    # Institution: prefer the normalizer's prefix mapping, else the source's own
    # institution label (e.g. FJP "Baltimore", KTIV holding library).
    institution = _first(
        inst_info.get("institution"),
        pgp_frag.get("library"),
        fjp0.get("institution") if fjp0.get("institution") != "Unknown" else None,
    )

    return {
        "canonical_id": cid,
        "shelfmark_display": display,
        "institution": institution,
        "collection": _first(inst_info.get("collection"), fjp0.get("collection")),
        "subcollection": inst_info.get("subcollection"),
        "sources_present": sources_present,
        "description": description,
        "pgpids": [d.get("pgpid") for d in pgp_docs if d.get("pgpid")],
        "images": images,
        "sources": {
            "pgp": pgp,
            "fjp": fjp_recs,
            "ktiv": ktiv or None,
        },
    }


def record_has_image(record: dict) -> bool:
    """Return True if a merged record points to any image (FJP files or KTIV scan).

    :param record: A merged record from :func:`build_merged_record`.
    :returns: True when at least one image pointer exists.
    """
    images = record.get("images") or {}
    if images.get("fjp"):
        return True
    ktiv = images.get("ktiv") or {}
    return bool(ktiv.get("pnx_id") or ktiv.get("friedberg_image_no"))


def record_is_rich(record: dict) -> bool:
    """Return True if a record carries substantive metadata worth imaging.

    Used to prioritise the image-gap worklist: a fragment we already have a
    description / transcription / scholarly catalog for, but no image, is a
    high-value KTIV scrape target.

    :param record: A merged record.
    :returns: True when description, transcription or scholarship is present.
    """
    if record.get("description"):
        return True
    sources = record.get("sources") or {}
    ktiv = sources.get("ktiv") or {}
    if ktiv.get("scholarly_entries") or ktiv.get("full_catalog"):
        return True
    for fjp in sources.get("fjp") or []:
        if fjp.get("transcriptions") or fjp.get("description"):
            return True
    return False


def compute_coverage(
    pgp_records: Dict[str, dict],
    fjp_by_cid: Dict[str, List[Tuple[str, dict]]],
    ktiv_by_cid: Dict[str, dict],
) -> dict:
    """Compute cross-source coverage stats for the report.

    Tracks two things the project cares about as KTIV grows: how much of PGP is
    already mirrored in FJP/KTIV (the shrinking PGP-only gap), and how the
    growing KTIV set breaks down against the frozen PGP/FJP sets.

    :param pgp_records: PGP canonical-id -> block.
    :param fjp_by_cid: FJP canonical-id -> records.
    :param ktiv_by_cid: KTIV canonical-id -> record.
    :returns: A coverage dict embedded in ``merge_report.json``.
    """
    pgp_ids = set(pgp_records)
    fjp_ids = set(fjp_by_cid)
    ktiv_ids = set(ktiv_by_cid)

    pgp_in_fjp = len(pgp_ids & fjp_ids)
    pgp_in_ktiv = len(pgp_ids & ktiv_ids)
    pgp_in_either = len(pgp_ids & (fjp_ids | ktiv_ids))

    ktiv_inst: collections.Counter = collections.Counter()
    for doc in ktiv_by_cid.values():
        head = (doc.get("shelf_mark") or "").split(" Ms.")[0].split(",")[0].strip()
        ktiv_inst[head or "?"] += 1

    return {
        "pgp": {
            "total": len(pgp_ids),
            "covered_by_fjp": pgp_in_fjp,
            "covered_by_ktiv": pgp_in_ktiv,
            "covered_by_either": pgp_in_either,
            "pgp_only_gap": len(pgp_ids) - pgp_in_either,
        },
        "ktiv": {
            "distinct": len(ktiv_ids),
            "matching_pgp": pgp_in_ktiv,
            "matching_fjp": len(ktiv_ids & fjp_ids),
            "matching_pgp_or_fjp": len(ktiv_ids & (pgp_ids | fjp_ids)),
            "ktiv_only_new": len(ktiv_ids - pgp_ids - fjp_ids),
            "by_institution": dict(ktiv_inst.most_common()),
        },
    }


def diff_against_snapshot(state: dict, prev: Optional[dict]) -> dict:
    """Compute what changed between this run's *state* and the *prev* snapshot.

    Surfaces the deltas the project tracks as KTIV grows: newly scraped
    KTIV-only fragments, PGP fragments that just gained FJP/KTIV coverage, and
    records that just gained an image. The first run (no ``prev``) reports the
    current sets as the baseline with empty deltas.

    :param state: Current snapshot (id lists) produced this run.
    :param prev: Previous ``merge_state.json`` contents, or ``None``.
    :returns: A diff dict written to ``merge_diff.json``.
    """
    if prev is None:
        return {"baseline": True, "note": "no previous merge_state.json; baseline run"}

    cur_all = set(state["all_ids"])
    prev_all = set(prev.get("all_ids", []))
    cur_ktiv_only = set(state["ktiv_only_ids"])
    prev_ktiv_only = set(prev.get("ktiv_only_ids", []))
    cur_pgp_cov = set(state["pgp_covered_ids"])
    prev_pgp_cov = set(prev.get("pgp_covered_ids", []))
    cur_no_img = set(state["no_image_ids"])
    prev_no_img = set(prev.get("no_image_ids", []))

    newly_imaged = sorted((prev_no_img - cur_no_img) & cur_all)
    return {
        "baseline": False,
        "previous_total": len(prev_all),
        "current_total": len(cur_all),
        "new_records": sorted(cur_all - prev_all),
        "removed_records": sorted(prev_all - cur_all),
        "new_ktiv_only": sorted(cur_ktiv_only - prev_ktiv_only),
        "newly_pgp_covered": sorted(cur_pgp_cov - prev_pgp_cov),
        "newly_imaged": newly_imaged,
        "counts": {
            "new_records": len(cur_all - prev_all),
            "removed_records": len(prev_all - cur_all),
            "new_ktiv_only": len(cur_ktiv_only - prev_ktiv_only),
            "newly_pgp_covered": len(cur_pgp_cov - prev_pgp_cov),
            "newly_imaged": len(newly_imaged),
        },
    }


def merge(out_dir: str = DEFAULT_OUT_DIR) -> dict:
    """Run the full merge and write the JSONL, report, image-gap worklist and diff.

    Outputs in *out_dir*: ``merged_shelfmarks.jsonl`` (the corpus),
    ``image_gap_targets.jsonl`` (records lacking images — KTIV scrape worklist),
    ``merge_report.json`` (stats + coverage + image gap + run-over-run diff
    summary), ``merge_diff.json`` (full per-id deltas vs the previous run), and
    ``merge_state.json`` (id-set snapshot used for the next run's diff).

    :param out_dir: Output directory.
    :returns: The report dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    pgp_records, alias = load_pgp()
    fjp_by_cid, fjp_stats = load_fjp(alias)
    ktiv_by_cid, ktiv_stats = load_ktiv(alias)

    all_ids = set(pgp_records) | set(fjp_by_cid) | set(ktiv_by_cid)
    counts: collections.Counter = collections.Counter()
    no_image_ids: List[str] = []
    no_image_rich = 0
    gap_by_inst: collections.Counter = collections.Counter()
    pgp_covered_ids: List[str] = []
    ktiv_only_ids: List[str] = []

    out_path = os.path.join(out_dir, MERGED_JSONL)
    gap_path = os.path.join(out_dir, IMAGE_GAP_FILE)
    with open(out_path, "w", encoding="utf-8") as out, \
            open(gap_path, "w", encoding="utf-8") as gap:
        for cid in sorted(all_ids):
            pgp = pgp_records.get(cid)
            fjp = fjp_by_cid.get(cid, [])
            ktiv = ktiv_by_cid.get(cid)
            record = build_merged_record(cid, pgp, fjp, ktiv)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[",".join(record["sources_present"])] += 1

            if pgp and (fjp or ktiv):
                pgp_covered_ids.append(cid)
            if record["sources_present"] == ["ktiv"]:
                ktiv_only_ids.append(cid)

            if not record_has_image(record):
                no_image_ids.append(cid)
                rich = record_is_rich(record)
                if rich:
                    no_image_rich += 1
                gap_by_inst[record.get("institution") or "(unknown)"] += 1
                gap.write(json.dumps({
                    "canonical_id": cid,
                    "shelfmark_display": record["shelfmark_display"],
                    "institution": record["institution"],
                    "collection": record["collection"],
                    "sources_present": record["sources_present"],
                    "has_rich_metadata": rich,
                }, ensure_ascii=False) + "\n")

    state = {
        "all_ids": sorted(all_ids),
        "pgp_covered_ids": sorted(pgp_covered_ids),
        "ktiv_only_ids": sorted(ktiv_only_ids),
        "no_image_ids": sorted(no_image_ids),
    }
    prev = _load_json(os.path.join(out_dir, STATE_FILE))
    diff = diff_against_snapshot(state, prev)
    _write_json(os.path.join(out_dir, DIFF_FILE), diff)
    _write_json(os.path.join(out_dir, STATE_FILE), state)

    report = {
        "total_records": len(all_ids),
        "pgp_fragments": len(pgp_records),
        "fjp": fjp_stats,
        "ktiv": ktiv_stats,
        "alias_index_size": len(alias),
        "fjp_matched_to_pgp": sum(1 for c in fjp_by_cid if c in pgp_records),
        "fjp_union_only": sum(1 for c in fjp_by_cid if c not in pgp_records),
        "ktiv_matched_to_pgp": sum(1 for c in ktiv_by_cid if c in pgp_records),
        "source_combinations": dict(counts),
        "coverage": compute_coverage(pgp_records, fjp_by_cid, ktiv_by_cid),
        "image_gap": {
            "records_without_image": len(no_image_ids),
            "without_image_but_rich": no_image_rich,
            "by_institution": dict(gap_by_inst.most_common()),
            "worklist": gap_path,
        },
        "diff_summary": diff.get("counts", {"baseline": True}),
        "output": out_path,
    }
    _write_json(os.path.join(out_dir, REPORT_FILE), report)
    return report


def _load_json(path: str) -> Optional[dict]:
    """Load a JSON file if it exists, else return ``None``.

    :param path: File path.
    :returns: Parsed dict or ``None``.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: dict) -> None:
    """Write *data* to *path* as indented UTF-8 JSON.

    :param path: Destination path.
    :param data: JSON-serialisable dict.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    """CLI entry point: run the merge and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()
    report = merge(args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
