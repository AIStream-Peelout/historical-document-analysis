#!/usr/bin/env python3
"""Dry-run map of current -> new canonical ids for the one-id-per-Oxford-leaf fix.

READ-ONLY. Streams the CURRENT ``merged/merged_shelfmarks.jsonl`` (never
modified), recomputes every line's canonical id with the current code — the
Oxford pre-pass in :meth:`ShelfmarkNormalizer.parse_oxford`, the reordered
institution tokens, the guarded PGP alias map, per-item keys for volume-only
KTIV records and the re-keyed Bodleian scrape records — and writes one CSV row
per line whose id changes or that is Oxford:

    old_canonical_id, new_canonical_id, sources, change_type, line,
    shelfmark_display, note

``change_type`` is one of:

* ``unchanged``          — same id (possibly now joined by other lines).
* ``collapsed_into``     — the id changes and ≥2 current lines share the new id
  (one physical leaf served as several documents today).
* ``renamed``            — the id changes, nothing else lands on it.
* ``moved_institution``  — the institution token changes (NLI / BnF / Harvard
  records filed under ``Oxford_Bodleian``).
* ``kept_distinct``      — a volume-only KTIV record now keyed per item; extra
  rows are emitted for the volume-only records the current merge dropped.

A JSON summary (collapse counts, the PGP and non-Oxford regression checks, the
f-letter fixes, the KTIV volume-only records) is printed to stdout and
optionally written to ``--summary-out``. The PGP source files and the KTIV
Bodleian JSON files are read (never written) to rebuild the alias map and to
list volume-only KTIV records.

Run::

    python -m src.datasets.merging.oxford_id_map
    python -m src.datasets.merging.oxford_id_map --summary-out /tmp/summary.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.datasets.merging.bodleian_images import (  # noqa: E402
    bodleian_folio_verified,
    bodleian_images_verified,
    tei_description,
)
from src.datasets.merging.institution_tokens import OXFORD_TOKEN, _REGISTRY  # noqa: E402
from src.datasets.merging.merge_shelfmarks import (  # noqa: E402
    DEFAULT_OUT_DIR,
    KTIV_DIR,
    MERGED_JSONL,
    _ktiv_fallback_shelfmarks,
    bodleian_canonical_id,
    fjp_canonical_id,
    ktiv_record_ids,
    load_pgp,
    pgp_canonical_id,
)

DRYRUN_CSV = "oxford_id_map_dryrun.csv"
CSV_FIELDS = [
    "old_canonical_id", "new_canonical_id", "sources", "change_type",
    "line", "shelfmark_display", "note",
]
# KTIV Bodleian item files are named after the holding library.
KTIV_BODLEIAN_GLOB = os.path.join(KTIV_DIR, "ktiv_The_Bodleian_Libraries*.json")
_KNOWN_TOKENS = sorted({token for token, _ in _REGISTRY}, key=len, reverse=True)


def iter_merged(path: str) -> Iterator[Tuple[int, dict]]:
    """Stream ``(line_number, record)`` from a merged JSONL file.

    :param path: Path to ``merged_shelfmarks.jsonl``.
    :returns: Iterator of 1-based line numbers and parsed records.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if raw.strip():
                yield lineno, json.loads(raw)


def institution_of(cid: str) -> str:
    """Return the institution token a canonical id starts with.

    :param cid: A canonical id (``Jerusalem_NLI_Heb_38°11343``).
    :returns: The longest registry token prefixing *cid*, else its first token.
    """
    for token in _KNOWN_TOKENS + ["Harvard"]:
        if cid == token or cid.startswith(token + "_"):
            return token
    return cid.split("_", 1)[0]


def _pick(candidates: List[str], old: str) -> str:
    """Choose one id among a KTIV join page's constituent ids.

    :param candidates: New ids of every constituent.
    :param old: The line's current id.
    :returns: The candidate equal to *old* if any, else the one whose trailing
        digit tokens match *old*'s, else the first.
    """
    if old in candidates or not candidates:
        return old if old in candidates else ""
    old_digits = re.findall(r"\d+", old)[-2:]
    for cand in candidates:
        if re.findall(r"\d+", cand)[-2:] == old_digits:
            return cand
    return candidates[0]


def recompute_id(record: dict, alias: Dict[str, str]) -> Tuple[str, Optional[str]]:
    """Recompute a merged line's canonical id with the current code.

    Precedence follows the line's own display source: PGP, then KTIV, then the
    Bodleian scrape, then FJP (the matched constituent is ``shelfmark_display``
    on FJP-only lines).

    :param record: One merged record.
    :param alias: The (guarded) PGP alias map from :func:`load_pgp`.
    :returns: ``(new_cid, volume_resolution)``; ``volume_resolution`` is set for
        a volume-only KTIV record (``"sys_num"`` / ``"leaf_note"`` /
        ``"unresolved"``).
    """
    sources = record.get("sources") or {}
    old = record["canonical_id"]
    pgp = sources.get("pgp")
    if pgp:
        _, cid = pgp_canonical_id(pgp.get("fragment") or {})
        return alias.get(cid, cid), None
    ktiv = sources.get("ktiv")
    if ktiv:
        sm = ktiv.get("shelf_mark") or ""
        marks = [sm] if sm else _ktiv_fallback_shelfmarks(ktiv)
        ids = [(alias.get(cid, cid), res) for _, cid, res in ktiv_record_ids(ktiv, marks) if cid]
        chosen = _pick([cid for cid, _ in ids], old)
        return chosen, dict(ids).get(chosen)
    bodleian = sources.get("bodleian")
    if bodleian:
        cid = bodleian_canonical_id(bodleian)
        return alias.get(cid, cid), None
    fjp = sources.get("fjp") or []
    if fjp:
        cid = fjp_canonical_id(record.get("shelfmark_display") or "", fjp[0])
        return alias.get(cid, cid), None
    return old, None


def _is_oxford(cid: str) -> bool:
    """Return True for an id under the Oxford token.

    :param cid: Canonical id.
    :returns: Whether *cid* is ``Oxford_Bodleian`` or starts with it.
    """
    return cid == OXFORD_TOKEN or cid.startswith(OXFORD_TOKEN + "_")


def _lost_f_letter(old: str, new: str, display: str) -> bool:
    """Return True when the old id had dropped a Bodleian ``f.`` size letter.

    :param old: Current id.
    :param new: Recomputed id.
    :param display: The line's display shelfmark.
    :returns: Whether the new id restores an ``f`` the old id lacked.
    """
    parsed = ShelfmarkNormalizer.parse_oxford(display)
    return (
        parsed is not None and parsed.letter == "f"
        and f"_{parsed.letter}_{parsed.volume}" in new
        and f"_f_{parsed.volume}" not in old
    )


def ktiv_volume_only_records(ktiv_glob: str = KTIV_BODLEIAN_GLOB) -> List[dict]:
    """List the volume-only Bodleian KTIV records on disk and their new ids.

    :param ktiv_glob: Glob of KTIV Bodleian item files (read only).
    :returns: One dict per file whose shelfmark names only a volume:
        ``shelf_mark``, ``sys_num``, ``new_canonical_id``, ``resolved_by``,
        ``mtime``.
    """
    out: List[dict] = []
    for path in sorted(glob.glob(ktiv_glob)):
        if path.endswith("_transcription.json"):
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        sm = doc.get("shelf_mark") or ""
        parsed = ShelfmarkNormalizer.parse_oxford(sm)
        if parsed is None or parsed.leaf:
            continue
        (_, cid, resolution), = ktiv_record_ids(doc, [sm])
        out.append({
            "shelf_mark": sm,
            "sys_num": doc.get("sys_num"),
            "new_canonical_id": cid,
            "resolved_by": resolution,
            "mtime": os.path.getmtime(path),
        })
    return out


def build_id_map(merged_path: str) -> Tuple[List[dict], dict]:
    """Recompute every merged line's id and classify the change.

    :param merged_path: Path to the current ``merged_shelfmarks.jsonl``.
    :returns: ``(rows, summary)``: CSV rows (Oxford lines and every line whose
        id changes) and the summary dict.
    """
    _, alias, alias_skipped = load_pgp()
    lines: List[dict] = []
    for lineno, record in iter_merged(merged_path):
        old = record["canonical_id"]
        new, volume_resolution = recompute_id(record, alias)
        sources = record.get("sources") or {}
        bodleian = sources.get("bodleian")
        lines.append({
            "line": lineno,
            "old": old,
            "new": new,
            "sources": "+".join(record.get("sources_present") or []),
            "display": record.get("shelfmark_display") or "",
            "volume_resolution": volume_resolution,
            "ktiv_shelf_mark": (sources.get("ktiv") or {}).get("shelf_mark"),
            "ktiv_sys_num": (sources.get("ktiv") or {}).get("sys_num"),
            "bodleian_folio_verified": bodleian_folio_verified(bodleian) if bodleian else None,
            "bodleian_images_verified": bodleian_images_verified(bodleian) if bodleian else None,
            "bodleian_has_images": bool(bodleian and bodleian.get("images")),
            "tei_text_used": bool(
                bodleian and record.get("description")
                and record.get("description") == tei_description(bodleian.get("tei"))
            ),
        })

    by_new: Dict[str, List[dict]] = collections.defaultdict(list)
    for ln in lines:
        by_new[ln["new"]].append(ln)
    old_ids = {ln["old"] for ln in lines}

    rows: List[dict] = []
    change_counts: collections.Counter = collections.Counter()
    moved_to: collections.Counter = collections.Counter()
    f_fixes: List[dict] = []
    pgp_exceptions: List[dict] = []
    non_oxford_mismatch: List[dict] = []
    pgp_total = pgp_same = non_oxford_total = 0
    for ln in lines:
        old, new = ln["old"], ln["new"]
        note = ""
        if institution_of(new) != institution_of(old):
            change = "moved_institution"
            moved_to[institution_of(new)] += 1
            if len(by_new[new]) > 1 or (new in old_ids and new != old):
                note = "joins_existing_line"
        elif ln["volume_resolution"] in ("sys_num", "leaf_note"):
            change = "kept_distinct"
            note = f"volume_only:{ln['volume_resolution']}"
        elif new == old:
            change = "unchanged"
        elif len(by_new[new]) > 1:
            change = "collapsed_into"
        else:
            change = "renamed"
        change_counts[change] += 1

        if "pgp" in ln["sources"].split("+") and _is_oxford(old):
            pgp_total += 1
            if new == old:
                pgp_same += 1
            else:
                pgp_exceptions.append({"old": old, "new": new, "shelfmark": ln["display"]})
        if not _is_oxford(old) and not _is_oxford(new):
            non_oxford_total += 1
            if new != old:
                non_oxford_mismatch.append({"line": ln["line"], "old": old, "new": new})
        if _lost_f_letter(old, new, ln["display"]):
            f_fixes.append({"old": old, "new": new})
        if _is_oxford(old) or _is_oxford(new) or new != old:
            rows.append({
                "old_canonical_id": old, "new_canonical_id": new,
                "sources": ln["sources"], "change_type": change,
                "line": ln["line"], "shelfmark_display": ln["display"], "note": note,
            })

    # Volume-only KTIV records the current merge collapsed onto one line.
    volume_only = ktiv_volume_only_records()
    merged_mtime = os.path.getmtime(merged_path)
    old_by_shelf = {
        ln["ktiv_shelf_mark"]: ln for ln in lines
        if ln["ktiv_shelf_mark"] and _is_oxford(ln["old"])
    }
    extra_rows = 0
    seen = set()
    for rec in volume_only:
        host = old_by_shelf.get(rec["shelf_mark"])
        # (1)-copies share a sys_num: one row per distinct item.
        if host is None or rec["sys_num"] == host["ktiv_sys_num"] or rec["new_canonical_id"] in seen:
            continue
        seen.add(rec["new_canonical_id"])
        rows.append({
            "old_canonical_id": host["old"], "new_canonical_id": rec["new_canonical_id"],
            "sources": "ktiv", "change_type": "kept_distinct", "line": "",
            "shelfmark_display": rec["shelf_mark"],
            "note": "new_since_merge" if rec["mtime"] > merged_mtime else "dropped_by_current_merge",
        })
        extra_rows += 1

    oxford_new = collections.defaultdict(list)
    for ln in lines:
        if _is_oxford(ln["new"]):
            oxford_new[ln["new"]].append(ln)
    groups = {cid: grp for cid, grp in oxford_new.items() if len(grp) > 1}
    group_sizes = collections.Counter(len(g) for g in groups.values())
    group_sources = collections.Counter(
        " | ".join(sorted(g["sources"] for g in grp)) for grp in groups.values()
    )
    bod_lines = [ln for ln in lines if ln["bodleian_folio_verified"] is not None]
    distinct_vol_ids = {r["new_canonical_id"] for r in volume_only}

    summary = {
        "merged_file": merged_path,
        "lines_total": len(lines),
        "oxford_lines_current": sum(1 for ln in lines if _is_oxford(ln["old"])),
        "oxford_lines_after": len(oxford_new),
        "csv_rows": len(rows),
        "change_type_counts": dict(change_counts.most_common()),
        "physical_leaves_collapsing": len(groups),
        "surplus_ids_removed": sum(len(g) - 1 for g in groups.values()),
        "collapse_group_sizes": dict(sorted(group_sizes.items())),
        "collapse_group_sources": dict(group_sources.most_common()),
        "moved_out_of_oxford": dict(moved_to.most_common()),
        "f_letter_fixes": {"count": len(f_fixes), "ids": f_fixes},
        "ktiv_volume_only": {
            "records_on_disk": len(volume_only),
            "distinct_sys_nums": len({r["sys_num"] for r in volume_only}),
            "distinct_new_ids": len(distinct_vol_ids),
            "lines_in_current_merge": sum(
                1 for ln in lines if ln["volume_resolution"] in ("sys_num", "leaf_note")
            ),
            "extra_rows_for_dropped_records": extra_rows,
            "resolved_by": dict(collections.Counter(r["resolved_by"] for r in volume_only)),
        },
        "bodleian_precedence": {
            "lines_with_bodleian": len(bod_lines),
            "not_folio_verified": sum(1 for ln in bod_lines if not ln["bodleian_folio_verified"]),
            "images_no_longer_preferred": sum(
                1 for ln in bod_lines
                if ln["bodleian_has_images"] and not ln["bodleian_images_verified"]
            ),
            "tei_text_now_dropped": sum(
                1 for ln in bod_lines if ln["tei_text_used"] and not ln["bodleian_folio_verified"]
            ),
        },
        "pgp_alias_skipped": collections.Counter(s["reason"] for s in alias_skipped),
        "regression_pgp_oxford": {
            "lines": pgp_total,
            "identical": pgp_same,
            "exceptions": pgp_exceptions,
        },
        "regression_non_oxford": {
            "lines": non_oxford_total,
            "identical": non_oxford_total - len(non_oxford_mismatch),
            "mismatches": non_oxford_mismatch[:200],
        },
    }
    return rows, summary


def write_csv(rows: List[dict], path: str) -> None:
    """Write the dry-run id map CSV.

    :param rows: Rows from :func:`build_id_map`.
    :param path: Output CSV path.
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point: build the map, write the CSV, print the summary."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=os.path.join(DEFAULT_OUT_DIR, MERGED_JSONL),
                        help="current merged_shelfmarks.jsonl (read only)")
    parser.add_argument("--out", default=os.path.join(DEFAULT_OUT_DIR, DRYRUN_CSV),
                        help="dry-run CSV to write")
    parser.add_argument("--summary-out", default=None,
                        help="optional path for the JSON summary")
    args = parser.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.merged):
        parser.error("--out must not overwrite the merged file")

    rows, summary = build_id_map(args.merged)
    write_csv(rows, args.out)
    summary["csv"] = args.out
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
