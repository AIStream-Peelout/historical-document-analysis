"""Dry-run an edition-led manuscript assembly, preserving source and placement uncertainty for review."""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from rapidfuzz.distance import Levenshtein

from src.datasets.indexing.legacy_store.pdf_transcriptions import file_sha256
from src.finetuning.qwen_hebrew.ja_ar_build import edition_lines as el, textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.run_gate import load_candidates, load_streams


def view_key(placement: dict) -> tuple[str, int, tuple]:
    """Identify a photographic view in the row builder's crop-then-rotate convention.

    :param placement: A previous pair or an edition placement.
    :return: Photograph, rotation and crop.
    """
    return placement["image_sha256"], placement.get("rotation") or 0, tuple(placement.get("crop") or ())


def compare_views(edition: dict, group: dict, lines: list[dict], prior: list[dict]) -> list[dict]:
    """Compare each edition and exact view with its former answer, preserving source-only and prior-only cases.

    :param edition: Edition record.
    :param group: Manuscript record with its photographs.
    :param lines: Every printed baseline.
    :param prior: Previous assembly pairs for the same edition.
    :return: Per-view draft, paired comparison and unresolved line references.
    """
    by_view = defaultdict(list)
    for line in lines:
        if line["placement"]:
            by_view[view_key(line["placement"])].append(line)
    previous = defaultdict(list)
    for p in prior:
        if p.get("image_sha256") and p.get("lines") and p["decision"] in {"review", "accept"} and not p.get("dup_rank"):
            previous[view_key(p)].append(p)
    unplaced = [l["line_id"] for l in lines if not l["placement"] and l["kind"] == "source_or_gap"]
    photos = {im["sha256"]: im for im in group["photos"]}
    views = []
    for key in sorted(set(by_view) | set(previous)):
        photo, rotation, crop = key
        placed = by_view[key]
        usable = [l for l in placed if l["text"]]
        previous_pairs = previous[key]
        source_text = "\n".join(l["text"] for l in usable)
        old_text = [p["lines"] for p in previous_pairs]
        # Multiple former answers on the same view are alternatives, never a concatenated reference.
        comparisons = []
        new = Counter(textnorm.skeleton(l["text"]) for l in usable)
        for p, former in zip(previous_pairs, old_text):
            old = Counter(textnorm.skeleton(s) for s in former)
            remaining = new.copy()
            unmatched = []
            for i, s in enumerate(former):
                skel = textnorm.skeleton(s)
                if remaining[skel]:
                    remaining[skel] -= 1
                else:
                    unmatched.append({"index": i, "text": s})
            # A recovered line break alone changes no letters in this alignment.
            edits = Counter(op.tag for op in Levenshtein.editops(textnorm.skeleton("".join(former)),
                                                                 textnorm.skeleton(source_text)))
            comparisons.append({"pair_id": p["pair_id"], "old_lines": len(former), "old_reasons": p.get("reasons", []),
                                "old_decision": p["decision"],
                                "old_to_draft_letter_edits": dict(edits), "previous_lines_to_reconcile": unmatched,
                                "old_letters": sum(textnorm.letters(s) for s in former),
                                "same_letter_sequence_lines": sum((new & old).values()),
                                "new_or_changed_lines": sum((new-old).values()),
                                "prior_only_or_changed_lines": sum((old-new).values())})
        flags = {f for l in placed for f in l["flags"]} | {"verify_leaf_order_and_photo_coverage"}
        if unplaced:
            flags.add("edition_has_unplaced_lines")
        if len(usable) < len(placed):
            flags.add("placed_line_lacks_reading")
        views.append({"row_id": edition["row_id"], "document_group": group["document_group"],
                      "source": edition["source"], "ja_signal": textnorm.ja_signal(source_text),
                      "script": textnorm.script_family(source_text) if source_text else None,
                      "rnl_train_only": any(e["rnl"] for e in group["editions"]),
                      "image_sha256": photo, "image_path": photos[photo]["path"], "rotation": rotation,
                      "crop": list(crop) or None, "decision": "review", "release_approved": False,
                      "line_ids": [l["line_id"] for l in placed], "draft_line_ids": [l["line_id"] for l in usable],
                      "draft_text": source_text, "draft_sha256": el.digest(source_text),
                      "placed_without_reading": [l["line_id"] for l in placed if not l["text"]],
                      "unplaced_edition_lines": unplaced, "placement_methods": dict(Counter(l["placement"]["method"] for l in placed)),
                      "flags": sorted(flags),
                      "previous": comparisons})
    return views


def write_records(path: Path, rows: Iterable[dict]) -> None:
    """Write a private JSONL artifact.

    :param path: New destination file.
    :param rows: Records to serialize.
    """
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> dict:
    """Prepare a reproducible edition-line dry run and publish it only after all checks succeed.

    :param args: Parsed CLI parameters.
    :return: Run summary.
    """
    if args.out.exists():
        raise FileExistsError(args.out)
    inventory = load_candidates([args.inventory])
    priority = {r["document_group"] for r in load_candidates([args.priority])} if args.priority else set()
    known_groups = {d["document_group"] for d in inventory}
    if len(known_groups) != len(inventory) or priority - known_groups:
        raise ValueError("Duplicate manuscript or unknown priority document")
    docs = sorted(inventory, key=lambda d: (d["document_group"] not in priority, d["document_group"]))
    by_edition = defaultdict(list)
    for p in load_candidates([args.previous_pairs]):
        if p["document_group"] in known_groups:
            by_edition[p["row_id"]].append(p)
    bindings = defaultdict(list)
    for b in load_candidates([args.bindings]) if args.bindings else []:
        bindings[b["row_id"]].append(b)
    streams = load_streams(args.reads_dir)
    ledgers, views, doc_results = [], [], []
    verified_photos = set()
    for d in docs:
        group = d["document_group"]
        result = {"document_group": group, "documentary_lead": group in priority,
                  "overlap_witnesses": d["overlap_witnesses"], "photos": [p["sha256"] for p in d["photos"]],
                  "edition_ids": [e["row_id"] for e in d["editions"]], "missing_pdfs": []}
        doc_results.append(result)
        if d["overlap_witnesses"]:
            continue
        photos = {p["sha256"]: p for p in d["photos"]}
        for sha, p in photos.items():
            if (p["path"], sha) not in verified_photos:
                if file_sha256(p["path"]) != sha:
                    raise ValueError(f"Manuscript photograph changed: {p['path']}")
                verified_photos.add((p["path"], sha))
        group_streams = [s for sha in photos for s in streams.get(sha, [])]
        for edition in d["editions"]:
            rid = edition["row_id"]
            names = {s["pointer"]["file"] for s in edition["sides"]}
            if edition.get("pdf_file"):
                names.add(edition["pdf_file"])
            if len(names) != 1:
                raise ValueError(f"Cannot identify a single source PDF for {rid}: {names}")
            name, = names
            if not name or Path(name).name != name:
                raise ValueError(f"Invalid source filename for {rid}")
            paths = [root / name for root in args.pdf_dir if (root / name).is_file()]
            if not paths:
                result["missing_pdfs"].append(name)
                continue
            pdf_sha, lines = el.printed_lines(paths[0], edition)
            if any(file_sha256(str(p)) != pdf_sha for p in paths[1:]):
                raise ValueError(f"Conflicting source PDFs named {name}")
            el.apply_bindings(lines, bindings.pop(rid, []), pdf_sha, photos)
            el.place_lines(lines, group_streams)
            el.previous_locations(lines, by_edition[rid])
            el.place_block_edges(lines)
            el.place_labelled_folios(lines, photos, edition)
            for line in lines:
                if line["kind"] == "source_or_gap" and not line["placement"]:
                    line["flags"].append("unplaced_source")
            ledgers.append({"document_group": group, "row_id": rid, "pdf_path": str(paths[0]),
                            "pdf_sha256": pdf_sha, "rnl_train_only": edition["rnl"], "lines": lines})
            views.extend(compare_views(edition, d, lines, by_edition[rid]))
    if bindings:
        raise ValueError(f"Unconsumed source placements: {sorted(bindings)}")
    methods = Counter(l["placement"]["method"] if l["placement"] else "unplaced" for e in ledgers
                      for l in e["lines"] if l["kind"] == "source_or_gap" or l["placement"])
    summary = {"manuscripts": len(docs), "overlap_holds": sum(bool(d["overlap_witnesses"]) for d in docs),
               "priority_groups": len(priority), "pdfs_read": len(ledgers),
               "missing_pdfs": sum(len(d["missing_pdfs"]) for d in doc_results),
               "physical_baselines": sum(len(e["lines"]) for e in ledgers), "placement_methods": dict(methods),
               "views_with_draft": sum(bool(v["draft_text"]) for v in views), "release_approved": 0,
               "inputs": {k: {"path": str(p), "sha256": file_sha256(str(p))}
                          for k, p in (("inventory", args.inventory), ("priority", args.priority),
                                       ("previous_pairs", args.previous_pairs), ("bindings", args.bindings)) if p},
               "read_caches": [{"path": str(p / "reads/reads.jsonl"), "sha256": file_sha256(str(p / "reads/reads.jsonl"))}
                               for p in args.reads_dir if (p / "reads/reads.jsonl").exists()],
               "pdf_dirs": list(map(str, args.pdf_dir))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        write_records(stage / "documents.jsonl", doc_results)
        write_records(stage / "editions.jsonl", ledgers)
        write_records(stage / "review_views.jsonl", views)
        (stage / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.out)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the edition-led mode of the assembly command.

    :param argv: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path, help="screened manuscript and edition inventory JSONL")
    parser.add_argument("--pdf-dir", required=True, type=Path, nargs="+", help="directories containing original PDFs")
    parser.add_argument("--previous-pairs", required=True, type=Path, help="existing assembly to compare")
    parser.add_argument("--reads-dir", type=Path, nargs="*", default=[], help="existing read caches for placement only")
    parser.add_argument("--priority", type=Path, help="documentary catalogue leads to process first")
    parser.add_argument("--bindings", type=Path, help="hash-bound edition and photo inspections")
    parser.add_argument("--out", required=True, type=Path, help="new private dry-run output directory")
    print(json.dumps(run(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
